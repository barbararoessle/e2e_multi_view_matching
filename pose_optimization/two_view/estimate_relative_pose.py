import torch
from kornia.geometry.epipolar import normalize_points, normalize_transformation, motion_from_essential, \
    motion_from_essential_choose_solution, triangulate_points, symmetrical_epipolar_distance
from kornia.geometry.epipolar.projection import depth_from_point

from pose_optimization.two_view.bundle_adjust_gauss_newton_2_view import BundleAdjustGaussNewton2View
from pose_optimization.two_view.compute_pose_error import compute_rotation_error, compute_translation_error_as_angle

def normalize(kpts, intr):
    n_kpts = torch.zeros_like(kpts)
    fx, fy, cx, cy = intr[..., 0, 0], intr[..., 1, 1], intr[..., 0, 2], intr[..., 1, 2]
    n_kpts[..., 0] = (kpts[..., 0] - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
    n_kpts[..., 1] = (kpts[..., 1] - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
    return n_kpts

def get_kpts(data, result, id0, id1):
    if "keypoints" + str(id0) in data:
        keypoints0, keypoints1 = data["keypoints" + str(id0)], data["keypoints" + str(id1)]
    else:
        keypoints0, keypoints1 = data["keypoints{}_{}_{}".format(id0, id0, id1)], data["keypoints{}_{}_{}".format(id1, id0, id1)]
    matches = result["matches{}_{}_{}".format(id0, id0, id1)]
    intr0, intr1 = data["intr" + str(id0)], data["intr" + str(id1)]
    bs, n_kpts0, _ = keypoints0.shape
    dev = keypoints0.device
    # determine confidence
    batch_idx0 = torch.arange(bs, device=dev).unsqueeze(-1).expand(bs, n_kpts0)
    conf_key = "conf_scores_{}_{}".format(id0, id1)
    confidence = result[conf_key]
    confidence = (matches >= 0).float().unsqueeze(-1) * confidence
    keypoints1 = keypoints1[batch_idx0, matches]
    return keypoints0, keypoints1, intr0, intr1, confidence

# adapted from https://github.com/kornia/kornia
def find_fundamental(points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    r"""Compute the fundamental matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.

    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape, points2.shape)
    if not (len(weights.shape) == 2 and weights.shape[1] == points1.shape[1]):
        raise AssertionError(weights.shape)

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = torch.ones_like(x1)

    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]

    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)  # BxNx9

    # apply the weights to the linear system
    w_diag = torch.diag_embed(weights)
    X = w_diag @ X

    # compute eigevectors and retrieve the one with the smallest eigenvalue
    _, _, V = torch.svd(X)
    F_mat = V[..., -1].view(-1, 3, 3)

    # reconstruct and force the matrix to have rank2
    U, S, V = torch.svd(F_mat)
    rank_mask = torch.tensor([1.0, 1.0, 0.0], device=F_mat.device, dtype=F_mat.dtype)

    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
    F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)

    return normalize_transformation(F_est)

def estimate_relative_pose_w8pt(kpts0, kpts1, intr0, intr1, confidence, choose_closest=False, T_021=None, determine_inliers=False):
    if kpts0.shape[1] < 8:
        return None, None
    sum_conf = confidence.sum(dim=1, keepdim=True) + 1e-6
    confidence = confidence / sum_conf
    kpts0_norm = normalize(kpts0, intr0)
    kpts1_norm = normalize(kpts1, intr1)
    dev = intr0.device
    bs = intr0.shape[0]
    intr = torch.eye(3, device=dev).unsqueeze(0)
    Fs = find_fundamental(kpts0_norm, kpts1_norm, confidence.squeeze(-1))
    if choose_closest:
        Rs, ts = motion_from_essential(Fs)
        min_err = torch.full((bs,), 1e6, device=dev)
        min_err_pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
        for R, t in zip(Rs.permute(1, 0, 2, 3), ts.permute(1, 0, 2, 3)):
            pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
            pred_T021[:, :3, :3] = R
            pred_T021[:, :3, 3] = t.squeeze(-1)

            curr_err = compute_rotation_error(pred_T021, T_021, reduce=False) + compute_translation_error_as_angle(pred_T021, T_021, reduce=False)
            update_mask = curr_err < min_err
            min_err[update_mask] = curr_err[update_mask]
            min_err_pred_T021[update_mask] = pred_T021[update_mask]
    else:
        R, t, pts = motion_from_essential_choose_solution(Fs, intr, intr, kpts0_norm, kpts1_norm, mask=None)
        min_err_pred_T021 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
        min_err_pred_T021[:, :3, :3] = R
        min_err_pred_T021[:, :3, 3] = t.squeeze(-1)
    # check for positive depth
    P0 = torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)[:, :3, :]
    pts_3d = triangulate_points(P0, min_err_pred_T021[:, :3, :], kpts0_norm, kpts1_norm)
    depth0 = pts_3d[..., -1]
    depth1 = depth_from_point(min_err_pred_T021[:, :3, :3], min_err_pred_T021[:, :3, 3:], pts_3d)
    pos_depth_mask = torch.logical_and(depth0 > 0., depth1 > 0.)
    epi_err = None
    inliers = None
    if determine_inliers:
        epi_err = symmetrical_epipolar_distance(kpts0_norm, kpts1_norm, Fs)
        epi_err_sqrt = epi_err.sqrt()
        thresh = 3. / ((intr0[:, 0, 0] + intr0[:, 1, 1] + intr1[:, 0, 0] + intr1[:, 1, 1]) / 4.)
        inliers = torch.logical_and(pos_depth_mask, epi_err_sqrt <= thresh.unsqueeze(-1))
    info = {"kpts0_norm": kpts0_norm, "kpts1_norm": kpts1_norm, "confidence": confidence, "inliers": inliers, \
        "pos_depth_mask": pos_depth_mask}
    return min_err_pred_T021, info

def run_weighted_8_point(data, result, id0, id1, choose_closest=False, target_T_021=None):
    match_key = "matches{}_{}_{}".format(id0, id0, id1)
    if match_key in result and result[match_key].shape[1] != 0:
        kpts0, kpts1, intr0, intr1, confidence = get_kpts(data, result, id0, id1)
        return estimate_relative_pose_w8pt(kpts0, kpts1, intr0, intr1, confidence, choose_closest=choose_closest, T_021=target_T_021)
    else:
        return None, None

def run_bundle_adjust_2_view(kpts0_norm, kpts1_norm, confidence, init_T021, n_iterations, check_lu_info_strict=False, \
        check_precond_strict=False):
    bs = kpts0_norm.shape[0]
    ba = BundleAdjustGaussNewton2View(batch_size=bs, n_iterations=n_iterations, check_lu_info_strict=check_lu_info_strict, \
                                      check_precond_strict=check_precond_strict)
    extrinsics, valid_batch = ba.run(kpts0_norm, kpts1_norm, confidence.squeeze(-1), init_T021)
    return extrinsics[:, 1], valid_batch