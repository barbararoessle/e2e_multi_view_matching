from itertools import compress
import coloredlogs, logging
coloredlogs.install()

import torch
import pytorch3d
from pytorch3d import transforms
import kornia

class Observations(object):
    def __init__(self, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.dev = device
        self.reset()
    
    def reset(self):
        self.observation_ids = [torch.zeros((0, 2), dtype=int, device=self.dev) for _ in range(self.batch_size)] # img_id, pt_id
        self.observation_pt_conf = [torch.zeros((0, 3), device=self.dev) for _ in range(self.batch_size)] # x, y, conf

    def add_matches(self, id0, id1, kpts0, kpts1, confidence, valid, n_matches):
        for i_batch in range(self.batch_size):
            n_matches_batch = n_matches[i_batch]
            img_ids = torch.tensor((id0, id1), device=self.dev).repeat_interleave(n_matches_batch)
            pt_id_start = int(self.observation_pt_conf[i_batch].shape[0] // 2)
            pt_ids = torch.arange(pt_id_start, pt_id_start + n_matches_batch, device=self.dev).repeat(2)
            ids = torch.stack((img_ids, pt_ids), 1)
            self.observation_ids[i_batch] = torch.cat((self.observation_ids[i_batch], ids), 0)
            mask = valid[i_batch]
            pts_batch = torch.cat((kpts0[i_batch][mask], kpts1[i_batch][mask]), 0)
            conf_batch = confidence[i_batch][mask].repeat(2)
            observation_pt_conf_batch = torch.cat((pts_batch, conf_batch.unsqueeze(-1)), 1)
            self.observation_pt_conf[i_batch] = torch.cat((self.observation_pt_conf[i_batch], observation_pt_conf_batch), 0)

    def get_n_obs(self):
        return torch.tensor([len(ids) for ids in self.observation_ids], device=self.dev)
    
    def get_obs_pts(self):
        return self.observation_pt_conf

    def get_obs_ids(self):
        return self.observation_ids

    def normalize_confidences(self):
        for i_batch in range(self.batch_size):
            conf = self.observation_pt_conf[i_batch][:, 2:].clone()
            sum_conf = conf.sum(dim=0, keepdim=True).clamp(min=1e-6)
            self.observation_pt_conf[i_batch][:, 2:] = conf / (0.5 * sum_conf) # 0.5 because each match leads to 2 observations

def fill_J(J_cam_unfold, J_pts_unfold, extrinsics, points_3d, cam_ids, pt_3d_ids, conf, n_o, dev):
    without_0th_cam = cam_ids != 0
    Ap = torch.matmul(extrinsics[cam_ids, :3, :4], torch.cat((points_3d[pt_3d_ids], torch.ones((n_o, 1), device=dev)), -1).unsqueeze(-1)).squeeze(-1)
    piAp = Ap[:, :2] / Ap[:, 2].unsqueeze(-1)
    J_proj = torch.zeros((n_o, 2, 3), device=dev, dtype=J_cam_unfold.dtype)
    J_proj[:, 0, 0] = 1. / Ap[:, 2]
    J_proj[:, 0, 2] = -Ap[:, 0] / Ap[:, 2].pow(2)
    J_proj[:, 1, 1] = 1. / Ap[:, 2]
    J_proj[:, 1, 2] = -Ap[:, 1] / Ap[:, 2].pow(2)
    # fill J points 3d
    i_obs = torch.arange(n_o, device=dev)
    J_pts_unfold[i_obs, pt_3d_ids] = conf.unsqueeze(-1).unsqueeze(-1) * torch.matmul(J_proj, extrinsics[cam_ids, :3, :3])
    # fill J cam
    Ap_hat = pytorch3d.transforms.so3.hat(Ap[without_0th_cam])
    I_Ap_hat = torch.cat((torch.eye(3, device=dev).unsqueeze(0).expand(Ap_hat.shape[0], 3, 3), -Ap_hat), 2)
    J_proj_I_Ap_hat = torch.matmul(J_proj[without_0th_cam], I_Ap_hat)
    J_cam_unfold[i_obs[without_0th_cam], cam_ids[without_0th_cam] - 1] = conf[without_0th_cam].unsqueeze(-1).unsqueeze(-1) * J_proj_I_Ap_hat
    return piAp

def compute_A_b(extrinsics, points_3d, obs_ids, obs_pts_conf, n_obs, n_imgs):
    dev = extrinsics.device
    A = []
    b = []
    r_norm = []
    for extr, pts_3d, ids, pts, n_o in zip(extrinsics, points_3d, obs_ids, obs_pts_conf, n_obs):
        n_o = int(n_o)
        unknown_cam = (n_imgs - 1) * 6
        unknown_pts_3d = 3 * int(n_o // 2)
        unknown = unknown_cam + unknown_pts_3d
        if n_o > 0:
            J_cam = torch.zeros((n_o * 2, unknown_cam), device=dev, dtype=extrinsics.dtype)
            J_pts = torch.zeros((n_o * 2, unknown_pts_3d), device=dev, dtype=extrinsics.dtype)
            J_cam_unfold = J_cam.unfold(0, 2, 2).unfold(1, 6, 6)
            J_pts_unfold = J_pts.unfold(0, 2, 2).unfold(1, 3, 3)
            cam_ids = ids[:, 0]
            pt_3d_ids = ids[:, 1]
            kpt_coords = pts[:, :2]
            conf = pts[:, 2]
            piAp = fill_J(J_cam_unfold, J_pts_unfold, extr, pts_3d, cam_ids, pt_3d_ids, conf, n_o, dev)
            r = (conf.unsqueeze(-1) * (piAp - kpt_coords)).view(n_o * 2, 1)
            J = torch.cat((J_cam, J_pts), 1)
            JT = torch.transpose(J, 0, 1)
            A.append(torch.matmul(JT, J))
            b.append(torch.matmul(-JT, r))
            r_norm.append(r.pow(2).sum())
        else:
            A.append(torch.eye(unknown, device=dev))
            b.append(torch.zeros((unknown, 1), device=dev))
            r_norm.append(torch.zeros(device=dev))
    return A, b, r_norm

class BundleAdjustGaussNewton2View(object):
    def __init__(self, batch_size, n_iterations, jacobi_precond=True, check_lu_info_strict=False, check_precond_strict=False, \
            vary_lm_fact=True, lm_increase=1.5, lm_decrease=3.5):
        super().__init__()
        self.n_imgs = 2
        self.bs = batch_size
        self.n_it = n_iterations
        self.jacobi_precond = jacobi_precond
        self.check_lu_info_strict = check_lu_info_strict
        self.check_precond_strict = check_precond_strict
        self.vary_lm_fact = vary_lm_fact
        self.lm_increase = lm_increase
        self.lm_decrease = lm_decrease
    
    def triangulate_points(self, extrinsics, n_kpts0, n_kpts1, valid, valid_batch):
        valid_batch_ids = torch.arange(self.bs)[valid_batch]
        points_3d = [torch.zeros((0, 3), device=extrinsics.device, dtype=extrinsics.dtype) for _ in valid_batch_ids]
        for i_pose, i_b in enumerate(valid_batch_ids):
            id0 = 0
            id1 = 1
            mask = valid[i_b]
            pts_3d = kornia.geometry.epipolar.triangulate_points(extrinsics[i_pose, id0, :3], extrinsics[i_pose, id1, :3], \
                n_kpts0[i_b][mask], n_kpts1[i_b][mask])
            points_3d[i_pose] = torch.cat((points_3d[i_pose], pts_3d), 0)
        return points_3d

    def run(self, n_kpts0, n_kpts1, conf, extr1):
        dev = n_kpts0.device
        observations = Observations(self.bs, dev)
        valid = conf.detach().clone() > 0.
        n_matches = valid.sum(-1)
        observations.add_matches(0, 1, n_kpts0, n_kpts1, conf, valid, n_matches)
        observations.normalize_confidences()
        valid_batch = n_matches > 6
        n_valid_batches = valid_batch.sum()
        if n_valid_batches != self.bs:
            logging.warning("{} batches are excluded, not enough matches".format( \
                self.bs - n_valid_batches))
        n_obs = observations.get_n_obs()[valid_batch]
        extrinsics = torch.eye(4, device=dev).unsqueeze(0).unsqueeze(0).repeat(n_valid_batches, self.n_imgs, 1, 1)
        # init
        extrinsics[:, 1] = extr1[valid_batch]
        best_extrinsics = torch.eye(4, device=dev).unsqueeze(0).unsqueeze(0).repeat(n_valid_batches, self.n_imgs, 1, 1)

        # triangulate points
        points_3d = self.triangulate_points(extrinsics, n_kpts0, n_kpts1, valid, valid_batch)
        obs_ids = list(compress(observations.get_obs_ids(), valid_batch))
        obs_pts = list(compress(observations.get_obs_pts(), valid_batch))

        update_batch = torch.ones(n_valid_batches, device=dev, dtype=bool)
        damp_fact = torch.full((n_valid_batches,), 0.1, device=dev)
        for i in range(self.n_it + 1):
            # compute A, b
            A, b, r_norm = compute_A_b(extrinsics, points_3d, obs_ids, obs_pts, n_obs, self.n_imgs)
            r_norm = torch.tensor(r_norm, device=dev)
            if i == 0:
                best_r_norm = r_norm
                best_extrinsics[:] = extrinsics[:] # init anyway
            else:
                best_pose_mask = torch.logical_and(r_norm < best_r_norm, update_batch)
                best_r_norm[best_pose_mask] = r_norm[best_pose_mask]
                best_extrinsics[best_pose_mask] = extrinsics[best_pose_mask]
                if self.vary_lm_fact:
                    damp_fact[best_pose_mask] = damp_fact[best_pose_mask] / self.lm_decrease
                    damp_fact[torch.logical_not(best_pose_mask)] = damp_fact[torch.logical_not(best_pose_mask)] * self.lm_increase
            if i == self.n_it:
                break
            for i_b, (A_batch, b_batch) in enumerate(zip(A, b)):
                if not update_batch[i_b]:
                    continue
                if self.jacobi_precond:
                    diag_A = torch.diagonal(A_batch, dim1=-2, dim2=-1)
                    if (diag_A > 0.).all(-1):
                        diag_A = diag_A.clamp(min=1e-12)
                        inv_M = torch.diag_embed(1. / diag_A)
                        A_batch = inv_M @ A_batch
                        b_batch = inv_M @ b_batch
                    else:
                        logging.warn("Preconditioning failed at iteration {} due to 0 element in diagonal".format(i))
                        if self.check_precond_strict:
                            update_batch[i_b] = False
                            continue
                scale_mat = torch.eye(A_batch.shape[0], dtype=A_batch.dtype, device=A_batch.device)
                A_batch = A_batch + scale_mat * damp_fact[i_b]
                A_lu, pivots, info = torch.lu(A_batch, get_infos=True)
                if self.check_lu_info_strict:
                    update_batch[i_b] = torch.logical_and(info == 0, update_batch[i_b])
                if info == 0 and update_batch[i_b]:
                    # solve linear system
                    delta_x = torch.lu_solve(b_batch, A_lu, pivots).squeeze(-1)
                    # update extrinsics
                    unknown_cam = (self.n_imgs - 1) * 6
                    delta_extr = delta_x[:unknown_cam].view(-1, 6)
                    delta_extr = pytorch3d.transforms.se3_exp_map(delta_extr).permute(0, 2, 1)
                    delta_extr = delta_extr.view(self.n_imgs - 1, 4, 4)
                    extrinsics[i_b, 1:] = delta_extr @ extrinsics[i_b, 1:].clone()
                    # update points 3d
                    delta_pts_3d = delta_x[unknown_cam:].view(-1, 3)
                    points_3d[i_b] = points_3d[i_b] + delta_pts_3d

        return best_extrinsics, valid_batch
