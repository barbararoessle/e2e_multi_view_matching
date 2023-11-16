import logging

import numpy as np
import torch
import cv2
from scipy.sparse.csgraph import minimum_spanning_tree

from models.models.utils import estimate_pose
from pose_optimization.two_view.estimate_relative_pose import run_bundle_adjust_2_view, estimate_relative_pose_w8pt, \
    normalize

def estimate_relative_pose_w8pt_ba(intr0, intr1, mkpts0, mkpts1, conf):
    pred_T021, info = estimate_relative_pose_w8pt(torch.from_numpy(mkpts0).cuda().unsqueeze(0), torch.from_numpy(mkpts1).cuda().unsqueeze(0), \
        torch.from_numpy(intr0).cuda().unsqueeze(0), torch.from_numpy(intr1).cuda().unsqueeze(0), torch.from_numpy(conf).cuda().unsqueeze(0), \
        determine_inliers=True)
    if pred_T021 is None:
        return False, None, None, None
    confidence = info["confidence"]
    confidence[torch.logical_not(info["pos_depth_mask"])] = 0.
    pred_T021_refine, valid_refine = run_bundle_adjust_2_view(info["kpts0_norm"], info["kpts1_norm"], confidence, pred_T021, \
        n_iterations=10)
    pred_T021[valid_refine] = pred_T021_refine
    return True, pred_T021[0, :3, :3].cpu().numpy(), pred_T021[0, :3, 3].cpu().numpy(), info["inliers"].squeeze(0).cpu().numpy()

def estimate_relative_pose_ransac_ba(intr0, intr1, mkpts0, mkpts1, conf):
    ret = estimate_pose(mkpts0, mkpts1, intr0, intr1, thresh=1.0)
    if ret is None:
        return False, None, None, None
    else:
        inlier_mask = ret[2]
        pred_T021 = torch.eye(4).unsqueeze(0).cuda()
        pred_T021[0, :3, 3] = torch.from_numpy(ret[1]).cuda().unsqueeze(0)
        pred_T021[0, :3, :3] = torch.from_numpy(ret[0]).cuda().unsqueeze(0)
        confidence = torch.from_numpy(conf[inlier_mask]).cuda().unsqueeze(0)
        intr0 = torch.from_numpy(intr0).cuda().unsqueeze(0)
        intr1 = torch.from_numpy(intr1).cuda().unsqueeze(0)
        kpts0_norm = normalize(torch.from_numpy(mkpts0[inlier_mask]).cuda().unsqueeze(0), intr0)
        kpts1_norm = normalize(torch.from_numpy(mkpts1[inlier_mask]).cuda().unsqueeze(0), intr1)
        pred_T021_refine, valid_refine = run_bundle_adjust_2_view(kpts0_norm, kpts1_norm, confidence, pred_T021, \
                n_iterations=10)
        pred_T021[valid_refine] = pred_T021_refine
        return True, pred_T021[0, :3, :3].cpu().numpy(), pred_T021[0, :3, 3].cpu().numpy(), inlier_mask

def estimate_relative_pose_ransac(intr0, intr1, mkpts0, mkpts1):
    success = True
    R = None
    t = None
    inliers = None
    ret = estimate_pose(mkpts0, mkpts1, intr0, intr1, thresh=1.0)
    if ret is not None:
        R, t, inliers = ret
    else:
        success = False
    return success, R, t, inliers

def normalize_confidences(obs_xyc):
    conf = obs_xyc[:, 2:]
    sum_conf = conf.sum(axis=0, keepdims=True) + 1e-3
    obs_xyc[:, 2:] = conf / (0.5 * sum_conf) # 0.5 because each match leads to 2 observations
    return obs_xyc

def initialize_bundle_adjust(n_images, data, result, file_path, conf_thresh=0., rel_pose_method="w8pt_ba"):
    min_inliers = 20
    pair_wise_data = dict()
    match_graph = np.zeros((n_images, n_images), dtype=int)
    # collect relevant data
    for id1 in range(n_images):
        for id0 in range(id1):
            matches_key = "matches{}_{}_{}".format(id0, id0, id1)
            if matches_key not in result:
                continue
            # get all matching data
            if "keypoints" + str(id0) in data:
                kpts0, kpts1 = data["keypoints" + str(id0)][0].cpu().numpy(), data["keypoints" + str(id1)][0].cpu().numpy()
            else:
                kpts0, kpts1 = data["keypoints{}_{}_{}".format(id0, id0, id1)][0].cpu().numpy(), \
                    data["keypoints{}_{}_{}".format(id1, id0, id1)][0].cpu().numpy()
            matches = result["matches{}_{}_{}".format(id0, id0, id1)][0].cpu().numpy()
            intr0, intr1 = data["intr" + str(id0)][0].numpy(), data["intr" + str(id1)][0].numpy()
            # get confidences
            conf_key = "conf_scores_{}_{}".format(id0, id1)
            confidence = result[conf_key][0].cpu().numpy()
            
            # determine valid keypoints
            valid = (matches >= 0) & np.all(confidence > conf_thresh, -1)

            pair_wise_data["mkpts{}_{}_{}".format(id0, id0, id1)] = kpts0[valid]
            pair_wise_data["mkpts{}_{}_{}".format(id1, id0, id1)] = kpts1[matches[valid]]
            confidence = confidence[valid]
            pair_wise_data["conf{}_{}_{}".format(id0, id0, id1)] = confidence
            pair_wise_data["conf{}_{}_{}".format(id1, id0, id1)] = confidence
            pair_wise_data["intr{}".format(id0)] = intr0
            pair_wise_data["intr{}".format(id1)] = intr1

    # estimate relative poses
    for id1 in range(n_images):
        for id0 in range(id1):
            matches_key = "mkpts{}_{}_{}".format(id0, id0, id1)
            if matches_key not in pair_wise_data:
                continue
            mkpts0 = pair_wise_data["mkpts{}_{}_{}".format(id0, id0, id1)]
            mkpts1 = pair_wise_data["mkpts{}_{}_{}".format(id1, id0, id1)]
            intr0 = pair_wise_data["intr{}".format(id0)]
            intr1 = pair_wise_data["intr{}".format(id1)]
            if rel_pose_method == "ransac":
                success, R, t, inliers = estimate_relative_pose_ransac(intr0, intr1, mkpts0, mkpts1)
                inlier_count = inliers.sum() if success else 0
            elif rel_pose_method == "w8pt_ba":
                success, R, t, inliers = estimate_relative_pose_w8pt_ba(intr0, intr1, mkpts0, mkpts1, pair_wise_data["conf{}_{}_{}".format(id0, id0, id1)])
                if success:
                    inlier_count = inliers.sum()
                    inliers = np.full_like(inliers, True)
                else:
                    inlier_count = 0
            elif rel_pose_method == "ransac_ba":
                success, R, t, inliers = estimate_relative_pose_ransac_ba(intr0, intr1, mkpts0, mkpts1, pair_wise_data["conf{}_{}_{}".format(id0, id0, id1)])
                inlier_count = inliers.sum() if success else 0
            else:
                logging.error("Relative pose estimation method {} is not defined".format(rel_pose_method))
            
            pair_wise_data["inlier_count{}_{}".format(id0, id1)] = inlier_count
            if success:
                mkpts0 = mkpts0[inliers]
                mkpts1 = mkpts1[inliers]
                pair_wise_data["mkpts{}_{}_{}".format(id0, id0, id1)] = mkpts0
                pair_wise_data["mkpts{}_{}_{}".format(id1, id0, id1)] = mkpts1
                pair_wise_data["conf{}_{}_{}".format(id0, id0, id1)] = pair_wise_data["conf{}_{}_{}".format(id0, id0, id1)][inliers]
                pair_wise_data["conf{}_{}_{}".format(id1, id0, id1)] = pair_wise_data["conf{}_{}_{}".format(id1, id0, id1)][inliers]
                rel_pose = np.eye(4)
                rel_pose[:3, :3] = R
                rel_pose[:3, 3] = t
                pair_wise_data["rel_pose{}_{}".format(id0, id1)] = rel_pose
                match_graph[id0, id1] = inliers.sum()

    # compute absolute poses from the relative poses of the maximum spanning tree of the match graph with inlier counts as edge weights
    max_inliers = np.amax(match_graph)
    non_zero_mask = match_graph != 0
    match_graph[non_zero_mask] = max_inliers - match_graph[non_zero_mask] + 1
    min_spanning_tree = minimum_spanning_tree(match_graph).toarray().astype(int)
    pair_wise_data["abs_init_pose0"] = np.eye(4)
    n_abs_poses = 1
    row, col = np.nonzero(min_spanning_tree)

    pairs_on_spanning_tree = []
    for _ in range(n_images):
        for r, c in zip(row, col):
            if r < c:
                id0 = r
                id1 = c
            else:
                id0 = c
                id1 = r
            pairs_on_spanning_tree.append((id0, id1))
            if "abs_init_pose{}".format(id1) not in pair_wise_data and "abs_init_pose{}".format(id0) in pair_wise_data:
                pair_wise_data["abs_init_pose{}".format(id1)] = pair_wise_data["abs_init_pose{}".format(id0)] @ \
                    np.linalg.inv(pair_wise_data["rel_pose{}_{}".format(id0, id1)])
                n_abs_poses += 1
            elif "abs_init_pose{}".format(id0) not in pair_wise_data and "abs_init_pose{}".format(id1) in pair_wise_data:
                pair_wise_data["abs_init_pose{}".format(id0)] = pair_wise_data["abs_init_pose{}".format(id1)] @ \
                    pair_wise_data["rel_pose{}_{}".format(id0, id1)]
                n_abs_poses += 1
        if n_abs_poses == n_images:
            break

    extr = [np.eye(4),]
    for id in range(1, n_images):
        abs_pose_key = "abs_init_pose{}".format(id)
        if abs_pose_key in pair_wise_data:
            extr.append(np.linalg.inv(pair_wise_data[abs_pose_key]))
        else:
            extr.append(np.eye(4))
    extr = np.array(extr)

    # write the initialization problem to file
    with open(file_path, 'w') as f:
        for id in range(n_images):
            R, t = extr[id, :3, :3], extr[id, :3, 3]
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(id, R[0, 0], R[1, 0], \
                R[2, 0], R[0, 1], R[1, 1], R[2, 1], R[0, 2], R[1, 2], R[2, 2]))
        for id1 in range(n_images):
            for id0 in range(id1):
                rel_pose_key = "rel_pose{}_{}".format(id0, id1)
                if rel_pose_key in pair_wise_data:
                    n_inliers = pair_wise_data["inlier_count{}_{}".format(id0, id1)]
                    if n_inliers >= min_inliers or (id0, id1) in pairs_on_spanning_tree:
                        T_021 = pair_wise_data[rel_pose_key]
                        R_021 = T_021[:3, :3]
                        t_021 = -R_021.transpose() @ T_021[:3, 3]
                        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(id0, id1, R_021[0, 0], R_021[1, 0], \
                            R_021[2, 0], R_021[0, 1], R_021[1, 1], R_021[2, 1], R_021[0, 2], R_021[1, 2], R_021[2, 2], t_021[0], t_021[1], t_021[2]))
    return pair_wise_data

def write_bundle_adjust_problem(n_images, pair_wise_data, extrinsics, file_path):
    if extrinsics.ndim != 3:
        extrinsics = np.array([np.eye(4) for _ in range(n_images)])
    min_inliers = 0
    # triangulate 3d points
    n_3d_pts = 0
    observations_img_id = [] # img_id
    observations_pt_id = [] # pt_id
    observations_xyc = [] # x, y
    points_in_3d = [] # x, y, z
    for id1 in range(n_images):
        for id0 in range(id1):
            mkpts0_key = "mkpts{}_{}_{}".format(id0, id0, id1)
            if mkpts0_key in pair_wise_data:
                n_inliers = pair_wise_data["inlier_count{}_{}".format(id0, id1)]
                if n_inliers >= min_inliers:
                    # normalize keypoints
                    mkpts0 = pair_wise_data[mkpts0_key]
                    mkpts1 = pair_wise_data["mkpts{}_{}_{}".format(id1, id0, id1)]
                    #conf0 = np.ones_like(pair_wise_data["conf{}_{}_{}".format(id0, id0, id1)]) # unweighted ba
                    #conf1 = np.ones_like(pair_wise_data["conf{}_{}_{}".format(id1, id0, id1)]) # unweighted ba
                    conf0 = pair_wise_data["conf{}_{}_{}".format(id0, id0, id1)]
                    conf1 = pair_wise_data["conf{}_{}_{}".format(id1, id0, id1)]
                    intr0 = pair_wise_data["intr{}".format(id0)]
                    intr1 = pair_wise_data["intr{}".format(id1)]
                    mkpts0 = (mkpts0 - intr0[[0, 1], [2, 2]][None]) / intr0[[0, 1], [0, 1]][None]
                    mkpts1 = (mkpts1 - intr1[[0, 1], [2, 2]][None]) / intr1[[0, 1], [0, 1]][None]
                    # triangulate 3D points
                    if mkpts0.shape[0] != 0:
                        pts_3d = cv2.triangulatePoints(extrinsics[id0, :3, :], extrinsics[id1, :3, :], mkpts0.transpose(), mkpts1.transpose())
                        pts_3d = (pts_3d[:3] / pts_3d[3]).transpose()
                    else:
                        pts_3d = np.zeros((0, 3)) # create empty array
                    
                    for id, mkpts, conf in zip((id0, id1), (mkpts0, mkpts1), (conf0, conf1)):
                        observations_img_id.append(np.full(mkpts.shape[0], id, dtype=int))
                        observations_pt_id.append(np.arange(n_3d_pts, n_3d_pts + pts_3d.shape[0], dtype=int))
                        observations_xyc.append(np.concatenate((mkpts, conf), -1))
                    n_3d_pts += pts_3d.shape[0]
                    # add 3d points
                    points_in_3d.append(pts_3d)

    # write to file
    observations_img_id = np.concatenate(observations_img_id, 0)
    observations_pt_id = np.concatenate(observations_pt_id, 0)
    observations_xyc = np.concatenate(observations_xyc, 0)

    observations_xyc = normalize_confidences(observations_xyc)

    points_in_3d = np.concatenate(points_in_3d, 0)
    with open(file_path, 'w') as f:
        ref_cam = 0
        # write header with intrinsics as identity
        f.write("{},{},{},{},{},{},{},{}\n".format(n_images, ref_cam, n_3d_pts, 2 * n_3d_pts, 1., 1., 0., 0.))
        for id, pt_id, kpt in zip(observations_img_id, observations_pt_id, observations_xyc):
            if kpt.shape[0] == 3:
                f.write("{},{},{},{},{}\n".format(id, pt_id, kpt[0], kpt[1], kpt[2])) # last element is confidence
            elif kpt.shape[0] == 4:
                f.write("{},{},{},{},{},{}\n".format(id, pt_id, kpt[0], kpt[1], kpt[2], kpt[3])) # last 2 elements are confidence
            else:
                logging.error("Unexpected number of confidence values")
        for id in range(n_images): # 1st is fixed during ba
            R, t = extrinsics[id, :3, :3], extrinsics[id, :3, 3]
            f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(R[0, 0], R[1, 0], R[2, 0], 
                R[0, 1], R[1, 1], R[2, 1], R[0, 2], R[1, 2], R[2, 2], t[0], t[1], t[2]))
        for pt_3d in points_in_3d:
            f.write("{},{},{}\n".format(pt_3d[0], pt_3d[1], pt_3d[2]))

def read_bundle_adjust_result(file_path):
    extrinsics = [] # R, t world to cam
    with open(file_path, "r") as f:
        for line in f:
            w = line.split(',')
            # rotation is column major
            R = np.array([[float(w[0]), float(w[3]), float(w[6])], [float(w[1]), float(w[4]), float(w[7])], [float(w[2]), float(w[5]), float(w[8])]])
            t = np.array([float(w[9]), float(w[10]), float(w[11])])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            extrinsics.append(T)
    return extrinsics
