import argparse
import os
import json
import coloredlogs, logging
coloredlogs.install()

import numpy as np
import torch
import cv2

from helpers import add_generic_arguments, get_exp_dir, load_ckpt, to_gpu_matcher, run_super_point
from datasets.scannet import read_intrinsics, read_pose
from datasets.matching_dataset import resize_intrinsics
from models.models.multi_view_matcher import MultiViewMatcher
from models.models.superpoint import SuperPoint
from models.models.utils import estimate_pose, pose_auc, compute_pose_error, rotate_pose_inplane, rotate_intrinsics
from pose_optimization.two_view.estimate_relative_pose import normalize, run_bundle_adjust_2_view, estimate_relative_pose_w8pt

torch.set_grad_enabled(False)

class PairMatchingDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, input_files, img_size, dataset):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.rgb_paths = []
        self.scenes = []
        self.ids = []
        self.intrinsics = []
        self.rots = []
        self.T021s = []
        # extract pairs
        for f_id, f in enumerate(input_files):
            if "megadepth" == dataset:
                data = np.load(f, allow_pickle=True)
                pairs = data["pair_infos"]
                paths = data["image_paths"]
                intrs = data["intrinsics"]
                extrs = data["poses"]
                for pair, _, _ in pairs:
                    id0, id1 = pair[0], pair[1]
                    self.rgb_paths.append((os.path.join(self.data_dir, paths[id0]), os.path.join(self.data_dir, paths[id1])))
                    self.intrinsics.append((intrs[id0].copy(), intrs[id1].copy()))
                    extr0 = extrs[id0]
                    extr1 = extrs[id1]
                    self.T021s.append(extr1 @ np.linalg.inv(extr0))
                    self.rots.append((0, 0))
                    self.scenes.append("mega{}".format(f_id))
                    self.ids.append((int(id0), int(id1)))
            elif "yfcc100m" == dataset:
                with open(f, 'r') as in_f:
                    pairs = [l.split() for l in in_f.readlines()]
                for i, pair in enumerate(pairs):
                    path0, path1 = pair[:2]
                    rot0, rot1 = int(pair[2]), int(pair[3])
                    intr0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
                    intr1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
                    T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)
                    self.rgb_paths.append((os.path.join(self.data_dir, path0), os.path.join(self.data_dir, path1)))
                    self.intrinsics.append((intr0, intr1))
                    self.T021s.append(T_0to1)
                    self.rots.append((rot0, rot1))
                    self.scenes.append("yfcc100m")
                    self.ids.append((i, 2*i))
            elif "scannet" == dataset:
                data = np.load(f)
                data_split_dir = os.path.join(data_dir, "scans_test")
                for i, (scene, _, id0, id1) in enumerate(data["name"]):
                    scene_string = "scene0{}_00".format(scene)
                    scene_dir = os.path.join(data_split_dir, scene_string)
                    # make rgb paths
                    self.rgb_paths.append((os.path.join(scene_dir, "color", "{}.jpg".format(id0)), \
                        os.path.join(scene_dir, "color", "{}.jpg".format(id1))))
                    # read intrinsics
                    intr = read_intrinsics(data_split_dir, scene_string)
                    self.intrinsics.append((intr, intr.copy()))
                    # read poses
                    pose0 = read_pose(data_split_dir, scene_string, id0)
                    pose1 = read_pose(data_split_dir, scene_string, id1)
                    self.T021s.append(np.linalg.inv(pose1) @ pose0)
                    self.rots.append((0, 0))
                    self.scenes.append(scene_string)
                    self.ids.append((int(id0), int(id1)))

    def __getitem__(self, index):
        data = dict()

        for id in range(2):
            img = cv2.imread(self.rgb_paths[index][id], cv2.IMREAD_GRAYSCALE).astype('float32')
            rot = self.rots[index][id]
            intr = self.intrinsics[index][id]
            if rot != 0:
                img = np.rot90(img, k=rot)
                intr = rotate_intrinsics(intr, img.shape, rot)
            # resize
            if self.img_size != max(img.shape[1], img.shape[0]):
                if img.shape[0] >= img.shape[1]:
                    ar = img.shape[1] / img.shape[0]
                    resize_size = (self.img_size, int(ar * self.img_size))
                else:
                    ar = img.shape[0] / img.shape[1]
                    resize_size = (int(ar * self.img_size), self.img_size)
                fact_x, fact_y = resize_size[1] / img.shape[1], resize_size[0] / img.shape[0]
                intr = resize_intrinsics(intr, fact_x, fact_y)
                img = cv2.resize(img, (resize_size[1], resize_size[0]))
            img = torch.from_numpy(img / 255.).unsqueeze(0)
            data["image{}".format(id)] = img
            data["intr{}".format(id)] = intr

        T021 = self.T021s[index]
        rot0, rot1 = self.rots[index]
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T021
            if rot0 != 0:
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T021 = cam1_T_cam0
        data["T021"] = T021
        data["ids"] = [self.ids[index][0], self.ids[index][1]]
        data["scene"] = self.scenes[index]

        return data

    def __len__(self):
        return len(self.rgb_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = add_generic_arguments(parser)
    
    # add arguments specific to testing
    parser.add_argument('--exp_name', type=str, default=None, 
        help='Specify an experiment name to test on')
    parser.add_argument('--eval_mode', type=str, default="w8pt_ba", help='choose relative pose estimation method: ransac, ransac_ba, w8pt, w8pt_ba')
    parser.add_argument('--n_workers', type=int, default=2, help='number workers')
    opt = parser.parse_args()

    # load a specific checkpoint: 
    # * None loads 'model.ckpt' (use for pretrained models)
    # * epoch as an integer loads '<epoch>_model.ckpt'
    # * 'last' loads last saved checkpoint 'last_model.ckpt'
    # * 'best' loads model with lowest validation loss 'best_model.ckpt'
    model_id = None

    # for weighted pose estimation there is no need to filter by confidence, confidences are accurate enough
    match_threshold = 0.02 if "ransac" in opt.eval_mode else 0.0
    source_dir = os.path.dirname(__file__)

    loftr_assets_dir = os.path.join(source_dir, "assets")
    super_glue_assets_dir = os.path.join(source_dir, "models", "assets")
    if "megadepth" in opt.dataset:
        # eval on pairs defined in loftr
        loftr_assets_dir = os.path.join(loftr_assets_dir, "megadepth_test_1500_scene_info")
        input_files = [os.path.join(loftr_assets_dir, f) for f in 
            ["0015_0.1_0.3.npz", "0015_0.3_0.5.npz", "0022_0.1_0.3.npz", "0022_0.3_0.5.npz", "0022_0.5_0.7.npz"]]
        opt.max_keypoints = 2048
        opt.nms_radius = 3
        opt.keypoint_threshold = 0.005
        img_size = 1600
    elif "yfcc100m" in opt.dataset:
        # eval on pairs defined in superglue
        input_files = [os.path.join(super_glue_assets_dir, "yfcc_test_pairs_with_gt.txt"),]
        opt.max_keypoints = 2048
        opt.nms_radius = 3
        opt.keypoint_threshold = 0.005
        img_size = 1600
    elif "scannet" in opt.dataset:
        # eval on pairs defined in loftr (same as used in superglue)
        loftr_assets_dir = os.path.join(loftr_assets_dir, "scannet_test_1500")
        input_files = [os.path.join(loftr_assets_dir, "test.npz"),]
        opt.max_keypoints = 1024
        opt.nms_radius = 4
        opt.keypoint_threshold = 0.001
        img_size = 720

    exp_dir, _ = get_exp_dir(opt.checkpoint_dir, opt.exp_name)

    dataset_dir = os.path.join(opt.data_dir, opt.dataset)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    matcher = torch.nn.DataParallel(MultiViewMatcher({
        'multi_frame_matching' : False,
        'tuple_size' : 2,
        'conf_mlp' : True,
        }).eval().cuda(), device_ids=[0])
    matcher, _, _, _, _ = load_ckpt(exp_dir, matcher, model_id)

    super_point = SuperPoint({
        'nms_radius': opt.nms_radius,
        'keypoint_threshold': opt.keypoint_threshold,
        'max_keypoints': opt.max_keypoints,
        'remove_borders' : 0,
    }).eval().cuda()

    cannot_compute_pose = 0
    pose_errors = []
    test_dataset = PairMatchingDataset(dataset_dir, input_files, img_size, opt.dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=opt.n_workers)
    for i, data in enumerate(test_loader):
        # match
        run_super_point(opt, data, super_point, merge=(False if "megadepth" in opt.dataset or "yfcc100m" in opt.dataset else True))
        to_gpu_matcher(data, 2)
        pred = matcher(data)

        # find matching keypoints and confidences
        kpts0, kpts1 = data['keypoints0'].squeeze(0).cpu().numpy(), data['keypoints1'].squeeze(0).cpu().numpy()
        matches = pred['matches0_0_1'].squeeze(0).cpu().numpy()
        conf = pred['conf_scores_0_1'].squeeze(0).squeeze(-1).cpu().numpy() 
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        conf_mask = mconf > match_threshold

        K0 = data["intr0"].squeeze(0)[:3, :3].cpu().numpy()
        K1 = data["intr1"].squeeze(0)[:3, :3].cpu().numpy()
        T_0to1 = data["T021"].squeeze(0).cpu().numpy()

        if opt.eval_mode in ["ransac", "ransac_ba"]:
            thresh = 1.
            ret = estimate_pose(mkpts0[conf_mask], mkpts1[conf_mask], K0, K1, thresh)
            if "ba" in opt.eval_mode and ret is not None:
                inlier_mask = ret[2]
                pred_T021 = torch.eye(4).unsqueeze(0).cuda()
                pred_T021[0, :3, 3] = torch.from_numpy(ret[1]).cuda().unsqueeze(0)
                pred_T021[0, :3, :3] = torch.from_numpy(ret[0]).cuda().unsqueeze(0)
                confidence = torch.from_numpy(mconf[conf_mask][inlier_mask]).cuda().unsqueeze(0)
                intr0 = torch.from_numpy(K0).cuda().unsqueeze(0)
                intr1 = torch.from_numpy(K1).cuda().unsqueeze(0)
                kpts0_norm = normalize(torch.from_numpy(mkpts0[conf_mask][inlier_mask]).cuda().unsqueeze(0), intr0)
                kpts1_norm = normalize(torch.from_numpy(mkpts1[conf_mask][inlier_mask]).cuda().unsqueeze(0), intr1)
                pred_T021_refine, valid_refine = run_bundle_adjust_2_view(kpts0_norm, kpts1_norm, confidence.unsqueeze(-1), pred_T021, \
                        n_iterations=10)
                pred_T021[valid_refine] = pred_T021_refine
                ret = (pred_T021[0, :3, :3].cpu().numpy(), pred_T021[0, :3, 3].cpu().numpy(), inlier_mask) if pred_T021 is not None else None
        elif "w8pt" in opt.eval_mode:
            intr0 = torch.from_numpy(K0).cuda().unsqueeze(0)
            intr1 = torch.from_numpy(K1).cuda().unsqueeze(0)
            pred_T021, info = estimate_relative_pose_w8pt(torch.from_numpy(mkpts0[conf_mask]).cuda().unsqueeze(0), torch.from_numpy(mkpts1[conf_mask]).cuda().unsqueeze(0), \
                intr0, intr1, torch.from_numpy(mconf[conf_mask]).cuda().unsqueeze(0), determine_inliers=True)
            if "ba" in opt.eval_mode and pred_T021 is not None:
                confidence = info["confidence"]
                confidence[torch.logical_not(info["pos_depth_mask"])] = 0.
                pred_T021_refine, valid_refine = run_bundle_adjust_2_view(info["kpts0_norm"], info["kpts1_norm"], confidence.unsqueeze(-1), pred_T021, \
                    n_iterations=10)
                pred_T021[valid_refine] = pred_T021_refine
            ret = (pred_T021[0, :3, :3].cpu().numpy(), pred_T021[0, :3, 3].cpu().numpy(), None) if pred_T021 is not None else None

        if ret is None:
            err_t, err_R = np.inf, np.inf
            cannot_compute_pose += 1
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)
            logging.info("{:>4d}: rot {:>5.1f}deg\tt {:>5.1f}deg".format(i, err_R, err_t))

        pose_error = np.maximum(err_t, err_R)
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    result = {"AUC@5deg" : 100. * aucs[0], "AUC@10deg" : 100. * aucs[1], "AUC@20deg" : 100. * aucs[2]}
    logging.info("AUC@5deg:  {:.3f}%".format(result["AUC@5deg"]))
    logging.info("AUC@10deg: {:.3f}%".format(result["AUC@10deg"]))
    logging.info("AUC@20deg: {:.3f}%".format(result["AUC@20deg"]))

    test_json = os.path.join(exp_dir, "two_view_{}_{}.json".format(opt.eval_mode, opt.dataset))
    with open(test_json, 'w') as tf:
        json.dump(result, tf, indent=4)