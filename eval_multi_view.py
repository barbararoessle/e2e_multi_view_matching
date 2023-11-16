import os
import argparse
import json
import subprocess
import time
import datetime

import numpy as np
import torch
import coloredlogs, logging
coloredlogs.install()

from helpers import dict_to_namespace, run_super_point, to_gpu_matcher, add_generic_arguments, get_exp_dir, load_ckpt
from models.models.multi_view_matcher import MultiViewMatcher
from models.models.utils import compute_pose_error, pose_auc
from models.models.multi_view_matcher import MultiViewMatcher
from models.models.superpoint import SuperPoint
from datasets.matching_dataset import MatchingDataset
from pose_optimization.multi_view.bundle_adjust_io import initialize_bundle_adjust, write_bundle_adjust_problem, read_bundle_adjust_result

def eval_bundle_adjust(tuple_size, data, result, tmp_dir, pose_errors, verbose=False):
    os.makedirs(tmp_dir, exist_ok=True)

    # initialize bundle adjustment problem using relative pose estimation between all possible pairs 
    # and determine absolute poses through robust estimators for rotation and translation:
    # Chatterjee, A. and Govindu, V. Efficient and Robust Large- Scale Rotation Averaging International Conference on Computer Vision (ICCV), 2013.
    # O. Ozyesil and Singer, A. Robust Camera Location Estimation by Convex Programming In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015.
    file_path = os.path.join(tmp_dir, "ba_init_in.csv")
    pair_wise_data = initialize_bundle_adjust(tuple_size, data, result, file_path)
    build_dir = os.path.join(os.path.dirname(__file__), "pose_optimization", "multi_view", "bundle_adjustment", "build")
    cmd = os.path.join(build_dir, "ba_initializer {}".format(tmp_dir))
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        if verbose:
            print(line)
        pass
    process.wait()
    file_path = os.path.join(tmp_dir, "ba_init_out.csv")
    extrinsics = np.array(read_bundle_adjust_result(file_path))

    # run Ceres Solver to solve bundle adjustment
    file_path = os.path.join(tmp_dir, "ba_in.csv")
    write_bundle_adjust_problem(tuple_size, pair_wise_data, extrinsics, file_path)
    cmd = os.path.join(build_dir, "bundle_adjuster {}".format(tmp_dir))
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        if verbose:
            print(line)
        pass
    process.wait()
    file_path = os.path.join(tmp_dir, "ba_out.csv")
    extrinsics = read_bundle_adjust_result(file_path)

    # evaluate results
    for id1 in range(tuple_size):
        for id0 in range(id1):
            # compute pose metrics
            pose0, pose1 = data["pose{}".format(id0)][0].numpy(), data["pose{}".format(id1)][0].numpy()
            T_021 = np.linalg.inv(pose1) @ pose0
            T_021_pred = extrinsics[id1] @ np.linalg.inv(extrinsics[id0])
            err_t, err_R = compute_pose_error(T_021, T_021_pred[:3, :3], T_021_pred[:3, 3])
            pose_error = np.maximum(err_t, err_R)
            pose_errors[0].append(pose_error)
            pose_errors[1].append(err_t)
            pose_errors[2].append(err_R)
            logging.info("{} -> {}: rot {:>5.1f}deg\tt {:>5.1f}deg".format(id0, id1, err_R, err_t))

    return pose_errors

def write_result(pose_errors, file):
    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors[0], thresholds)
    t_aucs = pose_auc(pose_errors[1], thresholds)
    r_aucs = pose_auc(pose_errors[2], thresholds)
    metrics = dict()
    for thresh, auc, t_auc, r_auc in zip(thresholds, aucs, t_aucs, r_aucs):
        metrics["pose_AUC@{}deg".format(thresh)] = auc * 100.0
        metrics["transl_AUC@{}deg".format(thresh)] = t_auc * 100.0
        metrics["rot_AUC@{}deg".format(thresh)] = r_auc * 100.0
    print()
    for error_type in ["transl", "rot"]:
        for thresh in thresholds:
            k = "{}_AUC@{}deg".format(error_type, thresh)
            logging.info("{}: \t{:>6.3f}%".format(k, metrics[k]))
    
    with open(file, 'w') as tf:
        json.dump(metrics, tf, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate multi-view',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser = add_generic_arguments(parser)
    
    # add arguments specific to testing
    parser.add_argument('--exp_name', type=str, default=None, 
        help='Specify an experiment name to test on')
    parser.add_argument('--n_workers', type=int, default=2, help='number workers')
    parser.add_argument('--verbose', action='store_true', help='log bundle adjustment calls')
    opt = parser.parse_args()

    # load a specific checkpoint: 
    # * None loads 'model.ckpt' (use for pretrained models)
    # * epoch as an integer loads '<epoch>_model.ckpt'
    # * 'last' loads last saved checkpoint 'last_model.ckpt'
    # * 'best' loads model with lowest validation loss 'best_model.ckpt'
    model_id = None

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # at test time we can use more keypoints
    opt.max_keypoints = 1024
    if "megadepth" in opt.dataset:
        opt.max_keypoints = 2048

    # load networks
    exp_dir, config_json = get_exp_dir(opt.checkpoint_dir, opt.exp_name)
    tmp_dir = os.path.join(exp_dir, "{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')))
    with open(config_json, 'r') as cf:
        cfg_dict = json.load(cf)
    train_opt = dict_to_namespace(cfg_dict)
    opt.remove_borders = train_opt.remove_borders
    opt.nms_radius = train_opt.nms_radius
    opt.keypoint_threshold = train_opt.keypoint_threshold
    opt.cross_attention_layers = train_opt.cross_attention_layers
    opt.gnn_layers = train_opt.gnn_layers
    matcher = torch.nn.DataParallel(MultiViewMatcher({ \
        "multi_frame_matching" : True, \
        'GNN_layers': (['self',] + ['cross',] * opt.cross_attention_layers) * opt.gnn_layers}).eval().cuda(), device_ids=[0])
    matcher, _, _, _, _ = load_ckpt(exp_dir, matcher, model_id)
    
    super_point = SuperPoint({
        'nms_radius': opt.nms_radius,
        'keypoint_threshold': opt.keypoint_threshold,
        'max_keypoints': opt.max_keypoints,
        'remove_borders' : opt.remove_borders,
    }).eval().cuda()
    
    # load dataset
    dataset_dir = os.path.join(opt.data_dir, opt.dataset)
    test_dataset = MatchingDataset(dataset_dir, split="test", tuple_size=opt.tuple_size, shuffle_tuple=False)

    test_tuple_file = os.path.join(os.path.dirname(__file__), "assets", "{}_test_5tuples.csv".format(opt.dataset.split("_")[0]))
    test_dataset.read_sampled_tuples(test_tuple_file)
    logging.info("Loaded {} test samples".format(len(test_dataset)))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=opt.n_workers)
    
    # start testing
    with torch.no_grad():
        pose_errors = ([], [], [])
        for i, data in enumerate(test_loader):
            print()
            logging.info("{}".format(i))
            
            run_super_point(opt, data, super_point)
            to_gpu_matcher(data, opt.tuple_size)
            result = matcher(data)

            pose_errors = eval_bundle_adjust(opt.tuple_size, data, result, tmp_dir, pose_errors, opt.verbose)

    test_json = os.path.join(exp_dir, "multi_view_{}.json".format(opt.dataset.split("_")[0]))
    write_result(pose_errors, test_json)
