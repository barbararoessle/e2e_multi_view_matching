import os
import time
import copy
import argparse
import coloredlogs, logging
coloredlogs.install()

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from pose_optimization.two_view.estimate_relative_pose import run_weighted_8_point
from pose_optimization.two_view.compute_pose_error import compute_rotation_error, compute_translation_error_as_angle

def add_generic_arguments(parser):
    parser.add_argument('--data_dir', type=str, default=None, help='path to directory containing dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='path to directory containing checkpoints')
    parser.add_argument('--dataset', type=str, default="scannet", help='dataset')
    parser.add_argument('--tuple_size', type=int, default=5, help='number of images to be matched')
    return parser

def get_exp_dir(checkpoint_dir, exp_name):
    exp_dir = os.path.join(checkpoint_dir, exp_name)
    config_json = os.path.join(exp_dir, "cfg.json")
    return exp_dir, config_json

def save_ckpt(epoch, model, optimizer, val_loss, pose_match_ratio, file_name, exp_dir):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'min_val_loss': val_loss,
        'pose_match_ratio': pose_match_ratio,}, os.path.join(exp_dir, file_name))

def load_ckpt(checkpoint_dir, model, model_id=None, file_suffix="model", local_rank=0, optimizer=None):
    logging.info("Loading checkpoint {}".format(checkpoint_dir))
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    filename = None
    if model_id is None:
        filename = "{}.ckpt".format(file_suffix)
    elif isinstance(model_id, int):
        filename = "{:0>6}_{}.ckpt".format(model_id, file_suffix)
    elif isinstance(model_id, str):
        filename = "{}_{}.ckpt".format(model_id, file_suffix)
    else:
        logging.error("Cannot load model {}, which is neither integer nor string".format(model_id))
    state_dict = torch.load(os.path.join(checkpoint_dir, filename), map_location=map_location)
    missing, unexpected = model.load_state_dict(state_dict['model'], strict=False)
    if len(missing) > 0:
        logging.warn("Missing keys: {}".format(missing))
    if len(unexpected) > 0:
        logging.warn("Unexpected keys: {}".format(unexpected))

    if optimizer is not None:
        old_state_dict = state_dict['optimizer']
        new_state_dict = optimizer.state_dict()
        for n in range(len(old_state_dict['param_groups']), len(new_state_dict['param_groups'])):
            old_state_dict['param_groups'].append(copy.deepcopy(new_state_dict['param_groups'][n]))
        optimizer.load_state_dict(old_state_dict)
    pose_match_ratio = state_dict['pose_match_ratio'] if 'pose_match_ratio' in state_dict else 0.
    return model, optimizer, state_dict['epoch'], state_dict['min_val_loss'], pose_match_ratio

def get_parameters(model, whitelist_key=None, blacklist_key=None):
    params = []
    for n, param in model.named_parameters():
        if whitelist_key is not None and whitelist_key not in n:
            continue
        if blacklist_key is not None and blacklist_key in n:
            continue
        params.append(param)
    return params

def to_gpu_super_point(data, tuple_size, merge=True):
    keys = []
    for i in range(tuple_size):
        keys += ["image" + str(i),]
    all_images = [data[k].cuda() for k in keys]
    if merge:
        return [torch.cat(all_images, 0),]
    else:
        return all_images

def run_super_point(opt, data, super_point, merge=True):
    curr_tuple_size = len(data["ids"])
    images = to_gpu_super_point(data, curr_tuple_size, merge=merge)
    with torch.no_grad():
        pred = super_point({"image" : images})
        for k, v in pred.items():
            if len(v) != curr_tuple_size:  # batch size > 1
                res = torch.stack(v)
                res_shape = res.shape
                res = res.view(curr_tuple_size, opt.batch_size, *res_shape[1:])
            else: # batch size 1
                res = [v_i.unsqueeze(0) for v_i in v]
            for m in range(curr_tuple_size):
                data[k + str(m)] = res[m]

def to_gpu_matcher(data, tuple_size):
    keys = []
    for i in range(tuple_size):
        keys += ["keypoints" + str(i),]
        for k in range(i):
            keys += ["gt_indices_{}_{}".format(k, i), "gt_weights_{}_{}".format(k, i)]
    for k in keys:
        if k in data:
            data[k] = data[k].cuda()

def to_gpu_compute_gt(data, tuple_size):
    keys = []
    for i in range(tuple_size):
        keys += ["intr" + str(i), "keypoints" + str(i), "depth" + str(i), "pose" + str(i),]
    for k in keys:
        data[k] = data[k].cuda()

def transform_kpts(kpts0, d, K0, K1, T_021):
    kpts0to1 = (K1 @ T_021 @ torch.linalg.inv(K0) @ torch.cat((kpts0 * d, d, torch.ones_like(d)), axis=-1).unsqueeze(-1))
    depths0to1 = kpts0to1[..., 2, :] # bs, n_kpts, 1
    kpts0to1 = kpts0to1[..., :2, 0] / depths0to1 # bs, n_kpts, 2
    return depths0to1, kpts0to1

def compute_gt_matches_of_image_pair(kpts0, kpts1, K0, K1, T0to1, depth0, depth1, max_matched_reproj_err, min_unmatched_reproj_err):
    # transform keypoints to other image
    bs, n_kpts, _ = kpts0.shape
    batch_idx = torch.arange(bs).unsqueeze(-1).expand(bs, n_kpts)
    kpts0 = kpts0.long()
    kpts1 = kpts1.long()
    d0 = depth0[batch_idx, kpts0[..., 1], kpts0[..., 0]].unsqueeze(-1) # use narrow?
    d1 = depth1[batch_idx, kpts1[..., 1], kpts1[..., 0]].unsqueeze(-1) # bs, n_kpts, 1
    K0 = K0.unsqueeze(1) # bs, 1, 4, 4
    K1 = K1.unsqueeze(1) # bs, 1, 4, 4
    T0to1 = T0to1.unsqueeze(1) # bs, 1, 4, 4
    depths0to1, kpts0to1 = transform_kpts(kpts0, d0, K0, K1, T0to1)
    depths1to0, kpts1to0 = transform_kpts(kpts1, d1, K1, K0, torch.linalg.inv(T0to1))

    # compute mean of the error between keypoints in both directions
    errors = torch.sqrt(((kpts1to0.unsqueeze(2).expand(bs, n_kpts, n_kpts, 2) - kpts0.unsqueeze(1))**2).sum(3)).transpose(1, 2)
    errors += torch.sqrt(((kpts0to1.unsqueeze(2).expand(bs, n_kpts, n_kpts, 2) - kpts1.unsqueeze(1))**2).sum(3))
    errors /= 2.0 # bs, n_kps, n_kps (row index for kpts0, col index for kpts1)

    row_mins = torch.argmin(errors, dim=2) # for each kpt0 the index to the closest kpt1
    col_mins = torch.argmin(errors, dim=1) # for each kpt1 the index to the closest kpt0

    n_kpts_bin = n_kpts + 1 # including dust bin
    match_indices0 = torch.full((bs, n_kpts_bin), -1, device=kpts0.device)
    match_indices1 = torch.full_like(match_indices0, -1)
    match_weights0 = torch.full((bs, n_kpts_bin), 0., device=kpts0.device)
    match_weights1 = torch.full_like(match_weights0, 0.)

    # for each kpt0 check if it matches
    i0s = torch.arange(n_kpts, device=kpts0.device).unsqueeze(0).expand(bs, n_kpts)
    i1s = row_mins
    # check if error is minimal in both directions
    min_for_both = col_mins[batch_idx, i1s] == i0s
    # check if reprojection error is small
    small_reproj_err_10 = errors[batch_idx, i0s, i1s] <= max_matched_reproj_err
    # check for valid depth
    d0 = d0.squeeze(-1)
    d1 = d1.squeeze(-1)
    valid_d0 = d0 > 1e-6
    match_d1 = d1[batch_idx, i1s]
    valid_d1 = match_d1 > 1e-6
    match_mask = torch.logical_and(min_for_both, small_reproj_err_10)
    match_mask = torch.logical_and(match_mask, valid_d0)
    match_mask = torch.logical_and(match_mask, valid_d1)
    # check that relative error is small
    match_mask[match_mask.clone()] = torch.logical_and( \
        (torch.abs(depths0to1.squeeze(-1)[match_mask] - match_d1[match_mask]) / match_d1[match_mask]) < 0.1, \
        (torch.abs(depths1to0.squeeze(-1)[batch_idx[match_mask], i1s[match_mask]] - d0[match_mask]) / d0[match_mask]) < 0.1)
    match_indices0[:, :-1][match_mask] = i1s[match_mask]
    match_indices1[batch_idx[match_mask], i1s[match_mask]] = i0s[match_mask]
    match_count = match_mask.sum(1)

    # for each kpt0 that does not match, drop those we are not certain that it does not match
    no_match_mask = torch.logical_not(match_mask)
    invalid_depth = torch.logical_or(torch.logical_not(valid_d0), torch.logical_not(valid_d1))
    small_reproj_err_15 = errors[batch_idx, i0s, i1s] <= min_unmatched_reproj_err
    drop_mask = torch.logical_and(no_match_mask, torch.logical_or(invalid_depth, small_reproj_err_15))
    match_weights0[:, :-1][drop_mask] = -1
    drop_count = drop_mask.sum(1)

    # for each kpt1 that does not match, drop those we are not certain that it does not match
    i1s = torch.arange(n_kpts, device=kpts0.device).unsqueeze(0).expand(bs, n_kpts)
    i0s = col_mins
    no_match_mask = match_indices1[:, :-1] == -1
    valid_d1 = d1 > 1e-6
    match_d0 = d0[batch_idx, i0s]
    valid_d0 = match_d0 > 1e-6
    invalid_depth = torch.logical_or(torch.logical_not(valid_d0), torch.logical_not(valid_d1))
    small_reproj_err_15 = errors[batch_idx, i0s, i1s] <= min_unmatched_reproj_err
    drop_mask = torch.logical_and(no_match_mask, torch.logical_or(invalid_depth, small_reproj_err_15))
    match_weights1[:, :-1][drop_mask] = -1
    drop_count += drop_mask.sum(1)
    # weigh entries to balance classes
    match_weight = 2. * match_count / (2. * torch.full_like(match_count, n_kpts) - drop_count)
    unmatch_weight = .5 / (1. - match_weight)
    match_weight = .5 / match_weight
    invalid_weights = torch.logical_not(torch.logical_and(match_weight.isfinite(), unmatch_weight.isfinite()))
    match_weight[invalid_weights] = 0.
    unmatch_weight[invalid_weights] = 0.
    set_weight(match_weights0, match_indices0, match_weight, unmatch_weight)
    set_weight(match_weights1, match_indices1, match_weight, unmatch_weight)

    return torch.stack((match_indices0, match_indices1), 1), torch.stack((match_weights0, match_weights1), 1)

def set_weight(match_weights, match_indices, match_weight, unmatch_weight):
    reset = match_weights == -1
    no_reset = torch.logical_not(reset)
    invalid_match_index = match_indices == -1
    unmatch = torch.logical_and(no_reset, invalid_match_index)
    match = torch.logical_and(no_reset, torch.logical_not(invalid_match_index))
    match_weights[reset] = 0.
    match_weights[unmatch] = unmatch_weight.unsqueeze(-1).expand_as(match_weights)[unmatch]
    match_weights[match] = match_weight.unsqueeze(-1).expand_as(match_weights)[match]

def compute_gt_matches(opt, data):
    curr_tuple_size = len(data["ids"])
    to_gpu_compute_gt(data, curr_tuple_size)
    for m in range(curr_tuple_size):
        for k in range(m):
            T_k2m = torch.linalg.inv(data["pose" + str(m)]) @ data["pose" + str(k)]
            data["gt_indices_{}_{}".format(k, m)], data["gt_weights_{}_{}".format(k, m)] = \
                compute_gt_matches_of_image_pair(data["keypoints" + str(k)], data["keypoints" + str(m)], \
                data["intr" + str(k)], data["intr" + str(m)], T_k2m, data["depth" + str(k)], data["depth" + str(m)], \
                    opt.match_reproj_err, opt.unmatch_reproj_err)
    for m in range(curr_tuple_size):
        data.pop("depth" + str(m))

def compute_match_loss(log_p, gt_indices_0_1, gt_weights_0_1):
    bs, ft, _ = log_p.shape

    match_indices0 = gt_indices_0_1.narrow(1, 0, 1)
    match_indices1 = gt_indices_0_1.narrow(1, 1, 1)
    match_weights0 = gt_weights_0_1.narrow(1, 0, 1)
    match_weights1 = gt_weights_0_1.narrow(1, 1, 1)

    l0 = -log_p.reshape(bs*ft, ft)[range(bs*ft), match_indices0.reshape(bs*ft)]
    l1 = -log_p.transpose(1,2).reshape(bs*ft, ft)[range(bs*ft), match_indices1.reshape(bs*ft)]

    loss = torch.dot(l0, match_weights0.reshape(bs*ft)) + torch.dot(l1, match_weights1.reshape(bs*ft))

    return loss / bs

def run_matcher(opt, data, matcher):
    curr_tuple_size = len(data["ids"])
    matcher.module.config["full_output"] = opt.pose_loss
    result = matcher(data)
    match_loss = torch.zeros(1, device=data["pose0"].device)
    rot_loss = torch.zeros(1, device=match_loss.device)
    transl_loss = torch.zeros(1, device=match_loss.device)
    for id1 in range(curr_tuple_size):
        for id0 in range(id1):
            match_loss = match_loss + compute_match_loss(result["scores_{}_{}".format(id0, id1)], data["gt_indices_{}_{}".format(id0, id1)], \
                data["gt_weights_{}_{}".format(id0, id1)])
            if opt.pose_loss:
                target = torch.linalg.inv(data["pose{}".format(id1)]) @ data["pose{}".format(id0)]
                pred, _ = run_weighted_8_point(data, result, id0, id1, choose_closest=True, target_T_021=target)
                rot_loss = rot_loss + compute_rotation_error(pred, target)
                transl_loss = transl_loss + compute_translation_error_as_angle(pred, target)
    losses = {"match_loss": match_loss, "rot_loss": rot_loss, "transl_loss": transl_loss}
    return losses, result

def dict_to_namespace(dictionary):
    d = copy.deepcopy(dictionary) # make deep copy
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return argparse.Namespace(**d)

def namespace_to_dict(namespace):
    namespace_as_dict = copy.deepcopy(vars(namespace)) # make deep copy
    for k, v in namespace_as_dict.items():
        if isinstance(v, argparse.Namespace):
            namespace_as_dict[k] = namespace_to_dict(v)
    return namespace_as_dict

def setup_tb(checkpoint_dir, exp_name):
    if is_main_process():
        run_dir = os.path.join(checkpoint_dir, "runs", exp_name)
        tb_writer = SummaryWriter(log_dir=run_dir)
    else:
        tb_writer = None
    return tb_writer

def has_finite_gradients(net):
    for params in net.parameters():
        if params.grad is not None and not params.grad.isfinite().all():
            return False
    return True

class MeanTracker(object):
    def __init__(self):
        self.reset()

    def add(self, input, weight=1.):
        for key, l in input.items():
            if not key in self.mean_dict:
                self.mean_dict[key] = 0
            self.mean_dict[key] = (self.mean_dict[key] * self.total_weight + l * weight) / (self.total_weight + weight)
        self.total_weight += weight

    def has(self, key):
        return (key in self.mean_dict)

    def get(self, key):
        return self.mean_dict[key]
    
    def as_dict(self):
        return self.mean_dict
        
    def reset(self):
        self.mean_dict = dict()
        self.total_weight = 0

class TimeTracker(object):
    def __init__(self):
        super().__init__()
        self.start_epoch()

    def start_epoch(self):
        self.epoch_start_time = time.time()
        self.n_iterations = 0
        self.batch_time_sum = 0
        self.start_points = dict()
        self.end_points = dict()
        self.durations = dict()
        self.count = dict()

    def start_batch(self):
        self.batch_start_time = time.time()

    def finish_batch(self):
        self.n_iterations += 1
        self.batch_end_time = time.time()
        self.batch_time_sum += self.batch_end_time - self.batch_start_time

    def as_dict(self):
        return self.durations

    def get_batch_time(self):
        return self.batch_time_sum / self.n_iterations

    def get_iteration_time(self):
        return (self.batch_end_time - self.epoch_start_time) / self.n_iterations

# Helper functions for distributed training from https://github.com/zju3dv/NeuralRecon.
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()