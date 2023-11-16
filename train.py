import os
import argparse
import datetime, time
import json
import math
import coloredlogs, logging
coloredlogs.install()

import torch
import numpy as np

from helpers import add_generic_arguments, save_ckpt, load_ckpt, get_exp_dir, setup_tb, get_parameters, has_finite_gradients, \
    namespace_to_dict, dict_to_namespace, TimeTracker, MeanTracker, run_super_point, compute_gt_matches, run_matcher, \
    is_main_process, synchronize
from datasets.matching_dataset import MatchingDataset
from datasets.sampling import create_sequential_subsets
from models.models.superpoint import SuperPoint
from models.models.multi_view_matcher import MultiViewMatcher

def compute_number_pairs(tuple_size):
    return np.arange(tuple_size).sum()

def scale_lr(tuple_size, batch_size, n_gpus, lr, decay_rate, start_decay, end_decay, dataset):
    # scale learning rate and decay params based on tuple size, batch size and number of gpus
    orig_batch_size = 64.0 # single gpu
    orig_tuple_size = 2.
    if "megadepth" in dataset:
        orig_batch_size = 16.0
    fact = float(tuple_size * batch_size * n_gpus) / (orig_tuple_size * orig_batch_size)
    new_lr = math.sqrt(fact) * lr
    new_start_decay = int(start_decay / fact)
    new_end_decay = int(end_decay / fact)
    new_decay_rate = decay_rate ** fact
    return new_lr, new_decay_rate, new_start_decay, new_end_decay

def combine_losses(losses, n_pairs, pose_match_ratio, rot_weight, trans_weight):
    losses = {k : v / float(n_pairs) for k, v in losses.items()}
    pose_loss = rot_weight * losses["rot_loss"] + trans_weight * losses["transl_loss"]
    total_loss = (1. - pose_match_ratio) * losses["match_loss"] + pose_match_ratio * pose_loss
    return total_loss, losses

class Validator(object):
    def __init__(self, val_dataset, n_pairs):
        validate_on_at_least_n_samples = 60000
        val_sample_count = len(val_dataset)
        if val_sample_count < validate_on_at_least_n_samples:
            self.val_subsets = [val_dataset,]
            logging.info("Small validation set -> no need to create subsets")
        else:
            self.val_subsets = create_sequential_subsets(val_dataset, validate_on_at_least_n_samples)
            logging.info("Create {} validation subsets with length {} or {}".format(len(self.val_subsets), \
                len(self.val_subsets[0]), len(self.val_subsets[-1])))
        self.val_subset_index = 0
        self.n_pairs = n_pairs

    def initialize_metric(self, min_val_loss):
        self.min_val_loss = min_val_loss
    
    def next_subset_index(self):
        curr_subset_index = self.val_subset_index
        self.val_subset_index += 1
        if self.val_subset_index == len(self.val_subsets):
            self.val_subset_index = 0
        return curr_subset_index

    def validate(self, opt, super_point, matcher, optimizer, pose_match_ratio, tb_writer, exp_dir, epoch, step):
        with torch.no_grad():
            matcher.eval()
            
            curr_val_subset = self.val_subsets[self.next_subset_index()]
            if opt.n_gpus > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(curr_val_subset, shuffle=False)
                val_loader = torch.utils.data.DataLoader(
                    curr_val_subset,
                    batch_size=opt.batch_size,
                    sampler=val_sampler,
                    num_workers=opt.n_workers,
                    pin_memory=False,
                    drop_last=True
                )
            else:
                val_loader = torch.utils.data.DataLoader(curr_val_subset, opt.batch_size, shuffle=False, \
                    num_workers=opt.n_workers, drop_last=True)
            
            val_metrics = MeanTracker()
            time_tracker = TimeTracker()
            time_tracker.start_epoch()
            for data in val_loader:
                time_tracker.start_batch()
                run_super_point(opt, data, super_point)
                # compute ground truth matches
                compute_gt_matches(opt, data)
                losses, result = run_matcher(opt, data, matcher)
                val_loss, losses = combine_losses(losses, self.n_pairs, pose_match_ratio, opt.rot_weight, opt.trans_weight)
                curr_metrics_dict = {"val_loss" : val_loss.item()}
                if opt.pose_loss:
                    curr_metrics_dict.update({k : v.item() for k, v in losses.items()})
                val_metrics.add(curr_metrics_dict)
                time_tracker.finish_batch()

            mean_val_loss = val_metrics.get("val_loss")
            val_loss = torch.tensor([mean_val_loss,], device=torch.device('cuda'))

            if opt.n_gpus > 1:
                torch.distributed.all_reduce(val_loss)
            val_loss = float(val_loss) / float(opt.n_gpus)

            if is_main_process():
                # logging
                logging.info("batch time {:.3f}, it time {:.3f}, val loss {:.3f}".format(time_tracker.get_batch_time(), \
                    time_tracker.get_iteration_time(), val_loss))
                tb_writer.add_scalars("loss", {"val" : val_loss}, step)
                if opt.pose_loss:
                    mean_match_loss = val_metrics.get("match_loss")
                    mean_rot_loss = val_metrics.get("rot_loss")
                    mean_transl_loss = val_metrics.get("transl_loss")
                    logging.info("    (match loss {:.3f}, rot loss {:.3f}, trans loss {:.3f})".format( \
                        mean_match_loss, mean_rot_loss, mean_transl_loss))
                    tb_writer.add_scalars("match_loss", {"val" : mean_match_loss}, step)
                    tb_writer.add_scalars("rot_loss", {"val" : mean_rot_loss}, step)
                    tb_writer.add_scalars("transl_loss", {"val" : mean_transl_loss}, step)

                # save checkpoint
                file_names = ["last_model.ckpt", "{:0>6}_model.ckpt".format(epoch)]
                if self.min_val_loss > val_loss:
                    self.min_val_loss = val_loss
                    file_names.append("best_model.ckpt")
                for file_name in file_names:
                    save_ckpt(epoch, matcher, optimizer, val_loss, pose_match_ratio, file_name, exp_dir)
            
        matcher.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training multi-view matcher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser = add_generic_arguments(parser)

    # arguments specific to training
    parser.add_argument('--local_rank', type=int, default=0, help='node rank')
    parser.add_argument('--exp_name', type=str, default=None, 
        help='Specify an experiment name to resume training from, None will start a new training')
    parser.add_argument(
        '--init_exp_name', type=str, default=None, 
        help='experiment name from which parameters are loaded')
    parser.add_argument(
        '--batch_size', type=int, default=8, # 8 (tuple_size 5, 3 gpus) 12 (tuple_size 5, 2 gpus) 32 (tuple_size 2, 2 gpus)
        help='batch size')
    parser.add_argument(
        '--n_workers', type=int, default=5, # 5 (batch_size 8) 6 (batch_size 12) 14 (batch_size 32)
        help='number workers')
    parser.add_argument(
        '--pose_loss', action='store_true',
        help='apply a pose loss')
    parser.add_argument(
        '--final_pose_match_ratio', type=float, default=0.99,
        help='ratio of pose loss to match loss after gradual increase')
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='learning rate')
    parser.add_argument(
        '--decay_rate', type=float, default=0.999992,
        help='decay rate')
    parser.add_argument(
        '--n_epochs', type=int, default=1000,
        help='number of epochs')
    parser.add_argument(
        '--i_train', type=int, default=100,
        help='log train statistics every n iterations')

    opt = parser.parse_args()

    # set some dataset dependent params
    # * superpoint params
    # * number of samples to be drawn from each scene, 
    # * reprojection error thresholds for matching
    # * learning rate decay start and end iterations
    # * color jitter for data augmentation
    if "scannet" in opt.dataset:
        opt.remove_borders = 12 # calibration causes black image borders
        opt.max_keypoints = 400
        opt.nms_radius = 4
        opt.keypoint_threshold = 0.001
        opt.n_samples = 200
        opt.match_reproj_err = 5.
        opt.unmatch_reproj_err = 15.
        opt.start_decay = 1e5
        opt.end_decay = 9e5
        opt.color_jitter = 0.2
    elif "matterport" in opt.dataset:
        opt.remove_borders = 4 # calibration causes black image borders
        opt.max_keypoints = 400
        opt.nms_radius = 4
        opt.keypoint_threshold = 0.001
        opt.n_samples = None # scenes are very different size, None adapts to scene size
        opt.match_reproj_err = 5.
        opt.unmatch_reproj_err = 15.
        opt.start_decay = 1e5
        opt.end_decay = 9e5
        opt.color_jitter = 0.2
    elif "megadepth" in opt.dataset:
        opt.remove_borders = 0
        opt.max_keypoints = 1024
        opt.nms_radius = 3
        opt.keypoint_threshold = 0.005
        opt.n_samples = 100 if opt.tuple_size > 2 else 200
        opt.match_reproj_err = 5.
        opt.unmatch_reproj_err = 10.
        opt.start_decay = 5e4
        opt.end_decay = 9e5
        opt.color_jitter = None
    else:
        logging.error("Dataset {} is not supported.".format(opt.dataset))
        exit()
    
    # scale rotation and translation loss, so that their sum initially has the same magnitude as the match loss
    if opt.pose_loss:
        if "scannet" in opt.dataset:
            if opt.tuple_size == 2:
                opt.rot_weight = 1597.
                opt.trans_weight = 270.
            elif opt.tuple_size == 5:
                opt.rot_weight = 726.
                opt.trans_weight = 244.
            else:
                logging.error("Specify rotation and translation loss weighting for tuple size {}, \
                            so that their sum initially has the same magnitude as the match loss.".format(opt.tuple_size))
                exit()
        elif "matterport" in opt.dataset:
            if opt.tuple_size == 2:
                opt.rot_weight = 717.
                opt.trans_weight = 591.
            elif opt.tuple_size == 5:
                opt.rot_weight = 240.
                opt.trans_weight = 263.
            else:
                logging.error("Specify rotation and translation loss weighting for tuple size {}, \
                            so that their sum initially has the same magnitude as the match loss.".format(opt.tuple_size))
                exit()
        elif "megadepth" in opt.dataset:
            if opt.tuple_size == 2:
                opt.rot_weight = 710.
                opt.trans_weight = 348.
            elif opt.tuple_size == 5:
                opt.rot_weight = 661.
                opt.trans_weight = 366.
            else:
                logging.error("Specify rotation and translation loss weighting for tuple size {}, \
                            so that their sum initially has the same magnitude as the match loss.".format(opt.tuple_size))
                exit()
    else:
        opt.rot_weight = 0.
        opt.trans_weight = 0.

    # use gradient clipping with the pose loss
    if opt.pose_loss:
        opt.grad_clip = 0.1
    else:
        opt.grad_clip = -1.

    # with enough data (scannet, matterport), multi-view matching benefits from more cross attention
    if opt.tuple_size > 2 and ("scannet" in opt.dataset or "matterport" in opt.dataset):
        opt.gnn_layers = 7
        opt.cross_attention_layers = 3
    else:
        opt.gnn_layers = 9
        opt.cross_attention_layers = 1

    opt.n_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    logging.info("Number of gpus: {}".format(opt.n_gpus))
    if opt.n_gpus > 1:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    opt.lr, opt.decay_rate, opt.start_decay, opt.end_decay = scale_lr(opt.tuple_size, opt.batch_size, \
        opt.n_gpus, opt.lr, opt.decay_rate, opt.start_decay, opt.end_decay, opt.dataset)
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    resume = opt.exp_name is not None
    if resume:
        tmp_local_rank = opt.local_rank
        tmp_n_gpu = opt.n_gpus
        exp_dir, config_json = get_exp_dir(opt.checkpoint_dir, opt.exp_name)
        with open(config_json, 'r') as cf:
            cfg_dict = json.load(cf)
        opt = dict_to_namespace(cfg_dict)
        if opt.n_gpus != tmp_n_gpu:
            logging.error("Resume training with same number of GPUs")
            exit()
        opt.local_rank = tmp_local_rank
        opt.n_gpus = tmp_n_gpu
    else:
        if is_main_process():
            opt.exp_name = "{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'))
            exp_dir, config_json = get_exp_dir(opt.checkpoint_dir, opt.exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            with open(config_json, 'w') as cf:
                json.dump(namespace_to_dict(opt), cf, indent=4)
        else:
            exp_dir = None
    print(opt)
    tb_writer = setup_tb(opt.checkpoint_dir, opt.exp_name)

    # setup dataset
    dataset_dir = os.path.join(opt.data_dir, opt.dataset)
    train_dataset = MatchingDataset(dataset_dir, split="train", tuple_size=opt.tuple_size, n_samples=opt.n_samples, \
        jitter=opt.color_jitter)
    val_dataset = MatchingDataset(dataset_dir, split="val", tuple_size=opt.tuple_size, n_samples=opt.n_samples)
    logging.info("Loaded {} train and {} val samples".format(len(train_dataset), len(val_dataset)))
    if opt.n_gpus > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=train_sampler,
            num_workers=opt.n_workers,
            pin_memory=False,
            drop_last=True
        )
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, \
            num_workers=opt.n_workers, drop_last=True)
    
    n_batches = len(train_loader)
    
    # setup networks
    super_point = SuperPoint({
        'nms_radius': opt.nms_radius,
        'keypoint_threshold': opt.keypoint_threshold,
        'max_keypoints': opt.max_keypoints,
        'remove_borders' : opt.remove_borders,
        'fill_with_random_keypoints' : True,
    }).eval().cuda()

    matcher_cfg = {
        'multi_frame_matching': opt.tuple_size > 2,
        'GNN_layers': (['self',] + ['cross',] * opt.cross_attention_layers) * opt.gnn_layers,
        'conf_mlp': True if opt.pose_loss else False,
    }
    matcher = MultiViewMatcher(matcher_cfg).train().cuda()
    if opt.n_gpus > 1:
        matcher = torch.nn.parallel.DistributedDataParallel(
            matcher, device_ids=[opt.local_rank], output_device=opt.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=False
        )
    else:
        matcher = torch.nn.DataParallel(matcher, device_ids=[0])
    
    # setup optimizer
    optimizer = torch.optim.Adam(get_parameters(matcher, blacklist_key="conf_mlp"), lr=opt.lr)
    if opt.pose_loss:
        optimizer.add_param_group({"params" : get_parameters(matcher, whitelist_key="conf_mlp"), "lr" : 1e-4})

    # load initial state
    start_epoch = 0
    min_val_loss = 1e6
    pose_match_ratio = 0.
    pose_match_ratio_increment = 2.5e-5
    successful_updates = 0
    if resume:
        logging.info("Resume training, loading network weights and optimizer state from {}.".format(opt.exp_name))
        matcher, optimizer, epoch, min_val_loss, pose_match_ratio = load_ckpt(exp_dir, matcher, model_id="last", local_rank=opt.local_rank, \
            optimizer=optimizer)
        start_epoch = epoch + 1
    elif opt.init_exp_name is not None:
        init_exp_dir, init_config_json = get_exp_dir(opt.checkpoint_dir, opt.init_exp_name)
        # for the 2nd training stage (with pose loss) the optimizer state from the 1st stage is loaded
        if opt.pose_loss:
            logging.info("Loading network weights and optimizer state from {}.".format(opt.init_exp_name))
            matcher, optimizer, epoch, _, pose_match_ratio = load_ckpt(init_exp_dir, matcher, model_id="best", local_rank=opt.local_rank, \
                optimizer=optimizer)
            start_epoch = epoch + 1
        else:
            logging.info("Loading network weights from {}.".format(opt.init_exp_name))
            matcher, _, _, _, _ = load_ckpt(init_exp_dir, matcher, local_rank=opt.local_rank)
    
    # setup validation
    n_pairs = compute_number_pairs(opt.tuple_size)
    validator = Validator(val_dataset, n_pairs)
    validator.initialize_metric(min_val_loss)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt.decay_rate)

    # train
    time_tracker = TimeTracker()
    for epoch in range(start_epoch, opt.n_epochs):
        if epoch > 0:
            train_dataset.start_epoch()
        if train_sampler is not None: # shuffle correctly in distributed mode
            train_sampler.set_epoch(epoch)

        time_tracker.start_epoch()

        train_metrics = MeanTracker()
        for i, data in enumerate(train_loader):
            time_tracker.start_batch()
            step = epoch * n_batches + (i + 1)

            run_super_point(opt, data, super_point)

            compute_gt_matches(opt, data)

            losses, _ = run_matcher(opt, data, matcher)
            if opt.pose_loss and pose_match_ratio < opt.final_pose_match_ratio:
                pose_match_ratio += pose_match_ratio_increment
                pose_match_ratio = min(pose_match_ratio, 1.)
            train_loss, losses = combine_losses(losses, n_pairs, pose_match_ratio, opt.rot_weight, opt.trans_weight)

            optimizer.zero_grad()
                
            train_loss.backward()
            if not opt.pose_loss or has_finite_gradients(matcher):
                if opt.grad_clip > 0.:
                    torch.nn.utils.clip_grad_value_(matcher.parameters(), opt.grad_clip)
                optimizer.step()
                successful_updates += 1
            
            curr_metrics_dict = {"train_loss" : train_loss.item()}
            if opt.pose_loss:
                curr_metrics_dict.update({k : v.item() for k, v in losses.items()})
            train_metrics.add(curr_metrics_dict)
            time_tracker.finish_batch()

            if step > opt.start_decay and step < opt.end_decay:
                lr_scheduler.step()

            if is_main_process() and (i + 1) % opt.i_train == 0:
                mean_train_loss = train_metrics.get("train_loss")
                logging.info("Epoch {:>3d}({:>4.1f}%), batch time {:.3f}, it time {:.3f}, train loss {:.3f}".format(epoch, \
                    float(i) / float(n_batches) * 100., time_tracker.get_batch_time(), time_tracker.get_iteration_time(), \
                    mean_train_loss))
                tb_writer.add_scalars("loss", {"train" : mean_train_loss}, step)
                if opt.pose_loss:
                    mean_match_loss = train_metrics.get("match_loss")
                    mean_rot_loss = train_metrics.get("rot_loss")
                    mean_transl_loss = train_metrics.get("transl_loss")
                    logging.info("    (match loss {:.1f}, rot loss {:.4f}, trans loss {:.4f}, successful updates {})".format( \
                        mean_match_loss, mean_rot_loss, mean_transl_loss, successful_updates))
                    tb_writer.add_scalars("match_loss", {"train" : mean_match_loss}, step)
                    tb_writer.add_scalars("rot_loss", {"train" : mean_rot_loss}, step)
                    tb_writer.add_scalars("transl_loss", {"train" : mean_transl_loss}, step)
                train_metrics.reset()
                successful_updates = 0

        validator.validate(opt, super_point, matcher, optimizer, pose_match_ratio, tb_writer, exp_dir, epoch, step)
