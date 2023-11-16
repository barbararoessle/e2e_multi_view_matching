import os
import argparse
import json
import shutil
from multiprocessing import Pool
import coloredlogs, logging
coloredlogs.install()

import cv2
import h5py
import numpy as np

from datasets.scannet import get_scenes
from datasets.matching_dataset import resize_intrinsics

class ConvertMegadepthScene(object):
    def __init__(self, scene_info_dir, dataset_dir, out_dataset_dir, image_size, valid_list):
        self.scene_info_dir = scene_info_dir
        self.dataset_dir = dataset_dir
        self.out_dataset_dir = out_dataset_dir
        self.image_size = image_size
        self.valid_list = valid_list

    def __call__(self, scene_info_file):
        scene = scene_info_file.split('.')[0]
        split_dir = "scans"
        if scene in train_scenes or scene in val_scenes:
            upper_overlap_limit = 0.7 # from SuperGlue
        elif scene in test_scenes:
            split_dir = split_dir + "_test"
            upper_overlap_limit = 0.4 # from SuperGlue
        else:
            return
        logging.info("Start processing scene {}".format(scene))
        info = np.load(os.path.join(self.scene_info_dir, scene_info_file), allow_pickle=True)
        img_paths = info["image_paths"]
        n_entries = len(img_paths)
        depth_paths = info["depth_paths"]
        assert len(depth_paths) == n_entries
        intrinsics = info["intrinsics"]
        assert len(intrinsics) == n_entries
        poses = info["poses"]
        assert len(poses) == n_entries
        overlap_matrix = info["overlap_matrix"]
        assert overlap_matrix.shape[0] == n_entries and overlap_matrix.shape[1] == n_entries
        count = 0
        sum_overlapping_imgs = 0
        matches = dict()
        valid_path_mask = np.array([False if p is None else True for p in img_paths], dtype=bool)
        n_valid_paths = valid_path_mask.sum()
        img_ids = np.zeros(n_entries, dtype=int)
        img_ids[valid_path_mask] = np.arange(n_valid_paths)
        for i, (rgb_path, depth_path, intr_3x3, pose) in enumerate(zip(img_paths, depth_paths, intrinsics, poses)):
            if rgb_path is not None and depth_path is not None:
                rgb_filename = rgb_path.split('/')[-1]
                file_id = "{}/{}".format(scene, rgb_filename)
                if file_id in valid_list:
                    # read rgb and depth
                    rgb_path = os.path.join(self.dataset_dir, rgb_path)
                    depth_path = os.path.join(self.dataset_dir, '/'.join(depth_path.split('/')[-5:]))
                    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                    depth = np.array(h5py.File(depth_path, 'r')["depth"])
                    if bgr.shape[0] == depth.shape[0] and bgr.shape[1] == depth.shape[1]:
                        # compute overlap
                        overlap_row = overlap_matrix[i, :]
                        overlap_row_valid = overlap_row > 0.
                        overlap_col = overlap_matrix[:, i]
                        overlap_col_valid = overlap_col > 0.
                        overlap = (overlap_row + overlap_col) * 0.5
                        overlap_in_range = (overlap >= 0.1) & (overlap <= upper_overlap_limit)
                        mask = (overlap_in_range & overlap_col_valid & overlap_row_valid)
                        overlapping_imgs = img_ids[mask & valid_path_mask]
                        matches[str(count)] = overlapping_imgs.tolist()
                        sum_overlapping_imgs += len(overlapping_imgs)
                        # resize to smaller dimension to specified size while keeping the aspect ratio
                        h, w = bgr.shape[0], bgr.shape[1]
                        if w > h:
                            new_h, new_w = self.image_size, int(self.image_size * float(w) / float(h))
                        else:
                            new_h, new_w = int(self.image_size * float(h) / float(w)), self.image_size
                        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                        intr = np.eye(4)
                        intr[:3, :3] = intr_3x3
                        intr = resize_intrinsics(intr, float(new_w) / float(w), float(new_h) / float(h))
                        # write color
                        scene_dir = os.path.join(self.out_dataset_dir, split_dir, scene)
                        color_dir = os.path.join(scene_dir, "color")
                        os.makedirs(color_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(color_dir, str(count) + ".jpg"), bgr)
                        # write depth
                        depth_dir = os.path.join(scene_dir, "depth")
                        os.makedirs(depth_dir, exist_ok=True)
                        with h5py.File(os.path.join(depth_dir, str(count) + ".h5"), 'w') as hf:
                            hf.create_dataset("depth",  data=depth)
                        # write intrinsic
                        intr_dir = os.path.join(scene_dir, "intrinsic")
                        os.makedirs(intr_dir, exist_ok=True)
                        np.savetxt(os.path.join(intr_dir, str(count) + ".txt"), intr)
                        # write pose
                        pose_dir = os.path.join(scene_dir, "pose")
                        os.makedirs(pose_dir, exist_ok=True)
                        np.savetxt(os.path.join(pose_dir, str(count) + ".txt"), np.linalg.inv(pose))
                    else:
                        logging.error("RGB and depth files do not match in dimension")
                        exit()
                    count += 1
                else:
                    logging.warn("{} is not in valid list".format(file_id))
        overlap_dir = os.path.join(self.out_dataset_dir, "overlap", split_dir)
        os.makedirs(overlap_dir, exist_ok=True)
        json.dump(matches, open(os.path.join(overlap_dir, scene + ".json"), 'w'), indent=4)
        logging.info("Scene {} has {} valid files, {} overlap on average".format(scene, count, sum_overlapping_imgs / count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert MegaDepth into ScanNet format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_dir', type=str, default=None, help='path to dataset directory')
    parser.add_argument('--image_size', type=int, default=640, help='length of smaller image dimension')

    opt = parser.parse_args()

    out_dataset_dir = opt.dataset_dir + "_{}".format(opt.image_size)

    # read list of train/val/test scenes
    train_scenes = [str(s).zfill(4) for s in get_scenes(opt.dataset_dir, split="train")]
    val_scenes = [str(s).zfill(4) for s in get_scenes(opt.dataset_dir, split="val")]
    test_scenes = [str(s).zfill(4) for s in get_scenes(opt.dataset_dir, split="test")]

    # create output directories
    os.makedirs(os.path.join(out_dataset_dir, "scans"), exist_ok=True)
    os.makedirs(os.path.join(out_dataset_dir, "scans_test"), exist_ok=True)
    os.makedirs(os.path.join(out_dataset_dir, "overlap", "scans"), exist_ok=True)
    os.makedirs(os.path.join(out_dataset_dir, "overlap", "scans_test"), exist_ok=True)

    shutil.copyfile(os.path.join(opt.dataset_dir, "megadepth_train.txt"), os.path.join(out_dataset_dir, "megadepth_train.txt"))
    shutil.copyfile(os.path.join(opt.dataset_dir, "megadepth_val.txt"), os.path.join(out_dataset_dir, "megadepth_val.txt"))
    shutil.copyfile(os.path.join(opt.dataset_dir, "megadepth_test.txt"), os.path.join(out_dataset_dir, "megadepth_test.txt"))

    # read valid file list from https://github.com/ubc-vision/COTR/blob/master/sample_data/jsons/megadepth_valid_list.json
    valid_list_path = os.path.join(opt.dataset_dir, "megadepth_valid_list.json")
    with open(valid_list_path, 'r') as jf:
        valid_list = list(json.load(jf))

    # iterate all scene info files
    scene_info_dir = os.path.join(opt.dataset_dir, "scene_info")
    tmp_scene_info_files = sorted(os.listdir(scene_info_dir))
    scene_info_files = []
    done_scene_jsons = os.listdir(os.path.join(out_dataset_dir, "overlap", "scans"))
    done_scene_jsons = done_scene_jsons + os.listdir(os.path.join(out_dataset_dir, "overlap", "scans_test"))
    for s in tmp_scene_info_files:
        if s.split('.')[0] + ".json" in done_scene_jsons:
            continue
        else:
            scene_info_files.append(s)
    logging.info("Scenes to be extracted from: {}".format(len(scene_info_files)))
    pool = Pool(12)
    scene_converter = ConvertMegadepthScene(scene_info_dir, opt.dataset_dir, out_dataset_dir, opt.image_size, valid_list)
    pool.map(scene_converter, scene_info_files)
