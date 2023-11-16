import os
import json

import numpy as np
import pandas as pd
import torch
import torchvision
import coloredlogs, logging
coloredlogs.install()
import h5py

from .scannet import get_scenes, read_pose, read_intrinsics, read_rgb, read_depth

def resize_intrinsics(K, fact_x, fact_y):
    K[0, 0] *= fact_x
    K[1, 1] *= fact_y
    K[0, 2] *= fact_x
    K[1, 2] *= fact_y
    return K

def crop_intrinsics(K, crop_x, crop_y):
    K[0, 2] -= crop_x
    K[1, 2] -= crop_y
    return K

class MatchingDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, split, tuple_size, n_samples=None, jitter=None, \
            shuffle_tuple=True, preprocess_dir="overlap"):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.data_split_dir = os.path.join(data_dir, "scans" + ("_test" if split == "test" else ""))
        self.scenes = get_scenes(data_dir, split)
        if "megadepth" in self.data_split_dir:
            self.scenes = [str(s).zfill(4) for s in self.scenes]
        self.tuple_size = tuple_size
        self.shuffle_tuple = shuffle_tuple # should not matter, especially as long as we do not encode image ids in the gnn
        self.n_samples = n_samples
        self.exclude_set = set()
        self.preprocess_dir = preprocess_dir
        self.sampled_tuples = self.start_epoch(make_exclude_set=True)
        tmp_scenes = []
        for scene in self.scenes:
            if not scene in self.exclude_set:
                tmp_scenes.append(scene)
        self.scenes = tmp_scenes
        self.jitter = jitter

    def make_tuple(self, start_key, candidates):
        n_trials = 0
        result_tuple = []
        while len(result_tuple) < self.tuple_size and n_trials < 5 * self.tuple_size:
            key = start_key
            result_tuple = []
            for _ in range(self.tuple_size):
                next_key = np.random.choice(candidates[key], 1)[0]
                if next_key not in result_tuple:
                    result_tuple.append(next_key)
                key = next_key
            n_trials += 1
        result_tuple = list(result_tuple)
        if self.shuffle_tuple:
            np.random.shuffle(result_tuple)
        if len(result_tuple) < self.tuple_size:
            return None
        return result_tuple

    def start_epoch(self, make_exclude_set=False):
        tuples = []
        for scene in self.scenes:
            with open(os.path.join(self.data_dir, self.preprocess_dir, os.path.relpath( \
                self.data_split_dir, self.data_dir), scene + ".json")) as f:
                match_candidates = dict()
                loaded = json.load(f)
            if len(loaded) < self.tuple_size:
                continue
            for k, v in loaded.items():
                if len(v) > 0:
                    match_candidates[int(k)] = v
            # select samples per scene
            n_candidates = len(match_candidates)
            if self.n_samples is None:
                n_samples = int(n_candidates / self.tuple_size)
            else:
                n_samples = self.n_samples
            if n_candidates < n_samples:
                logging.info("Scene {} has only {} images. Proceeding with fewer images.".format(scene, n_candidates))
                n_samples = n_candidates
            start_images = np.random.choice(list(match_candidates.keys()), n_samples, replace=False)
            for start_image in start_images:
                result_tuple = self.make_tuple(start_image, match_candidates)
                n_trials = 0
                while result_tuple is None and (not make_exclude_set or n_trials < 3 * self.tuple_size):
                    alternative_start_image = np.random.choice(list(match_candidates.keys()), 1)[0]
                    result_tuple = self.make_tuple(alternative_start_image, match_candidates)
                    n_trials += 1
                if result_tuple is None:
                    if make_exclude_set:
                        logging.warn("Added scene {} to exclude list".format(scene))
                        self.exclude_set.add(scene)
                    else:
                        logging.error("Failed to make a tuple from alternative start image, scene {}".format(scene))
                        exit()
                else:
                    tuples.append((scene, result_tuple))

        self.sampled_tuples = tuples
        return self.sampled_tuples
    
    def apply_color_jitter(self, rgb, jitter_params):
        if jitter_params is None:
            jitter_params = self.get_color_jitter_params()
        fn_idx, brightness, contrast, saturation, hue = jitter_params
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                rgb = torchvision.transforms.functional.adjust_brightness(rgb, brightness)
            elif fn_id == 1 and contrast is not None:
                rgb = torchvision.transforms.functional.adjust_contrast(rgb, contrast)
            elif fn_id == 2 and saturation is not None:
                rgb = torchvision.transforms.functional.adjust_saturation(rgb, saturation)
            elif fn_id == 3 and hue is not None:
                rgb = torchvision.transforms.functional.adjust_hue(rgb, hue)
        return rgb
    
    def get_color_jitter_params(self):
        centered_at_1 = (1. - self.jitter, 1. + self.jitter)
        centered_at_0 = (-self.jitter, self.jitter)
        jitter_params = torchvision.transforms.ColorJitter.get_params(brightness=centered_at_1, contrast=centered_at_1, \
            saturation=centered_at_1, hue=centered_at_0)
        return jitter_params
    
    def crop(self, rgb, depth, intr, center=True):
        h, w = depth.shape
        # crop a square
        if w > h:
            if center:
                left = int((w - h) / 2.)
            else:
                left = np.random.randint(0, w - h + 1)
                assert left <= w - h
            right = left + h
            top, bottom = 0, h
        else:
            if center:
                top = int((h - w) / 2.)
            else:
                top = np.random.randint(0, h - w + 1)
                assert top <= h - w
            bottom = top + w
            left, right = 0, w
        intr = crop_intrinsics(intr, left, top)
        rgb = rgb[:, top:bottom, left:right]
        depth = depth[top:bottom, left:right]
        return rgb, depth, intr
    
    def __getitem__(self, index):
        data = dict()
        scene, ids = self.sampled_tuples[index]
        data["scene"] = scene
        data["ids"] = ids
        
        # read per-scene intrinsics
        if "scannet" in self.data_split_dir:
            curr_intr = read_intrinsics(self.data_split_dir, scene)
        
        # precompute jitter params for images of a tuple
        jitter_params = None
        if self.jitter is not None:
            jitter_params = self.get_color_jitter_params()
        
        for i, id in enumerate(ids):
            # read per-image intrinsics
            if "matterport" in self.data_split_dir or "megadepth" in self.data_split_dir:
                curr_intr = read_intrinsics(self.data_split_dir, scene, id)
            data["intr" + str(i)] = curr_intr.copy().astype(np.float32)

            # read pose
            pose = read_pose(self.data_split_dir, scene, id).astype(np.float32)
            data["pose" + str(i)] = pose

            # read rgb and depth
            rgb = torchvision.transforms.ToTensor()(read_rgb(self.data_split_dir, scene, id))
            if "megadepth" in self.data_split_dir:
                depth = np.array(h5py.File(os.path.join(self.data_split_dir, scene, "depth", str(id) + ".h5"), 'r')["depth"])
                # crop
                center = True if self.split == "test" else False
                rgb, depth, data["intr" + str(i)] = self.crop(rgb, depth, data["intr" + str(i)], center=center)
            else:
                depth = read_depth(self.data_split_dir, scene, id)
            data["depth" + str(i)] = depth
            # resize rgb in case of scene with large images in scannet
            if rgb.shape[2] == 1296 and rgb.shape[1] == 968:
                # pad last dim by 0, 0 and second to last by 2, 2
                rgb = torch.nn.functional.pad(rgb, (0, 0, 2, 2), "constant", 0)
                data["intr" + str(i)][1, 2] = data["intr" + str(i)][1, 2] + 2
            
            # resize rgb to the size of the depth map
            resize_size = depth.shape
            if resize_size[1] != rgb.shape[2] or resize_size[0] != rgb.shape[1]:
                fact_x, fact_y = resize_size[1] / rgb.shape[2], resize_size[0] / rgb.shape[1]
                data["intr" + str(i)] = resize_intrinsics(data["intr" + str(i)], fact_x, fact_y)
                rgb = torchvision.transforms.functional.resize(rgb, size=resize_size)

            # add color jitter
            if self.jitter is not None:
                rgb = self.apply_color_jitter(rgb, jitter_params)

            # convert to gray
            gray = torchvision.transforms.functional.rgb_to_grayscale(rgb)

            data["image" + str(i)] = gray

        return data

    def __len__(self):
        return len(self.sampled_tuples)

    def write_sampled_tuples(self, file_path):
        all_scenes = []
        all_ids = []
        for i in range(1500):
            scene, ids = self.sampled_tuples[i]
            all_scenes.append(scene)
            all_ids.append(ids)
        df = pd.DataFrame({'scene': all_scenes, 'ids': all_ids,})
        df.to_csv(file_path, index=False)

    def read_sampled_tuples(self, file_path):
        tmp_sampled_tuples = []
        df = pd.read_csv(file_path, dtype=str)
        for scene, ids in zip(df['scene'], df['ids']):
            tmp_sampled_tuples.append((scene, eval(ids)))
        self.sampled_tuples = tmp_sampled_tuples
