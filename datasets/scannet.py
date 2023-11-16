import os

import cv2
import numpy as np
import pandas as pd
import coloredlogs, logging
coloredlogs.install()

def get_scenes(data_dir, split):
    for file in os.listdir(data_dir):
        if file.endswith("_{}.txt".format(split)):
            split_txt = file
    split_txt = os.path.join(data_dir, split_txt)
    scenes = pd.read_csv(split_txt, names=["scenes"], header=None)
    return scenes["scenes"].tolist()

def read_intrinsics(data_split_dir, scene, type="intrinsic_color"):
    cam_intr = np.loadtxt(os.path.join(data_split_dir, scene, "intrinsic", "{}.txt".format(type)), delimiter=' ')
    if not np.all(np.isfinite(cam_intr)):
        logging.warning("Non-finite intrinsics for {}".format(scene))
        return None
    return cam_intr

def read_pose(data_split_dir, scene, id, verbose=True):
    cam_pose = np.loadtxt(os.path.join(data_split_dir, scene, "pose", str(id) + ".txt"), delimiter=' ')
    if not np.all(np.isfinite(cam_pose)):
        if verbose:
            logging.warning("Non-finite pose for {}, id {}".format(scene, id))
        return None
    return cam_pose

def read_depth(data_split_dir, scene, id):
    depth_img = cv2.imread(os.path.join(data_split_dir, scene, "depth", str(id) + ".png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_img /= 1000.  # depth is saved in 16-bit PNG in millimeters
    return depth_img

def read_rgb(data_split_dir, scene, id, gray=False):
    path = os.path.join(data_split_dir, scene, "color", str(id) + ".jpg")
    if gray:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)