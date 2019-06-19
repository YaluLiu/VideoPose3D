# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates, camera_calibration
from common.utils import wild2human36m_xianhui
import matplotlib.pyplot as plt
import json
import cv2

def xianhui_data_processing(data_path, img_paths, img_cal_paths, cam_id, chess_board_shape = (7, 5)):
    # camera calibration
    intrinsics, distortion = camera_calibration(chess_board_shape, img_cal_paths)
    print("calibrate camera")
    
    # load simulation data
    with open(data_path, 'r') as f:
        data = json.loads(f.read())
    
    imgs = np.array([plt.imread(img) for img in img_paths])
    _, h, w, _ = imgs.shape

    pts_3d, pts_2d = wild2human36m_xianhui(data, w, h)
    print("load 3d and 2d skeletons")
    
    # estimate extrinsics
    obj_pts = pts_3d.reshape(-1,3,1)
    img_pts = pts_2d.reshape(-1,2,1)
    _, rvec, tvec, _ = cv2.solvePnPRansac(obj_pts, img_pts, intrinsics, distortion)
    rmat, _ = cv2.Rodrigues(rvec)
    extrinsics = np.zeros((3,4))
    extrinsics[:,:3] = rmat
    extrinsics[:,3] = tvec.reshape(-1)
    print("estimate extrinsics")

    camera_params = {}
    camera_params["intrinsics"] = intrinsics
    camera_params["extrinsics"] = extrinsics
    camera_params["rvec"] = rvec
    camera_params["tvec"] = tvec
    camera_params["distortion"] = distortion
    camera_params["res_w"] = w
    camera_params["res_h"] = h
    camera_params["id"] = cam_id
    
    return pts_3d, pts_2d, camera_params, imgs

xianhui_skeleton = Skeleton(parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15],
       joints_left=[4, 5, 6, 11, 12, 13],
       joints_right=[1, 2, 3, 14, 15, 16])

class XianhuiDataset(MocapDataset):
    def __init__(self, cameras, data):
        super().__init__(fps=60, skeleton=xianhui_skeleton)
        self._cameras = cameras
        self._data = data