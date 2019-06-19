# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import hashlib
import cv2
import json
from common.camera import *

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def wild2human36m_xianhui(data, w, h):
    xianhui_kpts_name = sorted(list(data["1"].keys()))
    xianhui_kpts_dict = {}
    xianhui_kpts_inverse_dict = {}
    for i in range(len(xianhui_kpts_name)):
        xianhui_kpts_dict[xianhui_kpts_name[i]] = i
        xianhui_kpts_inverse_dict[i] = xianhui_kpts_name[i]

    human36m_kpts_name = ['Pelvis', 'RHip', 'RKnee', 'RAnkle','LHip','LKnee',
                        'LAnkle','Spine1','Neck', 'Head','Site','LShoulder',
                        'LElbow','LWrist','RShoulder', 'RElbow','RWrist']
    human36m_kpts_dict = {}
    human36m_kpts_inverse_dict = {}
    for i in range(17):
        human36m_kpts_dict[human36m_kpts_name[i]] = i
        human36m_kpts_inverse_dict[i] = human36m_kpts_name[i]
        
    pair = [(0,2),(1,19),(2,15),(3,12),
            (4,10),(5,6),(6,3),(7,0),(8,11),
            (9,1),(10,1),(11,9),(12,5),
            (13,4),(14,18),(15,14),(16,13)]

    pts_3d = []
    pts_2d = []

    for i in range(len(data)-1):
        pts_3d_ = []
        pts_2d_ = []

        for p in pair:
            # print("p : ", p)
            # print(xianhui_kpts_inverse_dict[p[1]])
            try:
                x = data[str(i+1)][xianhui_kpts_inverse_dict[p[1]]]["x"]
                y = data[str(i+1)][xianhui_kpts_inverse_dict[p[1]]]["z"]
                z = data[str(i+1)][xianhui_kpts_inverse_dict[p[1]]]["y"]
                x_2d = data[str(i+1)][xianhui_kpts_inverse_dict[p[1]]]["view_x"]
                y_2d = data[str(i+1)][xianhui_kpts_inverse_dict[p[1]]]["view_y"]
                pts_3d_.append([x,y,z])
                # pts_2d_.append([(x_2d), (y_2d)])
                pts_2d_.append([(x_2d)*w, (1-y_2d)*h])
            except:
                print(p)
                print(p[1])
                print(xianhui_kpts_inverse_dict)
                print(xianhui_kpts_inverse_dict[p[1]])
            
        pts_3d.append(pts_3d_)
        pts_2d.append(pts_2d_)
        
        
    pts_world = np.array(pts_3d)
    pts_2d_gt = np.array(pts_2d)

    return pts_world, pts_2d_gt

def wild2human36m(keypoints_wild, tag):
    # pair : [human3.6, wild]
    # 亚鲁 format
    selected_keypoints_id = [(10, 2), (9, 29), (8, 18), (14, 22), 
                             (11, 27), (15, 23), (12, 28), (16, 8), 
                             (13, 14), (7, 17), (0, 0), (1, 19), 
                             (2, 20), (3, 5), (4, 24), (5, 25), (6, 11)]

    pair_dict = {}
    for pair in selected_keypoints_id:
        pair_dict[pair[0]] = pair[1]
        
    # wild : use id as key
    keypoints_converted = {}
    for i, item in enumerate(keypoints_wild.keys()):
        if tag == "3d":
            keypoints_converted[keypoints_wild[item]["id"]] = {"name" : item,
                                                               "position" : (keypoints_wild[item]["x"], 
                                                                             keypoints_wild[item]["y"],
                                                                             keypoints_wild[item]["z"], )}
        if tag == "2d":
            keypoints_converted[keypoints_wild[item]["id"]] = {"name" : item,
                                                               "position" : (keypoints_wild[item]["x"], 
                                                                             keypoints_wild[item]["y"])}
            
    # insert neck since 亚鲁 format neck is head
    if tag == "3d":
        keypoints_converted[29] = {
            "name" : "neck_2",
            "position" : ((keypoints_converted[2]["position"][0] + keypoints_converted[18]["position"][0])/2,
                          (keypoints_converted[2]["position"][1] + keypoints_converted[18]["position"][1])/2,
                          (keypoints_converted[2]["position"][2] + keypoints_converted[18]["position"][2])/2,)
        }
    if tag == "2d":
        keypoints_converted[29] = {
            "name" : "neck_2",
            "position" : ((keypoints_converted[2]["position"][0] + keypoints_converted[18]["position"][0])/2,
                          (keypoints_converted[2]["position"][1] + keypoints_converted[18]["position"][1])/2)}    
    
    # human3.6 : use id as key
    keypoints_h36m = {}
    for pair in selected_keypoints_id:
        keypoints_h36m[pair[0]] = keypoints_converted[pair[1]]["position"]
    
    # convert to np array
    return np.array([keypoints_h36m[i] for i in range(17)])

def fetch(dataset, 
          keypoints, 
          subjects, 
          stride, 
          action_filter=None, 
          subset=1, 
          parse_3d_poses=True):

    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
                    if 'intrinsics' in cam:
                        out_camera_params.append(cam['intrinsics'])
                    
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
    
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    

    return out_camera_params, out_poses_3d, out_poses_2d