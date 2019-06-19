import os
import numpy as np
from common.xianhui_dataset import *
from common.camera import *

def load_and_preprocess_cmu_mocap(data_root_path):
    # load 3d data
    print(" load 3d data ")
    subjects = os.listdir(data_root_path)
    data = {}
    cameras = {}
    for subject in subjects:
        # actions
        data[subject] = {}
        for action in os.listdir(os.path.join(data_root_path, subject)):
            action_path = data_root_path + subject + "/" + action
            pts_3d = np.load(action_path + "/points_3d.npy")
            camera_paths = sorted([ os.path.join(action_path, f)+"/cam_parameter.npy" for f in os.listdir(action_path) if f.startswith("cam")])
            camera_params = []
            for camera_path in camera_paths:
                camera_param = np.load(camera_path, allow_pickle=True).item() 
                camera_param["intrinsics"] = camera_param["intrinsics"].reshape(-1)#np.array([camera_param["intrinsics"][0,0], camera_param["intrinsics"][1,1], camera_param["intrinsics"][1,2], camera_param["intrinsics"][0,2], 0, 0, 0, 0, 0])
                camera_params.append(camera_param)

            data[subject][action] = {
                "positions" : pts_3d,
                "cameras" : camera_params
            }
        cameras[subject] = camera_params

    dataset = XianhuiDataset(cameras, data)
    
    # for subject in sorted(dataset.subjects()):
    #     print(" subject : ", subject)
    #     print(" actions : ", sorted(dataset[subject].keys()))
    
    # preprocessing 3d data
    print(" processing 3d data ")
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]        
            positions_3d = []
            for cam in anim['cameras']:
                nframe = anim['positions'].shape[0]
                pts_3d_homo = np.ones((nframe, 17, 4))
                pts_3d_homo[... , :3] = anim['positions']
                pts_3d_homo = pts_3d_homo.reshape(-1, 4).T

                pts_3d_cam = world2camera_cv(pts_3d_homo, cam['rvec'], cam['tvec'])[:3]
                pos_camera = pts_3d_cam.T.reshape(nframe, 17, 3)
                pos_camera[:, 1:] -= pos_camera[:, :1]

                positions_3d.append(pos_camera)
                
            anim['positions_3d'] = positions_3d

    # load 2d keypoints
    print(" load 2d keypoints ")
    keypoints = {}
    for subject in subjects:
        # actions
        keypoints[subject] = {}
        for action in os.listdir(os.path.join(data_root_path, subject)):
            action_path = data_root_path + subject + "/" + action
            pts_2d_paths = sorted([ os.path.join(action_path, f) + "/points_2d.npy" for f in os.listdir(action_path) if f.startswith("cam")])
            keypoints[subject][action]= [np.load(pts_2d_path) for pts_2d_path in pts_2d_paths]

    print(" processing 2d keypoints ")
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return dataset, keypoints