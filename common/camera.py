# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import cv2

from common.utils import wrap
from common.quaternion import qrot, qinverse
from scipy.spatial.transform import rotation

def normalize_screen_coordinates_self(X, w, h): 
    assert X.shape[-1] == 2
    normalized_X = X/w*2
    normalized_X[:,0] = normalized_X[:,0] - 1
    normalized_X[:,1] = normalized_X[:,1] - h/w
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return normalized_X

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    if torch.is_tensor(X):
        if X.device.type == "cuda":
            return X/w*2 - torch.Tensor([1, h/w]).cuda()
        else:
            return X/w*2 - torch.Tensor([1, h/w])
    else:
        return X/w*2 - [1, h/w]
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio

def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2
    

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

def world2camera_cv(pts_3d_world_homo, rvec, tvec):
    rmat, _ = cv2.Rodrigues(rvec)
    extrinsics = np.zeros((4,4))
    extrinsics[:3,:3] = rmat
    extrinsics[:3,3] = tvec.reshape(-1)
    extrinsics[3,3] = 1
    
    pts_3d_cam = np.matmul(extrinsics, pts_3d_world_homo)
    
    return pts_3d_cam

def camera2world_cv(pts_3d_cam, rvec, tvec):
    # convert pts to homo
    nframe = pts_3d_cam.shape[0]
    pts_3d_cam_homo = np.ones((nframe, 17, 4))
    pts_3d_cam_homo[... , :3] = pts_3d_cam
    pts_3d_cam_homo = pts_3d_cam_homo.reshape(-1, 4).T

    rmat, _ = cv2.Rodrigues(rvec)
    extrinsics = np.zeros((4,4))
    extrinsics[:3,:3] = rmat
    extrinsics[:3,3] = tvec.reshape(-1)
    extrinsics[3,3] = 1
    
    pts_3d_world_homo = np.matmul(np.linalg.inv(extrinsics), pts_3d_cam_homo)
    
    return pts_3d_world_homo[:3].T.reshape(-1, 17, 3)

def camera2screen_self(input_3d_cam, cam, w=640, h=480):
    batch_size, nframe, njoints, nchannel = input_3d_cam.shape
    input_3d_cam_total = input_3d_cam.reshape(batch_size, nframe * njoints, nchannel)
    cam = cam.reshape(batch_size, 3, 3).transpose(1,2)
    # projection
    proj_2d_screen = torch.matmul(input_3d_cam_total, cam)
    proj_2d_screen[:,:,0] /= proj_2d_screen[:,:,2]
    proj_2d_screen[:,:,1] /= proj_2d_screen[:,:,2]
    proj_2d_screen = proj_2d_screen[:,:,:2].reshape(batch_size, nframe, njoints, 2)
    # normalization
    return normalize_screen_coordinates(proj_2d_screen[..., :2], w=w, h=h)

def camera2screen(pts_3d_cam, intrinsics):

    proj_2d = np.matmul(intrinsics, pts_3d_cam)
    proj_2d_homo = proj_2d / proj_2d[-1]
    
    return proj_2d_homo[:2].T

def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def extrinsic_matrix(R_q, t):
    # [xc,yc,zc,1] = matmul(ext, [xw, yw, zw, 1])

    extMat = np.zeros((4,4))
    R = rotation.from_quat(R_q).as_dcm()
    R = -np.flip(np.flip(R, axis=0), axis =1) * np.array([[1,-1,1],[-1,1,-1],[-1,1,-1]])
    extMat[:3,:3] = R
    extMat[3,3] = 1
    extMat[:3,3] = np.matmul(-R, t)

    return extMat

def project_to_2d_self(X, camera_params, batch_size, w, h):
    X = X.reshape(batch_size, -1, 3).transpose(1,2)
    cam = camera_params.reshape(batch_size, 3, 3)
    proj_2d = torch.matmul(cam, X)
    proj_2d_homo = torch.zeros(batch_size, 2, 17)
    proj_2d_homo[:,0] = proj_2d[:,0] / proj_2d[:,-1]
    proj_2d_homo[:,1] = proj_2d[:,1] / proj_2d[:,-1]
    proj_2d_homo = proj_2d_homo.transpose(1,2)

    return normalize_screen_coordinates(proj_2d_homo[...,:2], 
                                        w=w,
                                        h=h).unsqueeze(1)



def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    
    return f*XX + c

def camera_calibration(chess_board_shape, images):
    # camera calibration using opencv
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    c_columns = chess_board_shape[0]
    c_rows = chess_board_shape[1]
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((c_rows * c_columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:c_columns, 0:c_rows].T.reshape(-1,2) * 1000
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (c_columns,c_rows),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx, dist