# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import matplotlib.pyplot as plt

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

class Losses:
    def __init__(self):
        self.losses_2d_train_unlabeled = []
        self.losses_2d_train_labeled_eval = []
        self.losses_2d_train_unlabeled_eval = []
        self.losses_2d_valid = []
        self.losses_traj_train = []
        self.losses_traj_train_eval = []
        self.losses_traj_valid = []
        self.losses_3d_train = []
        self.losses_3d_train_eval = []
        self.losses_3d_valid = []

    def update(self, train_losses, eval_losses):
        self.losses_2d_train_unlabeled.append(train_losses["epoch_loss_2d_train_unlabeled"])
        self.losses_2d_train_labeled_eval.append(eval_losses["epoch_loss_2d_train_labeled_eval"])
        self.losses_2d_train_unlabeled_eval.append(eval_losses["epoch_loss_2d_train_unlabeled_eval"])
        self.losses_2d_valid.append(eval_losses["epoch_loss_2d_valid"]) #

        self.losses_traj_train.append(train_losses["epoch_loss_traj_train"])
        self.losses_traj_train_eval.append(eval_losses["epoch_loss_traj_train_eval"])
        self.losses_traj_valid.append(eval_losses["epoch_loss_traj_valid"]) #

        self.losses_3d_train.append(train_losses["epoch_loss_3d_train"])
        self.losses_3d_train_eval.append(eval_losses["epoch_loss_3d_train_eval"])
        self.losses_3d_valid.append(eval_losses["epoch_loss_3d_valid"])

    def visualize(self, start_epoch, epoch):
        epoch_x = np.arange(start_epoch+3, start_epoch+len(self.losses_3d_train)) + 1
        fig = plt.figure(figsize=(15,10))
        ax_3d = fig.add_subplot(1,3,1)
        ax_3d.plot(epoch_x, self.losses_3d_train[3:], '--', color='C0')
        ax_3d.plot(epoch_x, self.losses_3d_train_eval[3:], color='C0')
        ax_3d.plot(epoch_x, self.losses_3d_valid[3:], color='C1')
        ax_3d.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
        ax_3d.set_ylabel('MPJPE (m)')
        ax_3d.set_xlabel('Epoch')
        ax_3d.set_xlim((start_epoch+3, epoch))

        ax_traj = fig.add_subplot(1,3,2)
        ax_traj.plot(epoch_x, self.losses_traj_train[3:], '--', color='C0')
        ax_traj.plot(epoch_x, self.losses_traj_train_eval[3:], color='C0')
        ax_traj.plot(epoch_x, self.losses_traj_valid[3:], color='C1')
        ax_traj.legend(['traj. train', 'traj. train (eval)', 'traj. valid (eval)'])
        ax_traj.set_ylabel('Mean distance (m)')
        ax_traj.set_xlabel('Epoch')
        ax_traj.set_xlim((start_epoch+3, epoch))

        ax_2d = fig.add_subplot(1,3,3)
        ax_2d.plot(epoch_x, self.losses_2d_train_labeled_eval[3:], color='C0')
        ax_2d.plot(epoch_x, self.losses_2d_train_unlabeled[3:], '--', color='C1')
        ax_2d.plot(epoch_x, self.losses_2d_train_unlabeled_eval[3:], color='C1')
        ax_2d.plot(epoch_x, self.losses_2d_valid[3:], color='C2')
        ax_2d.legend(['2d train labeled (eval)', '2d train unlabeled', '2d train unlabeled (eval)', '2d valid (eval)'])
        ax_2d.set_ylabel('MPJPE (2D)')
        ax_2d.set_xlabel('Epoch')
        ax_2d.set_xlim((start_epoch+3, epoch))

        return fig