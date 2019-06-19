import time 
from tqdm import tqdm
import sys
sys.path.append("../")
from common.loss import *
from common.camera import *
import torch 

def train(models, generators, optimizer, pad, skip, bone_length_term, no_proj, epoch = None, cmu=False, width= 640, height = 480):
    # semi-supervised
    start_time =time.time()
    model_traj_train = models["model_traj_train"]
    model_pos_train = models["model_pos_train"]
    train_generator = generators["train_generator"]
    semi_generator = generators["semi_generator"]
    
    epoch_loss_3d_train = 0
    epoch_loss_traj_train = 0
    epoch_loss_2d_train_unlabeled = 0
    
    N = 0
    N_semi = 0
    c = 0
    
    model_pos_train.train()

    if epoch is None:
        desc = " train : "
    else:
        desc = " - epoch {:3d} train ".format(epoch)
    
    for (_, batch_3d, batch_2d), (cam_semi, _, batch_2d_semi) in \
                tqdm(zip(train_generator.next_epoch(), semi_generator.next_epoch()), desc=desc, total = train_generator.num_batches):
        
        if (c % 10 == 0) and (c !=0):
            ext = time.time() - start_time
            # print("-- {:5d} -- {:{width}.{prec}f}s".format(c, ext/c, width = 100, prec = 3))
        c += 1
        
        cam_semi = torch.from_numpy(cam_semi.astype('float32'))
        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        if torch.cuda.is_available():
            cam_semi = cam_semi.cuda()
            inputs_3d = inputs_3d.cuda()
        
        inputs_traj = inputs_3d[:, :, :1].clone()
        inputs_3d[:, :, 0] = 0 # root points set to zero in camera coordinates
        
        # Split point between labeled and unlabeled samples in the batch
        split_idx = inputs_3d.shape[0]
        
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        inputs_2d_semi = torch.from_numpy(batch_2d_semi.astype('float32'))
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()
            inputs_2d_semi = inputs_2d_semi.cuda()
        inputs_2d_cat =  torch.cat((inputs_2d, inputs_2d_semi), dim=0) if not skip else inputs_2d
        
        optimizer.zero_grad()
        
        # Compute 3D poses
        predicted_3d_pos_cat = model_pos_train(inputs_2d_cat)
        
        # mpjpe
        loss_3d_pos = mpjpe(predicted_3d_pos_cat[:split_idx], inputs_3d)
        epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
        N += inputs_3d.shape[0]*inputs_3d.shape[1]
        loss_total = loss_3d_pos
        
        # Compute global trajectory
        predicted_traj_cat = model_traj_train(inputs_2d_cat)
        if cmu:
            w = 1 / torch.norm(inputs_traj, dim=3) * 5
        else:
            w = 1 / inputs_traj[:, :, :, 2] # Weight inversely proportional to depth

        loss_traj = weighted_mpjpe(predicted_traj_cat[:split_idx], inputs_traj, w)
        epoch_loss_traj_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_traj.item()
        assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]
        loss_total += loss_traj
        if not skip:
            # Semi-supervised loss for unlabeled samples
            predicted_semi = predicted_3d_pos_cat[split_idx:]
            if pad > 0:
                target_semi = inputs_2d_semi[:, pad:-pad, :, :2].contiguous()
            else:
                target_semi = inputs_2d_semi[:, :, :, :2].contiguous()

            if not cmu:         
                projection_func = project_to_2d
                reconstruction_semi = projection_func(predicted_semi + predicted_traj_cat[split_idx:], cam_semi)
            else:
                # self data
                reconstruction_semi = camera2screen_self(predicted_semi + predicted_traj_cat[split_idx:], cam_semi, w = width, h = height)

            loss_reconstruction = mpjpe(reconstruction_semi, target_semi) # On 2D poses
            epoch_loss_2d_train_unlabeled += predicted_semi.shape[0]*predicted_semi.shape[1] * loss_reconstruction.item()
            if not no_proj:
                loss_total += loss_reconstruction

            # Bone length term to enforce kinematic constraints
            if bone_length_term:
                dists = predicted_3d_pos_cat[:, :, 1:] - predicted_3d_pos_cat[:, :, dataset.skeleton().parents()[1:]]
                bone_lengths = torch.mean(torch.norm(dists, dim=3), dim=1)
                penalty = torch.mean(torch.abs(torch.mean(bone_lengths[:split_idx], dim=0) \
                                             - torch.mean(bone_lengths[split_idx:], dim=0)))
                loss_total += penalty


            N_semi += predicted_semi.shape[0]*predicted_semi.shape[1]
                
        else:
            N_semi += 1 # To avoid division by zero
        
        loss_total.backward()
        optimizer.step()    
        
    ext = time.time() - start_time
    # print("-- train elapse -- {:{width}.{prec}f}s".format(ext, width = 100, prec = 3))
    
    losses = {}
    losses["epoch_loss_traj_train"] = epoch_loss_traj_train / N
    losses["epoch_loss_2d_train_unlabeled"] = epoch_loss_2d_train_unlabeled / N_semi
    losses["epoch_loss_3d_train"] = epoch_loss_3d_train / N

    return models, optimizer, losses