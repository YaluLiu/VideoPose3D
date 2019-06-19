import time 
from tqdm import tqdm
import sys
sys.path.append("../")
from common.loss import *
from common.camera import *
from collections import OrderedDict

def states_convert(model_data_parallel):
    states = OrderedDict()
    for k, v in model_data_parallel.state_dict().items():
        name = k.replace("module.", "")
        states[name]=v
        
    return states

def evaluate(models, generators, pad, cmu=False, width = 640, height = 480):
    start_time = time.time()
    model_traj_train = models["model_traj_train"]
    model_pos_train = models["model_pos_train"]
    model_pos = models["model_pos"]
    model_traj = models["model_traj"]
    
    train_generator_eval = generators["train_generator_eval"]
    test_generator = generators["test_generator"]
    semi_generator_eval = generators["semi_generator_eval"]
    with torch.no_grad():
        model_pos.load_state_dict(states_convert(model_pos_train))
        model_pos.eval()
        
        model_traj.load_state_dict(states_convert(model_traj_train))
        model_traj.eval()
        
        epoch_loss_3d_valid = 0
        epoch_loss_traj_valid = 0
        epoch_loss_2d_valid = 0
        N = 0
        
        # Evaluate on test set
        for cam, batch, batch_2d in tqdm(test_generator.next_epoch(), desc = " evaluate on test set : ", total = len(test_generator.poses_2d)):
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_traj = inputs_3d[:, :, :1].clone()
            inputs_3d[:, :, 0] = 0
            
            # Predict 3D poses
            predicted_3d_pos = model_pos(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]
            
            cam = torch.from_numpy(cam.astype('float32'))
            if torch.cuda.is_available():
                cam = cam.cuda()

            predicted_traj = model_traj(inputs_2d)
            loss_traj = mpjpe(predicted_traj, inputs_traj)
            epoch_loss_traj_valid += inputs_traj.shape[0]*inputs_traj.shape[1] * loss_traj.item()
            assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]

            if pad > 0:
                target = inputs_2d[:, pad:-pad, :, :2].contiguous()
            else:
                target = inputs_2d[:, :, :, :2].contiguous()
            
            if not cmu:
                reconstruction = project_to_2d(predicted_3d_pos + predicted_traj, cam)
            else:
                reconstruction = camera2screen_self(predicted_3d_pos + predicted_traj, cam, w = width, h = height)

            loss_reconstruction = mpjpe(reconstruction, target) # On 2D poses
            epoch_loss_2d_valid += reconstruction.shape[0]*reconstruction.shape[1] * loss_reconstruction.item()
            assert reconstruction.shape[0]*reconstruction.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]
        
        # Evaluate on training set, this time in evaluation mode
        epoch_loss_3d_train_eval = 0
        epoch_loss_traj_train_eval = 0
        epoch_loss_2d_train_labeled_eval = 0
        N = 0

        for cam, batch, batch_2d in tqdm(train_generator_eval.next_epoch(), desc = " evaluate on train set : ", total = len(train_generator_eval.poses_2d)):
            if batch_2d.shape[1] == 0:
                # This can only happen when downsampling the dataset
                continue

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_traj = inputs_3d[:, :, :1].clone()
            inputs_3d[:, :, 0] = 0

            # Compute 3D poses
            predicted_3d_pos = model_pos(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train_eval += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]
        
            cam = torch.from_numpy(cam.astype('float32'))
            if torch.cuda.is_available():
                cam = cam.cuda()
            predicted_traj = model_traj(inputs_2d)
            loss_traj = mpjpe(predicted_traj, inputs_traj)
            epoch_loss_traj_train_eval += inputs_traj.shape[0]*inputs_traj.shape[1] * loss_traj.item()
            assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]

            if pad > 0:
                target = inputs_2d[:, pad:-pad, :, :2].contiguous()
            else:
                target = inputs_2d[:, :, :, :2].contiguous()

            if not cmu:
                reconstruction = project_to_2d(predicted_3d_pos + predicted_traj, cam)
            else:
                reconstruction = camera2screen_self(predicted_3d_pos + predicted_traj, cam, w = width, h = height)

            loss_reconstruction = mpjpe(reconstruction, target)
            epoch_loss_2d_train_labeled_eval += reconstruction.shape[0]*reconstruction.shape[1] * loss_reconstruction.item()
            assert reconstruction.shape[0]*reconstruction.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]
        
        # Evaluate 2D loss on unlabeled training set (in evaluation mode)
        epoch_loss_2d_train_unlabeled_eval = 0
        N_semi = 0
        for cam, _, batch_2d in tqdm(semi_generator_eval.next_epoch(),  desc = " evaluate on unlabel train set : ", total=len(semi_generator_eval.poses_2d)):           
            cam = torch.from_numpy(cam.astype('float32'))
            inputs_2d_semi = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                cam = cam.cuda()
                inputs_2d_semi = inputs_2d_semi.cuda()

            predicted_3d_pos_semi = model_pos(inputs_2d_semi)
            predicted_traj_semi = model_traj(inputs_2d_semi)
            if pad > 0:
                target_semi = inputs_2d_semi[:, pad:-pad, :, :2].contiguous()
            else:
                target_semi = inputs_2d_semi[:, :, :, :2].contiguous()

            if not cmu:
                reconstruction_semi = project_to_2d(predicted_3d_pos_semi + predicted_traj_semi, cam)
            else:
                reconstruction_semi = camera2screen_self(predicted_3d_pos_semi + predicted_traj_semi, cam, w = width, h = height)

            loss_reconstruction_semi = mpjpe(reconstruction_semi, target_semi)

            epoch_loss_2d_train_unlabeled_eval += reconstruction_semi.shape[0]*reconstruction_semi.shape[1] \
                                                  * loss_reconstruction_semi.item()
            N_semi += reconstruction_semi.shape[0]*reconstruction_semi.shape[1]
    ext = time.time() - start_time
    # print("-- eval elapse -- {:{width}.{prec}f}s".format(ext, width = 100, prec = 3))    

    losses = {}
    losses["epoch_loss_3d_valid"] = epoch_loss_3d_valid / N
    losses["epoch_loss_2d_valid"] = epoch_loss_2d_valid / N
    losses["epoch_loss_3d_train_eval"] = epoch_loss_3d_train_eval / N
    losses["epoch_loss_traj_train_eval"] = epoch_loss_traj_train_eval / N
    losses["epoch_loss_traj_valid"] = epoch_loss_traj_valid / N
    losses["epoch_loss_2d_train_labeled_eval"] = epoch_loss_2d_train_labeled_eval / N
    losses["epoch_loss_2d_train_unlabeled_eval"] = epoch_loss_2d_train_unlabeled_eval / N_semi

    return losses