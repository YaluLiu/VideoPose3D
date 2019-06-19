import sys
sys.path.append("../")

from common.model import *

def build_models(num_joints_in, in_features, num_joints_out, filter_widths):
    # model pos
    model_pos_train = TemporalModelOptimized1f(num_joints_in,
                                               in_features,
                                               num_joints_out,
                                               filter_widths=filter_widths,
                                               causal = False,
                                               dropout = 0.25,
                                               channels = 1024)
    


    # model traj
    model_traj_train = TemporalModelOptimized1f(num_joints_in,
                                                in_features,
                                                1,
                                                filter_widths=filter_widths,
                                                causal =False,
                                                dropout = 0.25,
                                                channels = 1024)

    # model pos eval
    model_pos = TemporalModel(num_joints_in,
                              in_features,
                              num_joints_out,
                              filter_widths=filter_widths,
                              causal =False,
                              dropout = 0.25,
                              channels = 1024,
                              dense = False)

    # model traj eval
    model_traj = TemporalModel(num_joints_in,
                               in_features,
                                1,
                                filter_widths=filter_widths,
                                causal =False,
                                dropout = 0.25,
                                channels = 1024,
                                dense = False)

    return model_pos_train, model_pos, model_traj, model_traj_train