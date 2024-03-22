import os
import torch
import numpy
import sys
sys.path.append('/home/gridsan/phmine/BarrierNet/Driving/models/')
from BarrierNet.Driving.models.barrier_net import LitModel

model = LitModel()

batch = [0,0,0,0,0,0]
# b, t, c, h, w = img_seq.shape
img_seq = batch[0]

# gt_s, gt_d, gt_mu, v, delta, gt_kappa = state_seq[:,0], state_seq[:,1], 
# state_seq[:,2], state_seq[:,3], state_seq[:,4], state_seq[:,5]
state_seq = batch[1]

# gt_obs_s, gt_obs_d = obs_seq[:,0], obs_seq[:,1]
obs_seq = batch[2]


ctrl_seq = batch[3]

initial_state = model.get_initial_state(10) # 10 is hardcoded batchsize

print(initial_state)


#func_call = model.forward(batch, rnn_state):

