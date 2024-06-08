import torch
import numpy as np
from BarrierNet import Driving
from BarrierNet.Driving.models.barrier_net import LitModel

BnetModel = LitModel(model_type = "deri", output_mode = ['v', 'delta', 'a', 'omega'])

rnn_state = [0,0]
batch = []

img_seq = torch.rand(1, 1, 3, 64, 64) 
batch.append(img_seq)
#img_seq = batch[0]

state_seq = np.random.randn(1, 6) # num samples = 1
batch.append(state_seq)
#state_seq = batch[1]

obs_seq = np.random.randn(1, 2)
batch.append(obs_seq)
#obs_seq = batch[2]

ctrl_seq = np.zeros(10)
batch.append(ctrl_seq)
#ctrl_seq = batch[3]

#BnetModel.custom_setup()
result = BnetModel.forward(batch, rnn_state)

print(result)