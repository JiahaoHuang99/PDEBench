import numpy as np


data_path = '/media/ssd/data_temp/PDE/results/vis/OFormer/results_darcy_flow_vis.npz'

with np.load(data_path) as f:
    gt = f['gt']  # (H_FULL, W_FULL)
    in_seq = f['in_seq']  # (H_FULL, W_FULL)
    pred = f['pred']

print(gt.shape)
print(in_seq.shape)
print(pred.shape)