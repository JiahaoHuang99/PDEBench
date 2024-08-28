import os
import argparse

import torch
from tqdm import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


pdes = (
    "darcy_oformer",
    "dr2d_oformer",
    "sw2d_oformer",
)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def resize_with_zeros(original_array, new_shape):
    # Create an array of zeros with the new shape
    resized_array = np.ones(new_shape) * -1

    # Determine the step size for rows and columns
    row_step = new_shape[0] // original_array.shape[0]
    col_step = new_shape[1] // original_array.shape[1]

    # Copy the original values into the resized array
    for i in range(original_array.shape[0]):
        for j in range(original_array.shape[1]):
            resized_array[i * row_step, j * col_step] = original_array[i, j]

    return resized_array

def visualize_dr2d(path, save_path):

    nb = 0
    with np.load(path) as f:
        gt = f['gt'].reshape(128, 128, 101, 2).transpose(2, 0, 1, 3)  # (T, H, W, 2)
        pred = f['pred'].reshape(128, 128, 101, 2).transpose(2, 0, 1, 3)  # (T, H, W, 2)

    # save gt frames
    for i in tqdm(range(gt.shape[0])):
        plt.imshow(gt[i, ..., 0].squeeze())
        plt.colorbar()
        plt.axis('off')
        mkdir(os.path.join(save_path, 'frames'))
        plt.savefig(os.path.join(save_path, 'frames', 'gt_u_{}_{:03d}.png'.format(nb, i,)), dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(gt[i, ..., 1].squeeze())
        plt.colorbar()
        plt.axis('off')
        mkdir(os.path.join(save_path, 'frames'))
        plt.savefig(os.path.join(save_path, 'frames', 'gt_v_{}_{:03d}.png'.format(nb, i,)), dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()

    # save pred frames
    for i in tqdm(range(pred.shape[0])):
        plt.imshow(pred[i, ..., 0].squeeze())
        plt.colorbar()
        plt.axis('off')
        mkdir(os.path.join(save_path, 'frames'))
        plt.savefig(os.path.join(save_path, 'frames', 'pred_u_{}_{:03d}.png'.format(nb, i, )), dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(pred[i, ..., 1].squeeze())
        plt.colorbar()
        plt.axis('off')
        mkdir(os.path.join(save_path, 'frames'))
        plt.savefig(os.path.join(save_path, 'frames', 'pred_v_{}_{:03d}.png'.format(nb, i, )), dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()


def visualize_sw2d(path, save_path):

    nb = 0
    with np.load(path) as f:
        gt = f['gt'].reshape(128, 128, 101, 1).transpose(2, 0, 1, 3)  # (T, H, W, 1)
        pred = f['pred'].reshape(128, 128, 101, 1).transpose(2, 0, 1, 3)  # (T, H, W, 1)

    # save gt frames
    for i in tqdm(range(gt.shape[0])):
        plt.imshow(gt[i].squeeze())
        plt.colorbar()
        plt.axis('off')
        mkdir(os.path.join(save_path, 'frames'))
        plt.savefig(os.path.join(save_path, 'frames', 'gt_{}_{:03d}.png'.format(nb, i,)), dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()

    # save pred frames
    for i in tqdm(range(pred.shape[0])):
        plt.imshow(pred[i].squeeze())
        plt.colorbar()
        plt.axis('off')
        mkdir(os.path.join(save_path, 'frames'))
        plt.savefig(os.path.join(save_path, 'frames', 'pred_{}_{:03d}.png'.format(nb, i,)), dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()


def visualize_df2d(path, save_path, resize_nu=None, map_null_point=False):

    nb = 0
    with np.load(path) as f:
        gt = f['gt'][..., 0]  # (H_FULL, W_FULL)
        nu = f['input'][..., 0]  # (H_FULL, W_FULL)
        pred = f['pred'][..., 0]
        # err = np.abs(gt - pred)
        err = gt - pred

    if resize_nu is not None:
        nu = resize_with_zeros(nu, resize_nu)
    cmap = plt.cm.RdBu_r.copy()
    if map_null_point:
        cmap.set_under('black')
    plt.imshow(nu.squeeze(), cmap=cmap, vmax=1.0, vmin=0.0)
    # plt.title('diffusion coefficient nu')
    # plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(save_path, f'nu_{nb}.png'), dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(pred.squeeze(), cmap='RdBu_r', vmax=0.5, vmin=0.0)
    # plt.title('Data u')
    # plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(save_path, f'pred_{nb}.png'), dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(gt.squeeze(), cmap='RdBu_r', vmax=0.5, vmin=0.0)
    # plt.title('Ground Truth')
    # plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(save_path, f'gt_{nb}.png'), dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("plot saved")

    plt.imshow(err.squeeze(), cmap='RdBu_r', vmax=0.05, vmin=-0.05)
    # plt.title('Data u')
    # colorbar()
    plt.colorbar()
    # plt.axis('off')
    plt.savefig(os.path.join(save_path, f'err_{nb}.png'), dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()







if __name__ == "__main__":

    # ----------------------------------------------------
    # Mamba v.s. Transformer
    # ----------------------------------------------------


    # ------------------------
    # DR2D
    # ------------------------

    pde_name = "DR2D"
    model_name = "OFormer"
    submodel_name = "GT"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaOFormer/vis/npz/samples_dr2d/OFormer_DR2D_Cx1_Tx1/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_dr2d(data_path, save_path)

    pde_name = "DR2D"
    model_name = "OFormer"
    submodel_name = "MB"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaOFormer/vis/npz/samples_dr2d/OMamba_DR2D_Cx1_Tx1/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_dr2d(data_path, save_path)

    # ------------------------

    pde_name = "DR2D"
    model_name = "GalerkinTransformer"
    submodel_name = "GT"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGalerkinTransformer/vis/npz/dr2d_pdebench/dr2d_pdebench_gt_Cx1_test/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_dr2d(data_path, save_path)

    pde_name = "DR2D"
    model_name = "GalerkinTransformer"
    submodel_name = "MB"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGalerkinTransformer/vis/npz/dr2d_pdebench/dr2d_pdebench_mb_Cx1_test/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_dr2d(data_path, save_path)

    # ------------------------

    pde_name = "DR2D"
    model_name = "GNOT"
    submodel_name = "GT"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGNOT/vis/npz/dr2d_pdebench/gnot_dr2d_mse_test/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_dr2d(data_path, save_path)

    pde_name = "DR2D"
    model_name = "GNOT"
    submodel_name = "MB"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGNOT/vis/npz/dr2d_pdebench/mambagnot_dr2d_mse_test/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_dr2d(data_path, save_path)



    # ------------------------
    # SW2D
    # ------------------------

    pde_name = "SW2D"
    model_name = "OFormer"
    submodel_name = "GT"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaOFormer/vis/npz/samples_sw2d/OFormer_SW2D_Cx1_Tx1/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_sw2d(data_path, save_path)

    pde_name = "SW2D"
    model_name = "OFormer"
    submodel_name = "MB"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaOFormer/vis/npz/samples_sw2d/OMamba_SW2D_Cx1_Tx1/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_sw2d(data_path, save_path)

    # ------------------------

    pde_name = "SW2D"
    model_name = "GalerkinTransformer"
    submodel_name = "GT"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGalerkinTransformer/vis/npz/sw2d_pdebench/sw2d_pdebench_gt_Cx1_test/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_sw2d(data_path, save_path)

    pde_name = "SW2D"
    model_name = "GalerkinTransformer"
    submodel_name = "MB"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGalerkinTransformer/vis/npz/sw2d_pdebench/sw2d_pdebench_mb_Cx1_test/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_sw2d(data_path, save_path)

    # ------------------------

    pde_name = "SW2D"
    model_name = "GNOT"
    submodel_name = "GT"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGNOT/vis/npz/sw2d_pdebench/gnot_sw2d_mse_test/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_sw2d(data_path, save_path)

    pde_name = "SW2D"
    model_name = "GNOT"
    submodel_name = "MB"
    data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGNOT/vis/npz/sw2d_pdebench/mambagnot_sw2d_mse_test/vis_results.npz"
    save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    mkdir(save_path)
    visualize_sw2d(data_path, save_path)


    # ------------------------
    # DF2D
    # ------------------------
    #
    # pde_name = "DF2D"
    # model_name = "OFormer"
    # submodel_name = "GT"
    # data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaOFormer/vis/npz/samples_darcyflow/OFormer_DarcyFlow_beta1.0/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)
    #
    # pde_name = "DF2D"
    # model_name = "OFormer"
    # submodel_name = "ST"
    # data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaOFormer/vis/npz/samples_darcyflow/OGTFormer_DarcyFlow_beta1.0/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)
    #
    # pde_name = "DF2D"
    # model_name = "OFormer"
    # submodel_name = "MB"
    # data_path =  "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaOFormer/vis/npz/samples_darcyflow/OMamba_DarcyFlow_beta1.0/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)

    # ------------------------

    # pde_name = "DF2D"
    # model_name = "GalerkinTransformer"
    # submodel_name = "GT"
    # data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGalerkinTransformer/vis/npz/darcyflow_pdebench/darcyflow_pdebench_gt_Stanx2_test/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)
    #
    # pde_name = "DF2D"
    # model_name = "GalerkinTransformer"
    # submodel_name = "ST"
    # data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGalerkinTransformer/vis/npz/darcyflow_pdebench/darcyflow_pdebench_st_Stanx2_test/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)
    #
    # pde_name = "DF2D"
    # model_name = "GalerkinTransformer"
    # submodel_name = "MB"
    # data_path =  "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGalerkinTransformer/vis/npz/darcyflow_pdebench/darcyflow_pdebench_mb_Stanx2_test/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)

    # ------------------------

    # pde_name = "DF2D"
    # model_name = "GNOT"
    # submodel_name = "GT"
    # data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGNOT/vis/npz/darcyflow_pdebench/gnot_darcyflow2d_mse_test/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)
    #
    # pde_name = "DF2D"
    # model_name = "GNOT"
    # submodel_name = "ST"
    # data_path = "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGNOT/vis/npz/darcyflow_pdebench/stgnot_darcyflow2d_mse_test/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)
    #
    # pde_name = "DF2D"
    # model_name = "GNOT"
    # submodel_name = "MB"
    # data_path =  "/media/NAS06/jiahao/Mamba_Transformer_PDE/MambaGNOT/vis/npz/darcyflow_pdebench/mambagnot_darcyflow2d_mse_test/vis_results.npz"
    # save_path = os.path.join('.', 'vis', 'png', model_name, pde_name, submodel_name)
    # mkdir(save_path)
    # visualize_df2d(data_path, save_path)

    # ------------------------