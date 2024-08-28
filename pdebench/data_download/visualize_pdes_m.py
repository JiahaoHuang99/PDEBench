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

def visualize_dr2d_oformer(path, save_path):

    for nb in range(1):
        with np.load(path) as f:
            gt = f['gt'][nb].reshape(128, 128, 101, 2).transpose(2, 0, 1, 3)  # (T, H, W, 2)
            pred = f['pred'][nb].reshape(128, 128, 101, 2).transpose(2, 0, 1, 3)  # (T, H, W, 2)
            in_seq = f['in_seq'][..., :-2][nb].reshape(128, 128, 10, 2).transpose(2, 0, 1, 3)  # (T, H, W, 2)

        # save gt gif
        # fig, ax = plt.subplots(1, 2)
        # ims = []
        # for i in tqdm(range(gt.shape[0])):
        #     im1 = ax[0].imshow(gt[i, ..., 0].squeeze(), animated=True)
        #     im2 = ax[1].imshow(gt[i, ..., 1].squeeze(), animated=True)
        #     if i == 0:
        #         ax[0].imshow(gt[0, ..., 0].squeeze())  # show an initial one first
        #         ax[1].imshow(gt[0, ..., 1].squeeze())  # show an initial one first
        #     ims.append([im1, im2])
        # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        # writer = animation.PillowWriter(fps=15, bitrate=1800)
        # ani.save(os.path.join(save_path, f"gt_{nb}.gif"), writer=writer)
        # plt.close()
        # print("Animation saved")

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

        # # save pred gif
        # fig, ax = plt.subplots(1, 2)
        # ims = []
        # for i in tqdm(range(pred.shape[0])):
        #     im1 = ax[0].imshow(pred[i, ..., 0].squeeze(), animated=True)
        #     im2 = ax[1].imshow(pred[i, ..., 1].squeeze(), animated=True)
        #     if i == 0:
        #         ax[0].imshow(pred[0, ..., 0].squeeze())  # show an initial one first
        #         ax[1].imshow(pred[0, ..., 1].squeeze())  # show an initial one first
        #     ims.append([im1, im2])
        # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        # writer = animation.PillowWriter(fps=15, bitrate=1800)
        # ani.save(os.path.join(save_path, f"pred_{nb}.gif"), writer=writer)
        # plt.close()

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


def visualize_sw2d_oformer(path, save_path):

    for nb in range(1):
        with np.load(path) as f:
            gt = f['gt'][nb].transpose(2, 0, 1)  # (T, H, W)
            pred = f['pred'][nb].transpose(2, 0, 1)  # (T, H, W)
            in_seq = f['in_seq'][..., :-2][nb].transpose(2, 0, 1)  # (T, H, W)

        # # save pred gif
        # fig, ax = plt.subplots()
        # ims = []
        # for i in tqdm(range(pred.shape[0])):
        #     im = ax.imshow(pred[i].squeeze(), animated=True)
        #     if i == 0:
        #         ax.imshow(pred[0].squeeze())  # show an initial one first
        #     ims.append([im])
        # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        # writer = animation.PillowWriter(fps=15, bitrate=1800)
        # ani.save(os.path.join(save_path, f"pred_{nb}.gif"), writer=writer)
        # plt.close()

        # save pred frames
        for i in tqdm(range(pred.shape[0])):
            plt.imshow(pred[i].squeeze())
            plt.colorbar()
            plt.axis('off')
            mkdir(os.path.join(save_path, 'frames'))
            plt.savefig(os.path.join(save_path, 'frames', 'pred_{}_{:03d}.png'.format(nb, i,)), dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close()

        # # save gt
        # fig, ax = plt.subplots()
        # ims = []
        # for i in tqdm(range(gt.shape[0])):
        #     im = ax.imshow(gt[i].squeeze(), animated=True)
        #     if i == 0:
        #         ax.imshow(gt[0].squeeze())  # show an initial one first
        #     ims.append([im])
        # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        # writer = animation.PillowWriter(fps=15, bitrate=1800)
        # ani.save(os.path.join(save_path, f"gt_{nb}.gif"), writer=writer)
        # plt.close()

        # save gt frames
        for i in tqdm(range(gt.shape[0])):
            plt.imshow(gt[i].squeeze())
            plt.colorbar()
            plt.axis('off')
            mkdir(os.path.join(save_path, 'frames'))
            plt.savefig(os.path.join(save_path, 'frames', 'gt_{}_{:03d}.png'.format(nb, i,)), dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close()

        print("Animation saved")


def visualize_darcy_oformer(path, save_path, resize_nu=None, map_null_point=False):

    for nb in range(1):
        with np.load(path) as f:
            gt = f['gt'][nb]  # (H_FULL, W_FULL)
            nu = f['in_seq'][..., : -2][nb]  # (H_FULL, W_FULL)
            pred = f['pred'][nb]
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


def visualize_darcy_mgkn(path, save_path):

    for nb in range(1):
        with np.load(path) as f:
            gt = f['gt'][nb]  # (H_FULL, W_FULL)
            nu = f['in_seq'][..., : -2][nb]  # (H_FULL, W_FULL)
            pred = f['pred'][nb]
            # err = np.abs(gt - pred)
            err = gt - pred
        plt.imshow(nu.squeeze(), cmap='RdBu_r')
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

        plt.imshow(err.squeeze(), cmap='RdBu_r', vmax=0.02, vmin=-0.02)
        # plt.title('Data u')
        # colorbar()
        plt.colorbar()
        # plt.axis('off')
        plt.savefig(os.path.join(save_path, f'err_{nb}.png'), dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()



def visualize_airfoil_oformer(path, save_path, type_list):
    from matplotlib.tri import Triangulation
    with np.load(path) as f:
        gt_all = f['gt']  # (B, T, N, 4)
        pred_all = f['pred']  # (B, T, N, 4)
        coords_all = f['coords']  # (B, N, 2)
        cells_all = f['cells']  # (B, 10216, 2)
    batch_size = gt_all.shape[0]
    res_t = pred_all.shape[1]
    res_n = coords_all.shape[2]
    num_cell = cells_all.shape[1]

    for idx_b in range(1):
        for idx_type, type in enumerate(type_list):
            for idx_t in range(res_t):
                gt = gt_all[idx_b, idx_t, :, idx_type]
                pred = pred_all[idx_b, idx_t, :, idx_type]
                pos = coords_all[idx_b]
                triag = cells_all[idx_b]
                triag = Triangulation(pos[:, 0], pos[:, 1], triag)

                fig, ax = plt.subplots()
                ax.tripcolor(triag, gt)
                ax.axis('off')
                ax.set_xlim([-0.5 + 20, 1.5 + 20])
                ax.set_ylim([-1.4 + 19.96, 1.4 + 19.96])
                ax.set_aspect(0.5)
                # ax.colorbar()
                mkdir(os.path.join(save_path, 'gt', type))
                plt.savefig(os.path.join(save_path, 'gt', type, f'gt_{idx_b}_{idx_t}.png'), dpi=500, bbox_inches='tight', pad_inches=0)
                plt.close()

                fig, ax = plt.subplots()
                ax.tripcolor(triag, pred)
                ax.axis('off')
                ax.set_xlim([-0.5 + 20, 1.5 + 20])
                ax.set_ylim([-1.4 + 19.96, 1.4 + 19.96])
                ax.set_aspect(0.5)
                # ax.colorbar()
                mkdir(os.path.join(save_path, 'pred', type))
                plt.savefig(os.path.join(save_path, 'pred', type, f'pred_{idx_b}_{idx_t}.png'), dpi=500, bbox_inches='tight', pad_inches=0)
                plt.close()



def check_airfoil(path_1, path_2):
    from matplotlib.tri import Triangulation
    with np.load(path_1) as f:
        gt_all_1 = f['gt']  # (B, T, N, 4)
        pred_all_1 = f['pred']  # (B, T, N, 4)
        coords_all_1 = f['coords']  # (B, N, 2)
        cells_all_1 = f['cells']  # (B, 10216, 2)
    # batch_size = gt_all.shape[0]
    # res_t = pred_all.shape[1]
    # res_n = coords_all.shape[2]
    # num_cell = cells_all.shape[1]

    with np.load(path_2) as f:
        gt_all_2 = f['gt'][:, :, :, [2, 3, 0, 1]]  # (B, T, N, 4)
        pred_all_2 = f['pred'][:, :, :, [2, 3, 0, 1]]  # (B, T, N, 4)
        coords_all_2 = f['coords']  # (B, N, 2)
        cells_all_2 = f['cells']  # (B, 10216, 2)

    print(np.allclose(gt_all_1, gt_all_2, atol=1e-5))
    print(np.allclose(pred_all_1, pred_all_2, atol=1e-5))
    print(np.allclose(coords_all_1, coords_all_2, atol=1e-5))
    print(np.allclose(cells_all_1, cells_all_2, atol=1e-5))
    import torch
    c = 3
    b = 0
    t = 20

    gt_all_1 = torch.from_numpy(gt_all_1[:, t, :, c])
    pred_all_1 = torch.from_numpy(pred_all_1[:, t, :, c])
    gt_all_2 = torch.from_numpy(gt_all_2[:, t, :, c])
    pred_all_2 = torch.from_numpy(pred_all_2[:, t, :, c])


    # gt_all_1 = torch.from_numpy(gt_all_1[..., c])
    # pred_all_1 = torch.from_numpy(pred_all_1[..., c])
    # gt_all_2 = torch.from_numpy(gt_all_2[..., c])
    # pred_all_2 = torch.from_numpy(pred_all_2[..., c])


    print(gt_all_1.shape)
    print(pred_all_1.shape)
    print(gt_all_2.shape)
    print(pred_all_2.shape)

    l2_err_1 = torch.nn.MSELoss()(pred_all_1, gt_all_1).item()
    l2_err_2 = torch.nn.MSELoss()(pred_all_2, gt_all_2).item()
    print(l2_err_1)
    print(l2_err_2)
    print(0)

if __name__ == "__main__":
    # ----------------------------------------------------
    # Mamba v.s. Transformer
    pass

    # pde_name = "darcy_oformer"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/OFormer/results_darcy_flow_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow', 'OFormer')
    # mkdir(save_path)
    # visualize_darcy_oformer(data_path, save_path)


    # ----------------------------------------------------
    # PhysGTN

    # pde_name = "darcy_oformer"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/OFormer/results_darcy_flow_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow', 'OFormer')
    # mkdir(save_path)
    # visualize_darcy_oformer(data_path, save_path)

    # pde_name = "sw2d_oformer"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/OFormer/results_shallow_water_vis.npz"
    # save_path = os.path.join('.', 'vis', 'ShallowWater', 'OFormer')
    # mkdir(save_path)
    # visualize_sw2d_oformer(data_path, save_path)

    # pde_name = "dr2d_oformer"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/OFormer/results_diffusion_reaction_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DiffusionReaction', 'OFormer')
    # mkdir(save_path)
    # visualize_dr2d_oformer(data_path, save_path)


    # pde_name = "darcy_physgtn"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/PhysGTN/results_darcy_flow_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow', 'PhysGTN')
    # mkdir(save_path)
    # visualize_darcy_oformer(data_path, save_path)

    # pde_name = "sw2d_physgtn"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/PhysGTN/results_shallow_water_vis.npz"
    # save_path = os.path.join('.', 'vis', 'ShallowWater', 'PhysGTN')
    # mkdir(save_path)
    # visualize_sw2d_oformer(data_path, save_path)

    # pde_name = "dr2d_physgtn"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/PhysGTN/results_diffusion_reaction_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DiffusionReaction', 'PhysGTN')
    # mkdir(save_path)
    # visualize_dr2d_oformer(data_path, save_path)



    # pde_name = "darcy_mgkn"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/MGKN/MGKN_darcy_results_for_visualize.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow', 'MGKN')
    # mkdir(save_path)
    # visualize_darcy_mgkn(data_path, save_path)

    # pde_name = "darcy_geofno"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/GEOFNO/GeoFNO_darcy1.0_results_for_visualize.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow', 'GEOFNO')
    # mkdir(save_path)
    # visualize_darcy_oformer(data_path, save_path)

    # ----------------------------------------------------
    # ABS

    # pde_name = "darcy_oformer_abs_res64"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/OFormer/results_darcy_flow_ablation_query_res64_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow_Abl_QRes64', 'OFormer')
    # mkdir(save_path)
    # visualize_darcy_oformer(data_path, save_path, resize_nu=(128, 128), map_null_point=True)
    #
    # pde_name = "darcy_oformer_abs_res32"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/OFormer/results_darcy_flow_ablation_query_res32_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow_Abl_QRes32', 'OFormer')
    # mkdir(save_path)
    # visualize_darcy_oformer(data_path, save_path, resize_nu=(128, 128), map_null_point=True)
    #
    # pde_name = "darcy_physgtn_abs_res64"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/PhysGTN/results_darcy_flow_ablation_query_res64_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow_Abl_QRes64', 'PhysGTN')
    # mkdir(save_path)
    # visualize_darcy_oformer(data_path, save_path, resize_nu=(128, 128), map_null_point=True)
    #
    # pde_name = "darcy_physgtn_abs_res32"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/PhysGTN/results_darcy_flow_ablation_query_res32_vis.npz"
    # save_path = os.path.join('.', 'vis', 'DarcyFlow_Abl_QRes32', 'PhysGTN')
    # mkdir(save_path)
    # visualize_darcy_oformer(data_path, save_path, resize_nu=(128, 128), map_null_point=True)


    # ----------------------------------------------------
    # AIRFOIL

    # pde_name = "airfoil_oformer"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/OFormer/results_airfoil_vis.npz"
    # data_path = "/home/jh/physics_graph_transformer/physics_graph_transformer_comparison/OFormer/airfoil/tmp/samples/results_airfoil_vis.npz"
    # save_path = os.path.join('.', 'vis', 'Airfoil', 'OFormer')
    # mkdir(save_path)
    # visualize_airfoil_oformer(data_path, save_path, type_list=['vel_x', 'vel_y', 'dns', 'prs'])
    #
    #
    # pde_name = "airfoil_physgtn"
    # data_path = "/media/ssd/data_temp/PDE/results/vis/PhysGTN/results_airfoil_vis.npz"
    # data_path = "/home/jh/physics_graph_transformer/physics_graph_transformer/results/PhysGTN_Airfoil_DeepMind_M22b_LEncDep6Chnl_D2_PNG_Customx1_Customx1_Tx4_RelPE_InitO_BN_LPL_OCLR1/RUN_TEST/array/results_airfoil_vis.npz"
    # data_path = "/home/jh/physics_graph_transformer/physics_graph_transformer/results/PhysGTN_Airfoil_DeepMind_M23b_LEncDep6Chnl_D2_PNG_Customx1_Customx1_Tx4_RelPE_InitO_BN_LPL_OCLR1/RUN_TEST/array/results_airfoil_vis.npz"
    # data_path = "/home/jh/physics_graph_transformer/physics_graph_transformer/results/PhysGTN_Airfoil_DeepMind_M23b_LEncDep6Chnl_D2_PNG_Customx1_Customx1_Tx4_RelPE_InitO_BN_LPL_ROI0.1_OCLR1/RUN_TEST/array/results_airfoil_vis.npz"
    # data_path = "/home/jh/physics_graph_transformer/physics_graph_transformer/results/PhysGTN_Airfoil_DeepMind_M24b_LEncDep6Chnl_D3_PNG_Customx1_Customx1_Tx4_RelPE_InitO_BN_LPL_OCLR1/RUN_TEST/array/results_airfoil_vis.npz"
    # data_path = "/home/jh/physics_graph_transformer/physics_graph_transformer/results/PhysGTN_Airfoil_DeepMind_M25b_LEncDep6Chnl_D3_PNG_Customx1_Customx1_Tx4_RelPE_InitO_BN_LPL_ROI1_OCLR1/RUN_TEST/array/results_airfoil_vis.npz"
    # save_path = os.path.join('.', 'vis', 'Airfoil', 'PhysGTN_M25_ROI1')
    # mkdir(save_path)
    # visualize_airfoil_oformer(data_path, save_path, type_list=['dns', 'prs', 'vel_x', 'vel_y',])
    #
    # path_1 = "/home/jh/physics_graph_transformer/physics_graph_transformer_comparison/OFormer/airfoil/tmp/samples/results_airfoil_vis.npz"
    # path_2 = "/home/jh/physics_graph_transformer/physics_graph_transformer/results/PhysGTN_Airfoil_DeepMind_M24b_LEncDep6Chnl_D3_PNG_Customx1_Customx1_Tx4_RelPE_InitO_BN_LPL_OCLR1/RUN_TEST/array/results_airfoil_vis.npz"
    #
    # check_airfoil(path_1, path_2)
    # pass

