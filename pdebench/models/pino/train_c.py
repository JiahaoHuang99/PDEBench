import os
import sys
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import wandb

# torch.manual_seed(0)
# np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pdebench.models.pino.pino import FNO1d, FNO2d, FNO3d
from pdebench.models.pino.utils import PINODatasetMult
from pdebench.models.metrics import metrics
from pdebench.models.pino.util_loss import loss_fn_phys

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_training(if_training,
                 continue_training,
                 num_workers,
                 modes,
                 width,
                 initial_step,
                 t_train,
                 num_channels,
                 batch_size,
                 epochs,
                 learning_rate,
                 scheduler_step,
                 scheduler_gamma,
                 model_update,
                 flnm,
                 single_file,
                 reduced_resolution,
                 reduced_resolution_t,
                 reduced_resolution_phys,
                 reduced_resolution_t_phys,
                 reduced_batch,
                 plot,
                 channel_plot,
                 x_min,
                 x_max,
                 y_min,
                 y_max,
                 t_min,
                 t_max,
                 base_path='../data/',
                 training_type='autoregressive',
                 weight_loss_phys=0.2,
                 weight_loss_data=1.,
                 task_name='PI0.2'
                 ):

    print(f'Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}')
    
    ################################################################
    # load data
    ################################################################

    if single_file:
        raise NotImplementedError
        # filename
        model_name = flnm[:-5] + '_FNO'
        print("FNODatasetSingle")

        # Initialize the dataset and dataloader
        train_data = FNODatasetSingle(flnm,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder = base_path
                                )
        val_data = FNODatasetSingle(flnm,
                              reduced_resolution=reduced_resolution,
                              reduced_resolution_t=reduced_resolution_t,
                              reduced_batch=reduced_batch,
                              initial_step=initial_step,
                              if_test=True,
                              saved_folder = base_path
                              )

    else:
        # filename
        model_name = flnm + '_PINO_C' + f'_{task_name}'
    
        print("FNODatasetMult")
        train_data = PINODatasetMult(flnm,
                                     reduced_resolution=reduced_resolution,
                                     reduced_resolution_t=reduced_resolution_t,
                                     reduced_batch=reduced_batch,
                                     saved_folder=base_path)

        train_data_phys = PINODatasetMult(flnm,
                                          reduced_resolution=reduced_resolution_phys,
                                          reduced_resolution_t=reduced_resolution_t_phys,
                                          reduced_batch=reduced_batch,
                                          saved_folder=base_path)

        val_data = PINODatasetMult(flnm,
                                   reduced_resolution=reduced_resolution,
                                   reduced_resolution_t=reduced_resolution_t,
                                   reduced_batch=reduced_batch,
                                   if_test=True,
                                   saved_folder=base_path)

        val_data_phys = PINODatasetMult(flnm,
                                        reduced_resolution=reduced_resolution_phys,
                                        reduced_resolution_t=reduced_resolution_t_phys,
                                        reduced_batch=reduced_batch,
                                        if_test=True,
                                        saved_folder = base_path)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    train_loader_phys = torch.utils.data.DataLoader(train_data_phys, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    val_loader_phys = torch.utils.data.DataLoader(val_data_phys, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    
    _, _data, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)
    if dimensions == 4:
        model = FNO1d(num_channels=num_channels,
                      width=width,
                      modes=modes,
                      initial_step=initial_step).to(device)
    elif dimensions == 5:
        model = FNO2d(num_channels=num_channels,
                      width=width,
                      modes1=modes,
                      modes2=modes,
                      initial_step=initial_step).to(device)
    elif dimensions == 6:
        model = FNO3d(num_channels=num_channels,
                      width=width,
                      modes1=modes,
                      modes2=modes,
                      modes3=modes,
                      initial_step=initial_step).to(device)
        
    # Set maximum time step of the data to train
    if t_train > _data.shape[-2]:
        t_train = _data.shape[-2]

    model_path = model_name + ".pt"
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    loss_fn = nn.MSELoss(reduction="mean")

    ## Check here
    # from util_loss import LpLoss
    # loss_fn = LpLoss()

    loss_val_min = np.infty
    
    start_epoch = 0

    if not if_training:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        Lx, Ly, Lz = 1., 1., 1.
        errs = metrics(val_loader_phys, model, Lx, Ly, Lz, plot, channel_plot,
                       model_name, x_min, x_max, y_min, y_max,
                       t_min, t_max, initial_step=initial_step)
        pickle.dump(errs, open(model_name+'.pickle', "wb"))
        
        return

    # set wandb
    # set wandb logger
    os.environ['WANDB_MODE'] = 'online'
    wandb.init(project='PINO_DR',
               entity="jiahaohuang",
               name=f'PINO_C_{task_name}')
    wandb.define_metric("Epoch")
    wandb.define_metric('Learning Rate', step_metric="Epoch")
    wandb.define_metric('TRAIN LOSS/Loss_MSE',)
    wandb.define_metric('TRAIN LOSS/Loss_MSE_Data',)
    wandb.define_metric('TRAIN LOSS/Loss_MSE_Phys',)
    wandb.define_metric('TRAIN LOSS Epoch/Loss_MSE', step_metric="Epoch")
    wandb.define_metric('TRAIN LOSS Epoch/Loss_MSE_Data', step_metric="Epoch")
    wandb.define_metric('TRAIN LOSS Epoch/Loss_MSE_Phys', step_metric="Epoch")
    wandb.define_metric('VAL LOSS Epoch/Loss_MSE', step_metric="Epoch")
    wandb.define_metric('VAL LOSS Epoch/Loss_MSE_Data', step_metric="Epoch")
    wandb.define_metric('VAL LOSS Epoch/Loss_MSE_Phys', step_metric="Epoch")
    wandb.watch(model)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if continue_training:
        print('Restoring model (that is the network\'s weights) from file...')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()
        
        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        start_epoch = checkpoint['epoch']
        loss_val_min = checkpoint['loss']
    
    for ep in range(start_epoch, epochs):
        model.train()
        t1 = default_timer()
        train_l2_full = []
        train_l2_data_full = []
        train_l2_phys_full = []
        step = 0
        for batch, batch_phys in zip(train_loader, train_loader_phys):
            loss_data = 0
            xx, yy, grid = batch
            xx_p, yy_p, grid_p = batch_phys

            # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # yy: target tensor [b, x1, ..., xd, t, v]
            # grid: meshgrid [b, x1, ..., xd, dims]
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)

            xx_p = xx_p.to(device)
            yy_p = yy_p.to(device)
            grid_p = grid_p.to(device)

            # Initialize the prediction tensor
            pred = yy[..., :initial_step, :]
            pred_p = yy_p[..., :initial_step, :]

            # Extract shape of the input tensor for reshaping (i.e. stacking the
            # time and channels dimension together)
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            inp_p_shape = list(xx_p.shape)
            inp_p_shape = inp_p_shape[:-2]
            inp_p_shape.append(-1)
    
            if training_type in ['autoregressive']:
                # Autoregressive loop
                for t in range(initial_step, t_train):
                    
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = xx.reshape(inp_shape)

                    # Extract target at current time step
                    y = yy[..., t:t+1, :]

                    # Model run
                    im = model(inp, grid)

                    # Loss calculation
                    _batch = im.size(0)
                    loss_data += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)
        
                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                for t in range(initial_step, t_train):
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp_p = xx_p.reshape(inp_p_shape)

                    # Model run
                    im_p = model(inp_p, grid_p)

                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred_p = torch.cat((pred_p, im_p), -2)

                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    xx_p = torch.cat((xx_p[..., 1:, :], im_p), dim=-2)

                # Loss calculation
                _batch, _res, _, _res_t, _ch = pred.shape
                _batch_p, _res_p, _p, _res_t_p, _ch_p = pred_p.shape
                assert _batch == _batch_p
                assert _res == _
                assert _res_p == _p
                assert _ch == _ch_p

                loss_phys = loss_fn_phys(a=None,
                                         u=pred_p[..., initial_step:, :].reshape(_batch_p, _res_p*_res_p, (_res_t_p-initial_step), _ch_p),
                                         batchsize=_batch_p,
                                         resolution=_res_p,
                                         task='dr')

                loss_data = loss_data
                # loss_data = loss_data / (t_train - initial_step)
                loss = weight_loss_phys * loss_phys + weight_loss_data * loss_data

                l2_phys_step = loss_phys.item()
                l2_data_step = loss_data.item()
                l2_step = loss.item()

                train_l2_data_full.append(l2_data_step)
                train_l2_phys_full.append(l2_phys_step)
                train_l2_full.append(l2_step)

                # record loss and accuracy
                log_wandb = {'Step': step}
                log_wandb['TRAIN LOSS/Loss_MSE'] = l2_step
                log_wandb['TRAIN LOSS/Loss_MSE_Data'] = l2_data_step
                log_wandb['TRAIN LOSS/Loss_MSE_Phys'] = l2_phys_step
                wandb.log(log_wandb)

                # print('step: {}, train_loss_step: {:.4f}, train_loss_phys_step: {:.4f}, train_loss_data_step: {:.4f}'.format(step, l2_step, l2_phys_step, l2_data_step))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

        if ep % model_update == 0:
            val_l2_data_full = []
            val_l2_phys_full = []
            val_l2_full = []
            with torch.no_grad():
                for batch, batch_phys in zip(val_loader, val_loader_phys):
                    xx, yy, grid = batch
                    xx_p, yy_p, grid_p = batch_phys

                    loss_data = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)

                    xx_p = xx_p.to(device)
                    yy_p = yy_p.to(device)
                    grid_p = grid_p.to(device)

                    if training_type in ['autoregressive']:
                        pred = yy[..., :initial_step, :]
                        pred_p = yy_p[..., :initial_step, :]

                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)

                        inp_p_shape = list(xx_p.shape)
                        inp_p_shape = inp_p_shape[:-2]
                        inp_p_shape.append(-1)

                        for t in range(initial_step, yy.shape[-2]):
                            inp_p = xx_p.reshape(inp_p_shape)
                            y_p = yy_p[..., t:t+1, :]
                            im_p = model(inp_p, grid_p)

                            loss_data += loss_fn(im_p.reshape(_batch, -1), y_p.reshape(_batch, -1))

                            pred_p = torch.cat((pred_p, im_p), -2)

                            xx_p = torch.cat((xx_p[..., 1:, :], im_p), dim=-2)

                        _batch_p, _res_p, _p, _res_t_p, _ch_p = pred_p.shape

                        loss_phys = loss_fn_phys(a=None,
                                                 u=pred_p[..., initial_step:, :].reshape(_batch_p, _res_p * _res_p, (_res_t_p - initial_step), _ch_p),
                                                 batchsize=_batch_p,
                                                 resolution=_res_p,
                                                 task='dr')

                        loss_data = loss_data
                        # loss_data = loss_data / (t_train - initial_step)

                        loss = weight_loss_phys * loss_phys + weight_loss_data * loss_data

                        l2_phys_step = loss_phys.item()
                        l2_data_step = loss_data.item()
                        l2_step = loss.item()

                        val_l2_data_full.append(l2_data_step)
                        val_l2_phys_full.append(l2_phys_step)
                        val_l2_full.append(l2_step)

                train_l2_data_full_avg = np.mean(train_l2_data_full)
                train_l2_phys_full_avg = np.mean(train_l2_phys_full)
                train_l2_full_avg = np.mean(train_l2_full)
                val_l2_data_full_avg = np.mean(val_l2_data_full)
                val_l2_phys_full_avg = np.mean(val_l2_phys_full)
                val_l2_full_avg = np.mean(val_l2_full)

                # record loss and accuracy
                log_wandb = {'Epoch': ep}
                log_wandb['TRAIN LOSS Epoch/Loss_MSE'] = train_l2_full_avg
                log_wandb['TRAIN LOSS Epoch/Loss_MSE_Data'] = train_l2_data_full_avg
                log_wandb['TRAIN LOSS Epoch/Loss_MSE_Phys'] = train_l2_phys_full_avg
                log_wandb['VAL LOSS Epoch/Loss_MSE'] = val_l2_full_avg
                log_wandb['VAL LOSS Epoch/Loss_MSE_Data'] = val_l2_data_full_avg
                log_wandb['VAL LOSS Epoch/Loss_MSE_Phys'] = val_l2_phys_full_avg
                wandb.log(log_wandb)

                if val_l2_full_avg < loss_val_min:
                    loss_val_min = val_l2_full_avg
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_val_min
                        }, model_path)

                model_path_master, model_name = os.path.split(model_path)
                model_name= model_name.replace('.pt', f'_{ep}.pt')
                mkdir(os.path.join(model_path_master, 'pino_weight'))
                model_path_epoch = os.path.join(model_path_master, 'pino_weight', model_name)
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val_min
                }, model_path_epoch)

        t2 = default_timer()
        log_wandb['Learning Rate'] = scheduler.get_last_lr()[0]
        wandb.log(log_wandb)
        scheduler.step()
        print('epoch: {}, t_used: {:.5f};\n train_l2 epoch: {:.4f}, train_l2_data_epoch: {:.4f}, train_l2_phys_epoch: {:.4f}, \n val_l2 epoch: {:.4f}, val_l2_data_epoch: {:.4f}, val_l2_phys_epoch: {:.4f}, \n'
              .format(ep, t2 - t1,
                      train_l2_full_avg, train_l2_data_full_avg, train_l2_phys_full_avg,
                      val_l2_full_avg, val_l2_data_full_avg, val_l2_phys_full_avg))


if __name__ == "__main__":
    
    run_training()
    print("Done.")

