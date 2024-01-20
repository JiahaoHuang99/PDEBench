# SWE
# FNO
CUDA_VISIBLE_DEVICES='0' \
nohup python3 train_models_forward.py \
+args='config_rdb' \
++args.model_name='FNO' \
++args.epochs=300 \
++args.data_path='/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench/raw/' \
++args.if_training=True \
>> log_SW_FNO.txt &

# UNet
#CUDA_VISIBLE_DEVICES='3' \
#python3 train_models_forward.py \
#+args='config_rdb' \
#++args.model_name='Unet' \
#++args.data_path='/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench/raw/'
#++args.training_type='autoregressive' \
#++args.pushforward=True \
#++args.ar_mode=True \
#++args.if_training=True &

# PINN
#CUDA_VISIBLE_DEVICES='3' \
#python3 train_models_forward.py \
#+args='config_pinn_swe2d' \
#++args.model_name='PINN' \
#++args.root_path='/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench/raw/' \
#++args.filename='/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench/raw/2D_rdb_NA_NA.h5' \
#++args.if_training=True &