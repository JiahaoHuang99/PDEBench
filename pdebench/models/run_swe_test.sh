# SWE
# FNO
CUDA_VISIBLE_DEVICES='0' \
python3 train_models_forward.py \
+args='config_rdb' \
++args.model_name='FNO' \
++args.data_path='/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench/raw/'
++args.if_training=False

# UNet
#CUDA_VISIBLE_DEVICES='3' \
#python3 train_models_forward.py \
#+args='config_rdb' \
#++args.model_name='Unet' \
#++args.data_path='/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench/raw/'
#++args.training_type='autoregressive' \
#++args.pushforward=True \
#++args.ar_mode=True \
#++args.if_training=False