# Darcy Flow
# FNO
#CUDA_VISIBLE_DEVICES='0' \
#python3 train_models_forward.py \
#++args.filename='2D_DarcyFlow_beta1.0_Train.hdf5' \
#++args.model_name='FNO' \
#++args.data_path='/media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/' \
#++args.if_training=False \
#++args.reduced_resolution=2

# UNet
#CUDA_VISIBLE_DEVICES='3' \
#python3 train_models_forward.py \
#++args.filename='2D_DarcyFlow_beta1.0_Train.hdf5' \
#++args.model_name='Unet' \
#++args.data_path='/media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/' \
#++args.training_type='autoregressive' \
#++args.pushforward=True \
#++args.t_train=2 \
#++args.ar_mode=True \
#++args.if_training=False \
#++args.reduced_resolution=2