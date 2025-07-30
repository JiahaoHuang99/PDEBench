# Darcy Flow
# FNO
rm -r log_test_DF2D_FNO.txt
export CUDA_VISIBLE_DEVICES=0
nohup \
python3 train_models_forward.py \
++args.filename='2D_DarcyFlow_beta1.0_Train.hdf5' \
++args.model_name='FNO' \
++args.data_path='/media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/' \
++args.if_training=False \
++args.reduced_resolution=2 \
>> log_test_DF2D_FNO.txt &

# UNet
rm -r log_test_DF2D_UNet.txt
export CUDA_VISIBLE_DEVICES=1
nohup \
python3 train_models_forward.py \
++args.filename='2D_DarcyFlow_beta1.0_Train.hdf5' \
++args.model_name='Unet' \
++args.data_path='/media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/' \
++args.t_train=2 \
++args.if_training=False \
++args.reduced_resolution=2 \
>> log_test_DF2D_UNet.txt &

#++args.training_type='autoregressive' \
#++args.pushforward=True \
#++args.ar_mode=True \