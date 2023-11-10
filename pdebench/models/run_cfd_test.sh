# Darcy Flow
# FNO
CUDA_VISIBLE_DEVICES='0' \
python3 train_models_forward.py \
+args='config_2DCFD' \
++args.filename='2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5' \
++args.model_name='FNO' \
++args.data_path='/media/ssd/data_temp/PDE/data/CompressibleNavierStokes/CompressibleNavierStokes2D/PDEBench/raw/2D/CFD/2D_Train_Rand/' \
++args.if_training=False

CUDA_VISIBLE_DEVICES='3' \
python3 train_models_forward.py \
+args='config_2DCFD' \
++args.filename='2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5' \
++args.model_name='Unet' \
++args.data_path='/media/ssd/data_temp/PDE/data/CompressibleNavierStokes/CompressibleNavierStokes2D/PDEBench/raw/2D/CFD/2D_Train_Rand/' \
++args.training_type='autoregressive' \
++args.pushforward=True \
++args.ar_mode=True \
++args.if_training=False \

