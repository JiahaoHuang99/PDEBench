# SW2D
# FNO
rm log_test_SW2D_FNO.txt
export CUDA_VISIBLE_DEVICES=0
nohup \
python3 train_models_forward.py \
+args='config_rdb' \
++args.model_name='FNO' \
++args.data_path='/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench/raw/' \
++args.if_training=False \
>> log_test_SW2D_FNO.txt &

# UNet
rm log_test_SW2D_UNet.txt
export CUDA_VISIBLE_DEVICES=1
nohup \
python3 train_models_forward.py \
+args='config_rdb' \
++args.model_name='Unet' \
++args.data_path='/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench/raw/' \
++args.if_training=False \
>> log_test_SW2D_UNet.txt &

#++args.training_type='autoregressive' \
#++args.pushforward=True \
#++args.ar_mode=True \