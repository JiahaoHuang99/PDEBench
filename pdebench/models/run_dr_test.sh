# DR
# FNO
CUDA_VISIBLE_DEVICES='0' \
python3 train_models_forward.py \
+args='config_diff-react' \
++args.model_name='FNO' \
++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
++args.if_training=False

# UNet
#CUDA_VISIBLE_DEVICES='3' \
#python3 train_models_forward.py \
#+args='config_diff-react' \
#++args.model_name='Unet' \
#++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
#++args.training_type='autoregressive' \
#++args.if_training=False