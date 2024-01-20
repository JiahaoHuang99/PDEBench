# DR
# FNO

export PYTHONPATH=/home/jh/PDEBench/pdebench/models: $PYTHONPATH
export PYTHONPATH=/home/jh/PDEBench: $PYTHONPATH

rm log_DR_FNO.txt
CUDA_VISIBLE_DEVICES='0' \
nohup python3 train_models_forward.py \
+args='config_diff-react' \
++args.model_name='FNO' \
++args.epochs=300 \
++args.model_update=1 \
++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
++args.if_training=True \
++args.batch_size=2 \
++args.scheduler_step=5 \
>> log_DR_FNO.txt &

# UNet
#CUDA_VISIBLE_DEVICES='3' \
#nohup python3 train_models_forward.py \
#+args='config_diff-react' \
#++args.model_name='Unet' \
#++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
#++args.training_type='autoregressive' \
#++args.if_training=True

# PINN
#CUDA_VISIBLE_DEVICES='3' \
#nohup python3 train_models_forward.py \
#+args='config_pinn_diff-react' \
#++args.model_name='PINN' \
#++args.root_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
#++args.filename='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/2D_diff-react_NA_NA.h5' \
#++args.if_training=True