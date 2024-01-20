# DR
# PINO

export PYTHONPATH=/home/jh/PDEBench/pdebench/models: $PYTHONPATH
export PYTHONPATH=/home/jh/PDEBench: $PYTHONPATH

CUDA_VISIBLE_DEVICES='3' \
python3 train_models_forward.py \
+args='config_diff-react_phys' \
++args.model_name='PINO' \
++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
++args.if_training=False \
++args.weight_loss_phys=0.0 \
++args.weight_loss_data=1.0 \
++args.task_name='PI0'

CUDA_VISIBLE_DEVICES='3' \
python3 train_models_forward.py \
+args='config_diff-react_phys' \
++args.model_name='PINO' \
++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
++args.if_training=False \
++args.weight_loss_phys=0.2 \
++args.weight_loss_data=1.0 \
++args.task_name='PI0.2'
