# DR
# PINO

export PYTHONPATH=/home/jh/PDEBench/pdebench/models: $PYTHONPATH
export PYTHONPATH=/home/jh/PDEBench: $PYTHONPATH
#
#model_name='PINO'
#task_name='PI0_Cx4'
#gup_id=0
#rm log_DR_${model_name}_${task_name}.txt
#CUDA_VISIBLE_DEVICES=${gup_id} \
#nohup python3 train_models_forward.py \
#+args='config_diff-react_phys' \
#++args.model_name=${model_name} \
#++args.epochs=300 \
#++args.model_update=1 \
#++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
#++args.if_training=True \
#++args.weight_loss_phys=0.0 \
#++args.weight_loss_data=1.0 \
#++args.task_name=${task_name} \
#++args.reduced_resolution=4 \
#++args.batch_size=4 \
#>> log_DR_${model_name}_${task_name}.txt &

#model_name='PINO'
#task_name='PI0.2_Cx4'
#gup_id=2
#rm log_DR_${model_name}_${task_name}.txt
#CUDA_VISIBLE_DEVICES=${gup_id} \
#nohup python3 train_models_forward.py \
#+args='config_diff-react_phys' \
#++args.model_name=${model_name} \
#++args.epochs=300 \
#++args.model_update=1 \
#++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
#++args.if_training=True \
#++args.weight_loss_phys=0.2 \
#++args.weight_loss_data=1.0 \
#++args.task_name=${task_name} \
#++args.reduced_resolution=4 \
#++args.batch_size=2 \
#++args.scheduler_step=5 \
#>> log_DR_${model_name}_${task_name}.txt &
#
#model_name='PINO'
#task_name='PI0_Cx1'
#gup_id=1
#rm log_DR_${model_name}_${task_name}.txt
#CUDA_VISIBLE_DEVICES=${gup_id} \
#nohup python3 train_models_forward.py \
#+args='config_diff-react_phys' \
#++args.model_name=${model_name} \
#++args.epochs=300 \
#++args.model_update=1 \
#++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
#++args.if_training=True \
#++args.weight_loss_phys=0.0 \
#++args.weight_loss_data=1.0 \
#++args.task_name=${task_name} \
#++args.reduced_resolution=1 \
#++args.batch_size=2 \
#++args.scheduler_step=5 \
#>> log_DR_${model_name}_${task_name}.txt &

#model_name='PINO_A'
#task_name='PI0_Cx1'
#gup_id=3
#rm log_DR_${model_name}_${task_name}.txt
#CUDA_VISIBLE_DEVICES=${gup_id} \
#nohup python3 train_models_forward.py \
#+args='config_diff-react_phys' \
#++args.model_name=${model_name} \
#++args.epochs=300 \
#++args.model_update=1 \
#++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
#++args.if_training=True \
#++args.weight_loss_phys=0.0 \
#++args.weight_loss_data=1.0 \
#++args.task_name=${task_name} \
#++args.reduced_resolution=1 \
#++args.batch_size=2 \
#++args.scheduler_step=5 \
#>> log_DR_${model_name}_${task_name}.txt &


#model_name='PINO_B'
#task_name='PI0_Cx1'
#gup_id=3
#rm log_DR_${model_name}_${task_name}.txt
#CUDA_VISIBLE_DEVICES=${gup_id} \
#nohup python3 train_models_forward.py \
#+args='config_diff-react_phys' \
#++args.model_name=${model_name} \
#++args.epochs=300 \
#++args.model_update=1 \
#++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
#++args.if_training=True \
#++args.weight_loss_phys=0.0 \
#++args.weight_loss_data=1.0 \
#++args.task_name=${task_name} \
#++args.reduced_resolution=1 \
#++args.batch_size=2 \
#++args.scheduler_step=5 \
#>> log_DR_${model_name}_${task_name}.txt &

model_name='PINO'
task_name='PI0_Cx1'
gup_id=0
rm log_DR_${model_name}_${task_name}.txt
CUDA_VISIBLE_DEVICES=${gup_id} \
nohup python3 train_models_forward.py \
+args='config_diff-react_phys' \
++args.model_name=${model_name} \
++args.epochs=300 \
++args.model_update=1 \
++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
++args.if_training=True \
++args.weight_loss_phys=0.0 \
++args.weight_loss_data=1.0 \
++args.task_name=${task_name} \
++args.reduced_resolution=1 \
++args.batch_size=2 \
++args.scheduler_step=5 \
++args.lr=0.0001 \
>> log_DR_${model_name}_${task_name}.txt &


model_name='PINO'
task_name='PI0.2_Cx1'
gup_id=1
rm log_DR_${model_name}_${task_name}.txt
CUDA_VISIBLE_DEVICES=${gup_id} \
nohup python3 train_models_forward.py \
+args='config_diff-react_phys' \
++args.model_name=${model_name} \
++args.epochs=300 \
++args.model_update=1 \
++args.data_path='/media/ssd/data_temp/PDE/data/DiffusionReaction/PDEBench/raw/2D/diffusion-reaction/' \
++args.if_training=True \
++args.weight_loss_phys=0.2 \
++args.weight_loss_data=1.0 \
++args.task_name=${task_name} \
++args.reduced_resolution=1 \
++args.batch_size=2 \
++args.scheduler_step=5 \
++args.lr=0.0001 \
>> log_DR_${model_name}_${task_name}.txt &