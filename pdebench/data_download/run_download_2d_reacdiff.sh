rm log_download.txt

nohup python download_direct.py \
--root_folder /home/jh2446/rds/rds-ai323-trafficcam/physics_graph_transformer/physics_graph_transformer/data/DiffusionReaction/PDEBench/raw \
--pde_name 2d_reacdiff \
>> log_download.txt &