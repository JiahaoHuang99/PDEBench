rm log_download_swe.txt

nohup python download_direct.py \
--root_folder /home/jh2446/rds/rds-ai323-trafficcam/physics_graph_transformer/physics_graph_transformer/data/ShallowWater/PDEBench/raw \
--pde_name swe \
>> log_download_swe.txt &