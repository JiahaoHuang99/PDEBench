rm log_download_2d_cfd.txt

nohup python download_direct.py \
--root_folder /home/jh2446/rds/rds-ai323-trafficcam/physics_graph_transformer/physics_graph_transformer/data/CompressibleNavierStokes/CompressibleNavierStokes2D/PDEBench/raw \
--pde_name 2d_cfd \
>> log_download_2d_cfd.txt &