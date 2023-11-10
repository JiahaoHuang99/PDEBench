rm log_download_2d_cfd.txt

nohup python download_direct.py \
--root_folder /media/ssd/data_temp/PDE/data/CompressibleNavierStokes/CompressibleNavierStokes2D/PDEBench/raw \
--pde_name 2d_cfd \
>> log_download_2d_cfd.txt &