
# Install 
Error when using bash with 
```
OSError: /home/jh/miniconda3/envs/pde_bench/lib/python3.9/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: symbol cublasLtGetStatusString version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference
```

Do:
```
pip uninstall nvidia_cublas_cu11
```



# DO EVAL

go to models
```
cd pdebench/models
```

DF2D
```
bash run_darcy_test.sh
```

DR2D
```
bash run_dr_test.sh
```

SW2D
```
bash run_swe_test.sh
```
