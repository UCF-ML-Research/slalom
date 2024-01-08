make clean

cd App
make clean
make
make -f Makefile_cu
cd ..

cd SGXDNN
make clean
make
cd ..

make SGX_MODE=SIM

# python -u ./python/slalom/scripts/eval.py --model_name mobilenet --mode sgxdnn --batch_size 1 --max_num_batches 1 --use_sgx
# python -u ./python/slalom/scripts/eval_slalom.py --model_name mobilenet --batch_size 1 --max_num_batches 1 --use_sgx --blinding

# SGX: process one image per 0.0338 s
# tf-cpu: process one image per 0.1338 s

# python -u ./python/slalom/scripts/benchmark_softmax.py --model_name mobilenet --mode sgxdnn --batch_size 1 --max_num_batches 1 --use_sgx
# python -u ./python/slalom/scripts/benchmark_TEE_XY.py --use_sgx --dim_1 128 --dim_2 128 --dim_3 768
# python -u ./python/slalom/scripts/benchmark_QK.py --use_sgx
# python -u ./python/slalom/scripts/benchmark_AV.py --use_sgx
# python -u ./python/slalom/scripts/benchmark_XW.py --use_sgx --ratio 0.2
python -u ./python/slalom/scripts/benchmark_softmax.py --use_sgx