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
python -u ./python/slalom/scripts/benchmark_XW.py --use_sgx