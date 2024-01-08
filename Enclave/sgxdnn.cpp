#define USE_EIGEN_TENSOR

#include "sgxdnn_main.hpp"

#include "tensor_types.h"

#include "Enclave.h"
#include "Enclave_t.h"

#include "Crypto.h"

using namespace tensorflow;

void ecall_load_model_float(char* model_json, float** filters)
{
	load_model_float(model_json, filters);
}

void ecall_predict_float(float* input, float* output, int batch_size)
{
	predict_float(input, output, batch_size);
}

void ecall_input_exp(float* gpu_x_exp_raw, float* r_exp_raw, float* a_idx, float* r_a_idx, float* x_r_idx, float* output, float* integrity_gap)
{
	input_exp(gpu_x_exp_raw, r_exp_raw, a_idx, r_a_idx, x_r_idx, output, integrity_gap);
}

void ecall_input_softmax(float* x_exp_raw, float* output)
{
	input_softmax(x_exp_raw, output);
}

void ecall_input_gelu(float* x, float* output)
{
	input_gelu(x, output);
}

void ecall_input_TEE_XY(unsigned int* dim_1, unsigned int* dim_2, unsigned int* dim_3, float* x, float* y, float* output)
{
	input_TEE_XY(dim_1, dim_2, dim_3, x, y, output);
}

void ecall_input_TEE_softmax(float* x, float* output)
{
	input_TEE_softmax(x, output);
}

void ecall_input_layernorm(float* x, float* output)
{
	input_layernorm(x, output);
}

void ecall_input_QK(unsigned int* gpu_res, unsigned int* Q_selected_indices, unsigned int* K_selected_indices, unsigned int* permuted_QR_indices, unsigned int* permuted_KS_indices, unsigned int* permuted_dim, unsigned int* output)
{
	input_QK(gpu_res, Q_selected_indices, K_selected_indices, permuted_QR_indices, permuted_KS_indices, permuted_dim, output);
}

void ecall_input_XW(unsigned int* gpu_res, unsigned int* X_selected_indices, unsigned int* W_selected_indices, unsigned int* permuted_XR_indices, unsigned int* permuted_WS_indices, unsigned int* permuted_dim_X, unsigned int* permuted_dim_W, unsigned int* output)
{
	input_XW(gpu_res, X_selected_indices, W_selected_indices, permuted_XR_indices, permuted_WS_indices, permuted_dim_X, permuted_dim_W, output);
}

void ecall_input_AV(unsigned int* gpu_res, unsigned int* A_selected_indices, unsigned int* V_selected_indices, unsigned int* permuted_AR_indices, unsigned int* permuted_VS_indices, unsigned int* permuted_dim_A, unsigned int* permuted_dim_V, unsigned int* output)
{
	input_AV(gpu_res, A_selected_indices, V_selected_indices, permuted_AR_indices, permuted_VS_indices, permuted_dim_A, permuted_dim_V, output);
}

void ecall_load_model_float_verify(char* model_json, float** filters, int preproc)
{
	load_model_float_verify(model_json, filters, preproc);
}

void ecall_predict_verify_float(float* input, float* output, float** aux_data, int batch_size)
{
	predict_verify_float(input, output, aux_data, batch_size);
}

void ecall_slalom_relu(float *in, float *out, float* blind, int numElements, char* activation) {
	slalom_relu(in, out, blind, numElements, activation);
}

void ecall_slalom_maxpoolrelu(float *in, float *out, float *blind, 
                   			  long int dim_in[4], long int dim_out[4],
                   			  int window_rows, int window_cols,
                   			  int row_stride, int col_stride,
                   			  int is_padding_same) {
	slalom_maxpoolrelu(in, out, blind, dim_in, dim_out, window_rows, window_cols, row_stride, col_stride, is_padding_same);
}

void ecall_slalom_init(int integrity, int privacy, int batch_size) {
	slalom_init(integrity, privacy, batch_size);
}

void ecall_slalom_get_r(float* out, int size) {
	slalom_get_r(out, size);
}

void ecall_slalom_set_z(float* z, float* z_enc, int size) {
	slalom_set_z(z, z_enc, size);
}

void ecall_slalom_blind_input(float* inp, float* out, int size) {
	slalom_blind_input(inp, out, size);
}

void ecall_sgxdnn_benchmarks(int num_threads) {
	sgxdnn_benchmarks(num_threads);
}
