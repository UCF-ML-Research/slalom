	void input_QK(unsigned int* gpu_res, unsigned int* Q_selected_indices, unsigned int* K_selected_indices, 
				unsigned int* permuted_QR_indices, unsigned int* permuted_KS_indices, unsigned int* permuted_dim, unsigned int* output) {

		const unsigned int permuted_dim_value = *permuted_dim;
		model_int.mem_pool = new MemPool(2, 3211264 * sizeof(unsigned int));

		array2d gpu_res_dim = {permuted_dim_value, permuted_dim_value};
		array1d Q_selected_indices_dim = {128};
		array1d K_selected_indices_dim = {128};
		array1d permuted_QR_indices_dim = {permuted_dim_value};
		array1d permuted_KS_indices_dim = {permuted_dim_value};
		array2d output_dim = {permuted_dim_value, permuted_dim_value};

		int gpu_res_size = permuted_dim_value * permuted_dim_value;
		int Q_selected_indices_size = 128;
		int K_selected_indices_size = 128;
		int permuted_QR_indices_size = permuted_dim_value;
		int permuted_KS_indices_size = permuted_dim_value;
		int output_size = permuted_dim_value * permuted_dim_value;

		// Copy input into enclave
		unsigned int* gpu_res_copy = model_int.mem_pool->alloc<unsigned int>(gpu_res_size);
		std::copy(gpu_res, gpu_res + gpu_res_size, gpu_res_copy);

		unsigned int* Q_selected_indices_copy = model_int.mem_pool->alloc<unsigned int>(Q_selected_indices_size);
		std::copy(Q_selected_indices, Q_selected_indices + Q_selected_indices_size, Q_selected_indices_copy);

		unsigned int* K_selected_indices_copy = model_int.mem_pool->alloc<unsigned int>(K_selected_indices_size);
		std::copy(K_selected_indices, K_selected_indices + K_selected_indices_size, K_selected_indices_copy);

		unsigned int* permuted_QR_indices_copy = model_int.mem_pool->alloc<unsigned int>(permuted_QR_indices_size);
		std::copy(permuted_QR_indices, permuted_QR_indices + permuted_QR_indices_size, permuted_QR_indices_copy);

		unsigned int* permuted_KS_indices_copy = model_int.mem_pool->alloc<unsigned int>(permuted_KS_indices_size);
		std::copy(permuted_KS_indices, permuted_KS_indices + permuted_KS_indices_size, permuted_KS_indices_copy);

		// Map tensors
		auto map_gpu_res = TensorMap<unsigned int, 2>(gpu_res_copy, gpu_res_dim);
		auto map_Q_selected_indices = TensorMap<unsigned int, 1>(Q_selected_indices_copy, Q_selected_indices_dim);
		auto map_K_selected_indices = TensorMap<unsigned int, 1>(K_selected_indices_copy, K_selected_indices_dim);
		auto map_permuted_QR_indices = TensorMap<unsigned int, 1>(permuted_QR_indices_copy, permuted_QR_indices_dim);
		auto map_permuted_KS_indices = TensorMap<unsigned int, 1>(permuted_KS_indices_copy, permuted_KS_indices_dim);
		unsigned int* gpu_res_all = model_int.mem_pool->alloc<unsigned int>(permuted_dim_value * permuted_dim_value);
		unsigned int* gpu_res_1 = model_int.mem_pool->alloc<unsigned int>(128 * 128);
		unsigned int* gpu_res_2 = model_int.mem_pool->alloc<unsigned int>(128 * (permuted_dim_value - 128));
		unsigned int* gpu_res_3 = model_int.mem_pool->alloc<unsigned int>((permuted_dim_value - 128) * 128);
		unsigned int* gpu_res_4 = model_int.mem_pool->alloc<unsigned int>((permuted_dim_value - 128) * (permuted_dim_value - 128));
		// Correct the order of gpu_res based on permuted indices
		for (int i = 0; i < permuted_dim_value; ++i) {
			for (int j = 0; j < permuted_dim_value; ++j) {
				int corrected_i = static_cast<int>(map_permuted_QR_indices(i));
				int corrected_j = static_cast<int>(map_permuted_KS_indices(j));
				output[corrected_i * permuted_dim_value + corrected_j] = map_gpu_res(i, j);
			}
		}
		// Compute RK
		for (int i = 128; i < permuted_dim_value; ++i) {
			for (int j = 0; j < 128; ++j) {
				int coefficient_i = i - 128 + 128;
				int coefficient_j = static_cast<int>(map_K_selected_indices(j)) + 128;
				output[i * permuted_dim_value + j] = output[i * permuted_dim_value + j] + output[coefficient_i * permuted_dim_value + coefficient_j];
			}
		}
		// Compute QS
		for (int i = 0; i < 128; ++i) {
			for (int j = 128; j < permuted_dim_value; ++j) {
				int coefficient_i = static_cast<int>(map_Q_selected_indices(i)) + 128;
				int coefficient_j = j - 128 + 128;
				output[i * permuted_dim_value + j] = output[i * permuted_dim_value + j] + output[coefficient_i * permuted_dim_value + coefficient_j];
			}
		}
		// Compute RES
		for (int i = 0; i < 128; ++i) {
			for (int j = 0; j < 128; ++j) {
				int coefficient_R = static_cast<int>(map_Q_selected_indices(i)) + 128;
				int coefficient_S = static_cast<int>(map_K_selected_indices(j)) + 128;
				output[i * permuted_dim_value + j] = output[i * permuted_dim_value + j] + output[i * permuted_dim_value + coefficient_S]
														  + output[coefficient_R * permuted_dim_value + j]
														  - output[coefficient_R * permuted_dim_value + coefficient_S];
			}
		}
		printf("TEST12n");
		// Release memory
		model_int.mem_pool->release(gpu_res_copy);
		model_int.mem_pool->release(Q_selected_indices_copy);
		model_int.mem_pool->release(K_selected_indices_copy);
		model_int.mem_pool->release(permuted_QR_indices_copy);
		model_int.mem_pool->release(permuted_KS_indices_copy);
	}
