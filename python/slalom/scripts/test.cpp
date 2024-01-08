    void input_XW(unsigned long int eid, unsigned int* gpu_res, unsigned int* X_selected_indices, unsigned int* W_selected_indices, unsigned int* permuted_XR_indices, unsigned int* permuted_WS_indices, unsigned int* permuted_dim_X, unsigned int* permuted_dim_W, unsigned int* output) {
        auto start = std::chrono::high_resolution_clock::now();
        sgx_status_t ret = ecall_input_XW(eid, gpu_res, X_selected_indices, W_selected_indices, permuted_XR_indices, permuted_WS_indices, permuted_dim_X, permuted_dim_W, output);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        else {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            printf("Execution time: %f ms.\n", elapsed * 1000);
        }
    }