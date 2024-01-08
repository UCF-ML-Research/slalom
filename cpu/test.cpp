#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <random>

double measure_matrix_multiplication_time(int dim_1, int dim_2, int dim_3, int num_trials = 1000) {
    // 初始化随机数生成器
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // 随机生成两个矩阵
    Eigen::MatrixXd A(dim_1, dim_2);
    Eigen::MatrixXd B(dim_2, dim_3);
    for (int i = 0; i < dim_1; ++i) {
        for (int j = 0; j < dim_2; ++j) {
            A(i, j) = distribution(generator);
        }
    }
    for (int i = 0; i < dim_2; ++i) {
        for (int j = 0; j < dim_3; ++j) {
            B(i, j) = distribution(generator);
        }
    }

    // 测量矩阵乘法执行时间
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_trials; ++i) {
        Eigen::MatrixXd C = A * B;  // 矩阵乘法
    }
    auto end = std::chrono::high_resolution_clock::now();

    // 计算平均执行时间
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count() / num_trials;
}

int main() {
    int dim_1 = 100, dim_2 = 200, dim_3 = 50;
    double average_execution_time = measure_matrix_multiplication_time(dim_1, dim_2, dim_3);
    std::cout << "Average execution time: " << average_execution_time << " seconds" << std::endl;
    return 0;
}
