#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <iostream>
#include <omp.h>
#include "cblas.h"
#include "blis.h"

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

__attribute__((noinline))
void kernel(float *a, float *b, float *c, dim_t m, dim_t n, dim_t k) {
    // Leading dimensions for row-major storage
    dim_t lda = k;
    dim_t ldb = n;
    dim_t ldc = n;
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha,
                a, lda,
                b, ldb,
                beta, c, ldc);
}

__attribute__((noinline))
void batch_parallel_kernel(float **a_batch, float **b_batch, float **c_batch,
                           int batch_size, dim_t m, dim_t n, dim_t k) {
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        kernel(a_batch[i], b_batch[i], c_batch[i], m, n, k);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <batch_size> <m> <k> <n>" << std::endl;
        exit(1);
    }
    int batch_size = std::stoi(argv[1]);
    dim_t m = std::stoi(argv[2]);
    dim_t k = std::stoi(argv[3]);
    dim_t n = std::stoi(argv[4]);

    const char* inner_steps_env = std::getenv("CHERRYBENCH_LOOP_STEPS");
    if (inner_steps_env == nullptr) {
        std::cerr << "CHERRYBENCH_LOOP_STEPS is not set" << std::endl;
        exit(1);
    }
    const int inner_steps = std::stoi(inner_steps_env);

    bli_thread_set_num_threads(1); // keep AOCL single-threaded
    omp_set_num_threads(batch_size);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);


    float **a_batch = new float*[batch_size];
    float **b_batch = new float*[batch_size];
    float **c_batch = new float*[batch_size];
    for (int i = 0; i < batch_size; i++) {
        if (posix_memalign((void **)&a_batch[i], 128, sizeof(float) * m * k) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(1);
        }    
        if (posix_memalign((void **)&b_batch[i], 128, sizeof(float) * n * k) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(1);
        }    
        if (posix_memalign((void **)&c_batch[i], 128, sizeof(float) * m * n) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(1);
        }    
        for (dim_t j = 0; j < m * k; ++j) a_batch[i][j] = distribution(generator);
        for (dim_t j = 0; j < n * k; ++j) b_batch[i][j] = distribution(generator);
    }
    batch_parallel_kernel(a_batch, b_batch, c_batch, batch_size, m, n, k); // Warm-up
    for (unsigned int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int j = 0; j < inner_steps; j++)
            batch_parallel_kernel(a_batch, b_batch, c_batch, batch_size, m, n, k);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
            << "ns" << std::endl;
    }
    // Cleanup
    for (int i = 0; i < batch_size; i++) {
        free(a_batch[i]);
        free(b_batch[i]);
        free(c_batch[i]);
    }
    delete [] a_batch;
    delete [] b_batch;
    delete [] c_batch;
    return 0;
}
