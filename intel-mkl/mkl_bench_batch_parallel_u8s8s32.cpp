#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <iostream>
#include <omp.h>
#include "mkl.h"

constexpr MKL_INT32 alpha = 1;
constexpr MKL_INT32 beta = 1;

__attribute__((noinline))
void kernel(MKL_UINT8 *a, MKL_INT8 *b, MKL_INT32 *c, MKL_INT m, MKL_INT n, MKL_INT k) {
    // Leading dimensions for row-major storage
    MKL_INT lda = k;
    MKL_INT ldb = n;
    MKL_INT ldc = n;
    
    // Per the documentation, when using CblasRowMajor, the types for A and B are swapped,
    // so, despite the name, the following is really a u8s8s32 operation.
    MKL_INT32 co = 0;
    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
                       m, n, k,
                       alpha,
                       a, lda, 0,  // A matrix as unsigned int8
                       b, ldb, 0,  // B matrix as signed int8
                       beta,
                       c, ldc,
                       &co);
}

__attribute__((noinline))
void batch_parallel_kernel(MKL_UINT8 **a_batch, MKL_INT8 **b_batch, MKL_INT32 **c_batch, 
                          MKL_INT batch_size, MKL_INT m, MKL_INT n, MKL_INT k) {
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
    
    MKL_INT batch_size = std::stoi(argv[1]);
    MKL_INT m = std::stoi(argv[2]);
    MKL_INT k = std::stoi(argv[3]);
    MKL_INT n = std::stoi(argv[4]);
    
    const char* inner_steps_env = std::getenv("CHERRYBENCH_LOOP_STEPS");
    if (inner_steps_env == nullptr) {
        std::cerr << "CHERRYBENCH_LOOP_STEPS is not set" << std::endl;
        exit(1);
    }
    const int inner_steps = std::stoi(inner_steps_env);

    // Set number of OpenMP threads to match batch size
    omp_set_num_threads(batch_size);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> a_distribution(0, 255);      // uint8 range
    std::uniform_int_distribution<int> b_distribution(-128, 127);   // int8 range

    // Allocate arrays for each batch
    MKL_UINT8 **a_batch = new MKL_UINT8*[batch_size];
    MKL_INT8 **b_batch = new MKL_INT8*[batch_size];
    MKL_INT32 **c_batch = new MKL_INT32*[batch_size];
    
    for (int i = 0; i < batch_size; i++) {
        if (posix_memalign((void **)&a_batch[i], 128, sizeof(MKL_UINT8) * m * k) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(1);
        }
        if (posix_memalign((void **)&b_batch[i], 128, sizeof(MKL_INT8) * n * k) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(1);
        }
        if (posix_memalign((void **)&c_batch[i], 128, sizeof(MKL_INT32) * m * n) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(1);
        }
        
        // Initialize with random values
        for (int j = 0; j < m * k; ++j) {
            a_batch[i][j] = static_cast<MKL_UINT8>(a_distribution(generator));
        }
        for (int j = 0; j < n * k; ++j) {
            b_batch[i][j] = static_cast<MKL_INT8>(b_distribution(generator));
        }
    }

    // Warm-up
    batch_parallel_kernel(a_batch, b_batch, c_batch, batch_size, m, n, k);
    
    // Benchmark
    for (unsigned int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int j = 0; j < inner_steps; j++) {
            batch_parallel_kernel(a_batch, b_batch, c_batch, batch_size, m, n, k);
        }
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
    delete[] a_batch;
    delete[] b_batch;
    delete[] c_batch;
    
    return 0;
}
