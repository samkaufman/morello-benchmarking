#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <iostream>
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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <k> <n>" << std::endl;
        exit(1);
    }
    
    MKL_INT m = std::stoi(argv[1]);
    MKL_INT k = std::stoi(argv[2]);
    MKL_INT n = std::stoi(argv[3]);
    const char* inner_steps_env = std::getenv("CHERRYBENCH_LOOP_STEPS");
    if (inner_steps_env == nullptr) {
        std::cerr << "CHERRYBENCH_LOOP_STEPS is not set" << std::endl;
        exit(1);
    }
    const int inner_steps = std::stoi(inner_steps_env);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> a_distribution(0, 255);      // uint8 range
    std::uniform_int_distribution<int> b_distribution(-128, 127);   // int8 range

    MKL_UINT8 *a;
    MKL_INT8 *b;
    MKL_INT32 *c;
    if (posix_memalign((void **)&a, 128, sizeof(MKL_UINT8) * m * k) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    if (posix_memalign((void **)&b, 128, sizeof(MKL_INT8) * n * k) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    if (posix_memalign((void **)&c, 128, sizeof(MKL_INT32) * m * n) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    
    for (int i = 0; i < m * k; ++i) {
        a[i] = static_cast<MKL_UINT8>(a_distribution(generator));
    }
    for (int i = 0; i < n * k; ++i) {
        b[i] = static_cast<MKL_INT8>(b_distribution(generator));
    }

    kernel(a, b, c, m, n, k);  // Warm-up
    for (unsigned int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int j = 0; j < inner_steps; j++)
            kernel(a, b, c, m, n, k);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
             << "ns" << std::endl;
    }

    free(a);
    free(b);
    free(c);
    return 0;
}
