#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <iostream>
#include "blis.h"

constexpr int32_t alpha = 1;
constexpr int32_t beta = 1;
constexpr char storage = 'r'; // Row major.
constexpr char transa = 'n';  // No transpose.
constexpr char transb = 'n';
constexpr char reordera = 'n';
constexpr char reorderb = 'r'; // Reorder B matrix, equal to packing entire B matrix.

__attribute__((noinline))
void kernel(uint8_t *a, int8_t *b, int16_t *c, dim_t m, dim_t n, dim_t k) {
    // Leading dimensions.
    dim_t lda = k;
    dim_t ldb = n;
    dim_t ldc = n;
    
    aocl_gemm_u8s8s16os16(
        storage, transa, transb,
        m, n, k,
        alpha,
        a, lda, reordera,
        b, ldb, reorderb,
        beta,
        c, ldc,
        NULL);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <k> <n>" << std::endl;
        exit(1);
    }
    
    dim_t m = std::stoi(argv[1]);
    dim_t k = std::stoi(argv[2]);
    dim_t n = std::stoi(argv[3]);
    const char* inner_steps_env  = std::getenv("CHERRYBENCH_LOOP_STEPS");
    if (inner_steps_env == nullptr) {
        std::cerr << "CHERRYBENCH_LOOP_STEPS is not set" << std::endl;
        exit(1);
    }
    const int inner_steps = std::stoi(inner_steps_env);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<uint8_t> distribution_a(0, 255);
    std::uniform_int_distribution<int8_t> distribution_b(-128, 127);
    uint8_t *a;
    int8_t *b;
    int16_t *c;
    if (posix_memalign((void **)&a, 128, sizeof(uint8_t) * m * k) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    if (posix_memalign((void **)&b, 128, sizeof(int8_t) * n * k) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    if (posix_memalign((void **)&c, 128, sizeof(int16_t) * m * n) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    for (dim_t i = 0; i < m * k; ++i) {
        a[i] = distribution_a(generator);
    }
    for (dim_t i = 0; i < n * k; ++i) {
        b[i] = distribution_b(generator);
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