#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include "mkl_cblas.h"

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

float random_f32() {
    uint32_t bits = 0x3f800000u | ((uint32_t)rand() & 0x007fffffu);
    float value;
    memcpy(&value, &bits, sizeof(bits));
    return value;
}

__attribute__((noinline))
void kernel(float *a, float *b, float *c, MKL_INT m, MKL_INT n, MKL_INT k) {
    // Leading dimensions for row-major storage
    MKL_INT lda = k;
    MKL_INT ldb = n;
    MKL_INT ldc = n;
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha,
                a, lda,
                b, ldb,
                beta, c, ldc);
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

    float *a;
    float *b;
    float *c;
    if (posix_memalign((void **)&a, 128, sizeof(float) * m * k) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    if (posix_memalign((void **)&b, 128, sizeof(float) * n * k) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    if (posix_memalign((void **)&c, 128, sizeof(float) * m * n) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    for (int i = 0; i < m * k; ++i) {
        a[i] = random_f32();
    }
    for (int i = 0; i < n * k; ++i) {
        b[i] = random_f32();
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
