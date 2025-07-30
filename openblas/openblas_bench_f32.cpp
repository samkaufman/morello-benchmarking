#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <iostream>
#include <cblas.h>

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

__attribute__((noinline))
void kernel(float *a, float *b, float *c, int m, int n, int k) {
    // Leading dimensions for row-major storage
    int lda = k;
    int ldb = n;
    int ldc = n;
    
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
    
    int m = std::stoi(argv[1]);
    int k = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);
    
    const char* inner_steps_env = std::getenv("CHERRYBENCH_LOOP_STEPS");
    if (inner_steps_env == nullptr) {
        std::cerr << "CHERRYBENCH_LOOP_STEPS is not set" << std::endl;
        exit(1);
    }
    const int inner_steps = std::stoi(inner_steps_env);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    // Allocate matrices and fill with random data
    float *a, *b, *c;
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
        a[i] = distribution(generator);
    }
    for (int i = 0; i < n * k; ++i) {
        b[i] = distribution(generator);
    }

    // Benchmark
    kernel(a, b, c, m, n, k);
    for (unsigned int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int j = 0; j < inner_steps; j++) {
            kernel(a, b, c, m, n, k);
        }
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
             << "ns" << std::endl;
    }

    free(a);
    free(b);
    free(c);
    return 0;
}
