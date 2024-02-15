#include <chrono>
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>

typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> MatrixU32;

__attribute__((noinline))
void kernel(MatrixU32 &m, MatrixU32 &n, MatrixU32 &r)
{
    r.noalias() = m * n;
}

int main() {
    const char* inner_steps_env  = std::getenv("CHERRYBENCH_LOOP_STEPS");
    if (inner_steps_env == nullptr) {
        std::cerr << "CHERRYBENCH_LOOP_STEPS is not set" << std::endl;
        exit(1);
    }
    const int inner_steps = std::stoi(inner_steps_env);

    MatrixU32 m = MatrixU32::Random(PROBLEM_SIZE, PROBLEM_SIZE);
    MatrixU32 n = MatrixU32::Random(PROBLEM_SIZE, PROBLEM_SIZE);
    MatrixU32 r;

    kernel(m, n, r);  // Warm-up
    for (unsigned int i = 0; i < 10; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int j = 0; j < inner_steps; j++)
            kernel(m, n, r);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
             << "ns" << std::endl;
    }
}