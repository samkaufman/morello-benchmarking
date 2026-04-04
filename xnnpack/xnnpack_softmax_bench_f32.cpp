#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <random>

#include <pthreadpool.h>
#include <xnnpack.h>

__attribute__((noinline))
void kernel(xnn_operator_t softmax_op, pthreadpool_t threadpool) {
    // ultimately calls xnn_compute_floating_point_softmax in XNNPACK
    if (xnn_run_operator(softmax_op, threadpool) != xnn_status_success) {
        std::cerr << "xnn_run_operator failed" << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <batch_size> <channels> <num_threads>" << std::endl;
        exit(1);
    }

    const size_t batch_size = std::stoull(argv[1]);
    const size_t channels = std::stoull(argv[2]);
    const size_t num_threads = std::stoull(argv[3]);

    const char* inner_steps_env = std::getenv("CHERRYBENCH_LOOP_STEPS");
    if (inner_steps_env == nullptr) {
        std::cerr << "CHERRYBENCH_LOOP_STEPS is not set" << std::endl;
        exit(1);
    }
    const size_t inner_steps = std::stoull(inner_steps_env);

    const size_t elements = batch_size * channels;
    const size_t extra_elements = (XNN_EXTRA_BYTES + sizeof(float) - 1) / sizeof(float);

    float* input;
    float* output;
    if (posix_memalign((void**)&input, 128, sizeof(float) * (elements + extra_elements)) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }
    if (posix_memalign((void**)&output, 128, sizeof(float) * (elements + extra_elements)) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(1);
    }

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t i = 0; i < elements; ++i) {
        input[i] = distribution(generator);
    }

    pthreadpool_t threadpool = nullptr;
    if (num_threads > 1) {
        threadpool = pthreadpool_create(num_threads);
        if (threadpool == nullptr) {
            std::cerr << "pthreadpool_create failed" << std::endl;
            exit(1);
        }
    }

    if (xnn_initialize(nullptr) != xnn_status_success) {
        std::cerr << "xnn_initialize failed" << std::endl;
        exit(1);
    }

    xnn_operator_t softmax_op = nullptr;
    if (xnn_create_softmax_nc_f32(0, &softmax_op) != xnn_status_success) {
        std::cerr << "xnn_create_softmax_nc_f32 failed" << std::endl;
        exit(1);
    }
    if (xnn_reshape_softmax_nc_f32(softmax_op, channels, channels, channels, batch_size, threadpool) != xnn_status_success) {
        std::cerr << "xnn_reshape_softmax_nc_f32 failed" << std::endl;
        exit(1);
    }
    if (xnn_setup_softmax_nc_f32(softmax_op, input, output) != xnn_status_success) {
        std::cerr << "xnn_setup_softmax_nc_f32 failed" << std::endl;
        exit(1);
    }

    kernel(softmax_op, threadpool);
    for (unsigned int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < inner_steps; j++) {
            kernel(softmax_op, threadpool);
        }
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
             << "ns" << std::endl;
    }

    if (xnn_delete_operator(softmax_op) != xnn_status_success) {
        std::cerr << "xnn_delete_operator failed" << std::endl;
        exit(1);
    }
    if (threadpool != nullptr) {
        pthreadpool_destroy(threadpool);
    }
    if (xnn_deinitialize() != xnn_status_success) {
        std::cerr << "xnn_deinitialize failed" << std::endl;
        exit(1);
    }

    free(input);
    free(output);
    return 0;
}
