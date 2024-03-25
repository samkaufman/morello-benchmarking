#include "gemma.h"  // Gemma

int main(int argc, char** argv) {
  RunBenchmark(gcpp::Model::GEMMA_2B);
  return 0;
}
