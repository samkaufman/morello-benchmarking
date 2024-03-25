// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_H_

#include <algorithm>
#include <cctype>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "compression/compress.h"  // SfpStream/NuqStream
#include "configs.h"  // kSeqLen
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // hwy::bfloat16_t
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace gcpp {

// Allowable types for GEMMA_WEIGHT_T (can be specified at compilation time):
// float, hwy::bfloat16_t, SfpStream, NuqStream
#ifndef GEMMA_WEIGHT_T
#define GEMMA_WEIGHT_T SfpStream
#endif  // !GEMMA_WEIGHT_T
using WeightT = GEMMA_WEIGHT_T;

using EmbedderInputT = hwy::bfloat16_t;
constexpr size_t kPrefillBatchSize = 16;
constexpr bool kSystemPrompt = false;

struct KVCache {
  hwy::AlignedFreeUniquePtr<float[]>
      key_cache;  // batch_size * kSeqLen * kLayers * kKVHeads * kQKVDim
  hwy::AlignedFreeUniquePtr<float[]>
      value_cache;  // batch_size * kSeqLen * kLayers * kKVHeads * kQKVDim
};

// Model variants: see configs.h for details.
enum class Model { GEMMA_2B, GEMMA_7B };
enum class ModelTraining { GEMMA_IT, GEMMA_PT };

struct RuntimeConfig {
  size_t max_tokens;
  size_t max_generated_tokens;
  float temperature;
  int verbosity;
};

KVCache CreateKVCache(Model type);  // convenient workaround for now
KVCache CreateKVCache(size_t size_cache_pos, size_t seq_len);

void RunBenchmark(Model type);

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_H_
