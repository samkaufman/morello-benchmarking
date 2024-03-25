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

// Lightweight C++ implementation of the gemma model.

#include "compression/compress-inl.h"
// copybara:import_next_line:gemma_cpp
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "ops.h"

#ifndef GEMMA_ONCE
#define GEMMA_ONCE

#include <math.h>  // sqrtf
#include <stddef.h>
#include <stdio.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <new>

#include "compression/compress.h"
#include "configs.h"
#include "gemma.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

struct timespec ts_diff(struct timespec start, struct timespec end) {
  struct timespec temp;
  if ((end.tv_nsec - start.tv_nsec) < 0) {
    temp.tv_sec = end.tv_sec - start.tv_sec - 1;
    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  }
  return temp;
}

namespace gcpp {

template <class TConfig>
struct Layer {
  Layer() = default;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  static constexpr size_t kAttVecEinsumWSize = kHeads * kQKVDim * kModelDim;
  // 3x for (query, key, value)
  static constexpr size_t kQKVEinsumWSize = 3 * kHeads * kQKVDim * kModelDim;
  // 2x for (gelu gating vector, gated vector)
  static constexpr size_t kGatingEinsumWSize = 2 * kFFHiddenDim * kModelDim;

  std::array<float, kAttVecEinsumWSize> attn_vec_einsum_w;
  std::array<float, kQKVEinsumWSize> qkv_einsum_w;
  std::array<float, kGatingEinsumWSize> gating_einsum_w;
  std::array<float, kModelDim * kFFHiddenDim> linear_w;
  std::array<float, kModelDim> pre_attention_norm_scale;
  std::array<float, kModelDim> pre_ffw_norm_scale;
};

template <class TConfig>
struct Weights {
  Weights() = default;

  hwy::AlignedUniquePtr<Layer<TConfig>[]> layers;  // kLayers

  std::array<float, TConfig::kVocabSize * TConfig::kModelDim>
      embedder_input_embedding;

  std::array<float, TConfig::kModelDim> final_norm_scale;
};

template <class TConfig>
struct CompressedLayer {
  // No ctor/dtor, allocated via AllocateAligned.

  using TLayer = gcpp::Layer<TConfig>;

  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;

  // Compressed Parameters
  // We don't yet have an RMSNorm that accepts all WeightT.
  CompressedArray<hwy::bfloat16_t, kModelDim> c_pre_attention_norm_scale;
  CompressedArray<hwy::bfloat16_t, kModelDim> c_pre_ffw_norm_scale;
  CompressedArray<WeightT, TLayer::kGatingEinsumWSize> c_gating_einsum_w;
  CompressedArray<WeightT, kModelDim * kFFHiddenDim> c_linear_w;
  CompressedArray<WeightT, TLayer::kQKVEinsumWSize> c_qkv_einsum_w;
  CompressedArray<WeightT, TLayer::kAttVecEinsumWSize> c_attn_vec_einsum_w;
};

// Array instead of single large allocation for parallel mem init. Split out of
// CompressedWeights so that only these pointers are initialized, not the
// CompressedArray.
template <class TConfig>
struct CompressedLayerPointers {
  explicit CompressedLayerPointers(hwy::ThreadPool& pool) {
    pool.Run(0, TConfig::kLayers, [this](uint64_t task, size_t /*thread*/) {
      this->c_layers[task] = hwy::AllocateAligned<CompressedLayer<TConfig>>(1);
    });
  }

  using CLayer = CompressedLayer<TConfig>;
  std::array<hwy::AlignedFreeUniquePtr<CLayer[]>, TConfig::kLayers> c_layers;
};

template <class TConfig>
struct CompressedWeights {
  // No ctor/dtor, allocated via AllocateAligned.

  CompressedArray<EmbedderInputT, TConfig::kVocabSize * TConfig::kModelDim>
      c_embedder_input_embedding;

  CompressedArray<hwy::bfloat16_t, TConfig::kModelDim> c_final_norm_scale;

  // Must be last so that the other arrays remain aligned.
  CompressedLayerPointers<TConfig> c_layer_ptrs;

  const CompressedLayer<TConfig>* CLayer(size_t layer) const {
    return c_layer_ptrs.c_layers[layer].get();
  }
  CompressedLayer<TConfig>* CLayer(size_t layer) {
    return c_layer_ptrs.c_layers[layer].get();
  }
};

// Aligned.
template <class TConfig, size_t TBatchSize>
struct Activations {
  static constexpr size_t kBatchSize = TBatchSize;
  using LayerConfig = Layer<TConfig>;
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kQKVDim = TConfig::kQKVDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static constexpr size_t kKVHeads = TConfig::kKVHeads;
  static constexpr size_t kCachePosSize = TConfig::kLayers * kKVHeads * kQKVDim;
  static constexpr size_t kCacheLayerSize = kKVHeads * kQKVDim;

  std::array<float, kBatchSize * kModelDim> x;  // input
  std::array<float, kBatchSize * kModelDim> pre_att_rms_out;
  std::array<float, kBatchSize * kHeads * kQKVDim> q;  // query vector
  std::array<float, kBatchSize * kHeads * TConfig::kSeqLen>
      att;                                                   // attention vector
  std::array<float, kBatchSize * kHeads * kQKVDim> att_out;  // attention output
  std::array<float, kHeads * kBatchSize * kModelDim>
      att_post1;  // attention output after linear transformation, per head
  std::array<float, kBatchSize * kModelDim>
      att_post2;  // accumulation of attention outputs over heads
  std::array<hwy::bfloat16_t, kBatchSize * kModelDim> bf_pre_ffw_rms_out;
  std::array<float, kBatchSize * TConfig::kFFHiddenDim * 2> ffw_hidden;
  // bf_ version can't be used until GeluMulToBF16 issue in FFW() is resolved.
  // std::array<hwy::bfloat16_t, kBatchSize * 2 * TConfig::kFFHiddenDim>
  //     bf_ffw_hidden;
  std::array<float, kBatchSize * kModelDim> ffw_out;
};

template <class Config>
KVCache CreateKVCache() {
  return CreateKVCache(Config::kLayers * Config::kKVHeads * Config::kQKVDim,
                       Config::kSeqLen);
}

KVCache CreateKVCache(Model type) {
  switch (type) {
    case Model::GEMMA_2B:
      return CreateKVCache<ConfigGemma2B>();
    case Model::GEMMA_7B:
      return CreateKVCache<ConfigGemma7B>();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(type));
  }
}

}  // namespace gcpp
#endif  // GEMMA_ONCE

// SIMD code, compiled once per target.
namespace gcpp {
namespace HWY_NAMESPACE {

template <class TConfig, size_t kBatchSize>
HWY_NOINLINE void Attention(size_t batch_start, size_t batch_idx, size_t layer,
                            Activations<TConfig, kBatchSize>& activations,
                            const CompressedLayer<TConfig>* c_layer,
                            KVCache& kv_cache, hwy::ThreadPool& pool) {
  PROFILER_ZONE("Gen.Attention");
  const size_t pos = batch_start + batch_idx;
  HWY_DASSERT(batch_idx < kBatchSize);
  static constexpr size_t kQKVDim = gcpp::Activations<TConfig, 1>::kQKVDim;
  static constexpr size_t kCachePosSize =
      gcpp::Activations<TConfig, kBatchSize>::kCachePosSize;
  static constexpr size_t kCacheLayerSize =
      gcpp::Activations<TConfig, kBatchSize>::kCacheLayerSize;
  static constexpr size_t kModelDim =
      gcpp::Activations<TConfig, kBatchSize>::kModelDim;
  static constexpr size_t kHeads = TConfig::kHeads;
  static const float kQueryScale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(kQKVDim)));

  pool.Run(0, kHeads, [&](const uint64_t head, size_t /*thread*/) HWY_ATTR {
    // linear projections to QKV
    const size_t head_offset =
        3 * kQKVDim * kModelDim;  // 3x for QKV dimensions
    const size_t q_offset = head * head_offset + 0 * kQKVDim * kModelDim;
    const size_t k_offset = head * head_offset + 1 * kQKVDim * kModelDim;
    const size_t v_offset = head * head_offset + 2 * kQKVDim * kModelDim;

    float* HWY_RESTRICT q =
        activations.q.data() + head * kQKVDim + batch_idx * kHeads * kQKVDim;

    const size_t batch_offset = batch_idx * kModelDim;

    MatVecLoop<kQKVDim, kModelDim>(
        c_layer->c_qkv_einsum_w, q_offset,
        activations.pre_att_rms_out.data() + batch_offset, q);

    const size_t kv_offset =
        pos * kCachePosSize + layer * kCacheLayerSize + head * kQKVDim;

    TwoOfsMatVecLoop<kQKVDim, kModelDim>(
        c_layer->c_qkv_einsum_w, k_offset, v_offset,
        activations.pre_att_rms_out.data() + batch_offset,
        kv_cache.key_cache.get() + kv_offset,
        kv_cache.value_cache.get() + kv_offset);

    // Calculate scores
    float* HWY_RESTRICT head_att = activations.att.data() +
                                   head * TConfig::kSeqLen +
                                   batch_idx * kHeads * kQKVDim;

    Rope(q, kQKVDim, pos);
    Rope(kv_cache.key_cache.get() + kv_offset, kQKVDim, pos);
    MulByConst(kQueryScale, q, kQKVDim);
    // Compute Q dot K scores
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const size_t cache_offset =
          pos2 * kCachePosSize + layer * kCacheLayerSize + head * kQKVDim;
      const float* HWY_RESTRICT k2 = kv_cache.key_cache.get() + cache_offset;
      const float score = Dot(q, k2, kQKVDim);
      head_att[pos2] = score;
    }
    Softmax(head_att, pos + 1);

    // Weighted summation
    float* HWY_RESTRICT att_out = activations.att_out.data() + head * kQKVDim +
                                  batch_idx * kHeads * kQKVDim;
    hwy::ZeroBytes(att_out, kQKVDim * sizeof(*att_out));
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const size_t cache_offset =
          pos2 * kCachePosSize + layer * kCacheLayerSize + head * kQKVDim;
      float* HWY_RESTRICT v2 = kv_cache.value_cache.get() + cache_offset;
      MulByConstAndAdd(head_att[pos2], v2, att_out, kQKVDim);
    }
    // linear projection from kQKVDim back to kModelDim, sum projections
    // across heads
    float* HWY_RESTRICT head_out =
        head == 0
            ? activations.att_post2.data() + batch_idx * kModelDim
            : activations.att_post1.data() + head * kBatchSize * kModelDim;
    MatVecLoop<kModelDim, kQKVDim>(c_layer->c_attn_vec_einsum_w,
                                   head * kModelDim * kQKVDim, att_out,
                                   head_out);
  });

  // accumulate output across all heads into att_post2. head 0 already wrote
  // directly to att_post2.
  for (size_t head = 1; head < kHeads; ++head) {
    AddFrom(activations.att_post1.data() + head * kBatchSize * kModelDim,
            activations.att_post2.data() + batch_idx * kModelDim, kModelDim);
  }
}

template <typename TConfig, size_t kBatchSize>
HWY_NOINLINE void FFW(Activations<TConfig, kBatchSize>& activations,
                      size_t batch_idx, const CompressedLayer<TConfig>* c_layer,
                      hwy::ThreadPool& pool) {
  HWY_DASSERT(batch_idx < kBatchSize);
  static constexpr size_t kModelDim = TConfig::kModelDim;
  static constexpr size_t kFFHiddenDim = TConfig::kFFHiddenDim;
  const size_t hidden_offset = batch_idx * kFFHiddenDim * 2;

  {
    PROFILER_ZONE("Gen.FFW.GatedGELU");
    const hwy::bfloat16_t* HWY_RESTRICT vec =
        activations.bf_pre_ffw_rms_out.data() + batch_idx * kModelDim;
    float* HWY_RESTRICT out = activations.ffw_hidden.data() + hidden_offset;
    float* HWY_RESTRICT out_mul = out + kFFHiddenDim;

    // Same matrix, first and second half of rows. Could fuse into one MatVec,
    // but separating them could help on NUMA e.g. multiple sockets.
    MatVec<kFFHiddenDim, kModelDim>(c_layer->c_gating_einsum_w,
                                    kFFHiddenDim * kModelDim, vec, out_mul,
                                    pool);

    // Gate, will go through the nonlinearity.
    MatVec<kFFHiddenDim, kModelDim>(c_layer->c_gating_einsum_w, 0, vec, out,
                                    pool);

    namespace hn = hwy::HWY_NAMESPACE;
    using DF = hn::ScalableTag<float>;
    using VF = hn::Vec<DF>;
    hn::Transform1(DF(), out, kFFHiddenDim, out_mul,
                   [](DF df, VF v, VF mul)
                       HWY_ATTR { return hn::Mul(mul, Gelu(df, v)); });
  }

  PROFILER_ZONE("Gen.FFW\\GatedGELU");
  MatVec<kModelDim, kFFHiddenDim>(
      c_layer->c_linear_w, 0, activations.ffw_hidden.data() + hidden_offset,
      activations.ffw_out.data() + batch_idx * kModelDim, pool);
}

// __builtin_sqrt is not constexpr as of Clang 17.
#if HWY_COMPILER_GCC_ACTUAL && defined(HWY_HAVE_SCALAR_BF16_OPERATORS) && \
    HWY_HAVE_SCALAR_BF16_OPERATORS
#define GEMMA_CONSTEXPR_SQRT constexpr
static GEMMA_CONSTEXPR_SQRT HWY_INLINE float Sqrt(float x) {
  return __builtin_sqrt(x);
}
#else
#define GEMMA_CONSTEXPR_SQRT
static GEMMA_CONSTEXPR_SQRT HWY_INLINE float Sqrt(float x) { return sqrtf(x); }
#endif

template <typename TConfig>
GEMMA_CONSTEXPR_SQRT float EmbeddingScaling() {
  // Round to bf16 to match Gemma's Embedder, which casts before mul.
  return hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<hwy::bfloat16_t>(
      Sqrt(static_cast<float>(TConfig::kModelDim))));
}

template <class TConfig>
void DecodeLayer(size_t pos, size_t layer,
                 const CompressedLayer<TConfig>* c_layer,
                 Activations<TConfig, 1>& activations, KVCache& kv_cache,
                 hwy::ThreadPool& pool) {
  static constexpr size_t kModelDim = TConfig::kModelDim;
  RMSNorm(activations.x.data(), c_layer->c_pre_attention_norm_scale.data(),
          activations.pre_att_rms_out.data(), kModelDim);
  Attention<TConfig, 1>(pos, 0, layer, activations, c_layer, kv_cache, pool);
  AddFrom(activations.att_post2.data(), activations.x.data(), kModelDim);
  RMSNorm(activations.x.data(), c_layer->c_pre_ffw_norm_scale.data(),
          activations.bf_pre_ffw_rms_out.data(), kModelDim);
  FFW<TConfig, 1>(activations, /* batch_idx = */ 0, c_layer, pool);
  AddFrom(activations.ffw_out.data(), activations.x.data(), kModelDim);
}

template <class TConfig, size_t kBatchSize>
void RandomizeActivations(Activations<TConfig, kBatchSize>* acts) {
  std::default_random_engine gen;
  std::uniform_real_distribution<float> dis_real32(0, 10);
  std::uniform_int_distribution<int16_t> dis_int16(0, 10);

  for (auto& x : acts->x) x = dis_real32(gen);
  for (auto& x : acts->pre_att_rms_out) x = dis_real32(gen);
  for (auto& x : acts->q) x = dis_real32(gen);
  for (auto& x : acts->att) x = dis_real32(gen);
  for (auto& x : acts->att_out) x = dis_real32(gen);
  for (auto& x : acts->att_post1) x = dis_real32(gen);
  for (auto& x : acts->att_post2) x = dis_real32(gen);
  for (auto& x : acts->bf_pre_ffw_rms_out)
    x = hwy::bfloat16_t::FromBits(dis_int16(gen));
  for (auto& x : acts->ffw_hidden) x = dis_real32(gen);
  for (auto& x : acts->ffw_out) x = dis_real32(gen);
}


template <size_t kCapacity>
void RandomizeCompressedArrayBF16(CompressedArray<hwy::bfloat16_t, kCapacity> *arr) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int16_t> dis_int16(0, 10);
  auto n = arr->NumElements();
  for (size_t i = 0; i < n; i++)
    arr->data()[i] = hwy::bfloat16_t::FromBits(dis_int16(gen));
}

template <class TConfig>
void RandomizeCompressedWeights(CompressedWeights<TConfig> *c_weights) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int16_t> dis_int16(0, 10);

  RandomizeCompressedArrayBF16(&c_weights->c_embedder_input_embedding);
  RandomizeCompressedArrayBF16(&c_weights->c_final_norm_scale);

  for (auto& layer : c_weights->c_layer_ptrs.c_layers) {
    auto* c_layer = layer.get();
    RandomizeCompressedArrayBF16(&c_layer->c_pre_attention_norm_scale);
    RandomizeCompressedArrayBF16(&c_layer->c_pre_ffw_norm_scale);
    RandomizeCompressedArrayBF16(&c_layer->c_gating_einsum_w);
    RandomizeCompressedArrayBF16(&c_layer->c_linear_w);
    RandomizeCompressedArrayBF16(&c_layer->c_qkv_einsum_w);
    RandomizeCompressedArrayBF16(&c_layer->c_attn_vec_einsum_w);
  }
}

template <class TConfig>
void RunBenchmark() {
  const char* inner_steps_env = getenv("CHERRYBENCH_LOOP_STEPS");
  if (inner_steps_env == NULL) {
    fprintf(stderr,
            "Environment variable CHERRYBENCH_LOOP_STEPS is not set.\n");
    exit(1);
  }
  const int bench_samples = atoi(inner_steps_env);

  struct timespec start, end;

  hwy::ThreadPool pool(1);

  // Initialize buffers (which will be re-used).
  using CWeights = CompressedWeights<TConfig>;
  hwy::AlignedFreeUniquePtr<uint8_t[]> c_weights_u8 =
      hwy::AllocateAligned<uint8_t>(sizeof(CWeights));
  CWeights* c_weights = reinterpret_cast<CWeights*>(c_weights_u8.get());
  new (&c_weights->c_layer_ptrs) CompressedLayerPointers<TConfig>(pool);
  RandomizeCompressedWeights(c_weights);
  Activations<TConfig, 1> activations;
  RandomizeActivations<TConfig>(&activations);
  auto kv_cache =
      CreateKVCache(TConfig::kLayers * TConfig::kKVHeads * TConfig::kQKVDim,
                    TConfig::kSeqLen);

  for (int i = 0; i < 10; i++) {
    DecodeLayer(
      /*pos=*/0, /*layer=*/0, c_weights->CLayer((size_t)0), activations,
      kv_cache, pool);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (long long bench_itr = 0; bench_itr < bench_samples; ++bench_itr) {
      DecodeLayer(
        /*pos=*/0, /*layer=*/0, c_weights->CLayer((size_t)0), activations,
        kv_cache, pool);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    const struct timespec delta = ts_diff(start, end);
    const long long elapsed_ns = delta.tv_sec * 1000000000L + delta.tv_nsec;
    printf("%lldns\n", elapsed_ns);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp

namespace gcpp {

KVCache CreateKVCache(size_t size_cache_pos, size_t seq_len) {
  KVCache kv_cache = {};
  kv_cache.key_cache = hwy::AllocateAligned<float>(seq_len * size_cache_pos);
  kv_cache.value_cache = hwy::AllocateAligned<float>(seq_len * size_cache_pos);
  return kv_cache;
}

void RunBenchmark(Model type) {
  switch (type) {
    case Model::GEMMA_2B:
      return HWY_STATIC_DISPATCH(RunBenchmark)<ConfigGemma2B>();
    case Model::GEMMA_7B:
      return HWY_STATIC_DISPATCH(RunBenchmark)<ConfigGemma7B>();
    default:
      HWY_ABORT("Model type %d unknown.", static_cast<int>(type));
  }
}

}  // namespace gcpp
