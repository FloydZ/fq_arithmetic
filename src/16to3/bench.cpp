#include <benchmark/benchmark.h>
#include "arith.h"

static void BM_gf16to3_mul(benchmark::State& state) {
    uint16_t a = 1;
    uint16_t b = 2;
    for (auto _ : state) {
        a = gf16to3_mul(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf16to3v_mul(benchmark::State& state) {
    uint64_t a = 1;
    uint64_t b = 2;
    for (auto _ : state) {
        a = gf16to3v_mul(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

#ifdef USE_AVX2
static void BM_gf16to3v_mul_gf16_u256(benchmark::State& state) {
    __m256i a = _mm256_set1_epi8(1);
    uint64_t b = 2;
    for (auto _ : state) {
        a = gf16to3v_mul_gf16_u256(a, b);
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf16to3v_mul_u256(benchmark::State& state) {
    __m256i a = _mm256_set1_epi8(1);
    __m256i b = _mm256_set1_epi8(1);
    for (auto _ : state) {
        a = gf16to3v_mul_u256(a, b);
        a += b;
        b += a;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_gf16to3v_mul_gf16_u256);
BENCHMARK(BM_gf16to3v_mul_u256);
#endif

BENCHMARK(BM_gf16to3_mul);
// TODO BENCHMARK(BM_gf16to3v_mul);
BENCHMARK_MAIN();

