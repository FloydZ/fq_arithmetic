#include <benchmark/benchmark.h>
#include "arith.h"

static void BM_gf16v_mul(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf16v_mul(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

#ifdef USE_AVX2

static void BM_gf16v_mul_u128(benchmark::State& state) {
    const __m128i acc = {783246, 872346};
    __m128i a = {23874687,905389},
            b = {87326457823,234652734687};
    for (auto _ : state) {
        a = gf16v_mul_u128(a, b);
        a ^= acc;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf16v_mul_u256(benchmark::State& state) {
    const __m256i acc = {783246, 872346, 876502, 236478234023948};
    __m256i a = {23874687,905389,235645274,873465876},
            b = {87326457823,234652734687,8723468726347,78234687236478};
    for (auto _ : state) {
        a = gf16v_mul_u256(a, b);
        a ^= acc;
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_gf16v_mul_u128);
BENCHMARK(BM_gf16v_mul_u256);
#endif

BENCHMARK(BM_gf16v_mul);
BENCHMARK_MAIN();

