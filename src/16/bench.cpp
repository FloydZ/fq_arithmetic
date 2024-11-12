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


static void BM_gf16_matrix_transpose_16x16(benchmark::State& state) {
    auto *data1 = (uint8_t *)calloc(1, 256);
    auto *data2 = (uint8_t *)calloc(1, 256);
    uint8_t a = 0;
    for (auto _ : state) {
        gf16_transpose_16x16(data2, data1, 8);
        a += data2[8];
        gf16_transpose_16x16(data1, data2, 8);
        a += data1[8];

        benchmark::DoNotOptimize(a+=1);
    }

    free(data1); free(data2);
}

static void BM_gf16_matrix_transpose_64x64(benchmark::State& state) {
    auto *data1 = (uint8_t *)calloc(1, 2048);
    auto *data2 = (uint8_t *)calloc(1, 2048);
    uint8_t a = 0;
    for (auto _ : state) {
        gf16_transpose_64x64(data2, data1, 0);
        a += data2[8];
        gf16_transpose_64x64(data1, data2, 0);
        a += data1[8];

        benchmark::DoNotOptimize(a+=1);
    }

    free(data1); free(data2);
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

static void BM_gf16_matrix_transpose_64x64_avx2(benchmark::State& state) {
    auto *data1 = (uint8_t *)calloc(1, 2048);
    auto *data2 = (uint8_t *)calloc(1, 2048);
    uint8_t a = 0;
    for (auto _ : state) {
        gf16_transpose_64x64_avx2(data2, data1, 32);
        a += data2[8];
        gf16_transpose_64x64_avx2(data1, data2, 32);
        a += data1[8];

        benchmark::DoNotOptimize(a+=1);
    }

    free(data1); free(data2);
}

BENCHMARK(BM_gf16v_mul_u128);
BENCHMARK(BM_gf16v_mul_u256);
BENCHMARK(BM_gf16_matrix_transpose_64x64_avx2);
#endif

BENCHMARK(BM_gf16v_mul);
BENCHMARK(BM_gf16_matrix_transpose_16x16);
BENCHMARK(BM_gf16_matrix_transpose_64x64);
BENCHMARK_MAIN();

