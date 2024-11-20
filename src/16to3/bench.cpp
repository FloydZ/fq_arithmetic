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


static void BM_gf16to3_matrix_mul(benchmark::State& state) {
    const uint32_t nrows = 16, ncols = 16;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_rand(m1, nrows, ncols);
    gf16to3_matrix_rand(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_mul(m3, m1, m2, nrows, ncols, nrows);
        gf16to3_matrix_mul(m1, m2, m3, nrows, ncols, nrows);
        gf16to3_matrix_mul(m2, m3, m1, nrows, ncols, nrows);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
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



static void BM_gf16to3_matrix_mul_16x16(benchmark::State& state) {
    const uint32_t nrows = 16, ncols = 16;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_rand(m1, nrows, ncols);
    gf16to3_matrix_rand(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_mul_16x16(m3, m1, m2);
        gf16to3_matrix_mul_16x16(m1, m2, m3);
        gf16to3_matrix_mul_16x16(m2, m3, m1);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
}

BENCHMARK(BM_gf16to3v_mul_gf16_u256);
BENCHMARK(BM_gf16to3v_mul_u256);
BENCHMARK(BM_gf16to3_matrix_mul_16x16);

#endif

BENCHMARK(BM_gf16to3_matrix_mul);
BENCHMARK(BM_gf16to3_mul);
// TODO BENCHMARK(BM_gf16to3v_mul);
BENCHMARK_MAIN();

