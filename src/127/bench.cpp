#include <benchmark/benchmark.h>

#include "arith.h"
#include "matrix.h"

static long long cpucycles(void) noexcept {
#ifdef USE_AVX2
    unsigned long long result;
    asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
                 : "=a"(result)::"%rdx");
    return result;
#else
    return 0;
#endif
}


static void BM_gf127_matrix_transpose(benchmark::State& state) {
    const uint32_t nrows = state.range(0);
    const uint32_t ncols = state.range(0);
    gf127 *m1 = gf127_matrix_alloc(nrows, ncols);
    gf127 *m2 = gf127_matrix_alloc(nrows, ncols);
    gf127_matrix_rng(m1, nrows, ncols);
    gf127_matrix_rng(m2, nrows, ncols);

    gf127 c = 0;
    for (auto _ : state) {
        gf127_matrix_transpose(m2, m1, nrows, ncols);
        gf127_matrix_transpose(m1, m2, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }

    free(m1); free(m2);
}

static void BM_gf127_matrix_transpose_opt(benchmark::State& state) {
    const uint32_t nrows = state.range(0);
    const uint32_t ncols = state.range(0);
    gf127 *m1 = gf127_matrix_alloc(nrows, ncols);
    gf127 *m2 = gf127_matrix_alloc(nrows, ncols);
    gf127_matrix_rng(m1, nrows, ncols);
    gf127_matrix_rng(m2, nrows, ncols);

    gf127 c = 0;
    for (auto _ : state) {
        gf127_matrix_transpose_opt(m2, m1, nrows);
        gf127_matrix_transpose_opt(m1, m2, nrows);
        benchmark::DoNotOptimize(c+=m2[7]);
    }

    free(m1); free(m2);
}

#ifdef USE_AVX2
static void BM_gf127_matrix_transpose_opt_avx2(benchmark::State& state) {
    const uint32_t nrows = state.range(0);
    const uint32_t ncols = state.range(0);
    gf127 *m1 = gf127_matrix_alloc(nrows, ncols);
    gf127 *m2 = gf127_matrix_alloc(nrows, ncols);
    gf127_matrix_rng(m1, nrows, ncols);
    gf127_matrix_rng(m2, nrows, ncols);

    gf127 c = 0;
    for (auto _ : state) {
        gf127_matrix_transpose_opt_avx2(m2, m1, nrows);
        gf127_matrix_transpose_opt_avx2(m1, m2, nrows);
        benchmark::DoNotOptimize(c+=m2[7]);
    }

    free(m1); free(m2);
}

static void BM_gf127_matrix_transpose_opt_avx2_buffered(benchmark::State& state) {
    const uint32_t nrows = state.range(0);
    const uint32_t ncols = state.range(0);
    gf127 *m1 = gf127_matrix_alloc(nrows, ncols);
    gf127 *m2 = gf127_matrix_alloc(nrows, ncols);
    gf127_matrix_rng(m1, nrows, ncols);
    gf127_matrix_rng(m2, nrows, ncols);

    gf127 c = 0;
    for (auto _ : state) {
        gf127_matrix_transpose_opt_avx2_buffered(m2, m1, nrows);
        gf127_matrix_transpose_opt_avx2_buffered(m1, m2, nrows);
        benchmark::DoNotOptimize(c+=m2[7]);
    }

    free(m1); free(m2);
}

static void BM_gf127v_mul_u256(benchmark::State& state) {
    uint64_t c = 0;
    __m256i a = _mm256_set1_epi8(1);
    __m256i b = _mm256_set1_epi8(2);
    for (auto _ : state) {
        c -= cpucycles();
        a = gf127v_mul_u256(a, b);
        b = gf127v_mul_u256(a, b);
        a = gf127v_mul_u256(a, b);
        c += cpucycles();

        a = _mm256_add_epi16(a, b);
        b = _mm256_add_epi16(a, b);
        benchmark::DoNotOptimize(a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}
BENCHMARK(BM_gf127v_mul_u256);

// BENCHMARK(BM_gf127_matrix_transpose_opt_avx2)->DenseRange(64, 512, 64);
// BENCHMARK(BM_gf127_matrix_transpose_opt_avx2)->DenseRange(64, 512, 64);
// BENCHMARK(BM_gf127_matrix_transpose_opt_avx2_buffered)->DenseRange(64, 512, 64);
#endif

#ifdef USE_AVX512 

static void BM_gf127v_scalar_u512(benchmark::State& state) {
    uint64_t c = 0;
    __m512i a = _mm512_set1_epi8(18);
    __m512i table[2];
    uint8_t b = 1;

    for (auto _ : state) {
        c -= cpucycles();
        gf127v_scalar_u512_compute_table(table, b);
        a = gf127v_scalar_table_u512(a, table[0], table[1]);
        a = gf127v_scalar_table_u512(a, table[0], table[1]);
        a = gf127v_scalar_table_u512(a, table[0], table[1]);
        c += cpucycles();

        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b+=1);
        b %= 127;
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}
BENCHMARK(BM_gf127v_scalar_u512);

#endif

// BENCHMARK(BM_gf127_matrix_transpose)->DenseRange(64, 512, 64);
// BENCHMARK(BM_gf127_matrix_transpose_opt)->DenseRange(64, 512, 64);


BENCHMARK_MAIN();
