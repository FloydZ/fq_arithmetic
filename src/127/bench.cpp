#include <benchmark/benchmark.h>

#include "arith.h"
#include "matrix.h"

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
// BENCHMARK(BM_gf127_matrix_transpose_opt_avx2)->DenseRange(64, 512, 64);;
// BENCHMARK(BM_gf127_matrix_transpose_opt_avx2_buffered)->DenseRange(64, 512, 64);;
#endif

// BENCHMARK(BM_gf127_matrix_transpose)->DenseRange(64, 512, 64);;
BENCHMARK(BM_gf127_matrix_transpose_opt)->DenseRange(64, 512, 64);;


BENCHMARK_MAIN();
