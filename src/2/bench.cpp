#include <benchmark/benchmark.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

gf2 *v1, *v2, *v3;
const uint32_t N = 1000;

gf2 *m1, *m2, *m3;
const uint32_t nrows  = 32;
const uint32_t ncols  = 7;
const uint32_t ncols2 = 32;


static void BM_gf2_mul_u64(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf2_mul_u64(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf2_vector_eval(benchmark::State& state) {
    for (auto _ : state) {
        const gf2 t = rand();
        gf2_vector_eval(v1, t, N);
        benchmark::DoNotOptimize(v1[3] += 1);
    }
}

static void BM_gf2_vector_add(benchmark::State& state) {
    for (auto _ : state) {
        gf2_vector_add(v1, v2, v3, N);
        gf2_vector_add(v2, v3, v1, N);
        gf2_vector_add(v3, v1, v2, N);
        benchmark::DoNotOptimize(v3[7] += 1);
    }
}

static void BM_gf2_vector_scalar_add_v2(benchmark::State& state) {
    for (auto _ : state) {
        const gf2 t = rand() & 1u;
        gf2_vector_scalar_add_v2(v1, v2, t, v3, N);
        gf2_vector_scalar_add_v2(v2, v3, t, v1, N);
        gf2_vector_scalar_add_v2(v3, v1, t, v2, N);
        benchmark::DoNotOptimize(v3[7] += 1);
    }
}

static void BM_gf2_matrix_mul(benchmark::State& state) {
    for (auto _ : state) {
        gf2_matrix_mul(m1, m2, m3, nrows, ncols, ncols2);
        gf2_matrix_mul(m2, m3, m1, nrows, ncols, ncols2);
        gf2_matrix_mul(m3, m1, m2, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

// BENCHMARK(BM_gf2_vector_eval);
// BENCHMARK(BM_gf2_vector_add);
// BENCHMARK(BM_gf2_vector_scalar_add_v2);
BENCHMARK(BM_gf2_matrix_mul);
#ifdef USE_AVX2

static void BM_gf2_mul_u256(benchmark::State& state) {
    const __m256i acc = {783246, 872346, 876502, 236478234023948};
    __m256i a = {23874687,905389,235645274,873465876},
            b = {87326457823,234652734687,8723468726347,78234687236478};
    for (auto _ : state) {
        a = gf2_mul_u256(a, b);
        a ^= acc;
        benchmark::DoNotOptimize(a);
    }
}


static void BM_gf2_vector_eval_u256(benchmark::State& state) {
    for (auto _ : state) {
        const gf2 t = rand();
        gf2_vector_eval_u256(v1, t, N);
        benchmark::DoNotOptimize(v1[3] += 1);
    }
}

static void BM_gf2_vector_add_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf2_vector_add_u256(v1, v2, v3, N);
        gf2_vector_add_u256(v2, v3, v1, N);
        gf2_vector_add_u256(v3, v1, v2, N);
        benchmark::DoNotOptimize(v3[7] += 1);
    }
}

static void BM_gf2_vector_scalar_add_v2_u256(benchmark::State& state) {
    for (auto _ : state) {
        const gf2 t = rand() & 1u;
        gf2_vector_scalar_add_u256_v2(v1, v2, t, v3, N);
        gf2_vector_scalar_add_u256_v2(v2, v3, t, v1, N);
        gf2_vector_scalar_add_u256_v2(v3, v1, t, v2, N);
        benchmark::DoNotOptimize(v3[7] += 1);
    }
}

static void BM_gf2_matrix_mul_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf2_matrix_mul_u256(m1, m2, m3, nrows, ncols, ncols2);
        gf2_matrix_mul_u256(m2, m3, m1, nrows, ncols, ncols2);
        gf2_matrix_mul_u256(m3, m1, m2, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}


//BENCHMARK(BM_gf2_mul_u256);
//BENCHMARK(BM_gf2_vector_eval_u256);
//BENCHMARK(BM_gf2_vector_add_u256);
//BENCHMARK(BM_gf2_vector_scalar_add_v2_u256);
BENCHMARK(BM_gf2_matrix_mul_u256);
#endif

//BENCHMARK(BM_gf2_mul_u64);



int main(int argc, char** argv) {
    v1 = (gf2 *)malloc(N);
    v2 = (gf2 *)malloc(N);
    v3 = (gf2 *)malloc(N);

    m1 = gf2_matrix_alloc(nrows, ncols2);
    m2 = gf2_matrix_alloc(nrows, ncols2);
    m3 = gf2_matrix_alloc(nrows, ncols2);
    gf2_matrix_random(m1, nrows, ncols2);
    gf2_matrix_random(m2, nrows, ncols2);
    gf2_matrix_random(m3, nrows, ncols2);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    free(v1); free(v2); free(v3);
    free(m1); free(m2); free(m3);
    return 0;
}
