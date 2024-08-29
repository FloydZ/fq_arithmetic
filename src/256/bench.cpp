#include <benchmark/benchmark.h>
#include "arith.h"

static void BM_gf256_mul_u64(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf256v_mul_u64(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf256_scalar_u64(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf256v_scalar_u64(a, b);
        a += 1;
        b += 1;
		benchmark::DoNotOptimize(a); 
    }
}

static void BM_gf256_scalar_u64_v2(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf256v_scalar_u64_v2(a, b);
        a += 1;
        b += 1;
		benchmark::DoNotOptimize(a);
    }
}

#ifdef USE_AVX2
#define LIST_SIZE (1u << 13u)
gf256 *A, *B, *C;

static void BM_gf256_mul_u256(benchmark::State& state) {
    v256 a = {0, 1, 2, 3, 4, 5, 6, 7}, b = {2, 12, 4, 18, 6, 17, 1, 9}, one = {1,13,2,5,3,12,18,9};
    for (auto _ : state) {
        a.v256 = gf256v_mul_u256(a.v256, b.v256);
        a.v256 = _mm256_add_epi32(a.v256, one.v256);
        benchmark::DoNotOptimize(a.v256);
    }
}
static void BM_gf256_mul_u256_v2(benchmark::State& state) {
    v256 a = {0, 1, 2, 3, 4, 5, 6, 7}, b = {2, 12, 4, 18, 6, 17, 1, 9}, one = {1,13,2,5,3,12,18,9};
    for (auto _ : state) {
        a.v256 = gf256v_mul_u256_v2(a.v256, b.v256);
        a.v256 = _mm256_add_epi32(a.v256, one.v256);
		benchmark::DoNotOptimize(a.v256); 
    }
}

static void BM_tbl32_gf256_mul_const(benchmark::State& state) {
    v256 a = {0, 1, 2, 3, 4, 5, 6, 7}, one = {1,13,2,5,3,12,18,9};
    uint32_t b = 2;
    for (auto _ : state) {
        a.v256 = tbl32_gf256_mul_const(b, a.v256);
        a.v256 = _mm256_add_epi32(a.v256, one.v256);
        b += 1;
		benchmark::DoNotOptimize(a.v256); 
    }
}

static void BM_gf256v_mul_scalar_avx2(benchmark::State& state) {
    v256 a = {0, 1, 2, 3, 4, 5, 6, 7}, one = {1,13,2,5,3,12,18,9};
    uint32_t b = 2;
    for (auto _ : state) {
        a.v256 = gf256v_mul_scalar_avx2(a.v256, b);
        a.v256 = _mm256_add_epi32(a.v256, one.v256);
        b += 1;
		benchmark::DoNotOptimize(a.v256); 
    }
}

static void BM_gf256v_vector_add_scalar_u256(benchmark::State& state) {
    for (size_t i = 0; i < state.range(0); ++i) {
        A[i] = rand(); B[i] = rand(); C[i] = rand();
    }

    for (auto _ : state) {
        gf256 s = rand();
        gf256_vector_add_scalar_u256(A, B, s, C, sizeof(gf256) * state.range(0));
        gf256_vector_add_scalar_u256(B, C, s, A, sizeof(gf256) * state.range(0));
        gf256_vector_add_scalar_u256(C, A, s, B, sizeof(gf256) * state.range(0));
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_gf256_mul_u256);
BENCHMARK(BM_gf256_mul_u256_v2);
BENCHMARK(BM_tbl32_gf256_mul_const);
BENCHMARK(BM_gf256v_mul_scalar_avx2);
BENCHMARK(BM_gf256v_vector_add_scalar_u256)->RangeMultiplier(2)->Range(32, LIST_SIZE);
#endif


#ifdef __AVX512VL__
static void BM_gf256v_vector_add_scalar_u512(benchmark::State& state) {
    for (size_t i = 0; i < state.range(0); ++i) {
        A[i] = rand(); B[i] = rand(); C[i] = rand();
    }

    for (auto _ : state) {
        gf256 s = rand();
        gf256_vector_add_scalar_u256(A, B, s, C, sizeof(gf256) * state.range(0));
        gf256_vector_add_scalar_u256(B, C, s, A, sizeof(gf256) * state.range(0));
        gf256_vector_add_scalar_u256(C, A, s, B, sizeof(gf256) * state.range(0));
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_gf256v_vector_add_scalar_u512)->RangeMultiplier(2)->Range(32, LIST_SIZE);
#endif

BENCHMARK(BM_gf256_mul_u64);
BENCHMARK(BM_gf256_scalar_u64);
BENCHMARK(BM_gf256_scalar_u64_v2);

int main(int argc, char** argv) {
    A = (gf256 *)malloc(sizeof(gf256) * LIST_SIZE);
    B = (gf256 *)malloc(sizeof(gf256) * LIST_SIZE);
    C = (gf256 *)malloc(sizeof(gf256) * LIST_SIZE);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    free(A); free(B); free(C);
    return 0;
}
