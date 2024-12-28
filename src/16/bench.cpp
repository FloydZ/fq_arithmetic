#include <benchmark/benchmark.h>
#include "arith.h"
#include "vector.h"
#include "matrix.h"

static long long cpucycles(void) noexcept {
    unsigned long long result;
    asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
                 : "=a"(result)::"%rdx");
    return result;
}

static void BM_gf16v_mul(benchmark::State& state) {
    uint64_t a = 1, c = 0;
    uint8_t b = 2;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf16v_mul(a, b);
        c += cpucycles();
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


static void BM_gf16_matrix_transpose_16x16(benchmark::State& state) {
    auto *data1 = (uint8_t *)calloc(1, 256);
    auto *data2 = (uint8_t *)calloc(1, 256);
    uint8_t a = 0;
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        gf16_transpose_16x16(data2, data1, 8);
        c += cpucycles();
        a += data2[8];
        gf16_transpose_16x16(data1, data2, 8);
        a += data1[8];

        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(data1); free(data2);
}

static void BM_gf16_matrix_transpose_64x64(benchmark::State& state) {
    auto *data1 = (uint8_t *)calloc(1, 2048);
    auto *data2 = (uint8_t *)calloc(1, 2048);
    uint8_t a = 0;
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        gf16_transpose_64x64(data2, data1, 0);
        c += cpucycles();
        a += data2[8];
        gf16_transpose_64x64(data1, data2, 0);
        a += data1[8];

        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(data1); free(data2);
}


static void BM_gf16_solve(benchmark::State& state) {
    auto *A = (uint8_t *)calloc(1, 2048*2);
    auto *b = (uint8_t *)calloc(1,  128*2);

    gf16_matrix_random(A, 78, 78);
    uint8_t a = 1;
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a+= gf16_solve(A+(8*(a&3)), b, 78, 78);
        c += cpucycles();
        a += A[8];
        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(A); free(b);
}
#ifdef USE_AVX2

static void BM_gf16v_mul_u128(benchmark::State& state) {
    const __m128i acc = {783246, 872346};
    __m128i a = {23874687,905389},
            b = {87326457823,234652734687};
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf16v_mul_u128(a, b);
        c += cpucycles();
        a ^= acc;
        benchmark::DoNotOptimize(a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf16v_mul_u256(benchmark::State& state) {
    const __m256i acc = {783246, 872346, 876502, 236478234023948};
    __m256i a = {23874687,905389,235645274,873465876},
            b = {87326457823,234652734687,8723468726347,78234687236478};
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf16v_mul_u256(a, b);
        c += cpucycles();
        a ^= acc;
        benchmark::DoNotOptimize(a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf16v_mul_full_u256(benchmark::State& state) {
    const __m256i acc = {783246, 872346, 876502, 236478234023948};
    __m256i a = {23874687,905389,235645274,873465876},
            b = {87326457823,234652734687,8723468726347,78234687236478};
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf16v_mul_full_u256(a, b);
        c += cpucycles();
        a ^= acc;
        benchmark::DoNotOptimize(a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf16_matrix_transpose_64x64_avx2(benchmark::State& state) {
    auto *data1 = (uint8_t *)calloc(1, 2048);
    auto *data2 = (uint8_t *)calloc(1, 2048);
    uint8_t a = 0;
    uint64_t c = 0;
    for (auto _ : state) {
        gf16_transpose_64x64_avx2(data2, data1, 32);
        a += data2[8];
        c -= cpucycles();
        gf16_transpose_64x64_avx2(data1, data2, 32);
        c += cpucycles();
        a += data1[8];

        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(data1); free(data2);
}

static void BM_gf16_solve_transpose(benchmark::State& state) {
    auto *A = (uint8_t *)calloc(1, 2048*2);
    auto *AT = (uint8_t *)calloc(1, 2048*2);
    auto *b = (uint8_t *)calloc(1,  128*2);
    gf16_matrix_random(A, 78, 78);
    uint8_t a = 1;
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a+= gf16_solve_transpose(A+(8*(a&3)), AT, b, 78, 78);
        c += cpucycles();
        a += A[8];
        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(A); free(b); free(AT);
}

BENCHMARK(BM_gf16v_mul_u128);
BENCHMARK(BM_gf16v_mul_u256);
BENCHMARK(BM_gf16v_mul_full_u256);
BENCHMARK(BM_gf16_matrix_transpose_64x64_avx2);
BENCHMARK(BM_gf16_solve_transpose);
#endif

BENCHMARK(BM_gf16v_mul);
BENCHMARK(BM_gf16_matrix_transpose_16x16);
BENCHMARK(BM_gf16_matrix_transpose_64x64);
BENCHMARK(BM_gf16_solve);
BENCHMARK_MAIN();

