#include <benchmark/benchmark.h>
#include "arith.h"

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


static void BM_gf2to32_mul(benchmark::State& state) {
    uint64_t c = 0;
    gf2to32 a = 1, b = 2;
    for (auto _ : state) {
        c -= cpucycles();
        a ^= gf2to32_mul(a, b);
        b ^= gf2to32_mul(a, b);
        a ^= gf2to32_mul(a, b);
        c += cpucycles();
        a += c;
        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}
static void BM_gf2to32_mul_v2(benchmark::State& state) {
    uint64_t c = 0;
    gf2to32 a = 1, b = 2;
    for (auto _ : state) {
        c -= cpucycles();
        a ^= gf2to32_mul_v2(a, b);
        b ^= gf2to32_mul_v2(a, b);
        a ^= gf2to32_mul_v2(a, b);
        c += cpucycles();
        a += c;
        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}


BENCHMARK(BM_gf2to32_mul);
BENCHMARK(BM_gf2to32_mul_v2);
#ifdef USE_AVX2
#include <immintrin.h>


static void BM_gf2to32_mul_u256(benchmark::State& state) {
    uint64_t c = 0;
    __m256i a = _mm256_set_epi64x(1,2,3,4);
    __m256i b = _mm256_set_epi64x(4,3,2,1);
    for (auto _ : state) {
        c -= cpucycles();
        a ^= gf2to32v_mul_u256(a, b);
        b ^= gf2to32v_mul_u256(a, b);
        a ^= gf2to32v_mul_u256(a, b);
        c += cpucycles();
        a += b;
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf2to32_mul_u256_v2(benchmark::State& state) {
    uint64_t c = 0;
    __m256i a = _mm256_set_epi64x(1,2,3,4);
    __m256i b = _mm256_set_epi64x(4,3,2,1);
    for (auto _ : state) {
        c -= cpucycles();
        a ^= gf2to32v_mul_u256_v2(a, b);
        b ^= gf2to32v_mul_u256_v2(a, b);
        a ^= gf2to32v_mul_u256_v2(a, b);
        c += cpucycles();
        a += b;
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

BENCHMARK(BM_gf2to32_mul_u256);
BENCHMARK(BM_gf2to32_mul_u256_v2);
#endif

BENCHMARK_MAIN();
