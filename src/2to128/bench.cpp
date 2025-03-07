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

#ifdef USE_AVX2
#include <immintrin.h>

static void BM_gf2to128_mul_avx(benchmark::State& state) {
    uint64_t c = 0;
    eff_t a = 1, b = 2;
    for (auto _ : state) {
        c -= cpucycles();
        a ^= mul_avx(a, b);
        b ^= mul_avx(a, b);
        a ^= mul_avx(a, b);
        c += cpucycles();
        a += c;
        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}
static void BM_gf2to128_mul_avx2(benchmark::State& state) {
    uint64_t c = 0;
    eff_t a = 1, b = 2;
    for (auto _ : state) {
        c -= cpucycles();
        a ^= mul_avx2(a, b);
        b ^= mul_avx2(a, b);
        a ^= mul_avx2(a, b);
        c += cpucycles();
        a += c;
        benchmark::DoNotOptimize(a+=1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf2to128_mul_u256(benchmark::State& state) {
    uint64_t c = 0;
    __m256i a = _mm256_set_epi64x(1,2,3,4);
    __m256i b = _mm256_set_epi64x(4,3,2,1);
    for (auto _ : state) {
        c -= cpucycles();
        a ^= gf2to128v_mul_u256(a, b);
        b ^= gf2to128v_mul_u256(a, b);
        a ^= gf2to128v_mul_u256(a, b);
        c += cpucycles();
        a += b;
        benchmark::DoNotOptimize(b+=a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}
#endif

BENCHMARK(BM_gf2to128_mul_avx);
BENCHMARK(BM_gf2to128_mul_avx2);
BENCHMARK(BM_gf2to128_mul_u256);
BENCHMARK_MAIN();

