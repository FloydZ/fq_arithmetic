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

static void BM_gf2to256_mul_u256(benchmark::State& state) {
    uint64_t c = 0;
    __m256i a = _mm256_set_epi64x(rand(), rand(), rand(), rand()),
            b = _mm256_set_epi64x(rand(), rand(), rand(), rand());

    for (auto _ : state) {
        c -= cpucycles();
        a ^= gf2to256v_mul_u256(a, b);
        b ^= gf2to256v_mul_u256(a, b);
        a ^= gf2to256v_mul_u256(a, b);
        c += cpucycles();
        benchmark::DoNotOptimize(a = _mm256_add_epi8(a, b));
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf2to256_mul_u256_v2(benchmark::State& state) {
    uint64_t c = 0;
    __m256i a = _mm256_set_epi64x(rand(), rand(), rand(), rand()),
            b = _mm256_set_epi64x(rand(), rand(), rand(), rand());

    for (auto _ : state) {
        c -= cpucycles();
        a ^= gf2to256v_mul_u256_v2(a, b);
        b ^= gf2to256v_mul_u256_v2(a, b);
        a ^= gf2to256v_mul_u256_v2(a, b);
        c += cpucycles();
        benchmark::DoNotOptimize(a = _mm256_add_epi8(a, b));
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}
#endif

BENCHMARK(BM_gf2to256_mul_u256);
BENCHMARK(BM_gf2to256_mul_u256_v2);
BENCHMARK_MAIN();
