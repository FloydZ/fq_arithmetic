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

static void BM_gf2to256_mul(benchmark::State& state) {
    uint64_t c = 0;
    gf2to256 a, b, d;
    for (auto _ : state) {
        c -= cpucycles();
        gf2to256_mul(d, a, b);
        gf2to256_mul(a, b, d);
        gf2to256_mul(b, d, a);
        c += cpucycles();
        a[0] += c;
        benchmark::DoNotOptimize(a[0]+=1);
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf2to256_mul_v2(benchmark::State& state) {
    uint64_t c = 0;
    gf2to256 a, b, d;
    for (auto _ : state) {
        c -= cpucycles();
        gf2to256_mul_v2(d, a, b);
        gf2to256_mul_v2(a, b, d);
        gf2to256_mul_v2(b, d, a);
        c += cpucycles();
        a[0] += c;
        benchmark::DoNotOptimize(a[0]+=1);
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
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

BENCHMARK(BM_gf2to256_mul_u256);
BENCHMARK(BM_gf2to256_mul_u256_v2);
#endif

BENCHMARK(BM_gf2to256_mul);
BENCHMARK(BM_gf2to256_mul_v2);
BENCHMARK_MAIN();
