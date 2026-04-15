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


static void BM_gf16to32_mul(benchmark::State& state) {
    uint64_t c = 0;
    gf16to32 d,a,b;
    gf16to32_set_uu(d, 1, 1); gf16to32_set_uu(a, 2, 2); gf16to32_set_uu(b, 3, 3);
    for (auto _ : state) {
        c -= cpucycles();
        gf16to32_mul(d, a, b);
        gf16to32_mul(a, b, b);
        gf16to32_mul(b, d, a);
        c += cpucycles();
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
}
static void BM_gf16to32_mul_v2(benchmark::State& state) {
    uint64_t c = 0;
    gf16to32 d,a,b;
    gf16to32_set_uu(d, 1, 1); gf16to32_set_uu(a, 2, 2); gf16to32_set_uu(b, 3, 3);
    for (auto _ : state) {
        c -= cpucycles();
        gf16to32_mul_v2(d, a, b);
        gf16to32_mul_v2(a, b, b);
        gf16to32_mul_v2(b, d, a);
        c += cpucycles();
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
}
#ifdef USE_AVX2
#include <immintrin.h>

static void BM_gf16to32_mul_u128(benchmark::State& state) {
    uint64_t c = 0;
    gf16to32 d,a,b;
    gf16to32_set_uu(d, 1, 1); gf16to32_set_uu(a, 2, 2); gf16to32_set_uu(b, 3, 3);
    for (auto _ : state) {
        c -= cpucycles();
        gf16to32_mul_u128(d, a, b);
        gf16to32_mul_u128(a, b, b);
        gf16to32_mul_u128(b, d, a);
        c += cpucycles();
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}
BENCHMARK(BM_gf16to32_mul_u128);
#endif
BENCHMARK(BM_gf16to32_mul);
BENCHMARK(BM_gf16to32_mul_v2);
BENCHMARK_MAIN();
