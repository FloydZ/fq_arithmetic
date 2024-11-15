#include <benchmark/benchmark.h>
#include "arith.h"



static long long cpucycles(void) noexcept {
    unsigned long long result;
    asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
            : "=a"(result)::"%rdx");
    return result;
}

static void BM_gf2to12_mul(benchmark::State& state) {
    uint64_t a = 1, c = 0;
    uint8_t b = 2;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf2to12_mul(a, b);
        c += cpucycles();
        a += 1 - b;
        b += 1 + a;
        benchmark::DoNotOptimize(a);
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf2to12_mul_v2(benchmark::State& state) {
    uint64_t a = 1, c = 0;
    uint8_t b = 2;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf2to12_mul_v2(a, b);
        c += cpucycles();
        a += 1 - b;
        b += 1 + a;
        benchmark::DoNotOptimize(a);
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

#ifdef USE_AVX2
static void BM_gf2to12v_mul_u256(benchmark::State& state) {
    const __m256i acc = {783246, 872346, 876502, 236478234023948};
    __m256i a = {23874687,905389,235645274,873465876},
            b = {87326457823,234652734687,8723468726347,78234687236478};
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf2to12v_mul_u256(a, b);
        c += cpucycles();
        a ^= acc;
        b ^= a;
        benchmark::DoNotOptimize(a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

BENCHMARK(BM_gf2to12v_mul_u256);
#endif

BENCHMARK(BM_gf2to12_mul);
BENCHMARK(BM_gf2to12_mul_v2);
BENCHMARK_MAIN();
