#include <benchmark/benchmark.h>
#include "arith.h"

const uint32_t N = 200;

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

static void BM_gf2to12_vector_add(benchmark::State& state) {
    gf2to12 *v1 = gf2to12_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12_vector_rand(v1, N);
    gf2to12_vector_rand(v2, N);

    uint64_t a = 1;
    for (auto _ : state) {
        gf2to12_vector_add(v1, v2, N);
        gf2to12_vector_add(v2, v1, N);

        benchmark::DoNotOptimize(a + v2[7]);
    }

    free(v1); free(v2);
}

static void BM_gf2to12_vector_add_gf2(benchmark::State& state) {
    gf2to12 *v1 = gf2to12_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12_vector_rand(v1, N);
    gf2to12_vector_rand(v2, N);

    uint64_t a = 1;
    for (auto _ : state) {
        gf2to12_vector_add_gf2(v1, (gf2 *)v2, N);
        gf2to12_vector_add_gf2(v2, (gf2 *)v1, N);

        benchmark::DoNotOptimize(a + v2[7]);
    }

    free(v1); free(v2);
}


#ifdef USE_AVX2
#include <immintrin.h>

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

static void BM_gf2to12v_mul_u256_v2(benchmark::State& state) {
    const __m256i acc = {78328972346, 878876872346, 87659787602, 236478234023948};
    __m256i b = {23874687,9053897621,235645274,873465876},
            a = {80007326457823,234652734687000,82873723468726347,78234687236478};
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf2to12v_mul_u256_v2(a, b);
        c += cpucycles();
        a ^= acc;
        b ^= a;
        benchmark::DoNotOptimize(a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf2to12_vector_add_u256(benchmark::State& state) {
    gf2to12 *v1 = gf2to12_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12_vector_rand(v1, N);
    gf2to12_vector_rand(v2, N);

    uint64_t a = 1;
    for (auto _ : state) {
        gf2to12_vector_add_u256(v1, v2, N);
        gf2to12_vector_add_u256(v2, v1, N);

        benchmark::DoNotOptimize(a + v2[7]);
    }

    free(v1); free(v2);
}

static void BM_gf2to12_vector_add_gf2_u256(benchmark::State& state) {
    gf2to12 *v1 = gf2to12_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12_vector_rand(v1, N);
    gf2to12_vector_rand(v2, N);

    uint64_t a = 1;
    for (auto _ : state) {
        gf2to12_vector_add_gf2_u256(v1, (gf2 *)v2, N);
        gf2to12_vector_add_gf2_u256(v2, (gf2 *)v1, N);

        benchmark::DoNotOptimize(a + v2[7]);
    }

    free(v1); free(v2);
}
BENCHMARK(BM_gf2to12v_mul_u256);
BENCHMARK(BM_gf2to12v_mul_u256_v2);
BENCHMARK(BM_gf2to12_vector_add_u256);
BENCHMARK(BM_gf2to12_vector_add_gf2_u256);
#endif

#ifdef USE_AVX512
static void BM_gf2to12v_mul_u256_avx512(benchmark::State& state) {
    const __m256i acc = {78328972346, 878876872346, 87659787602, 236478234023948};
    __m256i b = {23874687,9053897621,235645274,873465876},
            a = {80007326457823,234652734687000,82873723468726347,78234687236478};
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf2to12v_mul_u256_avx512(a, b);
        c += cpucycles();
        a ^= acc;
        b ^= a;
        benchmark::DoNotOptimize(a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf2to12v_mul_u512(benchmark::State& state) {
    const __m512i acc = {78328972346, 878876872346, 87659787602, 236478234023948, 78328972346, 878876872346, 87659787602, 236478234023948};
    __m512i b = {23874687,9053897621,235645274,873465876, 78328972346, 878876872346, 87659787602, 236478234023948},
            a = {80007326457823,234652734687000,82873723468726347,78234687236478, 78328972346, 878876872346, 87659787602, 236478234023948};
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a = gf2to12v_mul_u512(a, b);
        c += cpucycles();
        a ^= acc;
        b ^= a;
        benchmark::DoNotOptimize(a);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

BENCHMARK(BM_gf2to12v_mul_u256_avx512);
BENCHMARK(BM_gf2to12v_mul_u512);
#endif

BENCHMARK(BM_gf2to12_mul);
BENCHMARK(BM_gf2to12_mul_v2);
BENCHMARK(BM_gf2to12_vector_add);
BENCHMARK(BM_gf2to12_vector_add_gf2);
BENCHMARK_MAIN();
