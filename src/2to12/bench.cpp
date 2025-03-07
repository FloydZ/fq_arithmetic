#include <benchmark/benchmark.h>
#include "arith.h"
#include "vector.h"
#include "matrix.h"

const uint32_t N = 200;
const uint32_t nrows1 = 56;
const uint32_t ncols1 = 6;
const uint32_t ncols2 = 50;

gf2to12 *m1, *m2, *m3;

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
    gf2to12_vector_random(v1, N);
    gf2to12_vector_random(v2, N);

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
    gf2to12_vector_random(v1, N);
    gf2to12_vector_random(v2, N);

    uint64_t a = 1;
    for (auto _ : state) {
        gf2to12_vector_add_gf2(v1, (gf2 *)v2, N);
        gf2to12_vector_add_gf2(v2, (gf2 *)v1, N);

        benchmark::DoNotOptimize(a + v2[7]);
    }

    free(v1); free(v2);
}


static void BM_gf2to12_matrix_mul_gf2(benchmark::State& state) {
    for (auto _ : state) {
        gf2to12_matrix_mul_gf2(m1, (gf2 *)m2, m3, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul_gf2(m2, (gf2 *)m3, m1, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul_gf2(m3, (gf2 *)m1, m2, nrows1, ncols1, ncols2);

        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf2to12_matrix_mul(benchmark::State& state) {
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        gf2to12_matrix_mul(m1, m2, m3, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul(m2, m3, m1, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul(m3, m1, m2, nrows1, ncols1, ncols2);
        c += cpucycles();

        benchmark::DoNotOptimize(m3[7] += 1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
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
    gf2to12_vector_random(v1, N);
    gf2to12_vector_random(v2, N);

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
    gf2to12_vector_random(v1, N);
    gf2to12_vector_random(v2, N);

    uint64_t a = 1;
    for (auto _ : state) {
        gf2to12_vector_add_gf2_u256(v1, (gf2 *)v2, N);
        gf2to12_vector_add_gf2_u256(v2, (gf2 *)v1, N);

        benchmark::DoNotOptimize(a + v2[7]);
    }

    free(v1); free(v2);
}


static void BM_gf2to12_matrix_mul_gf2_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf2to12_matrix_mul_gf2_u256(m1, (gf2 *)m2, m3, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul_gf2_u256(m2, (gf2 *)m3, m1, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul_gf2_u256(m3, (gf2 *)m1, m2, nrows1, ncols1, ncols2);

        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf2to12_matrix_mul_u256(benchmark::State& state) {
    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        gf2to12_matrix_mul_u256(m1, m2, m3, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul_u256(m2, m3, m1, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul_u256(m3, m1, m2, nrows1, ncols1, ncols2);
        c += cpucycles();

        benchmark::DoNotOptimize(m3[7] += 1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

BENCHMARK(BM_gf2to12v_mul_u256);
BENCHMARK(BM_gf2to12v_mul_u256_v2);

//BENCHMARK(BM_gf2to12_vector_add_u256);
//BENCHMARK(BM_gf2to12_vector_add_gf2_u256);

BENCHMARK(BM_gf2to12_matrix_mul_gf2_u256);
BENCHMARK(BM_gf2to12_matrix_mul_u256);

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
// BENCHMARK(BM_gf2to12_vector_add);
// BENCHMARK(BM_gf2to12_vector_add_gf2);

BENCHMARK(BM_gf2to12_matrix_mul_gf2);
BENCHMARK(BM_gf2to12_matrix_mul);

int main(int argc, char** argv) {
    m1 = gf2to12_matrix_alloc(nrows1, ncols2);
    m2 = gf2to12_matrix_alloc(nrows1, ncols2);
    m3 = gf2to12_matrix_alloc(nrows1, ncols2);
    gf2to12_matrix_random(m1, nrows1, ncols2);
    gf2to12_matrix_random(m2, nrows1, ncols2);
    gf2to12_matrix_random(m3, nrows1, ncols2);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    free(m1); free(m2); free(m3);
    return 0;
}
