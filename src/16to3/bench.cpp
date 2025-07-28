#include <benchmark/benchmark.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

const uint32_t nrows = 16, ncols = 16;

static void BM_gf16to3_mul(benchmark::State& state) {
    uint16_t a = 1;
    uint16_t b = 2;
    for (auto _ : state) {
        a = gf16to3_mul(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf16to3v_mul(benchmark::State& state) {
    uint64_t a = 1;
    uint64_t b = 2;
    for (auto _ : state) {
        a = gf16to3v_mul(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf16to3_matrix_mul(benchmark::State& state) {
    const uint32_t nrows = 32, ncols = 6, ncols2 = 16;
    gf16to3 *m1 = gf16to3_matrix_alloc(32, 32);
    gf16to3 *m2 = gf16to3_matrix_alloc(32, 32);
    gf16to3 *m3 = gf16to3_matrix_alloc(32, 32);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_mul(m3, m1, m2, nrows, ncols, ncols2);
        gf16to3_matrix_mul(m1, m2, m3, nrows, ncols, ncols2);
        gf16to3_matrix_mul(m2, m3, m1, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
}

static void BM_gf16to3_matrix_add_multiple_3(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_add_multiple_3(m2, m1, c, m3, nrows, ncols);
        gf16to3_matrix_add_multiple_3(m1, m3, c, m2, nrows, ncols);
        gf16to3_matrix_add_multiple_3(m3, m2, c, m1, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2);
}

static void BM_gf16to3_matrix_add_multiple_2(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_add_multiple_2(m2, c, m1, nrows, ncols);
        gf16to3_matrix_add_multiple_2(m1, c, m2, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2);
}

static void BM_gf16to3_matrix_add_gf16(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_add_gf16(m3, m1, (const ff_t *)m2, nrows, ncols);
        gf16to3_matrix_add_gf16(m1, m2, (const ff_t *)m3, nrows, ncols);
        gf16to3_matrix_add_gf16(m2, m3, (const ff_t *)m1, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
}

static void BM_gf16to3_matrix_map_gf16(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_map_gf16(m1, (const ff_t *)m2, nrows, ncols);
        gf16to3_matrix_map_gf16(m2, (const ff_t *)m1, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2);
}

static void BM_gf16to3_matrix_mul_gf16(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_mul_gf16(m3, m1, (const ff_t *)m2, nrows, ncols, ncols);
        gf16to3_matrix_mul_gf16(m1, m2, (const ff_t *)m3, nrows, ncols, ncols);
        gf16to3_matrix_mul_gf16(m2, m3, (const ff_t *)m1, nrows, ncols, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
}

#ifdef USE_AVX2
static void BM_gf16to3v_mul_gf16_u256(benchmark::State& state) {
    __m256i a = _mm256_set1_epi8(1);
    uint64_t b = 2;
    for (auto _ : state) {
        a = gf16to3v_mul_gf16_u256(a, b);
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf16to3v_mul_u256(benchmark::State& state) {
    __m256i a = _mm256_set1_epi8(1);
    __m256i b = _mm256_set1_epi8(1);
    for (auto _ : state) {
        a = gf16to3v_mul_u256(a, b);
        a += b;
        b += a;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf16to3_matrix_mul_le32xCxC(benchmark::State& state) {
    const uint32_t nrows = 24, ncols = 6, ncols2 = 16;
    gf16to3 *m1 = gf16to3_matrix_alloc(32, 32);
    gf16to3 *m2 = gf16to3_matrix_alloc(32, 32);
    gf16to3 *m3 = gf16to3_matrix_alloc(32, 32);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_mul_le32xCxC(m3, m1, m2, nrows, ncols, ncols2);
        gf16to3_matrix_mul_le32xCxC(m1, m2, m3, nrows, ncols, ncols2);
        gf16to3_matrix_mul_le32xCxC(m2, m3, m1, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
}

static void BM_gf16to3_matrix_mul_16x16(benchmark::State& state) {
    const uint32_t nrows = 16, ncols = 16;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_mul_16x16(m3, m1, m2);
        gf16to3_matrix_mul_16x16(m1, m2, m3);
        gf16to3_matrix_mul_16x16(m2, m3, m1);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
}

static void BM_gf16to3_matrix_add_gf16_XxX(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_add_gf16_XxX(m3, m1, (const ff_t *)m2, nrows, ncols);
        gf16to3_matrix_add_gf16_XxX(m1, m2, (const ff_t *)m3, nrows, ncols);
        gf16to3_matrix_add_gf16_XxX(m2, m3, (const ff_t *)m1, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
}

static void BM_gf16to3_matrix_mul_gf16_XxX(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_mul_gf16_XxX(m3, m1, (const ff_t *)m2, nrows, ncols);
        gf16to3_matrix_mul_gf16_XxX(m1, m2, (const ff_t *)m3, nrows, ncols);
        gf16to3_matrix_mul_gf16_XxX(m2, m3, (const ff_t *)m1, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2); free(m3);
}

static void BM_gf16to3_matrix_add_multiple_3_XxX(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_add_multiple_3_XxX(m2, m1, c, m3, nrows, ncols);
        gf16to3_matrix_add_multiple_3_XxX(m1, m3, c, m2, nrows, ncols);
        gf16to3_matrix_add_multiple_3_XxX(m3, m2, c, m1, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2);
}

static void BM_gf16to3_matrix_add_multiple_2_XxX(benchmark::State& state) {
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);

    gf16to3 c = 0;
    for (auto _ : state) {
        gf16to3_matrix_add_multiple_2_XxX(m2, c, m1, nrows, ncols);
        gf16to3_matrix_add_multiple_2_XxX(m1, c, m2, nrows, ncols);
        benchmark::DoNotOptimize(c+=m2[7]);
    }
    free(m1); free(m2);
}

BENCHMARK(BM_gf16to3v_mul_gf16_u256);
BENCHMARK(BM_gf16to3v_mul_u256);
BENCHMARK(BM_gf16to3_matrix_mul_16x16);
BENCHMARK(BM_gf16to3_matrix_mul_le32xCxC);
BENCHMARK(BM_gf16to3_matrix_add_gf16_XxX);
BENCHMARK(BM_gf16to3_matrix_mul_gf16_XxX);
BENCHMARK(BM_gf16to3_matrix_add_multiple_2_XxX);
BENCHMARK(BM_gf16to3_matrix_add_multiple_3_XxX);

#endif


#ifdef USE_NEON

static void BM_gf16to3_vector_extend_gf16_x8(benchmark::State& state) {
    uint32_t tmp = 0;
    uint16x8_t ctr = {0};
    for (auto _ : state) {
        const uint16x8_t t1 = gf16to3_vector_extend_gf16_x8((uint8_t *)&tmp);
        tmp += 1;
        const uint16x8_t t2 = gf16to3_vector_extend_gf16_x8((uint8_t *)&tmp);
        tmp += 1;
        benchmark::DoNotOptimize(ctr = vaddq_u16(ctr, t1));
        benchmark::DoNotOptimize(ctr = vaddq_u16(ctr, t2));
    }
}

static void BM_gf16to3_vector_extend_gf16_x8_v2(benchmark::State& state) {
    uint32_t tmp = 0;
    uint16x8_t ctr = {0};
    for (auto _ : state) {
        const uint16x8_t t1 = gf16to3_vector_extend_gf16_x8_v2((uint8_t *)&tmp);
        tmp += 1;
        const uint16x8_t t2 = gf16to3_vector_extend_gf16_x8_v2((uint8_t *)&tmp);
        tmp += 1;
        benchmark::DoNotOptimize(ctr = vaddq_u16(ctr, t1));
        benchmark::DoNotOptimize(ctr = vaddq_u16(ctr, t2));
    }
}

BENCHMARK(BM_gf16to3_vector_extend_gf16_x8_v2);
BENCHMARK(BM_gf16to3_vector_extend_gf16_x8);
#endif

// BENCHMARK(BM_gf16to3_matrix_mul);
// BENCHMARK(BM_gf16to3_matrix_add_gf16);
// BENCHMARK(BM_gf16to3_matrix_map_gf16);
// BENCHMARK(BM_gf16to3_mul);
// BENCHMARK(BM_gf16to3_matrix_mul_gf16);
// BENCHMARK(BM_gf16to3_matrix_add_multiple_2);
// BENCHMARK(BM_gf16to3_matrix_add_multiple_3);
// TODO BENCHMARK(BM_gf16to3v_mul);
BENCHMARK_MAIN();

