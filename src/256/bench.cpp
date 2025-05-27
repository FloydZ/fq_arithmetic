#include <benchmark/benchmark.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

#define LIST_SIZE (1u << 10u)
gf256 *A, *B, *C;
gf256 *m1, *m2, *m3;
const uint32_t nrows = 16;
const uint32_t ncols = 16;
const uint32_t ncols2= 16;


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

static void BM_gf256_mul_u64(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf256v_mul_u64(a, b);
        a += 1;
        b += 1;
        benchmark::DoNotOptimize(a);
    }
}

static void BM_gf256_scalar_u64(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf256v_scalar_u64(a, b);
        a += 1;
        b += 1;
		benchmark::DoNotOptimize(a); 
    }
}

static void BM_gf256_scalar_u64_v2(benchmark::State& state) {
    uint64_t a = 1;
    uint8_t b = 2;
    for (auto _ : state) {
        a = gf256v_scalar_u64_v2(a, b);
        a += 1;
        b += 1;
		benchmark::DoNotOptimize(a);
    }
}


static void BM_gf256v_vector_set_to_gf2(benchmark::State& state) {
    for (auto _ : state) {
        gf256_vector_set_to_gf2(A, B, sizeof(gf256) * state.range(0));
        gf256_vector_set_to_gf2(B, A, sizeof(gf256) * state.range(0));
        benchmark::ClobberMemory();
    }
}


static void BM_gf256_matrix_add(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_add(m1, m2, m3, nrows, ncols);
        gf256_matrix_add(m2, m3, m1, nrows, ncols);
        gf256_matrix_add(m3, m1, m2, nrows, ncols);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_add_gf16(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_add_gf16(m1, m2, m3, nrows, ncols);
        gf256_matrix_add_gf16(m2, m3, m1, nrows, ncols);
        gf256_matrix_add_gf16(m3, m1, m2, nrows, ncols);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_add_multiple_gf16(benchmark::State& state) {
    for (auto _ : state) {
        const gf256 b = rand();
        gf256_matrix_add_multiple_gf16(m1, b, m2, nrows, ncols);
        gf256_matrix_add_multiple_gf16(m2, b, m1, nrows, ncols);
        benchmark::DoNotOptimize(m2[7] += 1);
    }
}

static void BM_gf256_matrix_add_multiple_2(benchmark::State& state) {
    for (auto _ : state) {
        const gf256 b = rand();
        gf256_matrix_add_multiple_2(m1, b, m2, nrows, ncols);
        gf256_matrix_add_multiple_2(m2, b, m1, nrows, ncols);
        benchmark::DoNotOptimize(m2[7] += 1);
    }
}

static void BM_gf256_matrix_product(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product(m1, m2, m3, nrows, ncols, ncols2);
        gf256_matrix_product(m2, m3, m1, nrows, ncols, ncols2);
        gf256_matrix_product(m3, m1, m2, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_product_8x8xC(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product(m1, m2, m3, 8, 8, ncols2);
        gf256_matrix_product(m2, m3, m1, 8, 8, ncols2);
        gf256_matrix_product(m3, m1, m2, 8, 8, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_product_16x16xC(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product(m1, m2, m3, 16, 16, ncols2);
        gf256_matrix_product(m2, m3, m1, 16, 16, ncols2);
        gf256_matrix_product(m3, m1, m2, 16, 16, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_product_gf16_1(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product_gf16_1(m1, m2, m3, nrows, ncols, ncols2);
        gf256_matrix_product_gf16_1(m2, m3, m1, nrows, ncols, ncols2);
        gf256_matrix_product_gf16_1(m3, m1, m2, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_product_gf16_2(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product_gf16_2(m1, m2, m3, nrows, ncols, ncols2);
        gf256_matrix_product_gf16_2(m2, m3, m1, nrows, ncols, ncols2);
        gf256_matrix_product_gf16_2(m3, m1, m2, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

#ifdef USE_AVX2

static void BM_gf256_sqr_u256(benchmark::State& state) {
    v256 a = {0, 1, 2, 3, 4, 5, 6, 7}, one = {1,13,2,5,3,12,18,9};
    // uint64_t c = 0;
    for (auto _ : state) {
        // c -= cpucycles();
        a.v256 = gf256v_sqr_u256(a.v256);
        //c += cpucycles();
        a.v256 = _mm256_add_epi32(a.v256, one.v256);
        benchmark::DoNotOptimize(a.v256);
    }
    // state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf256_mul_u256(benchmark::State& state) {
    v256 a = {0, 1, 2, 3, 4, 5, 6, 7}, b = {2, 12, 4, 18, 6, 17, 1, 9}, one = {1,13,2,5,3,12,18,9};

    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a.v256 = gf256v_mul_u256(a.v256, b.v256);
        c += cpucycles();
        a.v256 = _mm256_add_epi32(a.v256, one.v256);
        benchmark::DoNotOptimize(a.v256);
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf256_mul_u256_v2(benchmark::State& state) {
    v256 a = {0, 1, 2, 3, 4, 5, 6, 7}, b = {2, 12, 4, 18, 6, 17, 1, 9}, one = {1,13,2,5,3,12,18,9};

    uint64_t c = 0;
    for (auto _ : state) {
        c -= cpucycles();
        a.v256 = gf256v_mul_u256_v2(a.v256, b.v256);
        c += cpucycles();
        a.v256 = _mm256_add_epi32(a.v256, one.v256);
		benchmark::DoNotOptimize(a.v256); 
    }
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf256v_mul_scalar_avx2(benchmark::State& state) {
    v256 a = {0, 1, 2, 3, 4, 5, 6, 7}, one = {1,13,2,5,3,12,18,9};
    uint32_t b = 2;
    for (auto _ : state) {
        a.v256 = gf256v_mul_scalar_u256(a.v256, b);
        a.v256 = _mm256_add_epi32(a.v256, one.v256);
        b += 1;
		benchmark::DoNotOptimize(a.v256); 
    }
}


static void BM_gf256v_vector_set_to_gf2_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_vector_set_to_gf2_u256(A, B, sizeof(gf256) * state.range(0));
        gf256_vector_set_to_gf2_u256(B, A, sizeof(gf256) * state.range(0));
        benchmark::ClobberMemory();
    }
}

static void BM_gf256v_vector_set_to_gf16_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_vector_set_to_gf16_u256(A, B, sizeof(gf256) * state.range(0));
        gf256_vector_set_to_gf16_u256(B, A, sizeof(gf256) * state.range(0));
        benchmark::ClobberMemory();
    }
}

static void BM_gf256v_vector_set_to_gf16_u256_v2(benchmark::State& state) {
    for (auto _ : state) {
        gf256_vector_set_to_gf16_u256_v2(A, B, sizeof(gf256) * state.range(0));
        gf256_vector_set_to_gf16_u256_v2(B, A, sizeof(gf256) * state.range(0));
        benchmark::ClobberMemory();
    }
}

static void BM_gf256v_vector_add_scalar_u256(benchmark::State& state) {
    for (size_t i = 0; i < state.range(0); ++i) {
        A[i] = rand(); B[i] = rand(); C[i] = rand();
    }

    for (auto _ : state) {
        gf256 s = rand();
        gf256_vector_add_scalar_3_u256(A, B, s, C, sizeof(gf256) * state.range(0));
        gf256_vector_add_scalar_3_u256(B, C, s, A, sizeof(gf256) * state.range(0));
        gf256_vector_add_scalar_3_u256(C, A, s, B, sizeof(gf256) * state.range(0));
        benchmark::ClobberMemory();
    }
}


static void BM_gf256_matrix_add_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_add_u256(m1, m2, m3, nrows, ncols);
        gf256_matrix_add_u256(m2, m3, m1, nrows, ncols);
        gf256_matrix_add_u256(m3, m1, m2, nrows, ncols);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_add_gf16_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_add_gf16_u256(m1, m2, m3, nrows, ncols);
        gf256_matrix_add_gf16_u256(m2, m3, m1, nrows, ncols);
        gf256_matrix_add_gf16_u256(m3, m1, m2, nrows, ncols);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_add_multiple_gf16_u256(benchmark::State& state) {
    uint64_t c = 0;
    for (auto _ : state) {
        const gf256 b = rand();
        c -= cpucycles();
        gf256_matrix_add_multiple_gf16_u256(m1, b, m2, nrows, ncols);
        gf256_matrix_add_multiple_gf16_u256(m2, b, m1, nrows, ncols);
        c += cpucycles();
        benchmark::DoNotOptimize(m2[7] += 1);
    }

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_gf256_matrix_add_multiple_2_u256(benchmark::State& state) {
    for (auto _ : state) {
        const gf256 b = rand();
        gf256_matrix_add_multiple_2_u256(m1, b, m2, nrows, ncols);
        gf256_matrix_add_multiple_2_u256(m2, b, m1, nrows, ncols);
        benchmark::DoNotOptimize(m2[7] += 1);
    }
}

static void BM_gf256_matrix_product_gf16_1_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product_gf16_1_u256(m1, m2, m3, nrows, ncols, ncols2);
        gf256_matrix_product_gf16_1_u256(m2, m3, m1, nrows, ncols, ncols2);
        gf256_matrix_product_gf16_1_u256(m3, m1, m2, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_product_gf16_2_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product_gf16_2_u256(m1, m2, m3, nrows, ncols, ncols2);
        gf256_matrix_product_gf16_2_u256(m2, m3, m1, nrows, ncols, ncols2);
        gf256_matrix_product_gf16_2_u256(m3, m1, m2, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_product_8x8xC_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product_8x8xC_u256(m1, m2, m3, ncols2);
        gf256_matrix_product_8x8xC_u256(m2, m3, m1, ncols2);
        gf256_matrix_product_8x8xC_u256(m3, m1, m2, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_product_16x16xC_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product_16x16xC_u256(m1, m2, m3, ncols2);
        gf256_matrix_product_16x16xC_u256(m2, m3, m1, ncols2);
        gf256_matrix_product_16x16xC_u256(m3, m1, m2, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

static void BM_gf256_matrix_product_le32xBxle16_u256(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_product_le32xBxle16_u256(m1, m2, m3, nrows, ncols, ncols2);
        gf256_matrix_product_le32xBxle16_u256(m2, m3, m1, nrows, ncols, ncols2);
        gf256_matrix_product_le32xBxle16_u256(m3, m1, m2, nrows, ncols, ncols2);
        benchmark::DoNotOptimize(m3[7] += 1);
    }
}

// BENCHMARK(BM_gf256_sqr_u256);
// BENCHMARK(BM_gf256_mul_u256);
// BENCHMARK(BM_gf256_mul_u256_v2);
// BENCHMARK(BM_gf256v_mul_scalar_avx2);

// BENCHMARK(BM_gf256v_vector_add_scalar_u256)->RangeMultiplier(2)->Range(32, LIST_SIZE);
// BENCHMARK(BM_gf256v_vector_set_to_gf2_u256)->RangeMultiplier(2)->Range(15, LIST_SIZE);
// BENCHMARK(BM_gf256v_vector_set_to_gf16_u256   )->RangeMultiplier(2)->Range(15, LIST_SIZE);
// BENCHMARK(BM_gf256v_vector_set_to_gf16_u256_v2)->RangeMultiplier(2)->Range(15, LIST_SIZE);
// BENCHMARK(BM_gf256_matrix_add_u256);
// BENCHMARK(BM_gf256_matrix_add_gf16_u256);
BENCHMARK(BM_gf256_matrix_add_multiple_gf16_u256);
// BENCHMARK(BM_gf256_matrix_add_multiple_2_u256);
// BENCHMARK(BM_gf256_matrix_product_gf16_1_u256);
// BENCHMARK(BM_gf256_matrix_product_gf16_2_u256);
// BENCHMARK(BM_gf256_matrix_product_8x8xC_u256);
// BENCHMARK(BM_gf256_matrix_product_16x16xC_u256);
// BENCHMARK(BM_gf256_matrix_product_le32xBxle16_u256);
#endif

#ifdef USE_NEON

static void BM_gf256_matrix_transpose_32x32(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_transpose_32x32(m1, m2, nrows, ncols);
        gf256_matrix_transpose_32x32(m2, m1, nrows, ncols);
        benchmark::DoNotOptimize(m2[7] += 1);
    }
}

static void BM_gf256_matrix_transpose_32x32_v2(benchmark::State& state) {
    for (auto _ : state) {
        gf256_matrix_transpose_32x32_v2(m1, m2, nrows, ncols);
        gf256_matrix_transpose_32x32_v2(m2, m1, nrows, ncols);
        benchmark::DoNotOptimize(m2[7] += 1);
    }
}
BENCHMARK(BM_gf256_matrix_transpose_32x32);
BENCHMARK(BM_gf256_matrix_transpose_32x32_v2);
#endif

#ifdef __AVX512VL__
static void BM_gf256v_vector_add_scalar_u512(benchmark::State& state) {
    for (size_t i = 0; i < state.range(0); ++i) {
        A[i] = rand(); B[i] = rand(); C[i] = rand();
    }

    for (auto _ : state) {
        gf256 s = rand();
        gf256_vector_add_scalar_u256(A, B, s, C, sizeof(gf256) * state.range(0));
        gf256_vector_add_scalar_u256(B, C, s, A, sizeof(gf256) * state.range(0));
        gf256_vector_add_scalar_u256(C, A, s, B, sizeof(gf256) * state.range(0));
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_gf256v_vector_add_scalar_u512)->RangeMultiplier(2)->Range(32, LIST_SIZE);
#endif

//BENCHMARK(BM_gf256_mul_u64);
//BENCHMARK(BM_gf256_scalar_u64);
//BENCHMARK(BM_gf256_scalar_u64_v2);
// BENCHMARK(BM_gf256v_vector_set_to_gf2)->RangeMultiplier(2)->Range(15, LIST_SIZE);

// BENCHMARK(BM_gf256_matrix_add);
// BENCHMARK(BM_gf256_matrix_add_gf16);
BENCHMARK(BM_gf256_matrix_add_multiple_gf16);
// BENCHMARK(BM_gf256_matrix_add_multiple_2);
// BENCHMARK(BM_gf256_matrix_product_gf16_1);
// BENCHMARK(BM_gf256_matrix_product_gf16_2);
// BENCHMARK(BM_gf256_matrix_product);
// BENCHMARK(BM_gf256_matrix_product_8x8xC);
// BENCHMARK(BM_gf256_matrix_product_16x16xC);

int main(int argc, char** argv) {
    A = (gf256 *)malloc(sizeof(gf256) * LIST_SIZE);
    B = (gf256 *)malloc(sizeof(gf256) * LIST_SIZE);
    C = (gf256 *)malloc(sizeof(gf256) * LIST_SIZE);

    m1 = gf256_matrix_alloc(32, 32);
    m2 = gf256_matrix_alloc(32, 32);
    m3 = gf256_matrix_alloc(32, 32);
    gf256_matrix_random(m1, nrows, ncols);
    gf256_matrix_random(m2, nrows, ncols);
    gf256_matrix_random(m3, nrows, ncols);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    free(A); free(B); free(C);
    free(m1); free(m2); free(m3);
    return 0;
}
