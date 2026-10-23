#include <nvbench/nvbench.cuh>

#include <cstdint>

#include "arith.cu"
#include "../256to8/bench_cuda_common.cuh"

namespace gf256toX_cuda_bench {

struct Traits {
    typedef gf256toX_cuda::gf256toX T;
    static void vector_add(T *c, const T *a, const T *b, uint32_t n, cudaStream_t s) { gf256toX_cuda::gf256toX_vector_add(c, a, b, n, s); }
    static void vector_sub(T *c, const T *a, const T *b, uint32_t n, cudaStream_t s) { gf256toX_cuda::gf256toX_vector_sub(c, a, b, n, s); }
    static void vector_mul(T *c, const T *a, const T *b, uint32_t n, cudaStream_t s) { gf256toX_cuda::gf256toX_vector_mul(c, a, b, n, s); }
    static void matrix_add(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols, cudaStream_t s) {
        gf256toX_cuda::gf256toX_matrix_add(m1, m2, m3, nrows, ncols, s);
    }
    static void matrix_sub(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols, cudaStream_t s) {
        gf256toX_cuda::gf256toX_matrix_sub(m1, m2, m3, nrows, ncols, s);
    }
    static void matrix_mul(T *r, const T *A, const T *B, uint32_t n_rows1, uint32_t n_cols1, uint32_t n_cols2, cudaStream_t s) {
        gf256toX_cuda::gf256toX_matrix_mul(r, A, B, n_rows1, n_cols1, n_cols2, s);
    }
};

GF256TOWER_CUDA_BENCH_FUNCTIONS(bench_gf256toX, Traits)

} // namespace gf256toX_cuda_bench

using namespace gf256toX_cuda_bench;

// 2^22 elements = 192 MiB per buffer (D = 48)
NVBENCH_BENCH(bench_gf256toX_add).add_int64_power_of_two_axis("Elements", nvbench::range(14, 22, 4));
NVBENCH_BENCH(bench_gf256toX_sub).add_int64_power_of_two_axis("Elements", nvbench::range(14, 22, 4));
NVBENCH_BENCH(bench_gf256toX_mul).add_int64_power_of_two_axis("Elements", nvbench::range(14, 22, 4));

NVBENCH_BENCH(bench_gf256toX_matrix_add).add_int64_axis("N", {64, 256, 1024, 2048});
NVBENCH_BENCH(bench_gf256toX_matrix_sub).add_int64_axis("N", {64, 256, 1024, 2048});
NVBENCH_BENCH(bench_gf256toX_matrix_mul).add_int64_axis("N", {32, 64, 128, 256});
