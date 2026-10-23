#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "arith.h"   // C reference (plain C part only, no SIMD macro is defined)
#include "arith.cu"  // CUDA implementation (namespace gf256to24_cuda)
#include "../256to8/test_cuda_common.cuh"

// NOTE: a named namespace, as nvcc does not allow types of unnamed namespaces
// as template arguments of __global__ function templates
namespace gf256to24_cuda_test {

struct Traits {
    typedef gf256to24_cuda::gf256to24 T;
    static constexpr size_t n_random = 100000;
    static constexpr size_t n_device_random = 1u << 16u;

    static void ref_add(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint8_t x[24], y[24];  // the C functions do not promise to handle aliasing
        memcpy(x, a, 24); memcpy(y, b, 24);
        ::gf256to24_add(r, x, y);
    }
    static void ref_sub(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint8_t x[24], y[24];
        memcpy(x, a, 24); memcpy(y, b, 24);
        ::gf256to24_sub(r, x, y);
    }
    static void ref_mul(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint8_t x[24], y[24];
        memcpy(x, a, 24); memcpy(y, b, 24);
        ::gf256to24_mul(r, x, y);
    }

    __host__ __device__ static T add(const T &a, const T &b) { return gf256to24_cuda::gf256to24_add(a, b); }
    __host__ __device__ static T sub(const T &a, const T &b) { return gf256to24_cuda::gf256to24_sub(a, b); }
    __host__ __device__ static T mul(const T &a, const T &b) { return gf256to24_cuda::gf256to24_mul(a, b); }

    static void vector_add(T *c, const T *a, const T *b, uint32_t n) { gf256to24_cuda::gf256to24_vector_add(c, a, b, n); }
    static void vector_sub(T *c, const T *a, const T *b, uint32_t n) { gf256to24_cuda::gf256to24_vector_sub(c, a, b, n); }
    static void vector_mul(T *c, const T *a, const T *b, uint32_t n) { gf256to24_cuda::gf256to24_vector_mul(c, a, b, n); }
    static void matrix_add(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols) {
        gf256to24_cuda::gf256to24_matrix_add(m1, m2, m3, nrows, ncols);
    }
    static void matrix_sub(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols) {
        gf256to24_cuda::gf256to24_matrix_sub(m1, m2, m3, nrows, ncols);
    }
    static void matrix_mul(T *r, const T *A, const T *B, uint32_t n_rows1, uint32_t n_cols1, uint32_t n_cols2) {
        gf256to24_cuda::gf256to24_matrix_mul(r, A, B, n_rows1, n_cols1, n_cols2);
    }

    static std::vector<uint32_t> vector_lengths() { return gf256tower_cuda_test::lengths_full(); }
    static std::vector<gf256tower_cuda_test::Shape> mul_shapes() {
        // smaller than the spec shapes, as the C reference multiplication is
        // expensive; 7x700x3: more than one shared memory tile (170 elements)
        return {{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
                {64, 64, 16}, {65, 67, 33}, {7, 700, 3}, {300, 3, 2}};
    }
};

} // namespace gf256to24_cuda_test

GF256TOWER_CUDA_TESTS(GF256to24Cuda, gf256to24_cuda_test::Traits)
