#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "arith.h"   // C reference (plain C part only); defines EXTENSION_DEGREE
#include "arith.cu"  // CUDA implementation (namespace gf256toX_cuda)
#include "../256to8/test_cuda_common.cuh"

// NOTE: a named namespace, as nvcc does not allow types of unnamed namespaces
// as template arguments of __global__ function templates
namespace gf256toX_cuda_test {

constexpr size_t D = EXTENSION_DEGREE;

struct Traits {
    typedef gf256toX_cuda::gf256toX T;
    // the C reference multiplication is expensive (D^2 table based
    // multiplications), so fewer random pairs than for the other fields
    static constexpr size_t n_random = 10000;
    static constexpr size_t n_device_random = 1u << 12u;

    static void ref_add(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint8_t x[D], y[D];  // the C functions do not promise to handle aliasing
        memcpy(x, a, D); memcpy(y, b, D);
        ::gf256toX_add(r, x, y);
    }
    static void ref_sub(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint8_t x[D], y[D];
        memcpy(x, a, D); memcpy(y, b, D);
        ::gf256toX_sub(r, x, y);
    }
    static void ref_mul(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint8_t x[D], y[D];
        memcpy(x, a, D); memcpy(y, b, D);
        ::gf256toX_mul(r, x, y);
    }

    __host__ __device__ static T add(const T &a, const T &b) { return gf256toX_cuda::gf256toX_add(a, b); }
    __host__ __device__ static T sub(const T &a, const T &b) { return gf256toX_cuda::gf256toX_sub(a, b); }
    __host__ __device__ static T mul(const T &a, const T &b) { return gf256toX_cuda::gf256toX_mul(a, b); }

    static void vector_add(T *c, const T *a, const T *b, uint32_t n) { gf256toX_cuda::gf256toX_vector_add(c, a, b, n); }
    static void vector_sub(T *c, const T *a, const T *b, uint32_t n) { gf256toX_cuda::gf256toX_vector_sub(c, a, b, n); }
    static void vector_mul(T *c, const T *a, const T *b, uint32_t n) { gf256toX_cuda::gf256toX_vector_mul(c, a, b, n); }
    static void matrix_add(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols) {
        gf256toX_cuda::gf256toX_matrix_add(m1, m2, m3, nrows, ncols);
    }
    static void matrix_sub(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols) {
        gf256toX_cuda::gf256toX_matrix_sub(m1, m2, m3, nrows, ncols);
    }
    static void matrix_mul(T *r, const T *A, const T *B, uint32_t n_rows1, uint32_t n_cols1, uint32_t n_cols2) {
        gf256toX_cuda::gf256toX_matrix_mul(r, A, B, n_rows1, n_cols1, n_cols2);
    }

    static std::vector<uint32_t> vector_lengths() { return gf256tower_cuda_test::lengths_short(); }
    static std::vector<gf256tower_cuda_test::Shape> mul_shapes() {
        // much smaller than the spec shapes, as the C reference multiplication
        // is expensive; 7x300x3: more than one shared memory tile (85 elements
        // for D = 48); 300x3x2: more than one thread block of rows
        return {{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
                {64, 20, 9}, {7, 300, 3}, {300, 3, 2}};
    }
};

} // namespace gf256toX_cuda_test

GF256TOWER_CUDA_TESTS(GF256toXCuda, gf256toX_cuda_test::Traits)

/// X^(D-1) * X = X^D = X^5 + X^2 + X + 9 checks the reduction polynomial directly
TEST(GF256toXCuda, ReductionPolynomial) {
    using gf256toX_cuda::gf256toX;
    gf256toX a, x;
    memset(a.v, 0, sizeof(a.v));
    memset(x.v, 0, sizeof(x.v));
    a.v[gf256toX_cuda_test::D - 1] = 1;
    x.v[1] = 1;

    const gf256toX r = gf256toX_cuda::gf256toX_mul(a, x);
    for (size_t i = 0; i < gf256toX_cuda_test::D; i++) {
        const uint8_t expected = (i == 0) ? 9 : ((i == 1 || i == 2 || i == 5) ? 1 : 0);
        ASSERT_EQ(r.v[i], expected) << "coefficient " << i;
    }
}
