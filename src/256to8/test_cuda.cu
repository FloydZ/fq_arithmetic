#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "arith.h"   // C reference (plain C part only, no SIMD macro is defined)
#include "arith.cu"  // CUDA implementation (namespace gf256to8_cuda)
#include "test_cuda_common.cuh"

// NOTE: a named namespace, as nvcc does not allow types of unnamed namespaces
// as template arguments of __global__ function templates
namespace gf256to8_cuda_test {

struct Traits {
    typedef gf256to8_cuda::gf256to8 T;
    static constexpr size_t n_random = 100000;
    static constexpr size_t n_device_random = 1u << 16u;

    static void ref_add(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint64_t x, y;
        memcpy(&x, a, 8); memcpy(&y, b, 8);
        const uint64_t z = ::gf256to8_add(x, y);
        memcpy(r, &z, 8);
    }
    static void ref_sub(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint64_t x, y;
        memcpy(&x, a, 8); memcpy(&y, b, 8);
        const uint64_t z = ::gf256to8_sub(x, y);
        memcpy(r, &z, 8);
    }
    static void ref_mul(uint8_t *r, const uint8_t *a, const uint8_t *b) {
        uint64_t x, y;
        memcpy(&x, a, 8); memcpy(&y, b, 8);
        const uint64_t z = ::gf256to8_mul(x, y);
        memcpy(r, &z, 8);
    }

    __host__ __device__ static T add(const T &a, const T &b) { return gf256to8_cuda::gf256to8_add(a, b); }
    __host__ __device__ static T sub(const T &a, const T &b) { return gf256to8_cuda::gf256to8_sub(a, b); }
    __host__ __device__ static T mul(const T &a, const T &b) { return gf256to8_cuda::gf256to8_mul(a, b); }

    static void vector_add(T *c, const T *a, const T *b, uint32_t n) { gf256to8_cuda::gf256to8_vector_add(c, a, b, n); }
    static void vector_sub(T *c, const T *a, const T *b, uint32_t n) { gf256to8_cuda::gf256to8_vector_sub(c, a, b, n); }
    static void vector_mul(T *c, const T *a, const T *b, uint32_t n) { gf256to8_cuda::gf256to8_vector_mul(c, a, b, n); }
    static void matrix_add(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols) {
        gf256to8_cuda::gf256to8_matrix_add(m1, m2, m3, nrows, ncols);
    }
    static void matrix_sub(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols) {
        gf256to8_cuda::gf256to8_matrix_sub(m1, m2, m3, nrows, ncols);
    }
    static void matrix_mul(T *r, const T *A, const T *B, uint32_t n_rows1, uint32_t n_cols1, uint32_t n_cols2) {
        gf256to8_cuda::gf256to8_matrix_mul(r, A, B, n_rows1, n_cols1, n_cols2);
    }

    static std::vector<uint32_t> vector_lengths() { return gf256tower_cuda_test::lengths_full(); }
    static std::vector<gf256tower_cuda_test::Shape> mul_shapes() {
        // 7x4100x3: more than one shared memory tile (512 elements)
        return {{1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
                {64, 64, 64}, {129, 129, 65}, {7, 4100, 3}, {300, 3, 2}};
    }
};

} // namespace gf256to8_cuda_test

GF256TOWER_CUDA_TESTS(GF256to8Cuda, gf256to8_cuda_test::Traits)

/// the lower levels of the tower, used by 256to16/256to24/256toX as well
TEST(GF256to8Cuda, TowerLevelsMatchC) {
    for (uint32_t a = 0; a < 256; a++) {
        for (uint32_t b = 0; b < 256; b++) {
            ASSERT_EQ(gf256to8_cuda::gf256_mul(a, b), ::gf256_mul(a, b)) << a << " " << b;
        }
    }

    auto &gen = gf256tower_cuda_test::rng();
    for (uint32_t it = 0; it < 100000; it++) {
        const uint16_t a2 = gen(), b2 = gen();
        ASSERT_EQ(gf256to8_cuda::gf256to2_mul(a2, b2), ::gf256to2_mul(a2, b2)) << a2 << " " << b2;
        ASSERT_EQ(gf256to8_cuda::gf256to2_mul_gamma(a2), ::gf256to2_mul(a2, 0x2000)) << a2;

        const uint32_t a4 = gen(), b4 = gen();
        ASSERT_EQ(gf256to8_cuda::gf256to4_mul(a4, b4), ::gf256to4_mul(a4, b4)) << a4 << " " << b4;
        ASSERT_EQ(gf256to8_cuda::gf256to4_mul_gamma(a4), ::gf256to4_mul(a4, 0x20000000u)) << a4;

        const uint64_t a8 = ((uint64_t)gen() << 32u) | gen();
        ASSERT_EQ(gf256to8_cuda::gf256to8_mul_gamma(a8), ::gf256to8_mul(a8, 0x2000000000000000ull)) << a8;
    }
}
