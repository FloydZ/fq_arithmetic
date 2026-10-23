#pragma once

/// Shared googletest checks for the CUDA ports of the GF(256) tower fields
/// (256to8, 256to16, 256to24, 256toX). All of them have characteristic 2, are
/// stored as plain (non packed) arrays of fixed size elements, and every byte
/// string of the element size is a valid element.
///
/// A field provides a traits struct `Tr` with
///     typedef ... T;                                      device element type
///     static constexpr size_t n_random, n_device_random;  number of random scalar pairs
///     static void ref_add/ref_sub/ref_mul(uint8_t *r, const uint8_t *a, const uint8_t *b);  C reference
///     __host__ __device__ static T add/sub/mul(const T &a, const T &b);                       CUDA scalar ops
///     static void vector_add/vector_sub/vector_mul(T *c, const T *a, const T *b, uint32_t n);
///     static void matrix_add/matrix_sub(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols);
///     static void matrix_mul(T *r, const T *A, const T *B, uint32_t n_rows1, uint32_t n_cols1, uint32_t n_cols2);
///     static std::vector<uint32_t> vector_lengths();
///     static std::vector<gf256tower_cuda_test::Shape> mul_shapes();
/// and instantiates all tests with `GF256TOWER_CUDA_TESTS(SuiteName, Tr)`.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "../helper.cuh"

namespace gf256tower_cuda_test {

using fq_cuda::DeviceBuffer;

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };

inline std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

inline std::vector<uint8_t> random_bytes(const size_t n) {
    std::vector<uint8_t> v(n);
    for (auto &x : v) { x = rng()() & 0xFFu; }
    return v;
}

/// vector lengths of the spec
inline std::vector<uint32_t> lengths_full() {
    return {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};
}

/// vector lengths for fields with an expensive C reference multiplication
inline std::vector<uint32_t> lengths_short() {
    return {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097};
}

/// \return the element 1 (lowest byte 1, rest 0) as bytes
template<typename Tr>
std::vector<uint8_t> one_bytes() {
    std::vector<uint8_t> v(sizeof(typename Tr::T), 0);
    v[0] = 1;
    return v;
}

/// a[i], b[i]: all 9 pairs of {0, 1, all-ones}, followed by `n_random` random pairs
template<typename Tr>
void make_pairs(const size_t n_random,
                std::vector<uint8_t> &a,
                std::vector<uint8_t> &b) {
    constexpr size_t ES = sizeof(typename Tr::T);
    const std::vector<uint8_t> specials[3] = {
        std::vector<uint8_t>(ES, 0), one_bytes<Tr>(), std::vector<uint8_t>(ES, 0xFF),
    };

    a.clear();
    b.clear();
    for (uint32_t s = 0; s < 3; s++) {
        for (uint32_t t = 0; t < 3; t++) {
            a.insert(a.end(), specials[s].begin(), specials[s].end());
            b.insert(b.end(), specials[t].begin(), specials[t].end());
        }
    }

    const auto ra = random_bytes(n_random * ES), rb = random_bytes(n_random * ES);
    a.insert(a.end(), ra.begin(), ra.end());
    b.insert(b.end(), rb.begin(), rb.end());
}

/// compares the host side CUDA scalar operations with the C reference
template<typename Tr>
::testing::AssertionResult scalar_ok(const uint8_t *pa,
                                     const uint8_t *pb) {
    typedef typename Tr::T T;
    constexpr size_t ES = sizeof(T);
    T a, b;
    memcpy(&a, pa, ES);
    memcpy(&b, pb, ES);

    uint8_t ref[ES];
    const T r_add = Tr::add(a, b), r_sub = Tr::sub(a, b), r_mul = Tr::mul(a, b);
    Tr::ref_add(ref, pa, pb);
    if (memcmp(&r_add, ref, ES) != 0) { return ::testing::AssertionFailure() << "add"; }
    Tr::ref_sub(ref, pa, pb);
    if (memcmp(&r_sub, ref, ES) != 0) { return ::testing::AssertionFailure() << "sub"; }
    Tr::ref_mul(ref, pa, pb);
    if (memcmp(&r_mul, ref, ES) != 0) { return ::testing::AssertionFailure() << "mul"; }
    return ::testing::AssertionSuccess();
}

template<typename Tr>
void check_host_scalar() {
    constexpr size_t ES = sizeof(typename Tr::T);
    std::vector<uint8_t> a, b;
    make_pairs<Tr>(Tr::n_random, a, b);
    for (size_t i = 0; i < a.size() / ES; i++) {
        ASSERT_TRUE(scalar_ok<Tr>(&a[i * ES], &b[i * ES])) << "pair " << i;
    }
}

template<typename Tr>
__global__ void scalar_ops_kernel(typename Tr::T *r_add,
                                  typename Tr::T *r_sub,
                                  typename Tr::T *r_mul,
                                  const typename Tr::T *a,
                                  const typename Tr::T *b,
                                  const uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        r_add[i] = Tr::add(a[i], b[i]);
        r_sub[i] = Tr::sub(a[i], b[i]);
        r_mul[i] = Tr::mul(a[i], b[i]);
    }
}

/// device side scalar operations vs the C reference
template<typename Tr>
void check_device_scalar() {
    typedef typename Tr::T T;
    constexpr size_t ES = sizeof(T);
    std::vector<uint8_t> a, b;
    make_pairs<Tr>(Tr::n_device_random, a, b);
    const uint32_t n = (uint32_t)(a.size() / ES);

    DeviceBuffer da(a), db(b), dadd(a.size()), dsub(a.size()), dmul(a.size());
    scalar_ops_kernel<Tr><<<fq_cuda::grid_size(n), fq_cuda::BLOCK_SIZE>>>
        (dadd.as<T>(), dsub.as<T>(), dmul.as<T>(), da.as<T>(), db.as<T>(), n);
    FQ_CUDA_CHECK(cudaGetLastError());

    const auto add = dadd.download(), sub = dsub.download(), mul = dmul.download();
    uint8_t ref[ES];
    for (uint32_t i = 0; i < n; i++) {
        Tr::ref_add(ref, &a[i * ES], &b[i * ES]);
        ASSERT_EQ(0, memcmp(&add[i * ES], ref, ES)) << "add, pair " << i;
        Tr::ref_sub(ref, &a[i * ES], &b[i * ES]);
        ASSERT_EQ(0, memcmp(&sub[i * ES], ref, ES)) << "sub, pair " << i;
        Tr::ref_mul(ref, &a[i * ES], &b[i * ES]);
        ASSERT_EQ(0, memcmp(&mul[i * ES], ref, ES)) << "mul, pair " << i;
    }
}

/// vector add/sub/mul vs the C reference, plus an in-place multiplication
template<typename Tr>
void check_vector_ops() {
    typedef typename Tr::T T;
    constexpr size_t ES = sizeof(T);
    uint8_t ref[ES];

    for (const uint32_t n : Tr::vector_lengths()) {
        const auto a = random_bytes((size_t)n * ES), b = random_bytes((size_t)n * ES);
        DeviceBuffer da(a), db(b), dc(a.size());

        Tr::vector_add(dc.as<T>(), da.as<T>(), db.as<T>(), n);
        auto c = dc.download();
        for (size_t i = 0; i < n; i++) {
            Tr::ref_add(ref, &a[i * ES], &b[i * ES]);
            ASSERT_EQ(0, memcmp(&c[i * ES], ref, ES)) << "add n=" << n << " i=" << i;
        }

        Tr::vector_sub(dc.as<T>(), da.as<T>(), db.as<T>(), n);
        c = dc.download();
        for (size_t i = 0; i < n; i++) {
            Tr::ref_sub(ref, &a[i * ES], &b[i * ES]);
            ASSERT_EQ(0, memcmp(&c[i * ES], ref, ES)) << "sub n=" << n << " i=" << i;
        }

        Tr::vector_mul(dc.as<T>(), da.as<T>(), db.as<T>(), n);
        c = dc.download();
        for (size_t i = 0; i < n; i++) {
            Tr::ref_mul(ref, &a[i * ES], &b[i * ES]);
            ASSERT_EQ(0, memcmp(&c[i * ES], ref, ES)) << "mul n=" << n << " i=" << i;
        }

        // a = a*b in place
        Tr::vector_mul(da.as<T>(), da.as<T>(), db.as<T>(), n);
        ASSERT_EQ(da.download(), c) << "in-place mul n=" << n;
    }
}

template<typename Tr>
void check_matrix_add_sub() {
    typedef typename Tr::T T;
    constexpr size_t ES = sizeof(T);
    uint8_t ref[ES];

    for (const Shape &s : Tr::mul_shapes()) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const size_t n = (size_t)nrows * ncols;
        const auto A = random_bytes(n * ES), B = random_bytes(n * ES);
        DeviceBuffer dA(A), dB(B), dC(A.size());

        Tr::matrix_add(dC.as<T>(), dA.as<T>(), dB.as<T>(), nrows, ncols);
        auto C = dC.download();
        for (size_t i = 0; i < n; i++) {
            Tr::ref_add(ref, &A[i * ES], &B[i * ES]);
            ASSERT_EQ(0, memcmp(&C[i * ES], ref, ES)) << "add " << nrows << "x" << ncols;
        }

        Tr::matrix_sub(dC.as<T>(), dA.as<T>(), dB.as<T>(), nrows, ncols);
        C = dC.download();
        for (size_t i = 0; i < n; i++) {
            Tr::ref_sub(ref, &A[i * ES], &B[i * ES]);
            ASSERT_EQ(0, memcmp(&C[i * ES], ref, ES)) << "sub " << nrows << "x" << ncols;
        }

        // A - A = 0
        Tr::matrix_sub(dC.as<T>(), dA.as<T>(), dA.as<T>(), nrows, ncols);
        for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
    }
}

/// naive column major matrix multiplication with the C reference
template<typename Tr>
std::vector<uint8_t> ref_matrix_mul(const std::vector<uint8_t> &A,
                                    const std::vector<uint8_t> &B,
                                    const uint32_t n_rows1,
                                    const uint32_t n_cols1,
                                    const uint32_t n_cols2) {
    constexpr size_t ES = sizeof(typename Tr::T);
    std::vector<uint8_t> C((size_t)n_rows1 * n_cols2 * ES, 0);
    uint8_t acc[ES], t[ES];
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            memset(acc, 0, ES);
            for (uint32_t k = 0; k < n_cols1; k++) {
                Tr::ref_mul(t, &A[((size_t)k * n_rows1 + i) * ES], &B[((size_t)j * n_cols1 + k) * ES]);
                Tr::ref_add(acc, acc, t);
            }
            memcpy(&C[((size_t)j * n_rows1 + i) * ES], acc, ES);
        }
    }
    return C;
}

template<typename Tr>
void check_matrix_mul() {
    typedef typename Tr::T T;
    constexpr size_t ES = sizeof(T);

    for (const Shape &s : Tr::mul_shapes()) {
        const auto A = random_bytes((size_t)s.n_rows1 * s.n_cols1 * ES);
        const auto B = random_bytes((size_t)s.n_cols1 * s.n_cols2 * ES);
        const auto expected = ref_matrix_mul<Tr>(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(A), dB(B), dC(expected.size());
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, expected.size()));
        Tr::matrix_mul(dC.as<T>(), dA.as<T>(), dB.as<T>(), s.n_rows1, s.n_cols1, s.n_cols2);
        ASSERT_EQ(dC.download(), expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

/// A*I = A, entirely on the device
template<typename Tr>
void check_matrix_mul_identity() {
    typedef typename Tr::T T;
    constexpr size_t ES = sizeof(T);
    const auto one = one_bytes<Tr>();

    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_bytes((size_t)m * n * ES);
        std::vector<uint8_t> I((size_t)n * n * ES, 0);
        for (uint32_t i = 0; i < n; i++) {
            memcpy(&I[((size_t)i * n + i) * ES], one.data(), ES);
        }

        DeviceBuffer dA(A), dI(I), dC(A.size());
        Tr::matrix_mul(dC.as<T>(), dA.as<T>(), dI.as<T>(), m, n, n);
        ASSERT_EQ(dC.download(), A) << "A*I, n=" << n;
    }
}

/// A*(B + C) == A*B + A*C, entirely on the device
template<typename Tr>
void check_matrix_mul_distributive() {
    typedef typename Tr::T T;
    constexpr size_t ES = sizeof(T);
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_bytes((size_t)n_rows1 * n_cols1 * ES);
    const auto B = random_bytes((size_t)n_cols1 * n_cols2 * ES);
    const auto C = random_bytes((size_t)n_cols1 * n_cols2 * ES);
    const size_t out = (size_t)n_rows1 * n_cols2 * ES;

    DeviceBuffer dA(A), dB(B), dC(C), dBC(B.size()), dL(out), dAB(out), dAC(out), dR(out);
    Tr::matrix_add(dBC.as<T>(), dB.as<T>(), dC.as<T>(), n_cols1, n_cols2);
    Tr::matrix_mul(dL.as<T>(), dA.as<T>(), dBC.as<T>(), n_rows1, n_cols1, n_cols2);
    Tr::matrix_mul(dAB.as<T>(), dA.as<T>(), dB.as<T>(), n_rows1, n_cols1, n_cols2);
    Tr::matrix_mul(dAC.as<T>(), dA.as<T>(), dC.as<T>(), n_rows1, n_cols1, n_cols2);
    Tr::matrix_add(dR.as<T>(), dAB.as<T>(), dAC.as<T>(), n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

template<typename Tr>
void check_empty_shapes() {
    typedef typename Tr::T T;
    constexpr size_t ES = sizeof(T);

    DeviceBuffer d(16 * ES);
    Tr::vector_mul(d.as<T>(), d.as<T>(), d.as<T>(), 0);
    Tr::matrix_add(d.as<T>(), d.as<T>(), d.as<T>(), 0, 5);
    Tr::matrix_mul(d.as<T>(), d.as<T>(), d.as<T>(), 0, 3, 4);
    Tr::matrix_mul(d.as<T>(), d.as<T>(), d.as<T>(), 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(5 * 3 * ES);
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    Tr::matrix_mul(dC.as<T>(), d.as<T>(), d.as<T>(), 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}

} // namespace gf256tower_cuda_test

/// instantiates all shared tests for the traits struct `Tr`
#define GF256TOWER_CUDA_TESTS(Suite, Tr)                                                        \
    TEST(Suite, HostScalarMatchesC)    { gf256tower_cuda_test::check_host_scalar<Tr>(); }        \
    TEST(Suite, DeviceScalarOps)       { gf256tower_cuda_test::check_device_scalar<Tr>(); }      \
    TEST(Suite, VectorOps)             { gf256tower_cuda_test::check_vector_ops<Tr>(); }         \
    TEST(Suite, MatrixAddSub)          { gf256tower_cuda_test::check_matrix_add_sub<Tr>(); }     \
    TEST(Suite, MatrixMul)             { gf256tower_cuda_test::check_matrix_mul<Tr>(); }         \
    TEST(Suite, MatrixMulIdentity)     { gf256tower_cuda_test::check_matrix_mul_identity<Tr>(); } \
    TEST(Suite, MatrixMulDistributive) { gf256tower_cuda_test::check_matrix_mul_distributive<Tr>(); } \
    TEST(Suite, EmptyShapes)           { gf256tower_cuda_test::check_empty_shapes<Tr>(); }
