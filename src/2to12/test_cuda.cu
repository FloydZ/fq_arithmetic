#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "arith.h"   // C reference: gf2to12_mul
#include "arith.cu"  // CUDA implementation (namespace gf2to12_cuda)

static_assert(gf2to12_cuda::GF2TO12_CUDA_MODULUS == MODULUS, "CUDA and C use different polynomials");

namespace {

using fq_cuda::DeviceBuffer;
typedef uint16_t elem_t;

/// number of field elements
constexpr uint32_t Q = 1u << 12u;

std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

elem_t random_element() {
    return (elem_t)(rng()() & (Q - 1u));
}

std::vector<elem_t> random_vector(const size_t n) {
    std::vector<elem_t> v(n);
    for (auto &x : v) { x = random_element(); }
    return v;
}

std::vector<uint8_t> to_bytes(const std::vector<elem_t> &v) {
    std::vector<uint8_t> b(v.size() * sizeof(elem_t));
    if (!b.empty()) { memcpy(b.data(), v.data(), b.size()); }
    return b;
}

std::vector<elem_t> from_bytes(const std::vector<uint8_t> &b) {
    std::vector<elem_t> v(b.size() / sizeof(elem_t));
    if (!v.empty()) { memcpy(v.data(), b.data(), v.size() * sizeof(elem_t)); }
    return v;
}

/// reference: column major, entry (i,j) at j*nrows + i, same as matrix.h
std::vector<elem_t> ref_matrix_mul(const std::vector<elem_t> &A,
                                   const std::vector<elem_t> &B,
                                   const uint32_t n_rows1,
                                   const uint32_t n_cols1,
                                   const uint32_t n_cols2) {
    std::vector<elem_t> C((size_t)n_rows1 * n_cols2, 0);
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            elem_t acc = 0;
            for (uint32_t k = 0; k < n_cols1; k++) {
                acc ^= ::gf2to12_mul(A[(size_t)k * n_rows1 + i], B[(size_t)j * n_cols1 + k]);
            }
            C[(size_t)j * n_rows1 + i] = acc;
        }
    }
    return C;
}

/// all Q*Q pairs: thread t computes a = t / Q, b = t % Q
__global__ void scalar_ops_kernel(elem_t *out_add, elem_t *out_sub, elem_t *out_mul) {
    const size_t n = (size_t)Q * Q;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
        const elem_t a = (elem_t)(t / Q), b = (elem_t)(t % Q);
        out_add[t] = gf2to12_cuda::gf2to12_add(a, b);
        out_sub[t] = gf2to12_cuda::gf2to12_sub(a, b);
        out_mul[t] = gf2to12_cuda::gf2to12_mul(a, b);
    }
}

const uint32_t vector_lengths[] = {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 64, 64}, {129, 257, 65}, {7, 4100, 3}, {1000, 3, 300},
};

} // namespace

TEST(GF2to12Cuda, HostScalarMatchesC) {
    for (uint32_t a = 0; a < Q; a++) {
        for (uint32_t b = 0; b < Q; b++) {
            ASSERT_EQ(gf2to12_cuda::gf2to12_add(a, b), ::gf2to12_add(a, b));
            ASSERT_EQ(gf2to12_cuda::gf2to12_sub(a, b), ::gf2to12_sub(a, b));
            ASSERT_EQ(gf2to12_cuda::gf2to12_mul(a, b), ::gf2to12_mul(a, b)) << "a=" << a << " b=" << b;
        }
    }
}

TEST(GF2to12Cuda, DeviceScalarOps) {
    const size_t n = (size_t)Q * Q;
    DeviceBuffer d_add(n * sizeof(elem_t)), d_sub(n * sizeof(elem_t)), d_mul(n * sizeof(elem_t));
    scalar_ops_kernel<<<fq_cuda::grid_size(n), fq_cuda::BLOCK_SIZE>>>(d_add.as<elem_t>(), d_sub.as<elem_t>(), d_mul.as<elem_t>());
    FQ_CUDA_CHECK(cudaGetLastError());
    const auto add = from_bytes(d_add.download()), sub = from_bytes(d_sub.download()), mul = from_bytes(d_mul.download());
    for (size_t t = 0; t < n; t++) {
        const elem_t a = (elem_t)(t / Q), b = (elem_t)(t % Q);
        ASSERT_EQ(add[t], ::gf2to12_add(a, b));
        ASSERT_EQ(sub[t], ::gf2to12_sub(a, b));
        ASSERT_EQ(mul[t], ::gf2to12_mul(a, b)) << "a=" << a << " b=" << b;
    }
}

TEST(GF2to12Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        const auto a = random_vector(n), b = random_vector(n);
        DeviceBuffer da(to_bytes(a)), db(to_bytes(b)), dc(n * sizeof(elem_t));

        gf2to12_cuda::gf2to12_vector_add(dc.as<elem_t>(), da.as<elem_t>(), db.as<elem_t>(), n);
        auto c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf2to12_add(a[i], b[i])) << "add n=" << n; }

        gf2to12_cuda::gf2to12_vector_sub(dc.as<elem_t>(), da.as<elem_t>(), db.as<elem_t>(), n);
        c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf2to12_sub(a[i], b[i])) << "sub n=" << n; }

        gf2to12_cuda::gf2to12_vector_mul(dc.as<elem_t>(), da.as<elem_t>(), db.as<elem_t>(), n);
        c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf2to12_mul(a[i], b[i])) << "mul n=" << n << " i=" << i; }
    }
}

TEST(GF2to12Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_vector(n), b = random_vector(n);
    DeviceBuffer da(to_bytes(a)), db(to_bytes(b));
    gf2to12_cuda::gf2to12_vector_mul(da.as<elem_t>(), da.as<elem_t>(), db.as<elem_t>(), n);
    const auto c = from_bytes(da.download());
    for (uint32_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf2to12_mul(a[i], b[i])); }
}

TEST(GF2to12Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_vector((size_t)nrows * ncols), B = random_vector((size_t)nrows * ncols);
        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(gf2to12_cuda::gf2to12_matrix_bytes_size(nrows, ncols));

        gf2to12_cuda::gf2to12_matrix_add(dC.as<elem_t>(), dA.as<elem_t>(), dB.as<elem_t>(), nrows, ncols);
        auto C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], A[i] ^ B[i]) << nrows << "x" << ncols; }

        gf2to12_cuda::gf2to12_matrix_sub(dC.as<elem_t>(), dA.as<elem_t>(), dB.as<elem_t>(), nrows, ncols);
        C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], A[i] ^ B[i]) << nrows << "x" << ncols; }

        // A - A = 0
        gf2to12_cuda::gf2to12_matrix_sub(dC.as<elem_t>(), dA.as<elem_t>(), dA.as<elem_t>(), nrows, ncols);
        C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], 0u); }
    }
}

TEST(GF2to12Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_vector((size_t)s.n_rows1 * s.n_cols1);
        const auto B = random_vector((size_t)s.n_cols1 * s.n_cols2);
        const auto expected = ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(expected.size() * sizeof(elem_t));
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, dC.size));
        gf2to12_cuda::gf2to12_matrix_mul(dC.as<elem_t>(), dA.as<elem_t>(), dB.as<elem_t>(), s.n_rows1, s.n_cols1, s.n_cols2);
        ASSERT_EQ(from_bytes(dC.download()), expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

TEST(GF2to12Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_vector((size_t)m * n);
        std::vector<elem_t> I((size_t)n * n, 0);
        for (uint32_t i = 0; i < n; i++) { I[(size_t)i * n + i] = 1; }

        DeviceBuffer dA(to_bytes(A)), dI(to_bytes(I)), dC(A.size() * sizeof(elem_t));
        gf2to12_cuda::gf2to12_matrix_mul(dC.as<elem_t>(), dA.as<elem_t>(), dI.as<elem_t>(), m, n, n);
        ASSERT_EQ(from_bytes(dC.download()), A) << "A*I, n=" << n;
    }
}

TEST(GF2to12Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_vector((size_t)n_rows1 * n_cols1);
    const auto B = random_vector((size_t)n_cols1 * n_cols2), C = random_vector((size_t)n_cols1 * n_cols2);
    const size_t out = gf2to12_cuda::gf2to12_matrix_bytes_size(n_rows1, n_cols2);

    DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(to_bytes(C)), dBC(B.size() * sizeof(elem_t)),
                 dL(out), dAB(out), dAC(out), dR(out);
    gf2to12_cuda::gf2to12_matrix_add(dBC.as<elem_t>(), dB.as<elem_t>(), dC.as<elem_t>(), n_cols1, n_cols2);
    gf2to12_cuda::gf2to12_matrix_mul(dL.as<elem_t>(), dA.as<elem_t>(), dBC.as<elem_t>(), n_rows1, n_cols1, n_cols2);
    gf2to12_cuda::gf2to12_matrix_mul(dAB.as<elem_t>(), dA.as<elem_t>(), dB.as<elem_t>(), n_rows1, n_cols1, n_cols2);
    gf2to12_cuda::gf2to12_matrix_mul(dAC.as<elem_t>(), dA.as<elem_t>(), dC.as<elem_t>(), n_rows1, n_cols1, n_cols2);
    gf2to12_cuda::gf2to12_matrix_add(dR.as<elem_t>(), dAB.as<elem_t>(), dAC.as<elem_t>(), n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF2to12Cuda, EmptyShapes) {
    DeviceBuffer d(64);
    gf2to12_cuda::gf2to12_vector_mul(d.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 0);
    gf2to12_cuda::gf2to12_matrix_add(d.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 0, 5);
    gf2to12_cuda::gf2to12_matrix_mul(d.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 0, 3, 4);
    gf2to12_cuda::gf2to12_matrix_mul(d.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(gf2to12_cuda::gf2to12_matrix_bytes_size(5, 3));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf2to12_cuda::gf2to12_matrix_mul(dC.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
