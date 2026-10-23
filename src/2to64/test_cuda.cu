#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "arith.h"   // C reference: gf2to64_mul
#include "arith.cu"  // CUDA implementation (namespace gf2to64_cuda)

static_assert(gf2to64_cuda::GF2TO64_CUDA_MODULUS == MODULUS, "CUDA and C use different polynomials");

namespace {

using fq_cuda::DeviceBuffer;
typedef uint64_t elem_t;

std::mt19937_64 &rng() {
    static std::mt19937_64 gen(42);
    return gen;
}

elem_t random_element() {
    return (elem_t)rng()();
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
                acc ^= ::gf2to64_mul(A[(size_t)k * n_rows1 + i], B[(size_t)j * n_cols1 + k]);
            }
            C[(size_t)j * n_rows1 + i] = acc;
        }
    }
    return C;
}

__global__ void scalar_ops_kernel(elem_t *out_add, elem_t *out_sub, elem_t *out_mul,
                                  const elem_t *a, const elem_t *b, const size_t n) {
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
        out_add[t] = gf2to64_cuda::gf2to64_add(a[t], b[t]);
        out_sub[t] = gf2to64_cuda::gf2to64_sub(a[t], b[t]);
        out_mul[t] = gf2to64_cuda::gf2to64_mul(a[t], b[t]);
    }
}

/// random pairs plus all combinations of a few special values
void scalar_test_inputs(std::vector<elem_t> &a, std::vector<elem_t> &b) {
    const elem_t special[] = {0, 1, 2, 0x1Bull, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull,
                              0x7FFFFFFFFFFFFFFFull, 0xF000000000000000ull, 0x00000000FFFFFFFFull};
    for (const elem_t x : special) {
        for (const elem_t y : special) { a.push_back(x); b.push_back(y); }
    }
    for (uint32_t i = 0; i < 100000; i++) {
        a.push_back(random_element());
        b.push_back(random_element());
    }
}

const uint32_t vector_lengths[] = {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 64, 64}, {129, 257, 65}, {7, 4100, 3}, {1000, 3, 300},
};

} // namespace

TEST(GF2to64Cuda, HostScalarMatchesC) {
    std::vector<elem_t> a, b;
    scalar_test_inputs(a, b);
    for (size_t t = 0; t < a.size(); t++) {
        ASSERT_EQ(gf2to64_cuda::gf2to64_add(a[t], b[t]), ::gf2to64_add(a[t], b[t]));
        ASSERT_EQ(gf2to64_cuda::gf2to64_sub(a[t], b[t]), ::gf2to64_sub(a[t], b[t]));
        ASSERT_EQ(gf2to64_cuda::gf2to64_mul(a[t], b[t]), ::gf2to64_mul(a[t], b[t]))
            << std::hex << "a=" << a[t] << " b=" << b[t];
    }
}

TEST(GF2to64Cuda, DeviceScalarOps) {
    std::vector<elem_t> a, b;
    scalar_test_inputs(a, b);
    const size_t n = a.size();
    DeviceBuffer da(to_bytes(a)), db(to_bytes(b)), d_add(n * sizeof(elem_t)), d_sub(n * sizeof(elem_t)), d_mul(n * sizeof(elem_t));
    scalar_ops_kernel<<<fq_cuda::grid_size(n), fq_cuda::BLOCK_SIZE>>>(d_add.as<elem_t>(), d_sub.as<elem_t>(), d_mul.as<elem_t>(),
                                                                     da.as<elem_t>(), db.as<elem_t>(), n);
    FQ_CUDA_CHECK(cudaGetLastError());
    const auto add = from_bytes(d_add.download()), sub = from_bytes(d_sub.download()), mul = from_bytes(d_mul.download());
    for (size_t t = 0; t < n; t++) {
        ASSERT_EQ(add[t], ::gf2to64_add(a[t], b[t]));
        ASSERT_EQ(sub[t], ::gf2to64_sub(a[t], b[t]));
        ASSERT_EQ(mul[t], ::gf2to64_mul(a[t], b[t])) << std::hex << "a=" << a[t] << " b=" << b[t];
    }
}

TEST(GF2to64Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        const auto a = random_vector(n), b = random_vector(n);
        DeviceBuffer da(to_bytes(a)), db(to_bytes(b)), dc(n * sizeof(elem_t));

        gf2to64_cuda::gf2to64_vector_add(dc.as<elem_t>(), da.as<elem_t>(), db.as<elem_t>(), n);
        auto c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf2to64_add(a[i], b[i])) << "add n=" << n; }

        gf2to64_cuda::gf2to64_vector_sub(dc.as<elem_t>(), da.as<elem_t>(), db.as<elem_t>(), n);
        c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf2to64_sub(a[i], b[i])) << "sub n=" << n; }

        gf2to64_cuda::gf2to64_vector_mul(dc.as<elem_t>(), da.as<elem_t>(), db.as<elem_t>(), n);
        c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf2to64_mul(a[i], b[i])) << "mul n=" << n << " i=" << i; }
    }
}

TEST(GF2to64Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_vector(n), b = random_vector(n);
    DeviceBuffer da(to_bytes(a)), db(to_bytes(b));
    gf2to64_cuda::gf2to64_vector_mul(da.as<elem_t>(), da.as<elem_t>(), db.as<elem_t>(), n);
    const auto c = from_bytes(da.download());
    for (uint32_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf2to64_mul(a[i], b[i])); }
}

TEST(GF2to64Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_vector((size_t)nrows * ncols), B = random_vector((size_t)nrows * ncols);
        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(gf2to64_cuda::gf2to64_matrix_bytes_size(nrows, ncols));

        gf2to64_cuda::gf2to64_matrix_add(dC.as<elem_t>(), dA.as<elem_t>(), dB.as<elem_t>(), nrows, ncols);
        auto C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], A[i] ^ B[i]) << nrows << "x" << ncols; }

        gf2to64_cuda::gf2to64_matrix_sub(dC.as<elem_t>(), dA.as<elem_t>(), dB.as<elem_t>(), nrows, ncols);
        C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], A[i] ^ B[i]) << nrows << "x" << ncols; }

        // A - A = 0
        gf2to64_cuda::gf2to64_matrix_sub(dC.as<elem_t>(), dA.as<elem_t>(), dA.as<elem_t>(), nrows, ncols);
        C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], 0u); }
    }
}

TEST(GF2to64Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_vector((size_t)s.n_rows1 * s.n_cols1);
        const auto B = random_vector((size_t)s.n_cols1 * s.n_cols2);
        const auto expected = ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(expected.size() * sizeof(elem_t));
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, dC.size));
        gf2to64_cuda::gf2to64_matrix_mul(dC.as<elem_t>(), dA.as<elem_t>(), dB.as<elem_t>(), s.n_rows1, s.n_cols1, s.n_cols2);
        ASSERT_EQ(from_bytes(dC.download()), expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

TEST(GF2to64Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_vector((size_t)m * n);
        std::vector<elem_t> I((size_t)n * n, 0);
        for (uint32_t i = 0; i < n; i++) { I[(size_t)i * n + i] = 1; }

        DeviceBuffer dA(to_bytes(A)), dI(to_bytes(I)), dC(A.size() * sizeof(elem_t));
        gf2to64_cuda::gf2to64_matrix_mul(dC.as<elem_t>(), dA.as<elem_t>(), dI.as<elem_t>(), m, n, n);
        ASSERT_EQ(from_bytes(dC.download()), A) << "A*I, n=" << n;
    }
}

TEST(GF2to64Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_vector((size_t)n_rows1 * n_cols1);
    const auto B = random_vector((size_t)n_cols1 * n_cols2), C = random_vector((size_t)n_cols1 * n_cols2);
    const size_t out = gf2to64_cuda::gf2to64_matrix_bytes_size(n_rows1, n_cols2);

    DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(to_bytes(C)), dBC(B.size() * sizeof(elem_t)),
                 dL(out), dAB(out), dAC(out), dR(out);
    gf2to64_cuda::gf2to64_matrix_add(dBC.as<elem_t>(), dB.as<elem_t>(), dC.as<elem_t>(), n_cols1, n_cols2);
    gf2to64_cuda::gf2to64_matrix_mul(dL.as<elem_t>(), dA.as<elem_t>(), dBC.as<elem_t>(), n_rows1, n_cols1, n_cols2);
    gf2to64_cuda::gf2to64_matrix_mul(dAB.as<elem_t>(), dA.as<elem_t>(), dB.as<elem_t>(), n_rows1, n_cols1, n_cols2);
    gf2to64_cuda::gf2to64_matrix_mul(dAC.as<elem_t>(), dA.as<elem_t>(), dC.as<elem_t>(), n_rows1, n_cols1, n_cols2);
    gf2to64_cuda::gf2to64_matrix_add(dR.as<elem_t>(), dAB.as<elem_t>(), dAC.as<elem_t>(), n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF2to64Cuda, EmptyShapes) {
    DeviceBuffer d(64);
    gf2to64_cuda::gf2to64_vector_mul(d.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 0);
    gf2to64_cuda::gf2to64_matrix_add(d.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 0, 5);
    gf2to64_cuda::gf2to64_matrix_mul(d.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 0, 3, 4);
    gf2to64_cuda::gf2to64_matrix_mul(d.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(gf2to64_cuda::gf2to64_matrix_bytes_size(5, 3));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf2to64_cuda::gf2to64_matrix_mul(dC.as<elem_t>(), d.as<elem_t>(), d.as<elem_t>(), 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
