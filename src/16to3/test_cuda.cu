#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

// CPU reference. `16to3/arith.h` also includes `16/matrix.h`, which is not
// verified to be valid C++, so only the (C++ safe) GF(16) table arithmetic is
// included and `ref_mul` below mirrors `gf16to3_mul` of `16to3/arith.h`.
#include "../16/arith.h"
#include "arith.cu"  // CUDA implementation (namespace gf16to3_cuda)

namespace {

using fq_cuda::DeviceBuffer;
using gf16to3_cuda::gf16to3;

/// copy of `gf16to3_mul` (Karatsuba) of `16to3/arith.h`
gf16to3 ref_mul(const gf16to3 a, const gf16to3 b) {
    gf16to3 r;

    const gf16 a0 = a&0xF, a1 = (a>>4)&0xF, a2 = (a>>8)&0xF;
    const gf16 b0 = b&0xF, b1 = (b>>4)&0xF, b2 = (b>>8)&0xF;

    const gf16 p0 = ::gf16_mul(a0, b0);
    const gf16 p1 = ::gf16_mul(a1, b1);
    const gf16 p2 = ::gf16_mul(a2, b2);

    const gf16 a01 = ::gf16_add(a0, a1);
    const gf16 a12 = ::gf16_add(a1, a2);
    const gf16 a02 = ::gf16_add(a0, a2);
    const gf16 b01 = ::gf16_add(b0, b1);
    const gf16 b12 = ::gf16_add(b1, b2);
    const gf16 b02 = ::gf16_add(b0, b2);
    const gf16 p01 = ::gf16_mul(a01, b01);
    const gf16 p12 = ::gf16_mul(a12, b12);
    const gf16 p02 = ::gf16_mul(a02, b02);

    r = ::gf16_add(p1, p2);
    r = ::gf16_add(r, p12);
    r = ::gf16_add(r, p0);

    r^= p0 << 4;
    r^= p01 << 4;
    r^= p12 << 4;

    r^= p02 << 8;
    r^= p0 << 8;
    r^= p1 << 8;
    return r;
}

std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

gf16to3 random_element() { return (gf16to3)(rng()() & 0xFFFu); }

std::vector<gf16to3> random_vector(const size_t n) {
    std::vector<gf16to3> v(n);
    for (auto &x : v) { x = random_element(); }
    return v;
}

template<typename T>
std::vector<uint8_t> to_bytes(const std::vector<T> &v) {
    std::vector<uint8_t> b(v.size() * sizeof(T));
    if (!b.empty()) { memcpy(b.data(), v.data(), b.size()); }
    return b;
}

template<typename T>
std::vector<T> from_bytes(const std::vector<uint8_t> &b) {
    std::vector<T> v(b.size() / sizeof(T));
    if (!v.empty()) { memcpy(v.data(), b.data(), v.size() * sizeof(T)); }
    return v;
}

/// reference matrix multiplication, column major as in `16to3/matrix.h`
std::vector<gf16to3> ref_matrix_mul(const std::vector<gf16to3> &A,
                                    const std::vector<gf16to3> &B,
                                    const uint32_t n_rows1,
                                    const uint32_t n_cols1,
                                    const uint32_t n_cols2) {
    std::vector<gf16to3> C((size_t)n_rows1 * n_cols2, 0);
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf16to3 acc = 0;
            for (uint32_t k = 0; k < n_cols1; k++) {
                acc ^= ref_mul(A[(size_t)k * n_rows1 + i], B[(size_t)j * n_cols1 + k]);
            }
            C[(size_t)j * n_rows1 + i] = acc;
        }
    }
    return C;
}

/// all 2^24 pairs (a, b), a = t >> 12, b = t & 0xFFF
__global__ void scalar_ops_kernel(gf16to3 *out_add, gf16to3 *out_sub, gf16to3 *out_mul) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (1u << 24u)) { return; }
    const gf16to3 a = (gf16to3)(t >> 12u), b = (gf16to3)(t & 0xFFFu);
    out_add[t] = gf16to3_cuda::gf16to3_add(a, b);
    out_sub[t] = gf16to3_cuda::gf16to3_sub(a, b);
    out_mul[t] = gf16to3_cuda::gf16to3_mul(a, b);
}

const uint32_t vector_lengths[] = {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 64, 64}, {129, 257, 65}, {7, 4100, 3}, {300, 1025, 2}, {1000, 3, 300},
};

} // namespace

TEST(GF16to3Cuda, HostScalarMatchesReference) {
    for (uint32_t a = 0; a < 4096; a++) {
        for (uint32_t b = 0; b < 4096; b++) {
            ASSERT_EQ(gf16to3_cuda::gf16to3_add(a, b), (gf16to3)(a ^ b));
            ASSERT_EQ(gf16to3_cuda::gf16to3_sub(a, b), (gf16to3)(a ^ b));
            ASSERT_EQ(gf16to3_cuda::gf16to3_mul(a, b), ref_mul(a, b)) << "a=" << a << " b=" << b;
        }
    }
}

TEST(GF16to3Cuda, HostMulFieldProperties) {
    // 1 is neutral, X^3 = X + 1
    for (uint32_t a = 0; a < 4096; a++) {
        ASSERT_EQ(gf16to3_cuda::gf16to3_mul(a, 1), a);
    }
    const gf16to3 X = 0x010, X2 = 0x100;
    ASSERT_EQ(gf16to3_cuda::gf16to3_mul(X, X2), 0x011);
}

TEST(GF16to3Cuda, DeviceScalarOps) {
    const size_t n = 1u << 24u;
    DeviceBuffer d_add(n * sizeof(gf16to3)), d_sub(n * sizeof(gf16to3)), d_mul(n * sizeof(gf16to3));
    scalar_ops_kernel<<<(uint32_t)(n / 256), 256>>>(d_add.as<gf16to3>(), d_sub.as<gf16to3>(), d_mul.as<gf16to3>());
    FQ_CUDA_CHECK(cudaGetLastError());
    const auto add = from_bytes<gf16to3>(d_add.download());
    const auto sub = from_bytes<gf16to3>(d_sub.download());
    const auto mul = from_bytes<gf16to3>(d_mul.download());
    for (uint32_t t = 0; t < n; t++) {
        const gf16to3 a = t >> 12u, b = t & 0xFFFu;
        ASSERT_EQ(add[t], (gf16to3)(a ^ b));
        ASSERT_EQ(sub[t], (gf16to3)(a ^ b));
        ASSERT_EQ(mul[t], ref_mul(a, b)) << "a=" << a << " b=" << b;
    }
}

/// checks gf16to3_vector_{add,sub,mul}, also with an offset of one element
TEST(GF16to3Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        for (const uint32_t offset : {0u, 1u}) {
            const auto a = random_vector(n + offset), b = random_vector(n + offset);
            DeviceBuffer da(to_bytes(a)), db(to_bytes(b)), dc((n + offset) * sizeof(gf16to3));
            gf16to3 *pc = dc.as<gf16to3>() + offset;
            const gf16to3 *pa = da.as<gf16to3>() + offset, *pb = db.as<gf16to3>() + offset;

            gf16to3_cuda::gf16to3_vector_add(pc, pa, pb, n);
            auto c = from_bytes<gf16to3>(dc.download());
            for (size_t i = offset; i < n + offset; i++) { ASSERT_EQ(c[i], (gf16to3)(a[i] ^ b[i])) << "add n=" << n; }

            gf16to3_cuda::gf16to3_vector_sub(pc, pa, pb, n);
            c = from_bytes<gf16to3>(dc.download());
            for (size_t i = offset; i < n + offset; i++) { ASSERT_EQ(c[i], (gf16to3)(a[i] ^ b[i])) << "sub n=" << n; }

            gf16to3_cuda::gf16to3_vector_mul(pc, pa, pb, n);
            c = from_bytes<gf16to3>(dc.download());
            for (size_t i = offset; i < n + offset; i++) { ASSERT_EQ(c[i], ref_mul(a[i], b[i])) << "mul n=" << n; }
        }
    }
}

TEST(GF16to3Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_vector(n), b = random_vector(n);
    DeviceBuffer da(to_bytes(a)), db(to_bytes(b));
    gf16to3_cuda::gf16to3_vector_mul(da.as<gf16to3>(), da.as<gf16to3>(), db.as<gf16to3>(), n);
    const auto c = from_bytes<gf16to3>(da.download());
    for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ref_mul(a[i], b[i])); }
}

TEST(GF16to3Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_vector((size_t)nrows * ncols), B = random_vector((size_t)nrows * ncols);
        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(gf16to3_cuda::gf16to3_matrix_bytes_size(nrows, ncols));

        gf16to3_cuda::gf16to3_matrix_add(dC.as<gf16to3>(), dA.as<gf16to3>(), dB.as<gf16to3>(), nrows, ncols);
        auto C = from_bytes<gf16to3>(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], (gf16to3)(A[i] ^ B[i])) << nrows << "x" << ncols; }

        gf16to3_cuda::gf16to3_matrix_sub(dC.as<gf16to3>(), dA.as<gf16to3>(), dB.as<gf16to3>(), nrows, ncols);
        C = from_bytes<gf16to3>(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], (gf16to3)(A[i] ^ B[i])) << nrows << "x" << ncols; }

        // A - A = 0
        gf16to3_cuda::gf16to3_matrix_sub(dC.as<gf16to3>(), dA.as<gf16to3>(), dA.as<gf16to3>(), nrows, ncols);
        C = from_bytes<gf16to3>(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], 0); }
    }
}

TEST(GF16to3Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_vector((size_t)s.n_rows1 * s.n_cols1);
        const auto B = random_vector((size_t)s.n_cols1 * s.n_cols2);
        const auto expected = ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(expected.size() * sizeof(gf16to3));
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, dC.size));
        gf16to3_cuda::gf16to3_matrix_mul(dC.as<gf16to3>(), dA.as<gf16to3>(), dB.as<gf16to3>(),
                                         s.n_rows1, s.n_cols1, s.n_cols2);
        const auto C = from_bytes<gf16to3>(dC.download());
        ASSERT_EQ(C, expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

TEST(GF16to3Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_vector((size_t)m * n);
        std::vector<gf16to3> I((size_t)n * n, 0);
        for (uint32_t i = 0; i < n; i++) { I[(size_t)i * n + i] = 1; }

        DeviceBuffer dA(to_bytes(A)), dI(to_bytes(I)), dC(A.size() * sizeof(gf16to3));
        gf16to3_cuda::gf16to3_matrix_mul(dC.as<gf16to3>(), dA.as<gf16to3>(), dI.as<gf16to3>(), m, n, n);
        ASSERT_EQ(from_bytes<gf16to3>(dC.download()), A) << "A*I, n=" << n;
    }
}

TEST(GF16to3Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_vector((size_t)n_rows1 * n_cols1);
    const auto B = random_vector((size_t)n_cols1 * n_cols2), C = random_vector((size_t)n_cols1 * n_cols2);
    const size_t out = gf16to3_cuda::gf16to3_matrix_bytes_size(n_rows1, n_cols2);

    DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(to_bytes(C)), dBC(B.size() * sizeof(gf16to3)),
                 dL(out), dAB(out), dAC(out), dR(out);
    gf16to3_cuda::gf16to3_matrix_add(dBC.as<gf16to3>(), dB.as<gf16to3>(), dC.as<gf16to3>(), n_cols1, n_cols2);
    gf16to3_cuda::gf16to3_matrix_mul(dL.as<gf16to3>(), dA.as<gf16to3>(), dBC.as<gf16to3>(), n_rows1, n_cols1, n_cols2);
    gf16to3_cuda::gf16to3_matrix_mul(dAB.as<gf16to3>(), dA.as<gf16to3>(), dB.as<gf16to3>(), n_rows1, n_cols1, n_cols2);
    gf16to3_cuda::gf16to3_matrix_mul(dAC.as<gf16to3>(), dA.as<gf16to3>(), dC.as<gf16to3>(), n_rows1, n_cols1, n_cols2);
    gf16to3_cuda::gf16to3_matrix_add(dR.as<gf16to3>(), dAB.as<gf16to3>(), dAC.as<gf16to3>(), n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF16to3Cuda, EmptyShapes) {
    DeviceBuffer d(64);
    gf16to3 *p = d.as<gf16to3>();
    gf16to3_cuda::gf16to3_vector_mul(p, p, p, 0);
    gf16to3_cuda::gf16to3_matrix_add(p, p, p, 0, 5);
    gf16to3_cuda::gf16to3_matrix_mul(p, p, p, 0, 3, 4);
    gf16to3_cuda::gf16to3_matrix_mul(p, p, p, 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(gf16to3_cuda::gf16to3_matrix_bytes_size(5, 3));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf16to3_cuda::gf16to3_matrix_mul(dC.as<gf16to3>(), p, p, 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
