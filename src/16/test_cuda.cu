#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "arith.h"   // C reference: table based gf16_mul
#include "arith.cu"  // CUDA implementation (namespace gf16_cuda)

namespace {

using fq_cuda::DeviceBuffer;

std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

/// random nibble packed vector of n elements, padding nibble zero
std::vector<uint8_t> random_packed(const uint32_t n) {
    std::vector<uint8_t> v(gf16_cuda::gf16_matrix_bytes_per_column(n));
    for (auto &x : v) { x = rng()() & 0xFFu; }
    if (n & 1u) { v.back() &= 0x0Fu; }
    return v;
}

/// reference get/set, same layout as gf16_matrix_get/set in matrix.h
uint8_t ref_get(const std::vector<uint8_t> &m, const uint32_t nrows, const uint32_t i, const uint32_t j) {
    const uint8_t b = m[(size_t)gf16_cuda::gf16_matrix_bytes_per_column(nrows) * j + i / 2];
    return (i & 1u) ? (b >> 4u) : (b & 0xFu);
}

void ref_set(std::vector<uint8_t> &m, const uint32_t nrows, const uint32_t i, const uint32_t j, const uint8_t a) {
    uint8_t &b = m[(size_t)gf16_cuda::gf16_matrix_bytes_per_column(nrows) * j + i / 2];
    b = (i & 1u) ? (uint8_t)((b & 0x0Fu) | (a << 4u)) : (uint8_t)((b & 0xF0u) | (a & 0xFu));
}

std::vector<uint8_t> random_matrix(const uint32_t nrows, const uint32_t ncols) {
    std::vector<uint8_t> m(gf16_cuda::gf16_matrix_bytes_size(nrows, ncols), 0);
    for (uint32_t j = 0; j < ncols; j++) {
        for (uint32_t i = 0; i < nrows; i++) {
            ref_set(m, nrows, i, j, rng()() & 0xFu);
        }
    }
    return m;
}

std::vector<uint8_t> ref_matrix_mul(const std::vector<uint8_t> &A,
                                    const std::vector<uint8_t> &B,
                                    const uint32_t n_rows1,
                                    const uint32_t n_cols1,
                                    const uint32_t n_cols2) {
    std::vector<uint8_t> C(gf16_cuda::gf16_matrix_bytes_size(n_rows1, n_cols2), 0);
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            uint8_t acc = 0;
            for (uint32_t k = 0; k < n_cols1; k++) {
                acc ^= ::gf16_mul(ref_get(A, n_rows1, i, k), ref_get(B, n_cols1, k, j));
            }
            ref_set(C, n_rows1, i, j, acc);
        }
    }
    return C;
}

__global__ void scalar_ops_kernel(uint8_t *out_add, uint8_t *out_sub, uint8_t *out_mul) {
    const uint32_t t = threadIdx.x;
    const uint8_t a = (uint8_t)(t >> 4u), b = (uint8_t)(t & 0xFu);
    out_add[t] = gf16_cuda::gf16_add(a, b);
    out_sub[t] = gf16_cuda::gf16_sub(a, b);
    out_mul[t] = gf16_cuda::gf16_mul(a, b);
}

const uint32_t vector_lengths[] = {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 64, 64}, {129, 257, 65}, {7, 4100, 3}, {256, 4099, 2}, {1000, 3, 300},
};

} // namespace

TEST(GF16Cuda, HostScalarMatchesTable) {
    for (uint32_t a = 0; a < 16; a++) {
        for (uint32_t b = 0; b < 16; b++) {
            EXPECT_EQ(gf16_cuda::gf16_add(a, b), ::gf16_add(a, b));
            EXPECT_EQ(gf16_cuda::gf16_sub(a, b), ::gf16_sub(a, b));
            EXPECT_EQ(gf16_cuda::gf16_mul(a, b), ::gf16_mul(a, b)) << "a=" << a << " b=" << b;
        }
    }
}

TEST(GF16Cuda, HostPackedMulMatchesTable) {
    for (uint32_t it = 0; it < 10000; it++) {
        const uint32_t a = rng()(), b = rng()();
        const uint32_t c = gf16_cuda::gf16v_mul_u32(a, b);
        for (uint32_t l = 0; l < 8; l++) {
            const uint8_t al = (a >> (4 * l)) & 0xF, bl = (b >> (4 * l)) & 0xF;
            ASSERT_EQ((c >> (4 * l)) & 0xF, ::gf16_mul(al, bl));
        }
    }
}

TEST(GF16Cuda, DeviceScalarOps) {
    DeviceBuffer d_add(256), d_sub(256), d_mul(256);
    scalar_ops_kernel<<<1, 256>>>(d_add.ptr, d_sub.ptr, d_mul.ptr);
    FQ_CUDA_CHECK(cudaGetLastError());
    const auto add = d_add.download(), sub = d_sub.download(), mul = d_mul.download();
    for (uint32_t t = 0; t < 256; t++) {
        const uint8_t a = t >> 4u, b = t & 0xFu;
        EXPECT_EQ(add[t], a ^ b);
        EXPECT_EQ(sub[t], a ^ b);
        EXPECT_EQ(mul[t], ::gf16_mul(a, b)) << "a=" << (int)a << " b=" << (int)b;
    }
}

/// checks gf16_vector_{add,sub,mul}, aligned and with a 1-byte offset
/// (which forces the byte-wise kernel)
TEST(GF16Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        for (const uint32_t offset : {0u, 1u}) {
            const auto a = random_packed(n), b = random_packed(n);
            const size_t nbytes = a.size();

            std::vector<uint8_t> pa(offset, 0), pb(offset, 0);
            pa.insert(pa.end(), a.begin(), a.end());
            pb.insert(pb.end(), b.begin(), b.end());
            DeviceBuffer da(pa), db(pb), dc(nbytes + offset);

            gf16_cuda::gf16_vector_add(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            auto c = dc.download();
            for (size_t i = 0; i < nbytes; i++) { ASSERT_EQ(c[offset + i], a[i] ^ b[i]) << "add n=" << n; }

            gf16_cuda::gf16_vector_sub(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            c = dc.download();
            for (size_t i = 0; i < nbytes; i++) { ASSERT_EQ(c[offset + i], a[i] ^ b[i]) << "sub n=" << n; }

            gf16_cuda::gf16_vector_mul(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            c = dc.download();
            for (size_t i = 0; i < nbytes; i++) {
                ASSERT_EQ(c[offset + i], ::gf16v_mul(a[i], b[i])) << "mul n=" << n << " byte " << i;
            }
        }
    }
}

TEST(GF16Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_packed(n), b = random_packed(n);
    DeviceBuffer da(a), db(b);
    gf16_cuda::gf16_vector_mul(da.ptr, da.ptr, db.ptr, n);
    const auto c = da.download();
    for (size_t i = 0; i < a.size(); i++) { ASSERT_EQ(c[i], ::gf16v_mul(a[i], b[i])); }
}

TEST(GF16Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_matrix(nrows, ncols), B = random_matrix(nrows, ncols);
        DeviceBuffer dA(A), dB(B), dC(A.size());

        gf16_cuda::gf16_matrix_add(dC.ptr, dA.ptr, dB.ptr, nrows, ncols);
        auto C = dC.download();
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], A[i] ^ B[i]) << nrows << "x" << ncols; }

        gf16_cuda::gf16_matrix_sub(dC.ptr, dA.ptr, dB.ptr, nrows, ncols);
        C = dC.download();
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], A[i] ^ B[i]) << nrows << "x" << ncols; }

        // A + A = 0, A - A = 0
        gf16_cuda::gf16_matrix_sub(dC.ptr, dA.ptr, dA.ptr, nrows, ncols);
        C = dC.download();
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], 0); }
    }
}

TEST(GF16Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_matrix(s.n_rows1, s.n_cols1);
        const auto B = random_matrix(s.n_cols1, s.n_cols2);
        const auto expected = ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(A), dB(B), dC(expected.size());
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, expected.size()));
        gf16_cuda::gf16_matrix_mul(dC.ptr, dA.ptr, dB.ptr, s.n_rows1, s.n_cols1, s.n_cols2);
        const auto C = dC.download();
        ASSERT_EQ(C, expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

/// 1-byte offset on the operands forces the unaligned load path even for
/// column sizes that are multiples of 4 bytes
TEST(GF16Cuda, MatrixMulUnaligned) {
    const uint32_t n_rows1 = 64, n_cols1 = 33, n_cols2 = 17;
    const auto A = random_matrix(n_rows1, n_cols1);
    const auto B = random_matrix(n_cols1, n_cols2);
    const auto expected = ref_matrix_mul(A, B, n_rows1, n_cols1, n_cols2);

    std::vector<uint8_t> pA(1, 0);
    pA.insert(pA.end(), A.begin(), A.end());
    DeviceBuffer dA(pA), dB(B), dC(expected.size() + 1);
    gf16_cuda::gf16_matrix_mul(dC.ptr + 1, dA.ptr + 1, dB.ptr, n_rows1, n_cols1, n_cols2);
    const auto C = dC.download();
    ASSERT_EQ(std::vector<uint8_t>(C.begin() + 1, C.end()), expected);
}

TEST(GF16Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_matrix(m, n);
        std::vector<uint8_t> I(gf16_cuda::gf16_matrix_bytes_size(n, n), 0);
        for (uint32_t i = 0; i < n; i++) { ref_set(I, n, i, i, 1); }

        DeviceBuffer dA(A), dI(I), dC(A.size());
        gf16_cuda::gf16_matrix_mul(dC.ptr, dA.ptr, dI.ptr, m, n, n);
        ASSERT_EQ(dC.download(), A) << "A*I, n=" << n;
    }
}

TEST(GF16Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_matrix(n_rows1, n_cols1);
    const auto B = random_matrix(n_cols1, n_cols2), C = random_matrix(n_cols1, n_cols2);
    const size_t out = gf16_cuda::gf16_matrix_bytes_size(n_rows1, n_cols2);

    DeviceBuffer dA(A), dB(B), dC(C), dBC(B.size()), dL(out), dAB(out), dAC(out), dR(out);
    gf16_cuda::gf16_matrix_add(dBC.ptr, dB.ptr, dC.ptr, n_cols1, n_cols2);
    gf16_cuda::gf16_matrix_mul(dL.ptr, dA.ptr, dBC.ptr, n_rows1, n_cols1, n_cols2);
    gf16_cuda::gf16_matrix_mul(dAB.ptr, dA.ptr, dB.ptr, n_rows1, n_cols1, n_cols2);
    gf16_cuda::gf16_matrix_mul(dAC.ptr, dA.ptr, dC.ptr, n_rows1, n_cols1, n_cols2);
    gf16_cuda::gf16_matrix_add(dR.ptr, dAB.ptr, dAC.ptr, n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF16Cuda, EmptyShapes) {
    DeviceBuffer d(16);
    gf16_cuda::gf16_vector_mul(d.ptr, d.ptr, d.ptr, 0);
    gf16_cuda::gf16_matrix_add(d.ptr, d.ptr, d.ptr, 0, 5);
    gf16_cuda::gf16_matrix_mul(d.ptr, d.ptr, d.ptr, 0, 3, 4);
    gf16_cuda::gf16_matrix_mul(d.ptr, d.ptr, d.ptr, 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(gf16_cuda::gf16_matrix_bytes_size(5, 3));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf16_cuda::gf16_matrix_mul(dC.ptr, d.ptr, d.ptr, 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
