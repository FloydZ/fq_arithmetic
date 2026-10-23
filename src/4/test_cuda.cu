#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "arith.h"   // C reference: table based gf4_mul, gf4v_mul_u8
#include "arith.cu"  // CUDA implementation (namespace gf4_cuda)

namespace {

using fq_cuda::DeviceBuffer;

std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

/// reference get/set of a packed vector, same layout as gf4_vector_get/set in vector.h
uint8_t ref_vget(const std::vector<uint8_t> &v, const size_t off, const uint32_t i) {
    return (v[off + i / 4] >> (2u * (i % 4))) & 3u;
}

void ref_vset(std::vector<uint8_t> &v, const size_t off, const uint32_t i, const uint8_t a) {
    const uint32_t s = 2u * (i % 4);
    uint8_t &b = v[off + i / 4];
    b = (uint8_t)((b & ~(3u << s)) | ((a & 3u) << s));
}

/// random packed vector of n elements. The padding bits of the last byte are
/// random as well (the CUDA vector functions must ignore/clear them).
std::vector<uint8_t> random_packed(const uint32_t n) {
    std::vector<uint8_t> v(gf4_cuda::gf4_matrix_bytes_per_column(n));
    for (auto &x : v) { x = rng()() & 0xFFu; }
    return v;
}

/// reference get/set, same layout as gf4_matrix_get/set in matrix.h
uint8_t ref_get(const std::vector<uint8_t> &m, const uint32_t nrows, const uint32_t i, const uint32_t j) {
    return ref_vget(m, (size_t)gf4_cuda::gf4_matrix_bytes_per_column(nrows) * j, i);
}

void ref_set(std::vector<uint8_t> &m, const uint32_t nrows, const uint32_t i, const uint32_t j, const uint8_t a) {
    ref_vset(m, (size_t)gf4_cuda::gf4_matrix_bytes_per_column(nrows) * j, i, a);
}

/// random matrix with zero padding bits
std::vector<uint8_t> random_matrix(const uint32_t nrows, const uint32_t ncols) {
    std::vector<uint8_t> m(gf4_cuda::gf4_matrix_bytes_size(nrows, ncols), 0);
    for (uint32_t j = 0; j < ncols; j++) {
        for (uint32_t i = 0; i < nrows; i++) {
            ref_set(m, nrows, i, j, rng()() & 3u);
        }
    }
    return m;
}

std::vector<uint8_t> ref_matrix_mul(const std::vector<uint8_t> &A,
                                    const std::vector<uint8_t> &B,
                                    const uint32_t n_rows1,
                                    const uint32_t n_cols1,
                                    const uint32_t n_cols2) {
    std::vector<uint8_t> C(gf4_cuda::gf4_matrix_bytes_size(n_rows1, n_cols2), 0);
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            uint8_t acc = 0;
            for (uint32_t k = 0; k < n_cols1; k++) {
                acc = ::gf4_add(acc, ::gf4_mul(ref_get(A, n_rows1, i, k), ref_get(B, n_cols1, k, j)));
            }
            ref_set(C, n_rows1, i, j, acc);
        }
    }
    return C;
}

/// a[i] op b[i] for i < n with the C reference, padding bits zero
template<typename F>
std::vector<uint8_t> ref_vector_op(const std::vector<uint8_t> &a,
                                   const std::vector<uint8_t> &b,
                                   const uint32_t n,
                                   F f) {
    std::vector<uint8_t> c(a.size(), 0);
    for (uint32_t i = 0; i < n; i++) {
        ref_vset(c, 0, i, f(ref_vget(a, 0, i), ref_vget(b, 0, i)));
    }
    return c;
}

__global__ void scalar_ops_kernel(uint8_t *out_add, uint8_t *out_sub, uint8_t *out_mul) {
    const uint32_t t = threadIdx.x;
    const uint8_t a = (uint8_t)(t >> 2u), b = (uint8_t)(t & 3u);
    out_add[t] = gf4_cuda::gf4_add(a & 3u, b);
    out_sub[t] = gf4_cuda::gf4_sub(a & 3u, b);
    out_mul[t] = gf4_cuda::gf4_mul(a & 3u, b);
}

const uint32_t vector_lengths[] = {1, 2, 3, 4, 5, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4097, 100001};

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 64, 64}, {129, 257, 65}, {7, 4100, 3}, {15, 5, 3}, {16, 16, 16},
    {18, 9, 33}, {128, 300, 2}, {1000, 3, 300}, {64, 4097, 2},
};

} // namespace

TEST(GF4Cuda, HostScalarMatchesC) {
    for (uint32_t a = 0; a < 4; a++) {
        for (uint32_t b = 0; b < 4; b++) {
            EXPECT_EQ(gf4_cuda::gf4_add(a, b), ::gf4_add(a, b));
            EXPECT_EQ(gf4_cuda::gf4_sub(a, b), ::gf4_sub(a, b));
            EXPECT_EQ(gf4_cuda::gf4_mul(a, b), ::gf4_mul(a, b)) << "a=" << a << " b=" << b;
            EXPECT_EQ(gf4_cuda::gf4_mul(a, b), ::gf4_mul_v2(a, b)) << "a=" << a << " b=" << b;
        }
    }
}

/// all pairs of bytes (4 packed elements each)
TEST(GF4Cuda, HostPackedMulMatchesC) {
    for (uint32_t a = 0; a < 256; a++) {
        for (uint32_t b = 0; b < 256; b++) {
            ASSERT_EQ(gf4_cuda::gf4v_mul_u8(a, b), ::gf4v_mul_u8(a, b)) << "a=" << a << " b=" << b;
        }
    }

    for (uint32_t it = 0; it < 10000; it++) {
        const uint32_t a = rng()(), b = rng()();
        const uint32_t c = gf4_cuda::gf4v_mul_u32(a, b);
        for (uint32_t l = 0; l < 16; l++) {
            const uint8_t al = (a >> (2 * l)) & 3, bl = (b >> (2 * l)) & 3;
            ASSERT_EQ((c >> (2 * l)) & 3, ::gf4_mul(al, bl));
        }
    }
}

TEST(GF4Cuda, DeviceScalarOps) {
    DeviceBuffer d_add(16), d_sub(16), d_mul(16);
    scalar_ops_kernel<<<1, 16>>>(d_add.ptr, d_sub.ptr, d_mul.ptr);
    FQ_CUDA_CHECK(cudaGetLastError());
    const auto add = d_add.download(), sub = d_sub.download(), mul = d_mul.download();
    for (uint32_t t = 0; t < 16; t++) {
        const uint8_t a = t >> 2u, b = t & 3u;
        EXPECT_EQ(add[t], ::gf4_add(a, b));
        EXPECT_EQ(sub[t], ::gf4_sub(a, b));
        EXPECT_EQ(mul[t], ::gf4_mul(a, b)) << "a=" << (int)a << " b=" << (int)b;
    }
}

/// checks gf4_vector_{add,sub,mul}, aligned and with a 1-byte offset (which
/// forces the byte-wise kernel). The inputs have random padding bits, the
/// output padding bits must be zero, and the byte before the output (offset)
/// must not be touched.
TEST(GF4Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        for (const uint32_t offset : {0u, 1u}) {
            const auto a = random_packed(n), b = random_packed(n);
            const size_t nbytes = a.size();
            const auto e_add = ref_vector_op(a, b, n, [](uint8_t x, uint8_t y) { return ::gf4_add(x, y); });
            const auto e_sub = ref_vector_op(a, b, n, [](uint8_t x, uint8_t y) { return ::gf4_sub(x, y); });
            const auto e_mul = ref_vector_op(a, b, n, [](uint8_t x, uint8_t y) { return ::gf4_mul(x, y); });

            std::vector<uint8_t> pa(offset, 0), pb(offset, 0);
            pa.insert(pa.end(), a.begin(), a.end());
            pb.insert(pb.end(), b.begin(), b.end());
            DeviceBuffer da(pa), db(pb), dc(nbytes + offset);
            FQ_CUDA_CHECK(cudaMemset(dc.ptr, 0xA5, nbytes + offset));

            gf4_cuda::gf4_vector_add(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            auto c = dc.download();
            if (offset) { ASSERT_EQ(c[0], 0xA5); }
            for (size_t i = 0; i < nbytes; i++) { ASSERT_EQ(c[offset + i], e_add[i]) << "add n=" << n << " byte " << i; }

            gf4_cuda::gf4_vector_sub(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            c = dc.download();
            for (size_t i = 0; i < nbytes; i++) { ASSERT_EQ(c[offset + i], e_sub[i]) << "sub n=" << n << " byte " << i; }

            gf4_cuda::gf4_vector_mul(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            c = dc.download();
            for (size_t i = 0; i < nbytes; i++) { ASSERT_EQ(c[offset + i], e_mul[i]) << "mul n=" << n << " byte " << i; }
        }
    }
}

TEST(GF4Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_packed(n), b = random_packed(n);
    const auto expected = ref_vector_op(a, b, n, [](uint8_t x, uint8_t y) { return ::gf4_mul(x, y); });
    DeviceBuffer da(a), db(b);
    gf4_cuda::gf4_vector_mul(da.ptr, da.ptr, db.ptr, n);
    ASSERT_EQ(da.download(), expected);
}

TEST(GF4Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_matrix(nrows, ncols), B = random_matrix(nrows, ncols);
        std::vector<uint8_t> E(A.size(), 0);
        for (uint32_t j = 0; j < ncols; j++) {
            for (uint32_t i = 0; i < nrows; i++) {
                ref_set(E, nrows, i, j, ::gf4_add(ref_get(A, nrows, i, j), ref_get(B, nrows, i, j)));
            }
        }
        DeviceBuffer dA(A), dB(B), dC(A.size());

        gf4_cuda::gf4_matrix_add(dC.ptr, dA.ptr, dB.ptr, nrows, ncols);
        ASSERT_EQ(dC.download(), E) << nrows << "x" << ncols;

        gf4_cuda::gf4_matrix_sub(dC.ptr, dA.ptr, dB.ptr, nrows, ncols);
        ASSERT_EQ(dC.download(), E) << nrows << "x" << ncols;

        // A - A = 0
        gf4_cuda::gf4_matrix_sub(dC.ptr, dA.ptr, dA.ptr, nrows, ncols);
        for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
    }
}

TEST(GF4Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_matrix(s.n_rows1, s.n_cols1);
        const auto B = random_matrix(s.n_cols1, s.n_cols2);
        const auto expected = ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(A), dB(B), dC(expected.size());
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, expected.size()));
        gf4_cuda::gf4_matrix_mul(dC.ptr, dA.ptr, dB.ptr, s.n_rows1, s.n_cols1, s.n_cols2);
        ASSERT_EQ(dC.download(), expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

/// 1-byte offset on the operands forces the unaligned load path even for
/// column sizes that are multiples of 4 bytes
TEST(GF4Cuda, MatrixMulUnaligned) {
    const uint32_t n_rows1 = 128, n_cols1 = 33, n_cols2 = 17;
    const auto A = random_matrix(n_rows1, n_cols1);
    const auto B = random_matrix(n_cols1, n_cols2);
    const auto expected = ref_matrix_mul(A, B, n_rows1, n_cols1, n_cols2);

    std::vector<uint8_t> pA(1, 0);
    pA.insert(pA.end(), A.begin(), A.end());
    DeviceBuffer dA(pA), dB(B), dC(expected.size() + 1);
    gf4_cuda::gf4_matrix_mul(dC.ptr + 1, dA.ptr + 1, dB.ptr, n_rows1, n_cols1, n_cols2);
    const auto C = dC.download();
    ASSERT_EQ(std::vector<uint8_t>(C.begin() + 1, C.end()), expected);
}

TEST(GF4Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_matrix(m, n);
        std::vector<uint8_t> I(gf4_cuda::gf4_matrix_bytes_size(n, n), 0);
        for (uint32_t i = 0; i < n; i++) { ref_set(I, n, i, i, 1); }

        DeviceBuffer dA(A), dI(I), dC(A.size());
        gf4_cuda::gf4_matrix_mul(dC.ptr, dA.ptr, dI.ptr, m, n, n);
        ASSERT_EQ(dC.download(), A) << "A*I, n=" << n;
    }
}

TEST(GF4Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_matrix(n_rows1, n_cols1);
    const auto B = random_matrix(n_cols1, n_cols2), C = random_matrix(n_cols1, n_cols2);
    const size_t out = gf4_cuda::gf4_matrix_bytes_size(n_rows1, n_cols2);

    DeviceBuffer dA(A), dB(B), dC(C), dBC(B.size()), dL(out), dAB(out), dAC(out), dR(out);
    gf4_cuda::gf4_matrix_add(dBC.ptr, dB.ptr, dC.ptr, n_cols1, n_cols2);
    gf4_cuda::gf4_matrix_mul(dL.ptr, dA.ptr, dBC.ptr, n_rows1, n_cols1, n_cols2);
    gf4_cuda::gf4_matrix_mul(dAB.ptr, dA.ptr, dB.ptr, n_rows1, n_cols1, n_cols2);
    gf4_cuda::gf4_matrix_mul(dAC.ptr, dA.ptr, dC.ptr, n_rows1, n_cols1, n_cols2);
    gf4_cuda::gf4_matrix_add(dR.ptr, dAB.ptr, dAC.ptr, n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF4Cuda, EmptyShapes) {
    DeviceBuffer d(16);
    gf4_cuda::gf4_vector_mul(d.ptr, d.ptr, d.ptr, 0);
    gf4_cuda::gf4_matrix_add(d.ptr, d.ptr, d.ptr, 0, 5);
    gf4_cuda::gf4_matrix_mul(d.ptr, d.ptr, d.ptr, 0, 3, 4);
    gf4_cuda::gf4_matrix_mul(d.ptr, d.ptr, d.ptr, 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(gf4_cuda::gf4_matrix_bytes_size(7, 3));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf4_cuda::gf4_matrix_mul(dC.ptr, d.ptr, d.ptr, 7, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
