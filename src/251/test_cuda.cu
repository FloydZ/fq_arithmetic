#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "arith.h"   // C reference: gf251_add/sub/mul
#include "arith.cu"  // CUDA implementation (namespace gf251_cuda)

namespace {

using fq_cuda::DeviceBuffer;

constexpr uint32_t P = gf251_cuda::GF251_P;

std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

/// random vector of n canonical elements
std::vector<uint8_t> random_vector(const size_t n) {
    std::vector<uint8_t> v(n);
    for (auto &x : v) { x = rng()() % P; }
    return v;
}

/// reference get/set, same layout as gf251_matrix_get/set in matrix.h
uint8_t ref_get(const std::vector<uint8_t> &m, const uint32_t nrows, const uint32_t i, const uint32_t j) {
    return m[(size_t)j * nrows + i];
}

void ref_set(std::vector<uint8_t> &m, const uint32_t nrows, const uint32_t i, const uint32_t j, const uint8_t a) {
    m[(size_t)j * nrows + i] = a;
}

std::vector<uint8_t> random_matrix(const uint32_t nrows, const uint32_t ncols) {
    return random_vector(gf251_cuda::gf251_matrix_bytes_size(nrows, ncols));
}

std::vector<uint8_t> ref_matrix_mul(const std::vector<uint8_t> &A,
                                    const std::vector<uint8_t> &B,
                                    const uint32_t n_rows1,
                                    const uint32_t n_cols1,
                                    const uint32_t n_cols2) {
    std::vector<uint8_t> C(gf251_cuda::gf251_matrix_bytes_size(n_rows1, n_cols2), 0);
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            uint8_t acc = 0;
            for (uint32_t k = 0; k < n_cols1; k++) {
                acc = ::gf251_add(acc, ::gf251_mul(ref_get(A, n_rows1, i, k), ref_get(B, n_cols1, k, j)));
            }
            ref_set(C, n_rows1, i, j, acc);
        }
    }
    return C;
}

/// out[a*P + b] = op(a, b) for all a, b < P
__global__ void scalar_ops_kernel(uint8_t *out_add, uint8_t *out_sub, uint8_t *out_mul) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P * P) { return; }
    const uint8_t a = (uint8_t)(t / P), b = (uint8_t)(t % P);
    out_add[t] = gf251_cuda::gf251_add(a, b);
    out_sub[t] = gf251_cuda::gf251_sub(a, b);
    out_mul[t] = gf251_cuda::gf251_mul(a, b);
}

const uint32_t vector_lengths[] = {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 64, 64}, {129, 257, 65}, {7, 4100, 3}, {256, 4099, 2}, {1000, 3, 300},
};

} // namespace

TEST(GF251Cuda, HostScalarMatchesC) {
    for (uint32_t a = 0; a < P; a++) {
        for (uint32_t b = 0; b < P; b++) {
            ASSERT_EQ(gf251_cuda::gf251_add(a, b), ::gf251_add(a, b)) << "a=" << a << " b=" << b;
            ASSERT_EQ(gf251_cuda::gf251_sub(a, b), ::gf251_sub(a, b)) << "a=" << a << " b=" << b;
            ASSERT_EQ(gf251_cuda::gf251_mul(a, b), ::gf251_mul(a, b)) << "a=" << a << " b=" << b;
        }
    }
}

TEST(GF251Cuda, DeviceScalarOps) {
    DeviceBuffer d_add(P * P), d_sub(P * P), d_mul(P * P);
    scalar_ops_kernel<<<(P * P + 255) / 256, 256>>>(d_add.ptr, d_sub.ptr, d_mul.ptr);
    FQ_CUDA_CHECK(cudaGetLastError());
    const auto add = d_add.download(), sub = d_sub.download(), mul = d_mul.download();
    for (uint32_t a = 0; a < P; a++) {
        for (uint32_t b = 0; b < P; b++) {
            const uint32_t t = a * P + b;
            ASSERT_EQ(add[t], ::gf251_add(a, b)) << "a=" << a << " b=" << b;
            ASSERT_EQ(sub[t], ::gf251_sub(a, b)) << "a=" << a << " b=" << b;
            ASSERT_EQ(mul[t], ::gf251_mul(a, b)) << "a=" << a << " b=" << b;
        }
    }
}

/// checks gf251_vector_{add,sub,mul}, aligned and with a 1-byte offset
/// (which forces the byte-wise kernel)
TEST(GF251Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        for (const uint32_t offset : {0u, 1u}) {
            const auto a = random_vector(n), b = random_vector(n);

            std::vector<uint8_t> pa(offset, 0), pb(offset, 0);
            pa.insert(pa.end(), a.begin(), a.end());
            pb.insert(pb.end(), b.begin(), b.end());
            DeviceBuffer da(pa), db(pb), dc(n + offset);

            gf251_cuda::gf251_vector_add(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            auto c = dc.download();
            for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[offset + i], ::gf251_add(a[i], b[i])) << "add n=" << n; }

            gf251_cuda::gf251_vector_sub(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            c = dc.download();
            for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[offset + i], ::gf251_sub(a[i], b[i])) << "sub n=" << n; }

            gf251_cuda::gf251_vector_mul(dc.ptr + offset, da.ptr + offset, db.ptr + offset, n);
            c = dc.download();
            for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[offset + i], ::gf251_mul(a[i], b[i])) << "mul n=" << n; }
        }
    }
}

TEST(GF251Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_vector(n), b = random_vector(n);
    DeviceBuffer da(a), db(b);
    gf251_cuda::gf251_vector_mul(da.ptr, da.ptr, db.ptr, n);
    const auto c = da.download();
    for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf251_mul(a[i], b[i])); }
}

TEST(GF251Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_matrix(nrows, ncols), B = random_matrix(nrows, ncols);
        DeviceBuffer dA(A), dB(B), dC(A.size());

        gf251_cuda::gf251_matrix_add(dC.ptr, dA.ptr, dB.ptr, nrows, ncols);
        auto C = dC.download();
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], ::gf251_add(A[i], B[i])) << nrows << "x" << ncols; }

        gf251_cuda::gf251_matrix_sub(dC.ptr, dA.ptr, dB.ptr, nrows, ncols);
        C = dC.download();
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], ::gf251_sub(A[i], B[i])) << nrows << "x" << ncols; }

        // A - A = 0
        gf251_cuda::gf251_matrix_sub(dC.ptr, dA.ptr, dA.ptr, nrows, ncols);
        C = dC.download();
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], 0); }
    }
}

TEST(GF251Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_matrix(s.n_rows1, s.n_cols1);
        const auto B = random_matrix(s.n_cols1, s.n_cols2);
        const auto expected = ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(A), dB(B), dC(expected.size());
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, expected.size()));
        gf251_cuda::gf251_matrix_mul(dC.ptr, dA.ptr, dB.ptr, s.n_rows1, s.n_cols1, s.n_cols2);
        ASSERT_EQ(dC.download(), expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

/// all entries P-1 and an inner dimension spanning several tiles: maximal
/// accumulator values in the lazy reduction. (P-1)^2 = 1 mod P, so every
/// entry of the result is n_cols1 mod P.
TEST(GF251Cuda, MatrixMulLazyReductionWorstCase) {
    const uint32_t n_rows1 = 13, n_cols1 = 3 * gf251_cuda::GF251_CUDA_MUL_TILE_K + 5, n_cols2 = 3;
    const std::vector<uint8_t> A(gf251_cuda::gf251_matrix_bytes_size(n_rows1, n_cols1), P - 1);
    const std::vector<uint8_t> B(gf251_cuda::gf251_matrix_bytes_size(n_cols1, n_cols2), P - 1);
    DeviceBuffer dA(A), dB(B), dC(gf251_cuda::gf251_matrix_bytes_size(n_rows1, n_cols2));
    gf251_cuda::gf251_matrix_mul(dC.ptr, dA.ptr, dB.ptr, n_rows1, n_cols1, n_cols2);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, n_cols1 % P); }
}

/// 1-byte offset on the operands forces the unaligned load path even for
/// row counts that are multiples of 4
TEST(GF251Cuda, MatrixMulUnaligned) {
    const uint32_t n_rows1 = 64, n_cols1 = 33, n_cols2 = 17;
    const auto A = random_matrix(n_rows1, n_cols1);
    const auto B = random_matrix(n_cols1, n_cols2);
    const auto expected = ref_matrix_mul(A, B, n_rows1, n_cols1, n_cols2);

    std::vector<uint8_t> pA(1, 0);
    pA.insert(pA.end(), A.begin(), A.end());
    DeviceBuffer dA(pA), dB(B), dC(expected.size() + 1);
    gf251_cuda::gf251_matrix_mul(dC.ptr + 1, dA.ptr + 1, dB.ptr, n_rows1, n_cols1, n_cols2);
    const auto C = dC.download();
    ASSERT_EQ(std::vector<uint8_t>(C.begin() + 1, C.end()), expected);
}

TEST(GF251Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_matrix(m, n);
        std::vector<uint8_t> I(gf251_cuda::gf251_matrix_bytes_size(n, n), 0);
        for (uint32_t i = 0; i < n; i++) { ref_set(I, n, i, i, 1); }

        DeviceBuffer dA(A), dI(I), dC(A.size());
        gf251_cuda::gf251_matrix_mul(dC.ptr, dA.ptr, dI.ptr, m, n, n);
        ASSERT_EQ(dC.download(), A) << "A*I, n=" << n;
    }
}

TEST(GF251Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_matrix(n_rows1, n_cols1);
    const auto B = random_matrix(n_cols1, n_cols2), C = random_matrix(n_cols1, n_cols2);
    const size_t out = gf251_cuda::gf251_matrix_bytes_size(n_rows1, n_cols2);

    DeviceBuffer dA(A), dB(B), dC(C), dBC(B.size()), dL(out), dAB(out), dAC(out), dR(out);
    gf251_cuda::gf251_matrix_add(dBC.ptr, dB.ptr, dC.ptr, n_cols1, n_cols2);
    gf251_cuda::gf251_matrix_mul(dL.ptr, dA.ptr, dBC.ptr, n_rows1, n_cols1, n_cols2);
    gf251_cuda::gf251_matrix_mul(dAB.ptr, dA.ptr, dB.ptr, n_rows1, n_cols1, n_cols2);
    gf251_cuda::gf251_matrix_mul(dAC.ptr, dA.ptr, dC.ptr, n_rows1, n_cols1, n_cols2);
    gf251_cuda::gf251_matrix_add(dR.ptr, dAB.ptr, dAC.ptr, n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF251Cuda, EmptyShapes) {
    DeviceBuffer d(16);
    gf251_cuda::gf251_vector_mul(d.ptr, d.ptr, d.ptr, 0);
    gf251_cuda::gf251_matrix_add(d.ptr, d.ptr, d.ptr, 0, 5);
    gf251_cuda::gf251_matrix_mul(d.ptr, d.ptr, d.ptr, 0, 3, 4);
    gf251_cuda::gf251_matrix_mul(d.ptr, d.ptr, d.ptr, 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(gf251_cuda::gf251_matrix_bytes_size(5, 3));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf251_cuda::gf251_matrix_mul(dC.ptr, d.ptr, d.ptr, 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
