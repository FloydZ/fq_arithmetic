#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "arith.h"   // C reference: table based gf256_mul
#include "arith.cu"  // CUDA implementation (namespace gf256_cuda)

namespace {

using fq_cuda::DeviceBuffer;
using elem = uint8_t;

std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

elem random_elem() { return (elem)(rng()() & 0xFFu); }

std::vector<elem> random_vector(const size_t n) {
    std::vector<elem> v(n);
    for (auto &x : v) { x = random_elem(); }
    return v;
}

std::vector<uint8_t> to_bytes(const std::vector<elem> &v) {
    std::vector<uint8_t> r(v.size() * sizeof(elem));
    if (!r.empty()) { memcpy(r.data(), v.data(), r.size()); }
    return r;
}

std::vector<elem> from_bytes(const std::vector<uint8_t> &b, const size_t offset = 0) {
    std::vector<elem> r((b.size() - offset * sizeof(elem)) / sizeof(elem));
    if (!r.empty()) { memcpy(r.data(), b.data() + offset * sizeof(elem), r.size() * sizeof(elem)); }
    return r;
}

/// column major reference matrix, same layout as matrix.h
elem ref_get(const std::vector<elem> &m, const uint32_t nrows, const uint32_t i, const uint32_t j) {
    return m[(size_t)j * nrows + i];
}

std::vector<elem> ref_matrix_mul(const std::vector<elem> &A,
                                 const std::vector<elem> &B,
                                 const uint32_t n_rows1,
                                 const uint32_t n_cols1,
                                 const uint32_t n_cols2) {
    std::vector<elem> C((size_t)n_rows1 * n_cols2, 0);
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            elem acc = 0;
            for (uint32_t k = 0; k < n_cols1; k++) {
                acc ^= ::gf256_mul(ref_get(A, n_rows1, i, k), ref_get(B, n_cols1, k, j));
            }
            C[(size_t)j * n_rows1 + i] = acc;
        }
    }
    return C;
}

__global__ void scalar_ops_kernel(const elem *a, const elem *b,
                                  elem *out_add, elem *out_sub, elem *out_mul,
                                  const uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out_add[i] = gf256_cuda::gf256_add(a[i], b[i]);
        out_sub[i] = gf256_cuda::gf256_sub(a[i], b[i]);
        out_mul[i] = gf256_cuda::gf256_mul(a[i], b[i]);
    }
}

const uint32_t vector_lengths[] = {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 64, 64}, {129, 257, 65}, {7, 4100, 3}, {256, 1025, 2}, {5, 1024, 3}, {1000, 3, 300},
};

} // namespace

TEST(GF256Cuda, HostScalarMatchesC) {
    for (uint32_t a = 0; a < 256; a++) {
        for (uint32_t b = 0; b < 256; b++) {
            ASSERT_EQ(gf256_cuda::gf256_add(a, b), ::gf256_add(a, b));
            ASSERT_EQ(gf256_cuda::gf256_sub(a, b), ::gf256_sub(a, b));
            ASSERT_EQ(gf256_cuda::gf256_mul(a, b), ::gf256_mul(a, b)) << "a=" << a << " b=" << b;
        }
    }
}

TEST(GF256Cuda, HostPackedMulMatchesC) {
    for (uint32_t it = 0; it < 10000; it++) {
        const uint32_t a = rng()(), b = rng()();
        const uint32_t c = gf256_cuda::gf256v_mul_u32(a, b);
        for (uint32_t l = 0; l < 4; l++) {
            const elem al = (a >> (8 * l)) & 0xFF, bl = (b >> (8 * l)) & 0xFF;
            ASSERT_EQ((c >> (8 * l)) & 0xFF, ::gf256_mul(al, bl));
        }
    }
}

TEST(GF256Cuda, DeviceScalarOps) {
    // all 2^16 pairs
    const uint32_t n = 1u << 16u;
    std::vector<elem> a(n), b(n);
    for (uint32_t t = 0; t < n; t++) { a[t] = t >> 8u; b[t] = t & 0xFFu; }

    DeviceBuffer da(to_bytes(a)), db(to_bytes(b)), d_add(n), d_sub(n), d_mul(n);
    scalar_ops_kernel<<<fq_cuda::grid_size(n), fq_cuda::BLOCK_SIZE>>>
        (da.as<elem>(), db.as<elem>(), d_add.as<elem>(), d_sub.as<elem>(), d_mul.as<elem>(), n);
    FQ_CUDA_CHECK(cudaGetLastError());
    const auto add = from_bytes(d_add.download()), sub = from_bytes(d_sub.download()),
               mul = from_bytes(d_mul.download());
    for (uint32_t t = 0; t < n; t++) {
        ASSERT_EQ(add[t], a[t] ^ b[t]);
        ASSERT_EQ(sub[t], a[t] ^ b[t]);
        ASSERT_EQ(mul[t], ::gf256_mul(a[t], b[t])) << "a=" << (int)a[t] << " b=" << (int)b[t];
    }
}

/// checks gf256_vector_{add,sub,mul}, aligned and with a 1-element offset
/// (which forces the element-wise kernel)
TEST(GF256Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        for (const uint32_t offset : {0u, 1u}) {
            const auto a = random_vector(n), b = random_vector(n);

            std::vector<elem> pa(offset, 0), pb(offset, 0);
            pa.insert(pa.end(), a.begin(), a.end());
            pb.insert(pb.end(), b.begin(), b.end());
            DeviceBuffer da(to_bytes(pa)), db(to_bytes(pb)), dc((n + offset) * sizeof(elem));
            elem *c_ptr = dc.as<elem>() + offset;
            const elem *a_ptr = da.as<elem>() + offset, *b_ptr = db.as<elem>() + offset;

            gf256_cuda::gf256_vector_add(c_ptr, a_ptr, b_ptr, n);
            auto c = from_bytes(dc.download(), offset);
            for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[i], a[i] ^ b[i]) << "add n=" << n; }

            gf256_cuda::gf256_vector_sub(c_ptr, a_ptr, b_ptr, n);
            c = from_bytes(dc.download(), offset);
            for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[i], a[i] ^ b[i]) << "sub n=" << n; }

            gf256_cuda::gf256_vector_mul(c_ptr, a_ptr, b_ptr, n);
            c = from_bytes(dc.download(), offset);
            for (size_t i = 0; i < n; i++) {
                ASSERT_EQ(c[i], ::gf256_mul(a[i], b[i])) << "mul n=" << n << " i=" << i;
            }
        }
    }
}

TEST(GF256Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_vector(n), b = random_vector(n);
    DeviceBuffer da(to_bytes(a)), db(to_bytes(b));
    gf256_cuda::gf256_vector_mul(da.as<elem>(), da.as<elem>(), db.as<elem>(), n);
    const auto c = from_bytes(da.download());
    for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ::gf256_mul(a[i], b[i])); }
}

TEST(GF256Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_vector((size_t)nrows * ncols), B = random_vector((size_t)nrows * ncols);
        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(A.size() * sizeof(elem));

        gf256_cuda::gf256_matrix_add(dC.as<elem>(), dA.as<elem>(), dB.as<elem>(), nrows, ncols);
        auto C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], A[i] ^ B[i]) << nrows << "x" << ncols; }

        gf256_cuda::gf256_matrix_sub(dC.as<elem>(), dA.as<elem>(), dB.as<elem>(), nrows, ncols);
        C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], A[i] ^ B[i]) << nrows << "x" << ncols; }

        // A - A = 0
        gf256_cuda::gf256_matrix_sub(dC.as<elem>(), dA.as<elem>(), dA.as<elem>(), nrows, ncols);
        C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], 0); }
    }
}

TEST(GF256Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_vector((size_t)s.n_rows1 * s.n_cols1);
        const auto B = random_vector((size_t)s.n_cols1 * s.n_cols2);
        const auto expected = ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(expected.size() * sizeof(elem));
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, dC.size));
        gf256_cuda::gf256_matrix_mul(dC.as<elem>(), dA.as<elem>(), dB.as<elem>(), s.n_rows1, s.n_cols1, s.n_cols2);
        ASSERT_EQ(from_bytes(dC.download()), expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

/// a 1-element offset on matrix1/result forces the unaligned load path even
/// for column sizes that are multiples of 4 bytes
TEST(GF256Cuda, MatrixMulUnaligned) {
    const uint32_t n_rows1 = 64, n_cols1 = 33, n_cols2 = 17;
    const auto A = random_vector((size_t)n_rows1 * n_cols1);
    const auto B = random_vector((size_t)n_cols1 * n_cols2);
    const auto expected = ref_matrix_mul(A, B, n_rows1, n_cols1, n_cols2);

    std::vector<elem> pA(1, 0);
    pA.insert(pA.end(), A.begin(), A.end());
    DeviceBuffer dA(to_bytes(pA)), dB(to_bytes(B)), dC((expected.size() + 1) * sizeof(elem));
    gf256_cuda::gf256_matrix_mul(dC.as<elem>() + 1, dA.as<elem>() + 1, dB.as<elem>(), n_rows1, n_cols1, n_cols2);
    ASSERT_EQ(from_bytes(dC.download(), 1), expected);
}

TEST(GF256Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_vector((size_t)m * n);
        std::vector<elem> I((size_t)n * n, 0);
        for (uint32_t i = 0; i < n; i++) { I[(size_t)i * n + i] = 1; }

        DeviceBuffer dA(to_bytes(A)), dI(to_bytes(I)), dC(A.size() * sizeof(elem));
        gf256_cuda::gf256_matrix_mul(dC.as<elem>(), dA.as<elem>(), dI.as<elem>(), m, n, n);
        ASSERT_EQ(from_bytes(dC.download()), A) << "A*I, n=" << n;
    }
}

TEST(GF256Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_vector((size_t)n_rows1 * n_cols1);
    const auto B = random_vector((size_t)n_cols1 * n_cols2), C = random_vector((size_t)n_cols1 * n_cols2);
    const size_t out = (size_t)n_rows1 * n_cols2 * sizeof(elem);

    DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(to_bytes(C)), dBC(B.size() * sizeof(elem)),
                 dL(out), dAB(out), dAC(out), dR(out);
    gf256_cuda::gf256_matrix_add(dBC.as<elem>(), dB.as<elem>(), dC.as<elem>(), n_cols1, n_cols2);
    gf256_cuda::gf256_matrix_mul(dL.as<elem>(), dA.as<elem>(), dBC.as<elem>(), n_rows1, n_cols1, n_cols2);
    gf256_cuda::gf256_matrix_mul(dAB.as<elem>(), dA.as<elem>(), dB.as<elem>(), n_rows1, n_cols1, n_cols2);
    gf256_cuda::gf256_matrix_mul(dAC.as<elem>(), dA.as<elem>(), dC.as<elem>(), n_rows1, n_cols1, n_cols2);
    gf256_cuda::gf256_matrix_add(dR.as<elem>(), dAB.as<elem>(), dAC.as<elem>(), n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF256Cuda, EmptyShapes) {
    DeviceBuffer d(64);
    elem *p = d.as<elem>();
    gf256_cuda::gf256_vector_mul(p, p, p, 0);
    gf256_cuda::gf256_matrix_add(p, p, p, 0, 5);
    gf256_cuda::gf256_matrix_mul(p, p, p, 0, 3, 4);
    gf256_cuda::gf256_matrix_mul(p, p, p, 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(5 * 3 * sizeof(elem));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf256_cuda::gf256_matrix_mul(dC.as<elem>(), p, p, 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
