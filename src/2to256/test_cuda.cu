#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "arith.h"   // C reference: gf2to256 = uint64_t[4]
#include "arith.cu"  // CUDA implementation (namespace gf2to256_cuda)

namespace {

using fq_cuda::DeviceBuffer;
using elem = gf2to256_cuda::gf2to256;

std::mt19937_64 &rng() {
    static std::mt19937_64 gen(42);
    return gen;
}

elem random_elem() {
    elem r;
    for (auto &x : r.v) { x = rng()(); }
    return r;
}

bool equal(const elem &a, const elem &b) {
    return (a.v[0] == b.v[0]) && (a.v[1] == b.v[1]) && (a.v[2] == b.v[2]) && (a.v[3] == b.v[3]);
}

/// C reference multiplication
elem ref_mul(const elem &a, const elem &b) {
    elem r;
    ::gf2to256_mul(r.v, a.v, b.v);
    return r;
}

elem ref_add(const elem &a, const elem &b) {
    elem r;
    ::gf2to256_add(r.v, a.v, b.v);
    return r;
}

elem ref_sub(const elem &a, const elem &b) {
    elem r;
    ::gf2to256_sub(r.v, a.v, b.v);
    return r;
}

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

std::vector<elem> from_bytes(const std::vector<uint8_t> &b) {
    std::vector<elem> r(b.size() / sizeof(elem));
    if (!r.empty()) { memcpy(r.data(), b.data(), r.size() * sizeof(elem)); }
    return r;
}

/// reference matrix product, column major (same layout as matrix.h)
std::vector<elem> ref_matrix_mul(const std::vector<elem> &A,
                                 const std::vector<elem> &B,
                                 const uint32_t n_rows1,
                                 const uint32_t n_cols1,
                                 const uint32_t n_cols2) {
    std::vector<elem> C((size_t)n_rows1 * n_cols2);
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            elem acc = {{0, 0, 0, 0}};
            for (uint32_t k = 0; k < n_cols1; k++) {
                acc = ref_add(acc, ref_mul(A[(size_t)k * n_rows1 + i], B[(size_t)j * n_cols1 + k]));
            }
            C[(size_t)j * n_rows1 + i] = acc;
        }
    }
    return C;
}

__global__ void scalar_ops_kernel(elem *out_add, elem *out_sub, elem *out_mul,
                                  const elem *a, const elem *b, const uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out_add[i] = gf2to256_cuda::gf2to256_add(a[i], b[i]);
        out_sub[i] = gf2to256_cuda::gf2to256_sub(a[i], b[i]);
        out_mul[i] = gf2to256_cuda::gf2to256_mul(a[i], b[i]);
    }
}

/// a few structured operands: 0, 1, x, x^63, x^64, x^255, all ones, ...
std::vector<elem> special_elems() {
    const uint64_t ones = ~0ull, top = 1ull << 63u;
    return {
        {{0, 0, 0, 0}}, {{1, 0, 0, 0}}, {{2, 0, 0, 0}}, {{top, 0, 0, 0}}, {{0, 1, 0, 0}},
        {{0, 0, 0, 1}}, {{0, 0, 0, top}}, {{ones, 0, 0, 0}}, {{0, 0, 0, ones}},
        {{ones, ones, ones, ones}}, {{0x425, 0, 0, 0}}, {{top | 1, top | 1, top | 1, top | 1}},
    };
}

const uint32_t vector_lengths[] = {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};

/// small enough for the slow constant-time C reference
struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 20, 5}, {129, 9, 6}, {5, 700, 3}, {130, 3, 40},
};

} // namespace

TEST(GF2to256Cuda, HostScalarMatchesC) {
    const auto special = special_elems();
    for (const elem &a : special) {
        for (const elem &b : special) {
            ASSERT_TRUE(equal(gf2to256_cuda::gf2to256_mul(a, b), ref_mul(a, b)));
        }
    }

    for (uint32_t it = 0; it < 100000; it++) {
        const elem a = random_elem(), b = random_elem();
        ASSERT_TRUE(equal(gf2to256_cuda::gf2to256_add(a, b), ref_add(a, b)));
        ASSERT_TRUE(equal(gf2to256_cuda::gf2to256_sub(a, b), ref_sub(a, b)));
        ASSERT_TRUE(equal(gf2to256_cuda::gf2to256_mul(a, b), ref_mul(a, b))) << "iteration " << it;
    }
}

TEST(GF2to256Cuda, HostClmulMatchesShiftAndAdd) {
    for (uint32_t it = 0; it < 10000; it++) {
        const uint64_t a = rng()(), b = rng()();
        uint64_t lo, hi, elo = 0, ehi = 0;
        gf2to256_cuda::gf2to256_clmul64(lo, hi, a, b);
        for (uint32_t i = 0; i < 64; i++) {
            if ((b >> i) & 1u) {
                elo ^= a << i;
                ehi ^= i ? (a >> (64u - i)) : 0;
            }
        }
        ASSERT_EQ(lo, elo);
        ASSERT_EQ(hi, ehi);
    }
}

TEST(GF2to256Cuda, DeviceScalarOps) {
    auto a = random_vector(10000), b = random_vector(10000);
    const auto special = special_elems();
    for (size_t i = 0; i < special.size(); i++) {
        for (size_t j = 0; j < special.size(); j++) {
            a[i * special.size() + j] = special[i];
            b[i * special.size() + j] = special[j];
        }
    }
    const uint32_t n = (uint32_t)a.size();

    DeviceBuffer da(to_bytes(a)), db(to_bytes(b));
    DeviceBuffer d_add(n * sizeof(elem)), d_sub(n * sizeof(elem)), d_mul(n * sizeof(elem));
    scalar_ops_kernel<<<(n + 255) / 256, 256>>>(d_add.as<elem>(), d_sub.as<elem>(), d_mul.as<elem>(),
                                                da.as<elem>(), db.as<elem>(), n);
    FQ_CUDA_CHECK(cudaGetLastError());

    const auto add = from_bytes(d_add.download()), sub = from_bytes(d_sub.download()), mul = from_bytes(d_mul.download());
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_TRUE(equal(add[i], ref_add(a[i], b[i])));
        ASSERT_TRUE(equal(sub[i], ref_sub(a[i], b[i])));
        ASSERT_TRUE(equal(mul[i], ref_mul(a[i], b[i]))) << "element " << i;
    }
}

TEST(GF2to256Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        const auto a = random_vector(n), b = random_vector(n);
        DeviceBuffer da(to_bytes(a)), db(to_bytes(b)), dc(n * sizeof(elem));

        gf2to256_cuda::gf2to256_vector_add(dc.as<elem>(), da.as<elem>(), db.as<elem>(), n);
        auto c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_TRUE(equal(c[i], ref_add(a[i], b[i]))) << "add n=" << n; }

        gf2to256_cuda::gf2to256_vector_sub(dc.as<elem>(), da.as<elem>(), db.as<elem>(), n);
        c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_TRUE(equal(c[i], ref_sub(a[i], b[i]))) << "sub n=" << n; }

        gf2to256_cuda::gf2to256_vector_mul(dc.as<elem>(), da.as<elem>(), db.as<elem>(), n);
        c = from_bytes(dc.download());
        for (uint32_t i = 0; i < n; i++) { ASSERT_TRUE(equal(c[i], ref_mul(a[i], b[i]))) << "mul n=" << n << " i=" << i; }
    }
}

TEST(GF2to256Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_vector(n), b = random_vector(n);
    DeviceBuffer da(to_bytes(a)), db(to_bytes(b));
    gf2to256_cuda::gf2to256_vector_mul(da.as<elem>(), da.as<elem>(), db.as<elem>(), n);
    const auto c = from_bytes(da.download());
    for (uint32_t i = 0; i < n; i++) { ASSERT_TRUE(equal(c[i], ref_mul(a[i], b[i]))); }
}

TEST(GF2to256Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_vector((size_t)nrows * ncols), B = random_vector((size_t)nrows * ncols);
        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(gf2to256_cuda::gf2to256_matrix_bytes_size(nrows, ncols));

        gf2to256_cuda::gf2to256_matrix_add(dC.as<elem>(), dA.as<elem>(), dB.as<elem>(), nrows, ncols);
        auto C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_TRUE(equal(C[i], ref_add(A[i], B[i]))) << nrows << "x" << ncols; }

        gf2to256_cuda::gf2to256_matrix_sub(dC.as<elem>(), dA.as<elem>(), dB.as<elem>(), nrows, ncols);
        C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_TRUE(equal(C[i], ref_sub(A[i], B[i]))) << nrows << "x" << ncols; }

        // A - A = 0
        gf2to256_cuda::gf2to256_matrix_sub(dC.as<elem>(), dA.as<elem>(), dA.as<elem>(), nrows, ncols);
        for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
    }
}

TEST(GF2to256Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_vector((size_t)s.n_rows1 * s.n_cols1);
        const auto B = random_vector((size_t)s.n_cols1 * s.n_cols2);
        const auto expected = to_bytes(ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2));

        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(expected.size());
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, expected.size()));
        gf2to256_cuda::gf2to256_matrix_mul(dC.as<elem>(), dA.as<elem>(), dB.as<elem>(), s.n_rows1, s.n_cols1, s.n_cols2);
        ASSERT_EQ(dC.download(), expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

TEST(GF2to256Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_vector((size_t)m * n);
        std::vector<elem> I((size_t)n * n, elem{{0, 0, 0, 0}});
        for (uint32_t i = 0; i < n; i++) { I[(size_t)i * n + i] = elem{{1, 0, 0, 0}}; }

        DeviceBuffer dA(to_bytes(A)), dI(to_bytes(I)), dC(A.size() * sizeof(elem));
        gf2to256_cuda::gf2to256_matrix_mul(dC.as<elem>(), dA.as<elem>(), dI.as<elem>(), m, n, n);
        ASSERT_EQ(dC.download(), to_bytes(A)) << "A*I, n=" << n;
    }
}

TEST(GF2to256Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device (n_cols1 spans several tiles)
    const uint32_t n_rows1 = 77, n_cols1 = 600, n_cols2 = 31;
    const auto A = random_vector((size_t)n_rows1 * n_cols1);
    const auto B = random_vector((size_t)n_cols1 * n_cols2), C = random_vector((size_t)n_cols1 * n_cols2);
    const size_t out = gf2to256_cuda::gf2to256_matrix_bytes_size(n_rows1, n_cols2);

    DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(to_bytes(C)), dBC(B.size() * sizeof(elem));
    DeviceBuffer dL(out), dAB(out), dAC(out), dR(out);
    gf2to256_cuda::gf2to256_matrix_add(dBC.as<elem>(), dB.as<elem>(), dC.as<elem>(), n_cols1, n_cols2);
    gf2to256_cuda::gf2to256_matrix_mul(dL.as<elem>(), dA.as<elem>(), dBC.as<elem>(), n_rows1, n_cols1, n_cols2);
    gf2to256_cuda::gf2to256_matrix_mul(dAB.as<elem>(), dA.as<elem>(), dB.as<elem>(), n_rows1, n_cols1, n_cols2);
    gf2to256_cuda::gf2to256_matrix_mul(dAC.as<elem>(), dA.as<elem>(), dC.as<elem>(), n_rows1, n_cols1, n_cols2);
    gf2to256_cuda::gf2to256_matrix_add(dR.as<elem>(), dAB.as<elem>(), dAC.as<elem>(), n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF2to256Cuda, EmptyShapes) {
    DeviceBuffer d(128);
    gf2to256_cuda::gf2to256_vector_mul(d.as<elem>(), d.as<elem>(), d.as<elem>(), 0);
    gf2to256_cuda::gf2to256_matrix_add(d.as<elem>(), d.as<elem>(), d.as<elem>(), 0, 5);
    gf2to256_cuda::gf2to256_matrix_mul(d.as<elem>(), d.as<elem>(), d.as<elem>(), 0, 3, 4);
    gf2to256_cuda::gf2to256_matrix_mul(d.as<elem>(), d.as<elem>(), d.as<elem>(), 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(gf2to256_cuda::gf2to256_matrix_bytes_size(5, 3));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf2to256_cuda::gf2to256_matrix_mul(dC.as<elem>(), d.as<elem>(), d.as<elem>(), 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
