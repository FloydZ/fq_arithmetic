#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "arith.h"   // C reference: gf16to32_mul on uint8_t[16]
#include "arith.cu"  // CUDA implementation (namespace gf16to32_cuda)

/// the C type `gf16to32` (uint8_t[16]) and the CUDA struct share the name
using E = gf16to32_cuda::gf16to32;

namespace gf16to32_cuda {
// found by ADL, used by gtest
bool operator==(const gf16to32 &a, const gf16to32 &b) { return memcmp(&a, &b, sizeof(gf16to32)) == 0; }
} // namespace gf16to32_cuda

namespace {

using fq_cuda::DeviceBuffer;

/// a*b computed by the C implementation
E ref_mul(const E &a, const E &b) {
    uint8_t ra[16], rb[16], rr[16];
    memcpy(ra, &a, 16);
    memcpy(rb, &b, 16);
    ::gf16to32_mul(rr, ra, rb);
    E r;
    memcpy(&r, rr, 16);
    return r;
}

/// a+b computed by the C implementation
E ref_add(const E &a, const E &b) {
    uint8_t ra[16], rb[16], rr[16];
    memcpy(ra, &a, 16);
    memcpy(rb, &b, 16);
    ::gf16to32_add(rr, ra, rb);
    E r;
    memcpy(&r, rr, 16);
    return r;
}

E from_bytes16(const uint8_t (&b)[16]) {
    E r;
    memcpy(&r, b, 16);
    return r;
}

std::mt19937 &rng() {
    static std::mt19937 gen(42);
    return gen;
}

E random_element() {
    E r;
    for (auto &w : r.v) { w = rng()(); }
    return r;
}

std::vector<E> random_vector(const size_t n) {
    std::vector<E> v(n);
    for (auto &x : v) { x = random_element(); }
    return v;
}

std::vector<uint8_t> to_bytes(const std::vector<E> &v) {
    std::vector<uint8_t> b(v.size() * sizeof(E));
    if (!b.empty()) { memcpy(b.data(), v.data(), b.size()); }
    return b;
}

std::vector<E> from_bytes(const std::vector<uint8_t> &b) {
    std::vector<E> v(b.size() / sizeof(E));
    if (!v.empty()) { memcpy(v.data(), b.data(), v.size() * sizeof(E)); }
    return v;
}

/// reference matrix multiplication, column major as in `16to32/matrix.h`
std::vector<E> ref_matrix_mul(const std::vector<E> &A,
                              const std::vector<E> &B,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    std::vector<E> C((size_t)n_rows1 * n_cols2);
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            E acc = {{0, 0, 0, 0}};
            for (uint32_t k = 0; k < n_cols1; k++) {
                acc = ref_add(acc, ref_mul(A[(size_t)k * n_rows1 + i], B[(size_t)j * n_cols1 + k]));
            }
            C[(size_t)j * n_rows1 + i] = acc;
        }
    }
    return C;
}

__global__ void scalar_ops_kernel(E *out_add, E *out_sub, E *out_mul,
                                  const E *a, const E *b, const uint32_t n) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) { return; }
    out_add[t] = gf16to32_cuda::gf16to32_add(a[t], b[t]);
    out_sub[t] = gf16to32_cuda::gf16to32_sub(a[t], b[t]);
    out_mul[t] = gf16to32_cuda::gf16to32_mul(a[t], b[t]);
}

/// 0, 1, X, X^31, all-ones and a few random elements
std::vector<E> special_elements() {
    std::vector<E> s;
    s.push_back({{0, 0, 0, 0}});
    s.push_back({{1, 0, 0, 0}});
    s.push_back({{0x10, 0, 0, 0}});
    s.push_back({{0, 0, 0, 0x10000000u}});
    s.push_back({{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}});
    for (uint32_t i = 0; i < 8; i++) { s.push_back(random_element()); }
    return s;
}

const uint32_t vector_lengths[] = {1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 1000, 4097, 100001};

struct Shape { uint32_t n_rows1, n_cols1, n_cols2; };
// the CPU reference is slow (schoolbook over GF(16)), so the shapes are
// smaller than for the small fields; 5x300x2 and 7x600x3 cross the shared
// memory tile (256), 130x20x3 needs two thread blocks per column
const Shape mul_shapes[] = {
    {1, 1, 1}, {2, 3, 4}, {3, 5, 2}, {8, 8, 8}, {17, 9, 13}, {33, 40, 7},
    {64, 64, 16}, {130, 20, 3}, {5, 300, 2}, {7, 600, 3},
};

} // namespace

TEST(GF16to32Cuda, HostScalarMatchesReference) {
    const auto s = special_elements();
    for (const E &a : s) {
        for (const E &b : s) {
            ASSERT_EQ(gf16to32_cuda::gf16to32_add(a, b), ref_add(a, b));
            ASSERT_EQ(gf16to32_cuda::gf16to32_sub(a, b), ref_add(a, b));
            ASSERT_EQ(gf16to32_cuda::gf16to32_mul(a, b), ref_mul(a, b));
        }
    }

    for (uint32_t it = 0; it < 100000; it++) {
        const E a = random_element(), b = random_element();
        ASSERT_EQ(gf16to32_cuda::gf16to32_add(a, b), ref_add(a, b));
        ASSERT_EQ(gf16to32_cuda::gf16to32_mul(a, b), ref_mul(a, b)) << "iteration " << it;
    }
}

TEST(GF16to32Cuda, HostMulReduction) {
    // X^31 * X = X^32 = B*X^28 + X^21 + F*X^2 + 9
    const E x31 = {{0, 0, 0, 0x10000000u}}, x = {{0x10, 0, 0, 0}};
    const uint8_t expected[16] = {0x09, 0x0F, 0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0, 0, 0, 0x0B, 0};
    ASSERT_EQ(gf16to32_cuda::gf16to32_mul(x31, x), from_bytes16(expected));
    ASSERT_EQ(gf16to32_cuda::gf16to32_mul(x, x31), from_bytes16(expected));

    // 1 is neutral, multiplication is commutative and distributive
    const E one = {{1, 0, 0, 0}};
    for (uint32_t it = 0; it < 1000; it++) {
        const E a = random_element(), b = random_element(), c = random_element();
        ASSERT_EQ(gf16to32_cuda::gf16to32_mul(a, one), a);
        ASSERT_EQ(gf16to32_cuda::gf16to32_mul(a, b), gf16to32_cuda::gf16to32_mul(b, a));
        ASSERT_EQ(gf16to32_cuda::gf16to32_mul(a, gf16to32_cuda::gf16to32_add(b, c)),
                  gf16to32_cuda::gf16to32_add(gf16to32_cuda::gf16to32_mul(a, b), gf16to32_cuda::gf16to32_mul(a, c)));
    }
}

TEST(GF16to32Cuda, DeviceScalarOps) {
    const uint32_t n = 1u << 16u;
    auto a = random_vector(n), b = random_vector(n);
    const auto s = special_elements();
    for (size_t i = 0; i < s.size() * s.size(); i++) { a[i] = s[i / s.size()]; b[i] = s[i % s.size()]; }

    DeviceBuffer da(to_bytes(a)), db(to_bytes(b));
    DeviceBuffer d_add(n * sizeof(E)), d_sub(n * sizeof(E)), d_mul(n * sizeof(E));
    scalar_ops_kernel<<<n / 256, 256>>>(d_add.as<E>(), d_sub.as<E>(), d_mul.as<E>(), da.as<E>(), db.as<E>(), n);
    FQ_CUDA_CHECK(cudaGetLastError());
    const auto add = from_bytes(d_add.download()), sub = from_bytes(d_sub.download()), mul = from_bytes(d_mul.download());
    for (uint32_t t = 0; t < n; t++) {
        ASSERT_EQ(add[t], ref_add(a[t], b[t]));
        ASSERT_EQ(sub[t], ref_add(a[t], b[t]));
        ASSERT_EQ(mul[t], ref_mul(a[t], b[t])) << "t=" << t;
    }
}

/// checks gf16to32_vector_{add,sub,mul}, also with an offset of one element
TEST(GF16to32Cuda, VectorOps) {
    for (const uint32_t n : vector_lengths) {
        for (const uint32_t offset : {0u, 1u}) {
            const auto a = random_vector(n + offset), b = random_vector(n + offset);
            DeviceBuffer da(to_bytes(a)), db(to_bytes(b)), dc((n + offset) * sizeof(E));
            E *pc = dc.as<E>() + offset;
            const E *pa = da.as<E>() + offset, *pb = db.as<E>() + offset;

            gf16to32_cuda::gf16to32_vector_add(pc, pa, pb, n);
            auto c = from_bytes(dc.download());
            for (size_t i = offset; i < n + offset; i++) { ASSERT_EQ(c[i], ref_add(a[i], b[i])) << "add n=" << n; }

            gf16to32_cuda::gf16to32_vector_sub(pc, pa, pb, n);
            c = from_bytes(dc.download());
            for (size_t i = offset; i < n + offset; i++) { ASSERT_EQ(c[i], ref_add(a[i], b[i])) << "sub n=" << n; }

            gf16to32_cuda::gf16to32_vector_mul(pc, pa, pb, n);
            c = from_bytes(dc.download());
            for (size_t i = offset; i < n + offset; i++) { ASSERT_EQ(c[i], ref_mul(a[i], b[i])) << "mul n=" << n; }
        }
    }
}

TEST(GF16to32Cuda, VectorMulInPlace) {
    const uint32_t n = 1001;
    const auto a = random_vector(n), b = random_vector(n);
    DeviceBuffer da(to_bytes(a)), db(to_bytes(b));
    gf16to32_cuda::gf16to32_vector_mul(da.as<E>(), da.as<E>(), db.as<E>(), n);
    const auto c = from_bytes(da.download());
    for (size_t i = 0; i < n; i++) { ASSERT_EQ(c[i], ref_mul(a[i], b[i])); }
}

TEST(GF16to32Cuda, MatrixAddSub) {
    for (const Shape &s : mul_shapes) {
        const uint32_t nrows = s.n_rows1, ncols = s.n_cols2;
        const auto A = random_vector((size_t)nrows * ncols), B = random_vector((size_t)nrows * ncols);
        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(gf16to32_cuda::gf16to32_matrix_bytes_size(nrows, ncols));

        gf16to32_cuda::gf16to32_matrix_add(dC.as<E>(), dA.as<E>(), dB.as<E>(), nrows, ncols);
        auto C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], ref_add(A[i], B[i])) << nrows << "x" << ncols; }

        gf16to32_cuda::gf16to32_matrix_sub(dC.as<E>(), dA.as<E>(), dB.as<E>(), nrows, ncols);
        C = from_bytes(dC.download());
        for (size_t i = 0; i < A.size(); i++) { ASSERT_EQ(C[i], ref_add(A[i], B[i])) << nrows << "x" << ncols; }

        // A - A = 0
        gf16to32_cuda::gf16to32_matrix_sub(dC.as<E>(), dA.as<E>(), dA.as<E>(), nrows, ncols);
        for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
    }
}

TEST(GF16to32Cuda, MatrixMul) {
    for (const Shape &s : mul_shapes) {
        const auto A = random_vector((size_t)s.n_rows1 * s.n_cols1);
        const auto B = random_vector((size_t)s.n_cols1 * s.n_cols2);
        const auto expected = ref_matrix_mul(A, B, s.n_rows1, s.n_cols1, s.n_cols2);

        DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(expected.size() * sizeof(E));
        // garbage in the output must be overwritten, not accumulated
        FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xA5, dC.size));
        gf16to32_cuda::gf16to32_matrix_mul(dC.as<E>(), dA.as<E>(), dB.as<E>(), s.n_rows1, s.n_cols1, s.n_cols2);
        const auto C = from_bytes(dC.download());
        ASSERT_EQ(C, expected) << s.n_rows1 << "x" << s.n_cols1 << "x" << s.n_cols2;
    }
}

TEST(GF16to32Cuda, MatrixMulIdentity) {
    for (const uint32_t n : {1u, 5u, 16u, 63u, 128u}) {
        const uint32_t m = 2 * n + 1;
        const auto A = random_vector((size_t)m * n);
        std::vector<E> I((size_t)n * n, E{{0, 0, 0, 0}});
        for (uint32_t i = 0; i < n; i++) { I[(size_t)i * n + i] = E{{1, 0, 0, 0}}; }

        DeviceBuffer dA(to_bytes(A)), dI(to_bytes(I)), dC(A.size() * sizeof(E));
        gf16to32_cuda::gf16to32_matrix_mul(dC.as<E>(), dA.as<E>(), dI.as<E>(), m, n, n);
        ASSERT_EQ(from_bytes(dC.download()), A) << "A*I, n=" << n;
    }
}

TEST(GF16to32Cuda, MatrixMulDistributive) {
    // A*(B + C) == A*B + A*C, entirely on the device
    const uint32_t n_rows1 = 77, n_cols1 = 50, n_cols2 = 31;
    const auto A = random_vector((size_t)n_rows1 * n_cols1);
    const auto B = random_vector((size_t)n_cols1 * n_cols2), C = random_vector((size_t)n_cols1 * n_cols2);
    const size_t out = gf16to32_cuda::gf16to32_matrix_bytes_size(n_rows1, n_cols2);

    DeviceBuffer dA(to_bytes(A)), dB(to_bytes(B)), dC(to_bytes(C)), dBC(B.size() * sizeof(E)),
                 dL(out), dAB(out), dAC(out), dR(out);
    gf16to32_cuda::gf16to32_matrix_add(dBC.as<E>(), dB.as<E>(), dC.as<E>(), n_cols1, n_cols2);
    gf16to32_cuda::gf16to32_matrix_mul(dL.as<E>(), dA.as<E>(), dBC.as<E>(), n_rows1, n_cols1, n_cols2);
    gf16to32_cuda::gf16to32_matrix_mul(dAB.as<E>(), dA.as<E>(), dB.as<E>(), n_rows1, n_cols1, n_cols2);
    gf16to32_cuda::gf16to32_matrix_mul(dAC.as<E>(), dA.as<E>(), dC.as<E>(), n_rows1, n_cols1, n_cols2);
    gf16to32_cuda::gf16to32_matrix_add(dR.as<E>(), dAB.as<E>(), dAC.as<E>(), n_rows1, n_cols2);
    ASSERT_EQ(dL.download(), dR.download());
}

TEST(GF16to32Cuda, EmptyShapes) {
    DeviceBuffer d(256);
    E *p = d.as<E>();
    gf16to32_cuda::gf16to32_vector_mul(p, p, p, 0);
    gf16to32_cuda::gf16to32_matrix_add(p, p, p, 0, 5);
    gf16to32_cuda::gf16to32_matrix_mul(p, p, p, 0, 3, 4);
    gf16to32_cuda::gf16to32_matrix_mul(p, p, p, 3, 4, 0);
    FQ_CUDA_CHECK(cudaDeviceSynchronize());

    // n_cols1 == 0: the product is the zero matrix
    DeviceBuffer dC(gf16to32_cuda::gf16to32_matrix_bytes_size(5, 3));
    FQ_CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, dC.size));
    gf16to32_cuda::gf16to32_matrix_mul(dC.as<E>(), p, p, 5, 0, 3);
    for (const uint8_t x : dC.download()) { ASSERT_EQ(x, 0); }
}
