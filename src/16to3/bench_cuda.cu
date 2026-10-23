#include <nvbench/nvbench.cuh>

#include <cstdint>

#include "arith.cu"

namespace {

using gf16to3_cuda::gf16to3;

/// device buffer filled with random bytes
struct DeviceBuffer : fq_cuda::DeviceBuffer {
    explicit DeviceBuffer(const size_t n) : fq_cuda::DeviceBuffer(n) { fill_random(); }
};

enum class Op { add, sub, mul };

template<Op op>
void gf16to3_vector_op(gf16to3 *c, const gf16to3 *a, const gf16to3 *b, const uint32_t n, cudaStream_t s) {
    if constexpr (op == Op::add) { gf16to3_cuda::gf16to3_vector_add(c, a, b, n, s); }
    if constexpr (op == Op::sub) { gf16to3_cuda::gf16to3_vector_sub(c, a, b, n, s); }
    if constexpr (op == Op::mul) { gf16to3_cuda::gf16to3_vector_mul(c, a, b, n, s); }
}

/// element-wise gf16to3_add/sub/mul over `Elements` elements
template<Op op>
void bench_gf16to3_vector(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("Elements"));
    const size_t nbytes = (size_t)n * sizeof(gf16to3);
    DeviceBuffer a(nbytes), b(nbytes), c(nbytes);

    state.add_element_count(n, "Elements");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        gf16to3_vector_op<op>(c.as<gf16to3>(), a.as<gf16to3>(), b.as<gf16to3>(), n, launch.get_stream());
    });
}

void bench_gf16to3_add(nvbench::state &state) { bench_gf16to3_vector<Op::add>(state); }
void bench_gf16to3_sub(nvbench::state &state) { bench_gf16to3_vector<Op::sub>(state); }
void bench_gf16to3_mul(nvbench::state &state) { bench_gf16to3_vector<Op::mul>(state); }

/// N x N matrix addition/subtraction
template<Op op>
void bench_gf16to3_matrix_addsub(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t nbytes = gf16to3_cuda::gf16to3_matrix_bytes_size(n, n);
    DeviceBuffer A(nbytes), B(nbytes), C(nbytes);

    state.add_element_count((size_t)n * n, "Elements");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        if constexpr (op == Op::add) {
            gf16to3_cuda::gf16to3_matrix_add(C.as<gf16to3>(), A.as<gf16to3>(), B.as<gf16to3>(), n, n, launch.get_stream());
        } else {
            gf16to3_cuda::gf16to3_matrix_sub(C.as<gf16to3>(), A.as<gf16to3>(), B.as<gf16to3>(), n, n, launch.get_stream());
        }
    });
}

void bench_gf16to3_matrix_add(nvbench::state &state) { bench_gf16to3_matrix_addsub<Op::add>(state); }
void bench_gf16to3_matrix_sub(nvbench::state &state) { bench_gf16to3_matrix_addsub<Op::sub>(state); }

/// N x N times N x N matrix multiplication
void bench_gf16to3_matrix_mul(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t nbytes = gf16to3_cuda::gf16to3_matrix_bytes_size(n, n);
    DeviceBuffer A(nbytes), B(nbytes), C(nbytes);

    // one GF(16^3) multiply-accumulate per (i, j, k)
    state.add_element_count((size_t)n * n * n, "MACs");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        gf16to3_cuda::gf16to3_matrix_mul(C.as<gf16to3>(), A.as<gf16to3>(), B.as<gf16to3>(), n, n, n, launch.get_stream());
    });
}

} // namespace

// 2^26 elements = 128 MiB per buffer
NVBENCH_BENCH(bench_gf16to3_add).add_int64_power_of_two_axis("Elements", nvbench::range(16, 26, 2));
NVBENCH_BENCH(bench_gf16to3_sub).add_int64_power_of_two_axis("Elements", nvbench::range(16, 26, 2));
NVBENCH_BENCH(bench_gf16to3_mul).add_int64_power_of_two_axis("Elements", nvbench::range(16, 26, 2));

NVBENCH_BENCH(bench_gf16to3_matrix_add).add_int64_axis("N", {64, 256, 1024, 4096});
NVBENCH_BENCH(bench_gf16to3_matrix_sub).add_int64_axis("N", {64, 256, 1024, 4096});
NVBENCH_BENCH(bench_gf16to3_matrix_mul).add_int64_axis("N", {64, 128, 256, 512, 1024});
