#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <random>
#include <vector>

#include "arith.cu"

namespace {

/// device buffer filled with random canonical elements
struct DeviceBuffer : fq_cuda::DeviceBuffer {
    explicit DeviceBuffer(const size_t n) : fq_cuda::DeviceBuffer(n) {
        std::mt19937 gen(1);
        std::vector<uint8_t> host(n);
        for (auto &x : host) { x = gen() % gf31_cuda::GF31_P; }
        upload(host);
    }
};

enum class Op { add, sub, mul };

template<Op op>
void gf31_vector_op(uint8_t *c, const uint8_t *a, const uint8_t *b, const uint32_t n, cudaStream_t s) {
    if constexpr (op == Op::add) { gf31_cuda::gf31_vector_add(c, a, b, n, s); }
    if constexpr (op == Op::sub) { gf31_cuda::gf31_vector_sub(c, a, b, n, s); }
    if constexpr (op == Op::mul) { gf31_cuda::gf31_vector_mul(c, a, b, n, s); }
}

/// element-wise gf31_add/sub/mul over `Elements` elements (1 byte each)
template<Op op>
void bench_gf31_vector(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("Elements"));
    DeviceBuffer a(n), b(n), c(n);

    state.add_element_count(n, "Elements");
    state.add_global_memory_reads<uint8_t>(2 * (size_t)n);
    state.add_global_memory_writes<uint8_t>(n);
    state.exec([&](nvbench::launch &launch) {
        gf31_vector_op<op>(c.ptr, a.ptr, b.ptr, n, launch.get_stream());
    });
}

void bench_gf31_add(nvbench::state &state) { bench_gf31_vector<Op::add>(state); }
void bench_gf31_sub(nvbench::state &state) { bench_gf31_vector<Op::sub>(state); }
void bench_gf31_mul(nvbench::state &state) { bench_gf31_vector<Op::mul>(state); }

/// N x N matrix addition/subtraction
template<Op op>
void bench_gf31_matrix_addsub(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t nbytes = gf31_cuda::gf31_matrix_bytes_size(n, n);
    DeviceBuffer A(nbytes), B(nbytes), C(nbytes);

    state.add_element_count((size_t)n * n, "Elements");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        if constexpr (op == Op::add) {
            gf31_cuda::gf31_matrix_add(C.ptr, A.ptr, B.ptr, n, n, launch.get_stream());
        } else {
            gf31_cuda::gf31_matrix_sub(C.ptr, A.ptr, B.ptr, n, n, launch.get_stream());
        }
    });
}

void bench_gf31_matrix_add(nvbench::state &state) { bench_gf31_matrix_addsub<Op::add>(state); }
void bench_gf31_matrix_sub(nvbench::state &state) { bench_gf31_matrix_addsub<Op::sub>(state); }

/// N x N times N x N matrix multiplication
void bench_gf31_matrix_mul(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t nbytes = gf31_cuda::gf31_matrix_bytes_size(n, n);
    DeviceBuffer A(nbytes), B(nbytes), C(nbytes);

    // one GF(31) multiply-accumulate per (i, j, k)
    state.add_element_count((size_t)n * n * n, "MACs");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        gf31_cuda::gf31_matrix_mul(C.ptr, A.ptr, B.ptr, n, n, n, launch.get_stream());
    });
}

} // namespace

NVBENCH_BENCH(bench_gf31_add).add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
NVBENCH_BENCH(bench_gf31_sub).add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
NVBENCH_BENCH(bench_gf31_mul).add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

NVBENCH_BENCH(bench_gf31_matrix_add).add_int64_axis("N", {64, 256, 1024, 4096});
NVBENCH_BENCH(bench_gf31_matrix_sub).add_int64_axis("N", {64, 256, 1024, 4096});
NVBENCH_BENCH(bench_gf31_matrix_mul).add_int64_axis("N", {64, 128, 256, 512, 1024});
