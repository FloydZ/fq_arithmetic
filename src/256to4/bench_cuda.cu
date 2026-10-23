#include <nvbench/nvbench.cuh>

#include <cstdint>

#include "arith.cu"

namespace {

using elem = gf256to4_cuda::gf256to4;

/// device buffer filled with random bytes
struct DeviceBuffer : fq_cuda::DeviceBuffer {
    explicit DeviceBuffer(const size_t n) : fq_cuda::DeviceBuffer(n) { fill_random(); }
};

enum class Op { add, sub, mul };

/// element-wise gf256to4_add/sub/mul over `Elements` elements
template<Op op>
void bench_gf256to4_vector(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("Elements"));
    const size_t nbytes = (size_t)n * sizeof(elem);
    DeviceBuffer a(nbytes), b(nbytes), c(nbytes);

    state.add_element_count(n, "Elements");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        if constexpr (op == Op::add) { gf256to4_cuda::gf256to4_vector_add(c.as<elem>(), a.as<elem>(), b.as<elem>(), n, launch.get_stream()); }
        if constexpr (op == Op::sub) { gf256to4_cuda::gf256to4_vector_sub(c.as<elem>(), a.as<elem>(), b.as<elem>(), n, launch.get_stream()); }
        if constexpr (op == Op::mul) { gf256to4_cuda::gf256to4_vector_mul(c.as<elem>(), a.as<elem>(), b.as<elem>(), n, launch.get_stream()); }
    });
}

void bench_gf256to4_add(nvbench::state &state) { bench_gf256to4_vector<Op::add>(state); }
void bench_gf256to4_sub(nvbench::state &state) { bench_gf256to4_vector<Op::sub>(state); }
void bench_gf256to4_mul(nvbench::state &state) { bench_gf256to4_vector<Op::mul>(state); }

/// N x N matrix addition/subtraction
template<Op op>
void bench_gf256to4_matrix_addsub(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t nbytes = (size_t)n * n * sizeof(elem);
    DeviceBuffer A(nbytes), B(nbytes), C(nbytes);

    state.add_element_count((size_t)n * n, "Elements");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        if constexpr (op == Op::add) {
            gf256to4_cuda::gf256to4_matrix_add(C.as<elem>(), A.as<elem>(), B.as<elem>(), n, n, launch.get_stream());
        } else {
            gf256to4_cuda::gf256to4_matrix_sub(C.as<elem>(), A.as<elem>(), B.as<elem>(), n, n, launch.get_stream());
        }
    });
}

void bench_gf256to4_matrix_add(nvbench::state &state) { bench_gf256to4_matrix_addsub<Op::add>(state); }
void bench_gf256to4_matrix_sub(nvbench::state &state) { bench_gf256to4_matrix_addsub<Op::sub>(state); }

/// N x N times N x N matrix multiplication
void bench_gf256to4_matrix_mul(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t nbytes = (size_t)n * n * sizeof(elem);
    DeviceBuffer A(nbytes), B(nbytes), C(nbytes);

    // one GF(256^4) multiply-accumulate per (i, j, k)
    state.add_element_count((size_t)n * n * n, "MACs");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        gf256to4_cuda::gf256to4_matrix_mul(C.as<elem>(), A.as<elem>(), B.as<elem>(), n, n, n, launch.get_stream());
    });
}

} // namespace

NVBENCH_BENCH(bench_gf256to4_add).add_int64_power_of_two_axis("Elements", nvbench::range(16, 26, 2));
NVBENCH_BENCH(bench_gf256to4_sub).add_int64_power_of_two_axis("Elements", nvbench::range(16, 26, 2));
NVBENCH_BENCH(bench_gf256to4_mul).add_int64_power_of_two_axis("Elements", nvbench::range(16, 26, 2));

NVBENCH_BENCH(bench_gf256to4_matrix_add).add_int64_axis("N", {64, 256, 1024, 4096});
NVBENCH_BENCH(bench_gf256to4_matrix_sub).add_int64_axis("N", {64, 256, 1024, 4096});
NVBENCH_BENCH(bench_gf256to4_matrix_mul).add_int64_axis("N", {64, 128, 256, 512, 1024});
