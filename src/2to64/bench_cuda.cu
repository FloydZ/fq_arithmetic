#include <nvbench/nvbench.cuh>

#include <cstdint>

#include "arith.cu"

namespace {

typedef gf2to64_cuda::gf2to64 elem_t;

/// device buffer of `n` random field elements
struct RandomElements : fq_cuda::DeviceBuffer {
    explicit RandomElements(const size_t n) : fq_cuda::DeviceBuffer(n * sizeof(elem_t)) { fill_random(); }
    elem_t *get() const { return as<elem_t>(); }
};

enum class Op { add, sub, mul };

/// element-wise gf2to64_add/sub/mul over `Elements` elements
template<Op op>
void bench_gf2to64_vector(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("Elements"));
    RandomElements a(n), b(n), c(n);

    state.add_element_count(n, "Elements");
    state.add_global_memory_reads<elem_t>(2 * (size_t)n);
    state.add_global_memory_writes<elem_t>(n);
    state.exec([&](nvbench::launch &launch) {
        if constexpr (op == Op::add) { gf2to64_cuda::gf2to64_vector_add(c.get(), a.get(), b.get(), n, launch.get_stream()); }
        if constexpr (op == Op::sub) { gf2to64_cuda::gf2to64_vector_sub(c.get(), a.get(), b.get(), n, launch.get_stream()); }
        if constexpr (op == Op::mul) { gf2to64_cuda::gf2to64_vector_mul(c.get(), a.get(), b.get(), n, launch.get_stream()); }
    });
}

void bench_gf2to64_add(nvbench::state &state) { bench_gf2to64_vector<Op::add>(state); }
void bench_gf2to64_sub(nvbench::state &state) { bench_gf2to64_vector<Op::sub>(state); }
void bench_gf2to64_mul(nvbench::state &state) { bench_gf2to64_vector<Op::mul>(state); }

/// N x N matrix addition/subtraction
template<Op op>
void bench_gf2to64_matrix_addsub(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t elements = (size_t)n * n;
    RandomElements A(elements), B(elements), C(elements);

    state.add_element_count(elements, "Elements");
    state.add_global_memory_reads<elem_t>(2 * elements);
    state.add_global_memory_writes<elem_t>(elements);
    state.exec([&](nvbench::launch &launch) {
        if constexpr (op == Op::add) {
            gf2to64_cuda::gf2to64_matrix_add(C.get(), A.get(), B.get(), n, n, launch.get_stream());
        } else {
            gf2to64_cuda::gf2to64_matrix_sub(C.get(), A.get(), B.get(), n, n, launch.get_stream());
        }
    });
}

void bench_gf2to64_matrix_add(nvbench::state &state) { bench_gf2to64_matrix_addsub<Op::add>(state); }
void bench_gf2to64_matrix_sub(nvbench::state &state) { bench_gf2to64_matrix_addsub<Op::sub>(state); }

/// N x N times N x N matrix multiplication
void bench_gf2to64_matrix_mul(nvbench::state &state) {
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t elements = (size_t)n * n;
    RandomElements A(elements), B(elements), C(elements);

    // one GF(2^64) multiply-accumulate per (i, j, k)
    state.add_element_count((size_t)n * n * n, "MACs");
    state.add_global_memory_reads<elem_t>(2 * elements);
    state.add_global_memory_writes<elem_t>(elements);
    state.exec([&](nvbench::launch &launch) {
        gf2to64_cuda::gf2to64_matrix_mul(C.get(), A.get(), B.get(), n, n, n, launch.get_stream());
    });
}

} // namespace

NVBENCH_BENCH(bench_gf2to64_add).add_int64_power_of_two_axis("Elements", nvbench::range(16, 24, 4));
NVBENCH_BENCH(bench_gf2to64_sub).add_int64_power_of_two_axis("Elements", nvbench::range(16, 24, 4));
NVBENCH_BENCH(bench_gf2to64_mul).add_int64_power_of_two_axis("Elements", nvbench::range(16, 24, 4));

NVBENCH_BENCH(bench_gf2to64_matrix_add).add_int64_axis("N", {64, 256, 1024, 4096});
NVBENCH_BENCH(bench_gf2to64_matrix_sub).add_int64_axis("N", {64, 256, 1024, 4096});
NVBENCH_BENCH(bench_gf2to64_matrix_mul).add_int64_axis("N", {64, 128, 256, 512, 1024});
