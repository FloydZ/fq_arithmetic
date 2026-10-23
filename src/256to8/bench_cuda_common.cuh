#pragma once

/// Shared nvbench benchmarks for the CUDA ports of the GF(256) tower fields
/// (256to8, 256to16, 256to24, 256toX). A field provides a traits struct `Tr`:
///     typedef ... T;
///     static void vector_add/vector_sub/vector_mul(T *c, const T *a, const T *b, uint32_t n, cudaStream_t s);
///     static void matrix_add/matrix_sub(T *m1, const T *m2, const T *m3, uint32_t nrows, uint32_t ncols, cudaStream_t s);
///     static void matrix_mul(T *r, const T *A, const T *B, uint32_t n_rows1, uint32_t n_cols1, uint32_t n_cols2, cudaStream_t s);
/// and defines the benchmark functions with
/// `GF256TOWER_CUDA_BENCH_FUNCTIONS(prefix, Tr)`, which creates
///     prefix_add, prefix_sub, prefix_mul                       (axis "Elements")
///     prefix_matrix_add, prefix_matrix_sub, prefix_matrix_mul  (axis "N")
/// that still have to be registered with NVBENCH_BENCH(...).

#include <nvbench/nvbench.cuh>

#include <cstdint>

#include "../helper.cuh"

namespace gf256tower_cuda_bench {

enum class Op { add, sub, mul };

/// element-wise add/sub/mul over `Elements` elements
template<typename Tr, Op op>
void bench_vector(nvbench::state &state) {
    typedef typename Tr::T T;
    const auto n = static_cast<uint32_t>(state.get_int64("Elements"));
    const size_t nbytes = (size_t)n * sizeof(T);
    fq_cuda::DeviceBuffer a(nbytes), b(nbytes), c(nbytes);
    a.fill_random(1);
    b.fill_random(2);

    state.add_element_count(n, "Elements");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        if constexpr (op == Op::add) { Tr::vector_add(c.as<T>(), a.as<T>(), b.as<T>(), n, launch.get_stream()); }
        if constexpr (op == Op::sub) { Tr::vector_sub(c.as<T>(), a.as<T>(), b.as<T>(), n, launch.get_stream()); }
        if constexpr (op == Op::mul) { Tr::vector_mul(c.as<T>(), a.as<T>(), b.as<T>(), n, launch.get_stream()); }
    });
}

/// N x N matrix addition/subtraction
template<typename Tr, Op op>
void bench_matrix_addsub(nvbench::state &state) {
    typedef typename Tr::T T;
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t nbytes = (size_t)n * n * sizeof(T);
    fq_cuda::DeviceBuffer A(nbytes), B(nbytes), C(nbytes);
    A.fill_random(1);
    B.fill_random(2);

    state.add_element_count((size_t)n * n, "Elements");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        if constexpr (op == Op::add) {
            Tr::matrix_add(C.as<T>(), A.as<T>(), B.as<T>(), n, n, launch.get_stream());
        } else {
            Tr::matrix_sub(C.as<T>(), A.as<T>(), B.as<T>(), n, n, launch.get_stream());
        }
    });
}

/// N x N times N x N matrix multiplication
template<typename Tr>
void bench_matrix_mul(nvbench::state &state) {
    typedef typename Tr::T T;
    const auto n = static_cast<uint32_t>(state.get_int64("N"));
    const size_t nbytes = (size_t)n * n * sizeof(T);
    fq_cuda::DeviceBuffer A(nbytes), B(nbytes), C(nbytes);
    A.fill_random(1);
    B.fill_random(2);

    // one multiply-accumulate per (i, j, k)
    state.add_element_count((size_t)n * n * n, "MACs");
    state.add_global_memory_reads<uint8_t>(2 * nbytes);
    state.add_global_memory_writes<uint8_t>(nbytes);
    state.exec([&](nvbench::launch &launch) {
        Tr::matrix_mul(C.as<T>(), A.as<T>(), B.as<T>(), n, n, n, launch.get_stream());
    });
}

} // namespace gf256tower_cuda_bench

#define GF256TOWER_CUDA_BENCH_FUNCTIONS(prefix, Tr)                                                       \
    void prefix##_add(nvbench::state &s) { gf256tower_cuda_bench::bench_vector<Tr, gf256tower_cuda_bench::Op::add>(s); } \
    void prefix##_sub(nvbench::state &s) { gf256tower_cuda_bench::bench_vector<Tr, gf256tower_cuda_bench::Op::sub>(s); } \
    void prefix##_mul(nvbench::state &s) { gf256tower_cuda_bench::bench_vector<Tr, gf256tower_cuda_bench::Op::mul>(s); } \
    void prefix##_matrix_add(nvbench::state &s) {                                                          \
        gf256tower_cuda_bench::bench_matrix_addsub<Tr, gf256tower_cuda_bench::Op::add>(s); }                \
    void prefix##_matrix_sub(nvbench::state &s) {                                                          \
        gf256tower_cuda_bench::bench_matrix_addsub<Tr, gf256tower_cuda_bench::Op::sub>(s); }                \
    void prefix##_matrix_mul(nvbench::state &s) { gf256tower_cuda_bench::bench_matrix_mul<Tr>(s); }
