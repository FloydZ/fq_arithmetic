#pragma once

/// Shared helpers for the CUDA implementations in src/<field>/arith.cu and
/// their googletest tests / nvbench benchmarks.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

/// aborts with a message if a CUDA runtime call fails
#define FQ_CUDA_CHECK(call)                                                 \
    do {                                                                    \
        const cudaError_t fq_cuda_err__ = (call);                           \
        if (fq_cuda_err__ != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",                 \
                    cudaGetErrorString(fq_cuda_err__), __FILE__, __LINE__); \
            abort();                                                        \
        }                                                                   \
    } while (0)

namespace fq_cuda {

/// number of threads per block used by the element-wise kernels
constexpr uint32_t BLOCK_SIZE = 256;

/// \param work[in]: number of independent work items
/// \param block[in]: threads per block
/// \return number of blocks for a grid-stride loop over `work` items (>= 1)
static inline
uint32_t grid_size(const size_t work,
                   const uint32_t block = BLOCK_SIZE) {
    const size_t blocks = (work + block - 1) / block;
    const size_t max_blocks = 1u << 20u;
    return (uint32_t)(blocks < max_blocks ? (blocks ? blocks : 1) : max_blocks);
}

/// c[i] = op(a[i], b[i]) for i < n, grid-stride loop.
/// `Op` must provide `__device__ T operator()(const T &, const T &) const`.
template<typename T, typename Op>
__global__ void elementwise_kernel(T *c,
                                   const T *a,
                                   const T *b,
                                   const size_t n,
                                   const Op op) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = op(a[i], b[i]);
    }
}

/// enqueues c[i] = op(a[i], b[i]) for i < n on `stream`. c may alias a/b.
template<typename T, typename Op>
static inline
void elementwise(T *c,
                 const T *a,
                 const T *b,
                 const size_t n,
                 const Op op,
                 cudaStream_t stream) {
    if (n == 0) {
        return;
    }
    elementwise_kernel<T, Op><<<grid_size(n), BLOCK_SIZE, 0, stream>>>(c, a, b, n, op);
    FQ_CUDA_CHECK(cudaGetLastError());
}

/// RAII device buffer for tests and benchmarks (host side only)
struct DeviceBuffer {
    uint8_t *ptr = nullptr;
    size_t size = 0;

    explicit DeviceBuffer(const size_t n) : size(n) {
        FQ_CUDA_CHECK(cudaMalloc(&ptr, n ? n : 1));
    }
    explicit DeviceBuffer(const std::vector<uint8_t> &host) : DeviceBuffer(host.size()) {
        upload(host);
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    /// \return the buffer reinterpreted as a device pointer to T
    template<typename T> T *as() const { return reinterpret_cast<T *>(ptr); }

    void upload(const std::vector<uint8_t> &host) {
        if (host.size() > size) {
            fprintf(stderr, "DeviceBuffer::upload: %zu > %zu bytes\n", host.size(), size);
            abort();
        }
        FQ_CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size(), cudaMemcpyHostToDevice));
    }

    std::vector<uint8_t> download() const {
        std::vector<uint8_t> host(size);
        FQ_CUDA_CHECK(cudaMemcpy(host.data(), ptr, size, cudaMemcpyDeviceToHost));
        return host;
    }

    /// fills the buffer with random bytes (for benchmarks)
    void fill_random(const uint32_t seed = 1) {
        std::mt19937 gen(seed);
        std::vector<uint8_t> host(size);
        for (auto &x : host) { x = gen() & 0xFFu; }
        upload(host);
    }
};

} // namespace fq_cuda
