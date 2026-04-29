#include "memory_staging.cuh"
#include <stdexcept>

MemoryStaging::MemoryStaging() : h_buffer(nullptr), d_buffer(nullptr), allocated_bytes(0) {}

MemoryStaging::~MemoryStaging() {
    free_pinned();
}

void MemoryStaging::allocate_pinned(size_t bytes) {
    if (allocated_bytes > 0) {
        free_pinned();
    }
    
    cudaError_t err1 = cudaMallocHost(&h_buffer, bytes);
    if (err1 != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pinned memory (cudaMallocHost)");
    }
    
    cudaError_t err2 = cudaMalloc(&d_buffer, bytes);
    if (err2 != cudaSuccess) {
        cudaFreeHost(h_buffer);
        h_buffer = nullptr;
        throw std::runtime_error("Failed to allocate device memory (cudaMalloc)");
    }
    
    allocated_bytes = bytes;
}

void MemoryStaging::free_pinned() {
    if (h_buffer) {
        cudaFreeHost(h_buffer);
        h_buffer = nullptr;
    }
    if (d_buffer) {
        cudaFree(d_buffer);
        d_buffer = nullptr;
    }
    allocated_bytes = 0;
}

void MemoryStaging::async_h2d(cudaStream_t stream) {
    if (allocated_bytes == 0) return;
    cudaMemcpyAsync(d_buffer, h_buffer, allocated_bytes, cudaMemcpyHostToDevice, stream);
}

void MemoryStaging::async_d2h(cudaStream_t stream) {
    if (allocated_bytes == 0) return;
    cudaMemcpyAsync(h_buffer, d_buffer, allocated_bytes, cudaMemcpyDeviceToHost, stream);
}

uint64_t MemoryStaging::get_uva_handle() const {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_buffer));
}
