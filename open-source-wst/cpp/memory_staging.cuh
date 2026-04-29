#ifndef MEMORY_STAGING_CUH
#define MEMORY_STAGING_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <stddef.h>

class MemoryStaging {
public:
    MemoryStaging();
    ~MemoryStaging();

    void allocate_pinned(size_t bytes);
    void free_pinned();
    
    void async_h2d(cudaStream_t stream);
    void async_d2h(cudaStream_t stream);
    
    uint64_t get_uva_handle() const;

    // Public access for engine bindings
    float* h_buffer;
    float* d_buffer;
    size_t allocated_bytes;
};

#endif // MEMORY_STAGING_CUH
