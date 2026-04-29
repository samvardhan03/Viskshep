// wst_bridge.cu — CUDA implementation of the cxx FFI entry point.
// Compiled by nvcc as part of the `omni_wst_bridge` shared library target.
// This file is intentionally separate from wst_bindings.cu (the Python layer)
// so the two distribution targets (PyPI wheel and Rust sys crate) never interfere.
//
// TDD Reference: Section 2.1 — Zero-Cost cxx FFI Bridge
//
// ARCHITECTURE NOTE — Dynamic Template Instantiation Dispatcher
// =============================================================
// C++ templates are resolved at compile time. CUDA kernels parameterised by
// (J, Q) must be compiled for every configuration the user may request at
// runtime. We pre-instantiate a fixed dispatch matrix of (J, Q) pairs and
// route the runtime integers through the DISPATCH_WST_ENGINE macro. Adding a
// new (J, Q) configuration requires only a one-line macro invocation —
// no other code changes are necessary.

#include "wst_bridge.h"
#include "wst_kernel.cuh"
#include "jtfs_kernel.cuh"
#include "memory_staging.cuh"

#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Internal helper: map a host-side Plasma mmap ptr to a pinned CUDA buffer.
// The Plasma store allocates shared memory via shm_open / mmap. Before the
// kernel can read it, we register the pages as CUDA host memory so UVA
// can address them from device code without an explicit H2D memcpy.
// ---------------------------------------------------------------------------
static float* register_plasma_buffer(uint64_t plasma_ptr, size_t byte_count) {
    void* host_ptr = reinterpret_cast<void*>(plasma_ptr);
    cudaError_t err = cudaHostRegister(host_ptr, byte_count, cudaHostRegisterPortable);
    if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
        throw std::runtime_error(
            std::string("cudaHostRegister failed: ") + cudaGetErrorString(err));
    }
    return static_cast<float*>(host_ptr);
}

static void unregister_plasma_buffer(uint64_t plasma_ptr) {
    void* host_ptr = reinterpret_cast<void*>(plasma_ptr);
    // Best-effort unregister — if it was already registered elsewhere, ignore.
    cudaHostUnregister(host_ptr);
}

// ===========================================================================
// DISPATCH_WST_ENGINE — Dynamic Template Instantiation Dispatcher Macro
//
// For each pre-compiled (J, Q) pair, this macro:
//   1. Checks if the runtime `j` and `q` values match (J_VAL, Q_VAL).
//   2. Branches on `use_jtfs` to instantiate the correct engine class.
//   3. Executes H2D transfer, forward pass, and builds the WSTResult.
//   4. Returns immediately on match — no fallthrough to the next case.
//
// Usage:  DISPATCH_WST_ENGINE(8, 16)   // pre-compiles (J=8, Q=16) templates
// ===========================================================================
#define DISPATCH_WST_ENGINE(J_VAL, Q_VAL)                                      \
    if (j == (J_VAL) && q == (Q_VAL)) {                                        \
        if (!use_jtfs) {                                                       \
            /* Standard WST path */                                            \
            WSTEngine<HopperTag, J_VAL, Q_VAL> engine;                         \
            engine.initialise(signal_len, batch_size);                         \
                                                                               \
            float* d_input = nullptr;                                          \
            cudaMalloc(&d_input, input_bytes);                                 \
            cudaStream_t h2d_stream;                                           \
            cudaStreamCreate(&h2d_stream);                                     \
            cudaMemcpyAsync(d_input, h_input, input_bytes,                     \
                            cudaMemcpyHostToDevice, h2d_stream);               \
            cudaStreamSynchronize(h2d_stream);                                 \
            cudaStreamDestroy(h2d_stream);                                     \
                                                                               \
            engine.forward_pass(d_input, d_output, signal_len,                 \
                                batch_size, depth);                            \
            engine.destroy();                                                  \
            cudaFree(d_input);                                                 \
        } else {                                                               \
            /* JTFS path — separable 2D conv on dual CUDA streams */           \
            JTFSEngine<HopperTag, J_VAL, Q_VAL, 1> jtfs_engine;               \
            jtfs_engine.initialise(signal_len, batch_size);                    \
                                                                               \
            float* d_input = nullptr;                                          \
            cudaMalloc(&d_input, input_bytes);                                 \
            cudaStream_t h2d_stream;                                           \
            cudaStreamCreate(&h2d_stream);                                     \
            cudaMemcpyAsync(d_input, h_input, input_bytes,                     \
                            cudaMemcpyHostToDevice, h2d_stream);               \
            cudaStreamSynchronize(h2d_stream);                                 \
            cudaStreamDestroy(h2d_stream);                                     \
                                                                               \
            jtfs_engine.forward_pass(d_input, d_output, signal_len,            \
                                     batch_size, depth);                       \
            jtfs_engine.destroy();                                             \
            cudaFree(d_input);                                                 \
        }                                                                      \
        dispatched = true;                                                     \
    }

// ---------------------------------------------------------------------------
// run_wst_pipeline — Primary FFI entry point called by the Rust orchestrator.
//
// The (J, Q) values supplied at runtime are routed through the dispatch matrix
// below. If no pre-compiled instantiation matches, a std::invalid_argument
// exception is thrown with a diagnostic message listing all supported configs.
// ---------------------------------------------------------------------------
WSTResult run_wst_pipeline(
    uint64_t input_plasma_ptr,
    int32_t  signal_len,
    int32_t  batch_size,
    int32_t  j,
    int32_t  q,
    int32_t  depth,
    bool     use_jtfs
) {
    // --- Timing start ---
    auto t_start = std::chrono::high_resolution_clock::now();

    // --- Validate parameters ---
    if (signal_len <= 0 || batch_size <= 0 || j <= 0 || q <= 0 || depth <= 0) {
        throw std::runtime_error("run_wst_pipeline: invalid configuration parameters");
    }

    const size_t input_bytes  = static_cast<size_t>(signal_len) * batch_size * sizeof(float);
    const size_t output_elems = static_cast<size_t>(signal_len) * batch_size;
    const size_t output_bytes = output_elems * sizeof(float);

    // --- Register Plasma shared-memory buffer as CUDA host memory ---
    float* h_input = register_plasma_buffer(input_plasma_ptr, input_bytes);

    // --- Allocate persistent device memory for the output tensor ---
    // The Rust caller owns this pointer. It must call free_wst_result() when done.
    float* d_output = nullptr;
    cudaError_t err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess) {
        unregister_plasma_buffer(input_plasma_ptr);
        throw std::runtime_error(
            std::string("cudaMalloc for output tensor failed: ") + cudaGetErrorString(err));
    }

    // ===================================================================
    // DYNAMIC DISPATCH MATRIX
    //
    // Each DISPATCH_WST_ENGINE(J, Q) invocation pre-compiles the full
    // WSTEngine<HopperTag, J, Q> and JTFSEngine<HopperTag, J, Q, 1>
    // template instantiations. To add support for a new (J, Q) pair,
    // simply append a new macro call below — no other changes required.
    // ===================================================================
    bool dispatched = false;

    DISPATCH_WST_ENGINE(8,  16)
    DISPATCH_WST_ENGINE(10, 16)
    DISPATCH_WST_ENGINE(12, 16)
    DISPATCH_WST_ENGINE(8,   8)
    DISPATCH_WST_ENGINE(10,  8)

    if (!dispatched) {
        // Cleanup before throwing
        cudaFree(d_output);
        unregister_plasma_buffer(input_plasma_ptr);

        std::ostringstream oss;
        oss << "run_wst_pipeline: unsupported (J=" << j << ", Q=" << q
            << ") configuration. Pre-compiled dispatch matrix supports: "
               "(8,16), (10,16), (12,16), (8,8), (10,8). "
               "Add a DISPATCH_WST_ENGINE(" << j << ", " << q
            << ") invocation in wst_bridge.cu to enable this configuration.";
        throw std::invalid_argument(oss.str());
    }

    // Ensure all device work is complete before returning the output pointer
    cudaDeviceSynchronize();

    // Unregister the Plasma pages — the Plasma store manages their lifetime
    unregister_plasma_buffer(input_plasma_ptr);

    // --- Timing end ---
    auto t_end = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());

    return WSTResult {
        /* fingerprint_ptr */ reinterpret_cast<uint64_t>(d_output),
        /* coeff_count     */ output_elems,
        /* exec_time_us    */ elapsed_us
    };
}

// ---------------------------------------------------------------------------
// free_wst_result — Releases the device tensor allocated by run_wst_pipeline.
// The Rust orchestrator must call this after writing the tensor to Plasma.
// ---------------------------------------------------------------------------
void free_wst_result(WSTResult result) {
    if (result.fingerprint_ptr != 0) {
        cudaFree(reinterpret_cast<void*>(result.fingerprint_ptr));
    }
}
