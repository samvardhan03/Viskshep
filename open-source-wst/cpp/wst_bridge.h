// wst_bridge.h — C++ FFI entry point exposed to the Rust Phase 2 orchestrator
// via the `cxx` crate. This header is the ONLY interface that crosses the
// C++↔Rust language boundary. All GPU tensor payloads stay on-device (VRAM);
// only opaque CUdeviceptr handles (u64) are transmitted.
//
// Rust crate: crates/omni-wst-sys
// TDD Reference: Section 2.1 — Zero-Cost cxx FFI Bridge

#pragma once
#include "rust/cxx.h"
#include <cstdint>

// ---------------------------------------------------------------------------
// WSTResult — POD struct ABI-validated by cxx at compile time.
// All fields are plain uint64_t to guarantee identical representation in both
// C++ and Rust without any padding surprises.
// ---------------------------------------------------------------------------
struct WSTResult {
    /// CUdeviceptr pointing to the scattering coefficient tensor on GPU VRAM.
    /// The Rust orchestrator passes this directly to Arrow Plasma as an
    /// ObjectID source — no host-side memcpy is ever performed.
    uint64_t fingerprint_ptr;

    /// Total number of float32 scattering coefficients in the tensor.
    /// Allows the Rust side to know the buffer dimensions without dereferencing
    /// the GPU pointer.
    uint64_t coeff_count;

    /// Wall-clock execution time of the CUDA kernel cascade in microseconds.
    /// Used by the FinOps autoscaler (Section 4.2.2) for GPU cost attribution.
    uint64_t exec_time_us;
};

// ---------------------------------------------------------------------------
// run_wst_pipeline — Primary FFI entry point.
//
// Parameters:
//   input_plasma_ptr  — Raw mmap pointer from the Apache Arrow Plasma shared-
//                       memory store. This is a host-side address that has
//                       been pre-registered with CUDA via cudaHostRegister so
//                       it can be accessed from device kernels via UVA.
//   signal_len        — Number of float32 samples per signal in the batch.
//   batch_size        — Number of signals in the input batch.
//   J                 — Maximum wavelet scale (2^J samples of temporal support).
//   Q                 — Wavelets per octave (frequency resolution).
//   depth             — Scattering cascade depth (m in the Lipschitz bound).
//   use_jtfs          — If true, activates the Joint Time-Frequency Scattering
//                       phase-recovery pass using parallel CUDA streams.
//
// Returns a WSTResult with a stable CUdeviceptr to the output tensor.
// The caller (Rust orchestrator) owns the lifetime of this pointer and must
// call free_wst_result() when the tensor is no longer needed.
// ---------------------------------------------------------------------------
WSTResult run_wst_pipeline(
    uint64_t input_plasma_ptr,
    int32_t  signal_len,
    int32_t  batch_size,
    int32_t  J,
    int32_t  Q,
    int32_t  depth,
    bool     use_jtfs
);

// ---------------------------------------------------------------------------
// free_wst_result — Releases the device memory referenced by WSTResult.
// Must be called by the Rust orchestrator after the fingerprint has been
// written to the Plasma store. Failing to call this causes GPU memory leaks.
// ---------------------------------------------------------------------------
void free_wst_result(WSTResult result);
