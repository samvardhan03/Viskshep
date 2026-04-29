// wst_bindings.cu — Pybind11 bindings for the Vikshep WST C++/CUDA engine.
//
// Dual-path architecture:
//   GPU PATH: Uses WSTEngine<HopperTag, J, Q> templates via DISPATCH_FINGERPRINT
//   CPU PATH: Uses cpu_wst_engine.h — real Radix-2 FFT scattering cascade
//
// When compiled with -DVIKSHEP_CPU_ONLY, all CUDA headers and GPU code paths
// are excluded. The CPU engine handles all (J, Q) configurations at runtime.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// CPU engine — always available
#include "cpu_wst_engine.h"

#ifndef VIKSHEP_CPU_ONLY
// GPU headers — only when CUDA is available
#include "wst_kernel.cuh"
#include "jtfs_kernel.cuh"
#endif

#include <sstream>
#include <stdexcept>

namespace py = pybind11;

struct WSTConfigWrapper {
    int J;
    int Q;
    int depth;
    bool jtfs;
    float l1_norm_psi;
    
    WSTConfigWrapper(int j, int q, int d, bool jtf) : J(j), Q(q), depth(d), jtfs(jtf), l1_norm_psi(0.0f) {}
};

struct JTFSConfigWrapper {
    int J_fr;
    int Q_fr;
    JTFSConfigWrapper(int j, int q) : J_fr(j), Q_fr(q) {}
};

bool cuda_available() {
#ifndef VIKSHEP_CPU_ONLY
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    return (error_id == cudaSuccess && deviceCount > 0);
#else
    return false;
#endif
}

#ifndef VIKSHEP_CPU_ONLY
// ===========================================================================
// DISPATCH_FINGERPRINT — GPU path only. Routes runtime (J, Q) to the
// pre-compiled WSTEngine<HopperTag, J_VAL, Q_VAL> template instantiation.
// ===========================================================================
#define DISPATCH_FINGERPRINT(J_VAL, Q_VAL)                                     \
    if (cfg.J == (J_VAL) && cfg.Q == (Q_VAL)) {                               \
        WSTEngine<HopperTag, J_VAL, Q_VAL> engine;                            \
        engine.initialise(signal_len, batch_size);                             \
        cfg.l1_norm_psi = engine.compute_l1_norm_psi();                        \
                                                                               \
        float* ptr = static_cast<float*>(buf.ptr);                             \
        size_t out_elements = signal_len * batch_size;                         \
        auto result = py::array_t<float>(out_elements);                        \
        py::buffer_info res_buf = result.request();                            \
        float* res_ptr = static_cast<float*>(res_buf.ptr);                     \
                                                                               \
        engine.forward_pass(ptr, engine.d_output,                              \
                            signal_len, batch_size, cfg.depth);                \
                                                                               \
        cudaError_t err = cudaMemcpy(res_ptr, engine.d_output,                 \
            out_elements * sizeof(float), cudaMemcpyDeviceToHost);             \
        if (err != cudaSuccess) {                                              \
            engine.destroy();                                                  \
            throw std::runtime_error(                                          \
                "cudaMemcpyDeviceToHost failed in fingerprint");                \
        }                                                                      \
        engine.destroy();                                                      \
                                                                               \
        if (buf.ndim == 2) {                                                   \
            result.resize({batch_size, signal_len});                            \
        }                                                                      \
        return result;                                                         \
    }
#endif // !VIKSHEP_CPU_ONLY

// ===========================================================================
// CPU fallback — Real Wavelet Scattering Transform
//
// Uses cpu_wst_engine.h which implements:
//   - Radix-2 Cooley-Tukey FFT (unitary normalization)
//   - Analytic Morlet wavelet filter bank (peak ≤ 0.98)
//   - Full depth-m cascade: FFT → Ψ multiply → IFFT → |z| modulus
//
// NO MOCKS. Every operation is the true mathematical transform.
// Accepts any (J, Q) at runtime — no template dispatch needed.
// ===========================================================================
static py::array_t<float> cpu_fingerprint(
    py::buffer_info& buf,
    int signal_len,
    int batch_size,
    WSTConfigWrapper& cfg
) {
    // Build the Morlet filter bank for this (J, Q) configuration
    CPUFilterBank bank = build_cpu_morlet_bank(cfg.J, cfg.Q, signal_len);

    size_t out_elements = static_cast<size_t>(signal_len) * batch_size;
    auto result = py::array_t<float>(out_elements);
    py::buffer_info res_buf = result.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);
    float* in_ptr  = static_cast<float*>(buf.ptr);

    // Execute the real scattering cascade
    cpu_wst_forward(in_ptr, res_ptr, signal_len, batch_size,
                    cfg.depth, bank, cfg.l1_norm_psi);

    if (buf.ndim == 2) {
        result.resize({batch_size, signal_len});
    }
    return result;
}

py::array_t<float> fingerprint(py::array_t<float> signal, WSTConfigWrapper& cfg) {
    py::buffer_info buf = signal.request();
    
    int signal_len = 0;
    int batch_size = 1;
    
    if (buf.ndim == 1) {
        signal_len = buf.shape[0];
    } else if (buf.ndim == 2) {
        batch_size = buf.shape[0];
        signal_len = buf.shape[1];
    } else {
        throw std::runtime_error("Input signal must be 1D or 2D");
    }

#ifndef VIKSHEP_CPU_ONLY
    // ===================================================================
    // GPU PATH — Dynamic Template Dispatch
    // ===================================================================
    if (cuda_available()) {
        DISPATCH_FINGERPRINT(4,   4)
        DISPATCH_FINGERPRINT(6,   8)
        DISPATCH_FINGERPRINT(8,  16)
        DISPATCH_FINGERPRINT(10, 16)
        DISPATCH_FINGERPRINT(12, 16)
        DISPATCH_FINGERPRINT(8,   8)
        DISPATCH_FINGERPRINT(10,  8)

        // No GPU config matched — fall through to CPU engine
    }
#endif

    // ===================================================================
    // CPU PATH — Real Wavelet Scattering Transform (any J, Q)
    //
    // This is NOT a mock. It executes the full Radix-2 FFT scattering
    // cascade with Morlet wavelets. Accepts any (J, Q) at runtime.
    // ===================================================================
    return cpu_fingerprint(buf, signal_len, batch_size, cfg);
}

py::list scattering_paths(py::array_t<float> signal, WSTConfigWrapper& cfg) {
    py::list paths;
    auto fp = fingerprint(signal, cfg);
    paths.append(fp);
    return paths;
}

PYBIND11_MODULE(_vikshep_core, m) {
    m.doc() = "Vikshep C++/CUDA Mathematical Primitives — WST Engine";
    
    py::class_<WSTConfigWrapper>(m, "WSTConfig")
        .def(py::init<int, int, int, bool>(),
             py::arg("J"),
             py::arg("Q"),
             py::arg("depth"),
             py::arg("jtfs") = false)
        .def_readwrite("J", &WSTConfigWrapper::J)
        .def_readwrite("Q", &WSTConfigWrapper::Q)
        .def_readwrite("depth", &WSTConfigWrapper::depth)
        .def_readwrite("jtfs", &WSTConfigWrapper::jtfs)
        .def_readwrite("l1_norm_psi", &WSTConfigWrapper::l1_norm_psi);
        
    py::class_<JTFSConfigWrapper>(m, "JTFSConfig")
        .def(py::init<int, int>(),
             py::arg("J_fr"),
             py::arg("Q_fr"))
        .def_readwrite("J_fr", &JTFSConfigWrapper::J_fr)
        .def_readwrite("Q_fr", &JTFSConfigWrapper::Q_fr);
        
    m.def("fingerprint", &fingerprint, "Compute WST/JTFS fingerprint");
    m.def("scattering_paths", &scattering_paths, "Return scattering paths as a list of arrays");
    m.def("cuda_available", &cuda_available, "Return True if a CUDA device is accessible");
}
