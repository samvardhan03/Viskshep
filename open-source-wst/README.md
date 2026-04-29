# Vikshep: High-Performance Wavelet Scattering Primitives

**Vectorized Invariant Kernels for Scattering & High-performance Extraction Pipelines**

[![License](https://img.shields.io/badge/License-Apache_2.0-teal.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## Overview

**Vikshep** is a domain-agnostic Wavelet Scattering Transform (WST) engine that ships a **Zero-Copy C++/CUDA kernel** alongside a mathematically rigorous **CPU fallback** built on a Radix-2 Cooley-Tukey FFT with unitary normalization.

The scattering cascade constructs translation-invariant, Lipschitz-continuous signal representations:

$$S_0 x(t) = x * \phi(t) \qquad \text{(local average)}$$

$$S_1 x(t, \lambda_1) = |x * \psi_{\lambda_1}| * \phi(t) \qquad \text{(energy per band)}$$

$$S_2 x(t, \lambda_1, \lambda_2) = ||x * \psi_{\lambda_1}| * \psi_{\lambda_2}| * \phi(t) \qquad \text{(transient energy)}$$

The Lipschitz constant of the cascade satisfies:

$$\|Sx - Sy\|_2 \;\leq\; (0.98)^m \cdot \|x - y\|_2$$

where $m$ is the scattering depth and $0.98$ is the peak $\ell_1$ norm of the analytic Morlet filter bank.

---

## The Engine

| Layer | Technology | Details |
|-------|-----------|---------|
| **GPU Path** | CUDA C++ Templates | `WSTEngine<HopperTag, J, Q>` dispatched via `DISPATCH_FINGERPRINT` macro |
| **CPU Path** | Pure C++ (Radix-2 FFT) | `cpu_wst_forward()` — unitary Cooley-Tukey FFT, Morlet convolution, cascaded modulus |
| **Bindings** | pybind11 | Zero-copy NumPy ↔ C++ buffer protocol via `py::array_t<float>` |
| **Build** | scikit-build-core + CMake | Auto-detects CUDA; falls back to CPU-only `.so` on non-NVIDIA systems |

**No mocks. No hardcoded data. No scalar multipliers.** Every operation is the true mathematical transform.

---

## Installation

```bash
pip install vikshep
```

### From Source (with C++ compilation)

```bash
git clone https://github.com/samvardhan03/OmniPulse.git
cd OmniPulse/open-source-wst
pip install -e ".[dev]"
```

---

## Quick Start

### Native C++/CUDA Engine

```python
from vikshep import NativeWSTExtractor
import numpy as np

engine = NativeWSTExtractor(J=8, Q=16, depth=2)
signal = np.random.randn(4096).astype(np.float32)
coefficients = engine.fingerprint(signal)

print(f"Shape: {coefficients.shape}")       # (4096,)
print(f"L1 norm ψ: {engine.l1_norm_psi}")   # 0.98
print(f"CUDA: {engine.cuda_available()}")    # True/False
```

### Kymatio-Based Extractor

```python
from vikshep import WaveletScatteringExtractor
import numpy as np

extractor = WaveletScatteringExtractor(J=6, Q=(8, 1), sampling_rate=256.0)
signal = np.random.randn(4, 1024).astype(np.float32)  # (B, T)
result = extractor.transform(signal)

print(result.coefficients.shape)  # (4, P, T/2^J)
print(f"SNR: {result.snr_db:.2f} dB")
```

### MCP Server

```bash
vikshep-server
```

Exposes the `execute_wst` tool over stdio transport for agentic orchestrators.

---

## Architecture

```
vikshep/
├── _vikshep_core.cpython-*.so   ← Compiled C++/CUDA extension
├── core.py                      ← NativeWSTExtractor + WaveletScatteringExtractor
├── reduction.py                 ← PCA dimensionality reduction
├── foundation.py                ← TF-C contrastive alignment (PyTorch)
├── mcp_server.py                ← MCP tool server (FastMCP)
├── io.py                        ← .npy I/O utilities
└── utils.py                     ← SNR, anomaly detection helpers
```

---

## Mathematical Guarantees

The test suite rigorously validates:

- **Lipschitz Continuity**: $\|Sx - Sy\| \leq (0.98)^m \cdot \|x - y\|$ verified over 200 adversarial trials per depth
- **Exponential Decay**: Empirical Lipschitz ratio monotonically decreases with cascade depth
- **Determinism**: Identical inputs produce bit-identical outputs
- **Cross-Config Isolation**: Different $(J, Q)$ pairs produce distinct filter banks

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

Part of the [OmniPulse](https://github.com/samvardhan03/OmniPulse) monorepo.
