// cpu_wst_engine.h — Pure C++ CPU implementation of the Wavelet Scattering
// Transform. Zero external dependencies beyond <complex>, <vector>, <cmath>.
//
// Implements:
//   1. Radix-2 Cooley-Tukey FFT (unitary normalization: 1/√N on both dirs)
//   2. Analytic Morlet wavelet filter bank in the frequency domain
//   3. Depth-m scattering cascade: FFT → multiply → IFFT → |z| modulus
//
// NO MOCKS. Every operation is the real mathematical transform.

#ifndef CPU_WST_ENGINE_H
#define CPU_WST_ENGINE_H

#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using cfloat = std::complex<float>;
static constexpr float PI = 3.14159265358979323846f;

// ===========================================================================
// Radix-2 Cooley-Tukey FFT — in-place, unitary normalization
// ===========================================================================

static inline size_t next_power_of_2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Bit-reversal permutation
static void bit_reverse(std::vector<cfloat>& x) {
    size_t N = x.size();
    for (size_t i = 1, j = 0; i < N; ++i) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
}

// In-place Radix-2 Cooley-Tukey FFT
// inverse=false: forward DFT with 1/√N normalization
// inverse=true:  inverse DFT with 1/√N normalization
static void cpu_fft_inplace(std::vector<cfloat>& x, bool inverse) {
    size_t N = x.size();
    if (N == 0 || (N & (N - 1)) != 0) {
        throw std::runtime_error("cpu_fft: length must be a power of 2");
    }

    bit_reverse(x);

    // Butterfly stages
    for (size_t len = 2; len <= N; len <<= 1) {
        float angle = (inverse ? 1.0f : -1.0f) * 2.0f * PI / static_cast<float>(len);
        cfloat wlen(std::cos(angle), std::sin(angle));

        for (size_t i = 0; i < N; i += len) {
            cfloat w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                cfloat u = x[i + j];
                cfloat v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    // Unitary normalization: multiply by 1/√N
    float norm = 1.0f / std::sqrt(static_cast<float>(N));
    for (auto& val : x) {
        val *= norm;
    }
}

static std::vector<cfloat> cpu_fft(const std::vector<cfloat>& input) {
    auto x = input;
    cpu_fft_inplace(x, false);
    return x;
}

static std::vector<cfloat> cpu_ifft(const std::vector<cfloat>& input) {
    auto x = input;
    cpu_fft_inplace(x, true);
    return x;
}

// ===========================================================================
// Analytic Morlet Wavelet Filter Bank — frequency domain
//
// Ψ_λ(ω) = exp(-0.5 * (σ * (ω - ξ))²)
// where: ξ = π * 2^{-λ/Q}  (center frequency)
//        σ = 0.8 * 2^{λ/Q}  (bandwidth — inversely proportional to ξ)
//        λ ∈ [0, J*Q)        (wavelet index across all scales and octaves)
//
// Each wavelet is scaled so that max|Ψ_λ(ω)| ≤ psi_peak, ensuring the
// Lipschitz constant of each convolution layer is bounded by psi_peak.
// ===========================================================================

struct CPUFilterBank {
    std::vector<std::vector<float>> filters;  // [n_wavelets][signal_len] real-valued
    float l1_norm_psi;                        // max peak across all wavelets
    int n_wavelets;
};

static CPUFilterBank build_cpu_morlet_bank(int J, int Q, int signal_len) {
    CPUFilterBank bank;
    bank.n_wavelets = J * Q;
    bank.filters.resize(bank.n_wavelets, std::vector<float>(signal_len, 0.0f));
    bank.l1_norm_psi = 0.0f;

    // Peak scaling factor — ensures ||Ψ||_∞ < 1 for Lipschitz stability
    const float psi_peak = 0.98f;

    for (int lam = 0; lam < bank.n_wavelets; ++lam) {
        float ratio = static_cast<float>(lam) / static_cast<float>(Q);
        float xi    = PI * std::pow(2.0f, -ratio);            // center frequency
        float sigma = 0.8f * std::pow(2.0f, ratio);           // bandwidth

        float max_val = 0.0f;
        for (int k = 0; k < signal_len; ++k) {
            // Normalized frequency ∈ [0, 2π)
            float omega = 2.0f * PI * static_cast<float>(k) / static_cast<float>(signal_len);
            // Map to [-π, π) for symmetric wavelet
            if (omega > PI) omega -= 2.0f * PI;

            float diff = sigma * (omega - xi);
            float val  = std::exp(-0.5f * diff * diff);
            bank.filters[lam][k] = val;
            max_val = std::max(max_val, val);
        }

        // Normalize so peak = psi_peak (ensures Lipschitz bound)
        if (max_val > 1e-12f) {
            float scale = psi_peak / max_val;
            for (int k = 0; k < signal_len; ++k) {
                bank.filters[lam][k] *= scale;
            }
        }
    }

    bank.l1_norm_psi = psi_peak;
    return bank;
}

// ===========================================================================
// CPU WST Forward Pass — depth-m scattering cascade
//
// For each scattering layer d ∈ [0, depth):
//   1. X̂ = FFT(x)                              (forward transform)
//   2. Ŷ = X̂ · Ψ_d                              (frequency-domain filtering)
//   3. y = IFFT(Ŷ)                              (back to time domain)
//   4. x = |y| = √(Re²+Im²)                    (complex modulus nonlinearity)
//
// The wavelet index cycles through the filter bank: λ = d % n_wavelets.
// Output is the real-valued scattering coefficients after the final layer.
// ===========================================================================

static void cpu_wst_forward(
    const float* input,       // [batch_size * signal_len] float32 input
    float*       output,      // [batch_size * signal_len] float32 output
    int          signal_len,
    int          batch_size,
    int          depth,
    const CPUFilterBank& bank,
    float&       l1_norm_out  // reports the L1 norm for cfg.l1_norm_psi
) {
    size_t N = static_cast<size_t>(next_power_of_2(signal_len));
    l1_norm_out = bank.l1_norm_psi;

    for (int b = 0; b < batch_size; ++b) {
        const float* sig_in  = input  + b * signal_len;
        float*       sig_out = output + b * signal_len;

        // Initialize complex buffer from real input
        std::vector<cfloat> x(N, cfloat(0.0f, 0.0f));
        for (int i = 0; i < signal_len; ++i) {
            x[i] = cfloat(sig_in[i], 0.0f);
        }

        // Scattering cascade
        for (int d = 0; d < depth; ++d) {
            // Select wavelet — evenly sample across the J*Q filter bank.
            // This ensures different J values probe different scales, since
            // n_wavelets = J*Q and the stride changes with J.
            int lam = ((d + 1) * bank.n_wavelets) / (depth + 1);
            lam = std::min(lam, bank.n_wavelets - 1);
            const auto& psi = bank.filters[lam];

            // Step 1: Forward FFT
            cpu_fft_inplace(x, false);

            // Step 2: Pointwise multiply by Ψ_λ in frequency domain
            for (size_t k = 0; k < N; ++k) {
                float filt = (k < static_cast<size_t>(signal_len)) ? psi[k] : 0.0f;
                x[k] *= filt;
            }

            // Step 3: Inverse FFT
            cpu_fft_inplace(x, true);

            // Step 4: Complex modulus |z| = √(Re² + Im²)
            for (size_t k = 0; k < N; ++k) {
                float mag = std::abs(x[k]);
                x[k] = cfloat(mag, 0.0f);
            }
        }

        // Extract real-valued scattering coefficients
        for (int i = 0; i < signal_len; ++i) {
            sig_out[i] = x[i].real();
        }
    }
}

#endif // CPU_WST_ENGINE_H
