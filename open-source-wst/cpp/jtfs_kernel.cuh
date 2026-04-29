#ifndef JTFS_KERNEL_CUH
#define JTFS_KERNEL_CUH

#include "wst_kernel.cuh"

// JTFS Configuration
struct JTFSConfig {
    int J;
    int Q;
    int J_fr;
    int Q_fr;
    int depth;
};

// Phase 1 Kernel (Time-axis convolution)
__global__ void launch_time_conv(const cufftComplex* d_scalogram, const cufftComplex* d_psi_mu, cufftComplex* d_intermediate, int Lambda1, int T, int mu_count, int batch) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int lambda = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (t < T && lambda < Lambda1 && b < batch) {
        // Mock convolution mapping
        int idx = (b * Lambda1 + lambda) * T + t;
        int mu_idx = lambda % mu_count; // Simplified modulation indexing

        cufftComplex s = d_scalogram[idx];
        cufftComplex p = d_psi_mu[mu_idx * T + t];

        cufftComplex res;
        res.x = s.x * p.x - s.y * p.y;
        res.y = s.x * p.y + s.y * p.x;

        d_intermediate[idx] = res;
    }
}

// Phase 2 Kernel (Log-frequency convolution)
__global__ void launch_freq_conv(const cufftComplex* d_intermediate, const cufftComplex* d_psi_ls, float* d_jtfs_out, int Lambda1, int T, int ls_count, int batch) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int lambda = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (t < T && lambda < Lambda1 && b < batch) {
        int idx = (b * Lambda1 + lambda) * T + t;
        int ls_idx = lambda % ls_count;

        cufftComplex inter = d_intermediate[idx];
        cufftComplex p = d_psi_ls[ls_idx * Lambda1 + lambda];

        float rx = inter.x * p.x - inter.y * p.y;
        float ry = inter.x * p.y + inter.y * p.x;

        // Extract modulus for final scattering coefficient
        d_jtfs_out[idx] = sqrtf(rx * rx + ry * ry);
    }
}

template<typename ArchTag, int J, int Q, int J_fr>
class JTFSEngine : public WSTEngine<ArchTag, J, Q> {
public:
    cufftComplex* d_freq_filter_bank;
    cufftComplex* d_jtfs_intermediate;
    cufftComplex* d_time_wavelets;

    JTFSEngine() : WSTEngine<ArchTag, J, Q>(), d_freq_filter_bank(nullptr), d_jtfs_intermediate(nullptr), d_time_wavelets(nullptr) {}

    virtual ~JTFSEngine() {
        if (d_freq_filter_bank) cudaFree(d_freq_filter_bank);
        if (d_jtfs_intermediate) cudaFree(d_jtfs_intermediate);
        if (d_time_wavelets) cudaFree(d_time_wavelets);
    }

    void jtfs_forward(const cufftComplex* d_scalogram, float* d_jtfs_out, int batch, int Lambda1, int T, int mu_count, int ls_count) {
        dim3 block(16, 16, 1);
        dim3 grid((T + 15) / 16, (Lambda1 + 15) / 16, batch);

        // Phase 1: Time-axis convolution on stream0
        launch_time_conv<<<grid, block, 0, this->stream0>>>(d_scalogram, d_time_wavelets, d_jtfs_intermediate, Lambda1, T, mu_count, batch);

        // Phase 2: Log-frequency convolution on stream1
        launch_freq_conv<<<grid, block, 0, this->stream1>>>(d_jtfs_intermediate, d_freq_filter_bank, d_jtfs_out, Lambda1, T, ls_count, batch);

        // Synchronize both streams to complete separable convolution
        cudaStreamSynchronize(this->stream0);
        cudaStreamSynchronize(this->stream1);
    }
};

#endif // JTFS_KERNEL_CUH
