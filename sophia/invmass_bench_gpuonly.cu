// invariant-mass calculation on CPU vs GPU
// Compile : nvcc -O3 -arch=sm_80 invmass_bench.cu -o invmass_bench
// Usage   : ./invmass_bench jets.bin      (binary with N, then 4·N floats)
//           ./invmass_bench               (falls back to 1 M random jets)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel
__global__
void invmass_kernel(const float* __restrict__ pt,
                    const float* __restrict__ eta,
                    const float* __restrict__ phi,
                    const float* __restrict__ mass,
                    float* __restrict__ out,
                    std::size_t N)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float px  = pt[i]  * cosf(phi[i]);
        float py  = pt[i]  * sinf(phi[i]);
        float pz  = pt[i]  * sinhf(eta[i]);
        float E   = sqrtf(fmaxf(mass[i]*mass[i] + px*px + py*py + pz*pz, 0.f));
        float m2  = E*E - (px*px + py*py + pz*pz);
        out[i]    = (m2 > 0.f) ? sqrtf(m2) : -1.f;
    }
}

// Error-check helper
#define ck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

using clock_s = std::chrono::steady_clock;

int main(int argc, char** argv)
{
    const int RUNS = 6;
    std::size_t N  = 0;

    std::vector<float> h_pt, h_eta, h_phi, h_m, h_out;

    // Option A: load jets.bin
    if (argc == 2) {
        FILE* f = fopen(argv[1], "rb");
        if (!f) { perror("fopen"); return 1; }

        fread(&N, sizeof(std::size_t), 1, f);
        h_pt.resize(N);  h_eta.resize(N);  h_phi.resize(N);  h_m.resize(N);

        fread(h_pt.data(),  sizeof(float), N, f);
        fread(h_eta.data(), sizeof(float), N, f);
        fread(h_phi.data(), sizeof(float), N, f);
        fread(h_m.data(),   sizeof(float), N, f);
        fclose(f);

        printf("Loaded %zu jets from %s\n", N, argv[1]);
    }
    // Option B: generate random jets
    else {
        N = 1'000'000'000;
        h_pt.resize(N);  h_eta.resize(N);  h_phi.resize(N);  h_m.resize(N);

        for (std::size_t i = 0; i < N; ++i) {
            h_pt [i] =  50.f + 300.f * rand() / RAND_MAX;
            h_eta[i] = -2.5f +   5.f * rand() / RAND_MAX;
            h_phi[i] = -M_PI + 2.f * M_PI * rand() / RAND_MAX;
            h_m  [i] =  10.f +  40.f * rand() / RAND_MAX;
        }
        printf("Generated %zu random jets\n", N);
    }

    h_out.resize(N);

    // Allocate device buffers
    float *d_pt, *d_eta, *d_phi, *d_m, *d_out;
    ck(cudaMalloc(&d_pt , N * sizeof(float)));
    ck(cudaMalloc(&d_eta, N * sizeof(float)));
    ck(cudaMalloc(&d_phi, N * sizeof(float)));
    ck(cudaMalloc(&d_m  , N * sizeof(float)));
    ck(cudaMalloc(&d_out, N * sizeof(float)));

    ck(cudaMemcpy(d_pt , h_pt .data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_eta, h_eta.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_phi, h_phi.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_m  , h_m .data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Timing containers
    float  gpu_ms[RUNS]{};   // must be float for cudaEventElapsedTime

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    for (int r = 0; r < RUNS; ++r) {
        // GPU pass
        cudaEvent_t ev_start, ev_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);

        ck(cudaEventRecord(ev_start));
        invmass_kernel<<<grid, block>>>(d_pt, d_eta, d_phi, d_m, d_out, N);
        ck(cudaEventRecord(ev_stop));
        ck(cudaEventSynchronize(ev_stop));
        ck(cudaGetLastError());
        ck(cudaEventElapsedTime(&gpu_ms[r], ev_start, ev_stop));

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
    }

    // Per-run results
    printf("\nRun   GPU_ms\n");
    for (int r = 0; r < RUNS; ++r)
        printf("%2d   %7.3f\n", r + 1, gpu_ms[r]);

    // Averages: CPU (1-6) vs GPU (2-6)
    double gpu_mean = 0.0;
    for (int r = 1; r < RUNS; ++r)  gpu_mean += static_cast<double>(gpu_ms[r]);
    gpu_mean /= (RUNS - 1);

    printf("Average GPU time  (runs 2-6): %.3f ms\n", gpu_mean);

    // Cleanup
    cudaFree(d_pt);  cudaFree(d_eta);  cudaFree(d_phi);
    cudaFree(d_m);   cudaFree(d_out);
    cudaDeviceReset();
    return 0;
}
