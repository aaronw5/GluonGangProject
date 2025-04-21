/*
  invariant‑mass calculation on CPU vs GPU
  Compile:   nvcc -O3 -arch=sm_80 invmass_bench.cu -o invmass_bench
  Usage:     ./invmass_bench jets.bin          (binary with N, then 4*N floats; uses python generated bin as input)
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel stuff 
__global__
void invmass_kernel(const float* __restrict__ pt,
                    const float* __restrict__ eta,
                    const float* __restrict__ phi,
                    const float* __restrict__ mass,
                    float* __restrict__ out,
                    std::size_t N)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float px   = pt[i] * cosf(phi[i]);
        float py   = pt[i] * sinf(phi[i]);
        float pz   = pt[i] * sinhf(eta[i]);
        float E    = sqrtf(fmaxf(mass[i]*mass[i] + px*px + py*py + pz*pz, 0.f));
        float m2   = E*E - (px*px + py*py + pz*pz);
        out[i]     = (m2 > 0.f) ? sqrtf(m2) : -1.f;
    }
}

// Helpers
#define ck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error %s at %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

using clock_s = std::chrono::steady_clock;

// Main
int main(int argc, char** argv)
{
    const int   RUNS = 6;
    std::size_t N    = 0;

    std::vector<float> h_pt, h_eta, h_phi, h_m, h_out;

    // (A)  Load data from file <N> <pt[N]> <eta[N]> <phi[N]> <m[N]>
    if (argc == 2)
    {
        FILE* f = fopen(argv[1],"rb");
        if (!f) { perror("fopen"); return 1; }
        fread(&N, sizeof(std::size_t), 1, f);
        h_pt .resize(N); h_eta.resize(N); h_phi.resize(N); h_m.resize(N);
        fread(h_pt.data() , sizeof(float), N, f);
        fread(h_eta.data(), sizeof(float), N, f);
        fread(h_phi.data(), sizeof(float), N, f);
        fread(h_m.data()  , sizeof(float), N, f);
        fclose(f);
        printf("Loaded %zu jets from %s\n", N, argv[1]);
    }
    // (B)  Otherwise make random test data (1 M jets)
    else
    {
        N = 1'000'000;
        h_pt .resize(N);
        h_eta.resize(N);
        h_phi.resize(N);
        h_m  .resize(N);
        for (std::size_t i=0;i<N;++i)
        {
            h_pt [i] = 50.f  + 300.f * rand() / RAND_MAX; // GeV
            h_eta[i] =   -2.5f +  5.f  * rand() / RAND_MAX;
            h_phi[i] =  -M_PI + 2.f * M_PI * rand() / RAND_MAX;
            h_m  [i] = 10.f   + 40.f  * rand() / RAND_MAX;
        }
        printf("Generated %zu random jets\n", N);
    }

    h_out.resize(N);                              // CPU result

    // Allocate device memory once
    float *d_pt, *d_eta, *d_phi, *d_m, *d_out;
    ck(cudaMalloc(&d_pt , N*sizeof(float)));
    ck(cudaMalloc(&d_eta, N*sizeof(float)));
    ck(cudaMalloc(&d_phi, N*sizeof(float)));
    ck(cudaMalloc(&d_m  , N*sizeof(float)));
    ck(cudaMalloc(&d_out, N*sizeof(float)));

    ck(cudaMemcpy(d_pt , h_pt .data(), N*sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_eta, h_eta.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_phi, h_phi.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_m  , h_m .data(), N*sizeof(float), cudaMemcpyHostToDevice));

    // Timing containers
    double cpu_ms[RUNS]{}, gpu_ms[RUNS]{};

    // ==========================================================
    //              CPU + GPU BENCHMARK LOOPS
    // ==========================================================
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    for (int r=0; r<RUNS; ++r)
    {
        // CPU pass
        auto t0 = clock_s::now();
        for (std::size_t i=0;i<N;++i)
        {
            float px  = h_pt[i]  * std::cos(h_phi[i]);
            float py  = h_pt[i]  * std::sin(h_phi[i]);
            float pz  = h_pt[i]  * std::sinh(h_eta[i]);
            float E   = std::sqrt(std::max(h_m[i]*h_m[i] + px*px + py*py + pz*pz, 0.f));
            float m2  = E*E - (px*px + py*py + pz*pz);
            h_out[i]  = (m2 > 0.f) ? std::sqrt(m2) : -1.f;
        }
        auto t1 = clock_s::now();
        cpu_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();

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

    // Print summary
    printf("\nRun  CPU_ms   GPU_ms\n");
    for (int r=0;r<RUNS;++r)
        printf("%2d   %7.2f  %7.2f\n", r+1, cpu_ms[r], gpu_ms[r]);

    // Bonus: reveal warm‑up effect
    printf("\nGPU speed‑up (run 2 vs run 1): %.1f× faster\n",
           gpu_ms[0] / gpu_ms[1]);

    // Clean up
    cudaFree(d_pt); cudaFree(d_eta); cudaFree(d_phi); cudaFree(d_m); cudaFree(d_out);
    cudaDeviceReset();
    return 0;
}

