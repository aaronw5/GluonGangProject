// invariant-mass calculation on CPU vs GPU
// Compile : nvcc -O3 -arch=sm_80 invmass_bench.cu -o invmass_bench
// Usage   : ./invmass_bench jets.bin      (binary with N, then 4·N floats)
//           ./invmass_bench               (falls back to 1 M random jets)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

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

    // Timing containers
    double cpu_ms[RUNS]{};
    
    for (int r = 0; r < RUNS; ++r) {
        // CPU pass
        auto t0 = clock_s::now();
        for (std::size_t i = 0; i < N; ++i) {
            float px = h_pt[i] * std::cos(h_phi[i]);
            float py = h_pt[i] * std::sin(h_phi[i]);
            float pz = h_pt[i] * std::sinh(h_eta[i]);
            float E  = std::sqrt(std::max(h_m[i]*h_m[i] + px*px + py*py + pz*pz, 0.f));
            float m2 = E*E - (px*px + py*py + pz*pz);
            h_out[i] = (m2 > 0.f) ? std::sqrt(m2) : -1.f;
        }
        auto t1 = clock_s::now();
        cpu_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
	
    // Per-run results
    printf("\nRun  CPU_ms\n");
    for (int r = 0; r < RUNS; ++r)
        printf("%2d   %7.2f\n", r + 1, cpu_ms[r]);

    // Averages: CPU (1-6) vs GPU (2-6)
    double cpu_mean = 0.0;
    for (int r = 0; r < RUNS; ++r)  cpu_mean += cpu_ms[r];
    cpu_mean /= RUNS;

    printf("\nAverage CPU time  (runs 1-6): %.2f ms\n", cpu_mean);

    return 0;
}
