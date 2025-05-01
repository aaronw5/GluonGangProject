"""
Benchmark jet-mass calculation on CPU (NumPy) vs GPU (CuPy)
and collect basic resource-use metrics.

Extra deps:
  pip install psutil pynvml
"""

import time, pathlib, numpy as np, psutil, pynvml
import uproot, awkward as ak, cupy as cp

# 0) Config
FILE_PATH = pathlib.Path(
    "/home/nbrandma/DAOD_PHYSLITE.37621365._000015.pool.root.1"
).expanduser().resolve()
N_RUNS = 6                                         # timing passes

proc = psutil.Process()
pynvml.nvmlInit()
nv_dev = pynvml.nvmlDeviceGetHandleByIndex(0)

# 1) Load branches → Awkward (CPU backend)
br = {
    "pt":  "AnalysisJetsAuxDyn.pt",
    "eta": "AnalysisJetsAuxDyn.eta",
    "phi": "AnalysisJetsAuxDyn.phi",
    "m":   "AnalysisJetsAuxDyn.m",
}

with uproot.open(FILE_PATH) as f:
    raw = f["CollectionTree"].arrays(list(br.values()), library="ak")
jets_cpu = ak.zip({k: raw[v] for k, v in br.items()})

# Copy to GPU once
jets_gpu = ak.to_backend(jets_cpu, "cuda")

# 2) Helper: invariant mass (NumPy ufuncs dispatch)
def compute_mass(jets):
    px = jets.pt * np.cos(jets.phi)
    py = jets.pt * np.sin(jets.phi)
    pz = jets.pt * np.sinh(jets.eta)
    E  = np.sqrt(np.maximum(jets.m**2 + px**2 + py**2 + pz**2, 0))
    m2 = E**2 - (px**2 + py**2 + pz**2)
    return ak.where(m2 > 0, np.sqrt(m2), -1.0)

# 3) Metric helpers
def cpu_snapshot():
    return dict(rss=proc.memory_info().rss, threads=proc.num_threads())

def gpu_snapshot():
    util = pynvml.nvmlDeviceGetUtilizationRates(nv_dev)
    mem  = pynvml.nvmlDeviceGetMemoryInfo(nv_dev)
    return dict(util=util.gpu, mem_used=mem.used)

# print fixed GPU facts once 
dev_attrs = cp.cuda.Device(0).attributes
gpu_name  = pynvml.nvmlDeviceGetName(nv_dev)
if isinstance(gpu_name, bytes):        # NVML < 12 returns bytes
    gpu_name = gpu_name.decode()

print(f"\nGPU 0: {gpu_name}")
print(f"  SMs           : {dev_attrs['MultiProcessorCount']}")
print(f"  Core clock    : {dev_attrs['ClockRate']/1e3:.0f} MHz")
tot_vram = pynvml.nvmlDeviceGetMemoryInfo(nv_dev).total
print(f"  Total VRAM    : {tot_vram/1e9:.1f} GB\n")
print()

# 4) Timing + resource sampling loops
def run_suite(arr, is_gpu):
    times, rss0, rss1, mem0, mem1, util = [], [], [], [], [], []
    sync = (lambda: None) if not is_gpu else cp.cuda.Device(0).synchronize

    for _ in range(N_RUNS):
        sync()
        rss0.append(cpu_snapshot()['rss'])
        if is_gpu:
            g0 = gpu_snapshot(); mem0.append(g0['mem_used']); util.append(g0['util'])

        t0 = time.perf_counter()
        _  = compute_mass(arr)
        sync()
        times.append(time.perf_counter() - t0)

        rss1.append(cpu_snapshot()['rss'])
        if is_gpu:
            g1 = gpu_snapshot(); mem1.append(g1['mem_used'])
            util[-1] = (util[-1] + g1['util']) / 2  # crude avg

    return dict(time=np.array(times),
                rss0=np.array(rss0), rss1=np.array(rss1),
                mem0=np.array(mem0) if is_gpu else None,
                mem1=np.array(mem1) if is_gpu else None,
                util=np.array(util)  if is_gpu else None)

gpu = run_suite(jets_gpu, is_gpu=True)

# 5) Report
print("──────────  RESULTS  ──────────")
print(f"Runs per mode      : {N_RUNS}")
print(f"GPU mean time (2-{N_RUNS}): {gpu['time'][1:].mean():.4f} s")
print(f"Times per run      :    1   |    2   |    3   |    4   |    5   |    6   |")
print(f"GPU (s)            : {gpu['time'][0]:.4f} | {gpu['time'][1]:.4f} | {gpu['time'][2]:.4f} | {gpu['time'][3]:.4f} | {gpu['time'][4]:.4f} | {gpu['time'][5]:.4f} |")

print("GPU VRAM Δ per run (MB):",
      np.round((gpu['mem1'] - gpu['mem0']) / 1_048_576, 3))

print("GPU util sampled (%):", gpu['util'].astype(int))
