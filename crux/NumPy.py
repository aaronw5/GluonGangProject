"""
Benchmark jet-mass calculation on CPU (NumPy) vs GPU (CuPy)
and collect basic resource-use metrics.

Extra deps:
  pip install psutil pynvml
"""

import time, pathlib, numpy as np, psutil
import uproot, awkward as ak

# 0) Config
FILE_PATH = pathlib.Path(
    "/home/nbrandma/DAOD_PHYSLITE.37621365._000015.pool.root.1"
).expanduser().resolve()
N_RUNS = 6                                         # timing passes

proc = psutil.Process()

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

# 4) Timing + resource sampling loops
def run_suite(arr, is_gpu):
    times, rss0, rss1, mem0, mem1, util = [], [], [], [], [], []
    sync = (lambda: None)

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

cpu = run_suite(jets_cpu, is_gpu=False)

# 5) Report
print("──────────  RESULTS  ──────────")
print(f"Runs per mode      : {N_RUNS}")
print(f"CPU mean time      : {cpu['time'].mean():.4f} s")
print(f"Times per run      :    1   |    2   |    3   |    4   |    5   |    6   |")
print(f"CPU (s)            : {cpu['time'][0]:.4f} | {cpu['time'][1]:.4f} | {cpu['time'][2]:.4f} | {cpu['time'][3]:.4f} | {cpu['time'][4]:.4f} | {cpu['time'][5]:.4f} |")

print("CPU RSS Δ per run (MB):",
      np.round((cpu['rss1'] - cpu['rss0']) / 1_048_576, 3))
