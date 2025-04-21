#!/usr/bin/env python3
"""
Convert a PHYS‑LITE ROOT file to jets.bin for invmass_bench.cu

Usage
-----
python make_jets_bin.py /lstr/sahara/act/data/DAOD_PHYSLITE.37621365._000015.pool.root.1  jets.bin
"""

import sys, struct
import uproot, awkward as ak, numpy as np

def main(root_path: str, bin_path: str = "jets.bin") -> None:
    # 1. open ROOT & get branches
    tree = uproot.open(root_path)["CollectionTree"]
    vars_ak = tree.arrays(
        ["AnalysisJetsAuxDyn.pt",
         "AnalysisJetsAuxDyn.eta",
         "AnalysisJetsAuxDyn.phi",
         "AnalysisJetsAuxDyn.m"],
        library="ak"
    )

    # 2. flatten → NumPy float32
    pt   = ak.to_numpy(ak.flatten(vars_ak["AnalysisJetsAuxDyn.pt" ])).astype(np.float32, copy=False)
    eta  = ak.to_numpy(ak.flatten(vars_ak["AnalysisJetsAuxDyn.eta"])).astype(np.float32, copy=False)
    phi  = ak.to_numpy(ak.flatten(vars_ak["AnalysisJetsAuxDyn.phi"])).astype(np.float32, copy=False)
    mass = ak.to_numpy(ak.flatten(vars_ak["AnalysisJetsAuxDyn.m"  ])).astype(np.float32, copy=False)

    if not (pt.size == eta.size == phi.size == mass.size):
        raise RuntimeError("Branch sizes differ – double‑check the file")

    N = pt.size
    print(f"Writing {N:,} jets → {bin_path}")

    # 3. write as [uint64 N] [pt] [eta] [phi] [mass]
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", N))           # 8‑byte little‑endian length
        for arr in (pt, eta, phi, mass):
            arr.tofile(f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python make_jets_bin.py input.root [output.bin]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "jets.bin")

