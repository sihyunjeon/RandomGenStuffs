#!/usr/bin/env python3
import uproot
import awkward as ak
import numpy as np
import vector
vector.register_awkward()
import json
import get_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import h5py

def construct_vector(values):
    if "mass" not in values.keys():
        values["mass"] = values["pt"] * 0.
    return ak.zip({key: value for key, value in values.items()}, with_name="PtEtaPhiMLorentzVector")

def remove_overlaps(pair1, pair2):
    pairs = ak.cartesian([pair1, pair2], axis=1, nested=True)
    pair1 = pairs["0"]
    pair2 = pairs["1"]
    delta_eta = np.abs(pair1["eta"] - pair2["eta"])
    delta_phi = np.abs(pair1["phi"] - pair2["phi"])
    delta_phi = ak.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
    delta_r = np.sqrt(delta_phi**2 + delta_eta**2)
    overlap = ak.any(delta_r < 0.4, axis=2)
    return ~overlap

def process_file(file):
    events = uproot.open(file)["Events"]
    weight = events["genWeight"].array()

    keys = ["pt", "eta", "phi", "mass"]
    genphotons = construct_vector({k: events[f"GenIsolatedPhoton_{k}"].array() for k in keys})
    genjets = construct_vector({k: events[f"GenJet_{k}"].array() for k in keys})

    genphotons_mask = (genphotons["pt"] > 10) & (np.abs(genphotons["eta"]) < 1.4442)
    genphotons = genphotons[genphotons_mask]

    genjets_mask = (genjets["pt"] > 5) & (np.abs(genjets["eta"]) < 2.5)
    genjets = genjets[genjets_mask]
    genjets = genjets[remove_overlaps(genjets, genphotons)]
    events_mask = (ak.num(genjets, axis=1) > 0) & (ak.num(genphotons, axis=1) > 0)

    return {
        "genphoton": genphotons[events_mask][:,0],
        "genjet": genjets[events_mask][:,0],
        "weight": weight[events_mask],
        "weight_sum_filtered": np.sum(weight[events_mask]),
        "weight_sum_prefilter": np.sum(weight),
    }

def worker(input_file):
    return process_file(input_file)

def write_h5(dataset_name, batches, index):
    arrays = {}
    scalars = {"weight_sum_prefilter": 0.0, "weight_sum_filtered": 0.0}
    for batch in batches:
        for key, val in batch.items():
            if key in scalars:
                scalars[key] += val
            else:
                arrays.setdefault(key, []).append(val.to_numpy())

    with h5py.File(f"{dataset_name}_{index}.h5", "w") as wf:
        for key, val in arrays.items():
            wf.create_dataset(key, data=np.concatenate(val, axis=0))

def main():
    batch_size = 2
    inputs = {
        "Sherpa": [
"/GJ-4Jets-2NLO2LO_Bin-PTG-25_Par-BiasedPTG_TuneSherpaDef_13p6TeV_sherpaMEPS/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
        ],
        "aMCatNLO_100to200": [
"/GJ_PTG-100to200_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM"
        ],
        "aMCatNLO_200to400": [
"/GJ_PTG-200to400_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM"
        ],
        "aMCatNLO_400to600": [
"/GJ_PTG-400to600_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15_ext1-v2/NANOAODSIM"
        ],
        "aMCatNLO_600": [
"/GJ_PTG-600_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15_ext1-v2/NANOAODSIM"
        ],
    }
    input_data = get_dataset.Get(inputs)
    for dataset_name, info in input_data.items():
        input_files = info["files"]
        index = 0
        batches = []
        with Pool(cpu_count()) as pool:
            with tqdm(total=len(input_files), desc=f"{dataset_name}", unit="file") as pbar:
                for output in pool.imap_unordered(worker, input_files, chunksize=batch_size):
                    batches.append(output)
                    if len(batches) >= batch_size:
                        write_h5(dataset_name, batches, index)
                        batches.clear()
                        index = index + 1
                    pbar.update(1)
                if batches:
                    write_h5(dataset_name, batches, index) 
                    batches.clear()

if __name__ == "__main__":
    main()
