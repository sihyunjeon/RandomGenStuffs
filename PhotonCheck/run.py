#!/usr/bin/env python3
import os
import uproot
import awkward as ak
import numpy as np
import vector
vector.register_awkward()
import json
import get_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
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

def get_photon_corr(pt):
    with open("correction.json") as f:
        data = json.load(f)
    bin_edges = np.array(data["bin_edges"])
    ratios = np.array(data["ratios"])
    idx = np.digitize(pt, bin_edges) - 1
    idx = ak.where(idx > 40, 40, idx)
    return ak.flatten(ratios[idx])

def process_file(dataset_name, file):
    events = uproot.open(file)["Events"]
    weight = events["genWeight"].array()

    keys = ["pt", "eta", "phi", "mass"]
    genphotons = construct_vector({k: events[f"GenIsolatedPhoton_{k}"].array() for k in keys})
    genjets = construct_vector({k: events[f"GenJet_{k}"].array() for k in keys})

    genphotons_mask = (genphotons["pt"] > 230) & (np.abs(genphotons["eta"]) < 1.4442)
    genphotons = genphotons[genphotons_mask]

    genjets_mask = (genjets["pt"] > 20) & (np.abs(genjets["eta"]) < 2.5)
    genjets = genjets[genjets_mask]
    genjets = genjets[remove_overlaps(genjets, genphotons)]

    monojet_mask = (genjets["pt"] > 100)
    monojet = genjets[monojet_mask]

    weight_corr = ak.ones_like(weight)
    if "aMCatNLO" in dataset_name:
        lheparts = construct_vector({k: events[f"LHEPart_{k}"].array() for k in keys + ["pdgId"]})
        lheparts_mask = (lheparts["pdgId"] == 22)
        lhephotons_pt = lheparts[lheparts_mask]["pt"]
        weight_corr = get_photon_corr(lhephotons_pt)
    events_mask = (ak.num(monojet, axis=1) > 0) & (ak.num(genphotons, axis=1) > 0)
    return {
        "genphoton": genphotons[events_mask][:,0],
        "monojet": monojet[events_mask][:,0],
        "ngenjets": ak.num(genjets[events_mask], axis=1),
        "weight": weight[events_mask],
        "weight_corr": weight_corr[events_mask],
        "weight_sum_filtered": np.sum(weight[events_mask]),
        "weight_sum_prefilter": np.sum(weight),
    }

def worker(dataset_name, input_file):
    return process_file(dataset_name, input_file)

def write_h5(dataset_name, batches, index):
    arrays = {}
    scalars = {"weight_sum_prefilter": 0.0, "weight_sum_filtered": 0.0}
    for batch in batches:
        for key, val in batch.items():
            if key in scalars:
                scalars[key] += val
            else:
                arrays.setdefault(key, []).append(val.to_numpy())

    os.system(f"mkdir -p inputs/{dataset_name}")
    with h5py.File(f"inputs/{dataset_name}/data_{index}.h5", "w") as wf:
        for key, val in arrays.items():
            wf.create_dataset(key, data=np.concatenate(val, axis=0))
        for key, val in scalars.items():
            wf.create_dataset(key, data=val)

def main():
    batch_size = 20
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
        "MadGraph_100to200_1000": [
"/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
        ],
        "MadGraph_100to200_200to400": [
"/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
        ],
        "MadGraph_100to200_400to600": [
"/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM"
        ],
        "MadGraph_100to200_40to200": [
"/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-40to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
        ],
        "MadGraph_100to200_600to1000": [
"/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-600to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
        ],
        "MadGraph_200_1000": [
"/GJ-4Jets_dRGJ-0p25_PTG-200_HT-1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
        ],
        "MadGraph_200_400to600": [
"/GJ-4Jets_dRGJ-0p25_PTG-200_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
        ],
        "MadGraph_200_40to400": [
"/GJ-4Jets_dRGJ-0p25_PTG-200_HT-40to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
        ],
        "MadGraph_200_600to1000": [
"/GJ-4Jets_dRGJ-0p25_PTG-200_HT-600to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
        ],
    }
    input_data = get_dataset.Get(inputs)
    for dataset_name, info in input_data.items():
        input_files = info["files"]
        index = 0
        batches = []
        worker_func = partial(process_file, dataset_name)
        with Pool(8) as pool:
            with tqdm(total=len(input_files), desc=f"{dataset_name}", unit="file") as pbar:
                for output in pool.imap_unordered(worker_func, input_files):
                    batches.append(output)
                    if len(batches) >= batch_size:
                        write_h5(dataset_name, batches, index)
                        batches.clear()
                        index += 1
                    pbar.update(1)

                if batches:
                    write_h5(dataset_name, batches, index)
                    batches.clear()

if __name__ == "__main__":
    main()
