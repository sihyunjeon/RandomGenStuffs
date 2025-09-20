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
from multiprocessing import Pool
from functools import partial
import h5py
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class GJetProcessor:
    def __init__(self, analysis, batch_size=30, pool_size=30):
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.datasets = self._get_dataset_config()
        self.analysis = analysis
    
    def _get_dataset_config(self):
        """Dataset configuration - can be moved to external config file if needed"""
        return {
            "Sherpa": [
                "/GJ-4Jets-2NLO2LO_Bin-PTG-25_Par-BiasedPTG_TuneSherpaDef_13p6TeV_sherpaMEPS/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_100to200_1000": [
                "/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-1000-PTG-100to200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_100to200_200to400": [
                "/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-200to400-PTG-100to200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_100to200_400to600": [
                "/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-400to600-PTG-100to200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_100to200_40to200": [
                "/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-40to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-40to200-PTG-100to200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_100to200_600to1000": [
                "/GJ-4Jets_dRGJ-0p25_PTG-100to200_HT-600to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-600to1000-PTG-100to200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_200_1000": [
                "/GJ-4Jets_dRGJ-0p25_PTG-200_HT-1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-1000-PTG-200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_200_400to600": [
                "/GJ-4Jets_dRGJ-0p25_PTG-200_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-400to600-PTG-200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_200_40to400": [
                "/GJ-4Jets_dRGJ-0p25_PTG-200_HT-40to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-40to400-PTG-200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
            "MadGraph_200_600to1000": [
                "/GJ-4Jets_dRGJ-0p25_PTG-200_HT-600to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "/GJ-4Jets_Bin-HT-600to1000-PTG-200_Par-dRGJ-0p25_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
            ],
        }
    
    @staticmethod
    def construct_vector(values):
        """Construct 4-vector from pt, eta, phi, mass"""
        if "mass" not in values.keys():
            values["mass"] = values["pt"] * 0.
        return ak.zip({key: value for key, value in values.items()}, with_name="Momentum4D")
    
    @staticmethod
    def remove_overlaps(pair1, pair2, delta_r_cut=0.4):
        """Remove overlapping particles based on delta R"""
        pairs = ak.cartesian([pair1, pair2], axis=1, nested=True)
        pair1 = pairs["0"]
        pair2 = pairs["1"]
        delta_eta = np.abs(pair1["eta"] - pair2["eta"])
        delta_phi = np.abs(pair1["phi"] - pair2["phi"])
        delta_phi = ak.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
        delta_r = np.sqrt(delta_phi**2 + delta_eta**2)
        overlap = ak.any(delta_r < delta_r_cut, axis=2)
        return ~overlap
    
    @staticmethod
    def get_genphotons(particles, pt_cut=150, eta_cut=2.0):
        """Select generator-level photons with kinematic cuts"""
        particles_mask = (particles["pt"] > pt_cut) & (np.abs(particles["eta"]) < eta_cut)
        genphoton = particles[particles_mask][:,0:1]
        return genphoton
    
    def process_file(self, dataset_name, file_path):
        """Process a single ROOT file"""
        try: 
            with uproot.open(file_path, timeout=120) as rf:
                events = rf["Events"]
        except Exception as e:
            logger.warning(e)
            return None

        weight = events["genWeight"].array()

        keys = ["pt", "eta", "phi", "mass"]
                
        # Isolated photons
        genisolatedphotons = self.construct_vector({
            k: events[f"GenIsolatedPhoton_{k}"].array() for k in keys
        })
                
        # Generator particles
        genparts = self.construct_vector({
            k: events[f"GenPart_{k}"].array() for k in keys + ["pdgId", "status"]
        })
        genparts_mask = (genparts["status"] == 1) & (genparts["pdgId"] == 22) & (genparts["pt"] > 10)
        genparts = genparts[genparts_mask]

        # Select photons
        genisolatedphoton = self.get_genphotons(particles=genisolatedphotons)
        genpart = self.get_genphotons(genparts)
        genphoton = ak.where(ak.num(genisolatedphoton, axis=1) > 0, genisolatedphoton, genpart)

        # Jets
        genjets = self.construct_vector({k: events[f"GenJet_{k}"].array() for k in keys})
        event_mask = (ak.num(genphoton, axis=1) > 0)
        out = {}
        out["genphoton"] = genphoton
        out["weight"] = weight

        if self.analysis == "monojet":
            genjets_mask = (genjets["pt"] > 50) & (np.abs(genjets["eta"]) < 3.0)
            genjets = genjets[genjets_mask]
            genjet = genjets[self.remove_overlaps(genjets, genphoton)][:,0:1]
            event_mask = event_mask & (ak.num(genjet, axis=1) > 0)
            out["genjet"] = genjet

        elif self.analysis == "dijet":
            genjets_mask = (genjets["pt"] > 50)
            genjets = genjets[genjets_mask]
            genjets_pair = ak.combinations(genjets, 2, axis=1)
            genjet0 = genjets_pair["0"]
            genjet1 = genjets_pair["1"]
            gendijet = genjet0 + genjet1
            gendijet = self.construct_vector({
                "pt": gendijet.pt,
                "eta": gendijet.eta,
                "phi": gendijet.phi,
                "mass": gendijet.mass
            })
            gendijet_mask = (gendijet.mass > 800) & (genjet0.eta * genjet1.eta < 0) & (genjet0.pt > 100)
            genjet0 = genjet0[gendijet_mask][:,0:1]
            genjet1 = genjet1[gendijet_mask][:,0:1]
            event_mask = event_mask & (ak.num(gendijet_mask, axis=1) > 0)
            out["genjet0"] = genjet0
            out["genjet1"] = genjet1
            out["gendijet"] = gendijet

        for k, v in out.items():
            out[k] = v[event_mask]
        out["weight_total"] = np.sum(weight)
        out["weight_filtered"] = np.sum(weight[event_mask])

        return out
    
    def write_h5(self, dataset_name, batches, index):
        """Write processed data to HDF5 file"""
        # Filter out None results from failed files
        valid_batches = [batch for batch in batches if batch is not None]
        if not valid_batches:
            logger.warning(f"No valid batches for {dataset_name} index {index}")
            return
            
        arrays = {}
        scalars = {"weight_total": 0.0, "weight_filtered": 0.0}
        
        for batch in valid_batches:
            for key, val in batch.items():
                if key in scalars:
                    scalars[key] += val
                else:
                    arrays.setdefault(key, []).append(val)

        # Create output directory and file
        os.makedirs(f"inputs/{self.analysis}/{dataset_name}", exist_ok=True)
        with h5py.File(f"inputs/{self.analysis}/{dataset_name}/data_{index}.h5", "w") as wf:
            for key, val_list in arrays.items():
                concatenated = ak.concatenate(val_list, axis=0)
                if key == 'weight':
                    # Simple 1D array
                    wf.create_dataset(key, data=ak.to_numpy(concatenated))
                else:
                    # LorentzVector objects - extract components
                    for component in ['pt', 'eta', 'phi', 'mass']:
                        comp_data = ak.to_numpy(getattr(concatenated, component))
                        wf.create_dataset(f"{key}_{component}", data=comp_data)
            for key, val in scalars.items():
                wf.create_dataset(key, data=val)

    def process_dataset(self, dataset_name, input_files):
        """Process all files for a given dataset"""
        index = 0
        batches = []

        worker_func = partial(self.process_file, dataset_name)
        
        with Pool(self.pool_size) as pool:
            with tqdm(total=len(input_files), desc=f"{dataset_name}", unit="file") as pbar:
                for output in pool.imap_unordered(worker_func, input_files):
                    if output is not None:  # Only add successful results
                        batches.append(output)
                    
                    if len(batches) >= self.batch_size:
                        self.write_h5(dataset_name, batches, index)
                        batches.clear()
                        index += 1
                    
                    pbar.update(1)

                # Write remaining batches
                if batches:
                    self.write_h5(dataset_name, batches, index)
                    batches.clear()
    
    def run(self):
        """Main processing loop"""
        input_data = get_dataset.Get(self.datasets)
        
        for dataset_name, info in input_data.items():
            input_files = info["files"]
            self.process_dataset(dataset_name, input_files)


def main():
    for analysis in ["monojet"]:
        processor = GJetProcessor(analysis=analysis, batch_size=10, pool_size=10)
        processor.run()


if __name__ == "__main__":
    main()
