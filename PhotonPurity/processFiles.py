import os
import uproot
import subprocess
import h5py
from tqdm import tqdm
import numpy as np
import awkward as ak
import multiprocessing as mp
from functools import partial
import mplhep as hep
hep.style.use("CMS")

XRD_PATH = "root://eosuser.cern.ch//eos/user/s/shjeon/pyRATOutput/hinvisible__FullAnalysis/"
OUTPUT_PATH = "/home/shjeon-cern/work/users/shjeon-cern/"

def get_pt_ranges(hlt="MC"):
    pt_ranges = {
        "MC": [
            (160, 170), (170, 190), (190, 210), (210, 250), (250, 300), (300, 350), (350, 400), (400, 500), (500, 600), (600, 700), (700, 13600),
        ],
        "HLT_Photon150": [
            (160, 170), (170, 190), (190, 210), (210, 250), (250, 300), (300, 500), (500, 13600)
        ],
        "HLT_Photon175": [
            (190, 210), (210, 250), (250, 300), (300, 500), (500, 13600)
        ],
        "HLT_Photon200": [
            (210, 250), (250, 300), (300, 350), (350, 400), (400, 500), (500, 600), (600, 700), (700, 13600)
        ]
    }
    return pt_ranges[hlt]

def get_masks():
    return ["pass_alliso", "fail_oneiso", "fail_chiso", "fail_ecaliso", "fail_hcaliso"]

class ProcessFiles:
    def __init__(self):
        pass
    def PreRun(self, file, pt_range, hlt):
        self.file = file
        self.is_data = True if "EGamma" in self.file else False
        self.pt_min = pt_range[0]
        self.pt_max = pt_range[1]
        self.hlt = hlt
        try:
            with uproot.open(self.file) as rf:
                self.summed_weight = rf["cutflow"].values()[1]
            self.values = {}
        except:
            self.summed_weight = None
            self.values = None
    def Run(self):
        with uproot.open(self.file) as rf:
            if not "Events" in rf:
                return
            events = rf["Events"]

            common_mask = events["SinglePhoton_trigger_selection" if not self.is_data else self.hlt].array()
            common_mask = common_mask & events["Monojet_PhotonPurity_selection"].array()
            common_mask = common_mask & (events["jec_Nominal_TypeIPuppiMET_pt"].array() < 60)
            common_mask = common_mask & events["Noise_filter_selection"].array()
            common_mask = common_mask & events["DetectorMitigation_selection"].array()
            photons = ak.zip(
                {
                    var: events[f"Monojet_PhotonPurity_photons_{var}"].array() for var in ["pt", "sieie", "electronVeto", "isScEtaEB"] + get_masks()
                }
            )[common_mask]
            weights = events["weight_total"].array()[common_mask]
            #weights = events["weight_generator_nominal"].array()[common_mask]
        photons = photons[photons["electronVeto"] & photons["isScEtaEB"] & (photons["pt"] > self.pt_min) & (photons["pt"] < self.pt_max)]

        for _mask_name in get_masks():
            _mask = photons[_mask_name]
            self.values[_mask_name] = {
                "pt": ak.flatten(photons[_mask][:,0:1]["pt"]).to_numpy(),
                "sieie": ak.flatten(photons[_mask][:,0:1]["sieie"]).to_numpy(),
                "weight": weights[ak.any(_mask, axis=1) > 0].to_numpy()
            }
    def Out(self):
        return self.summed_weight, self.values

def AggregateResults(results):
    summed_weight = 0
    values = {mask_name: {"pt": [], "sieie": [], "weight": []} for mask_name in get_masks()}

    for result in results:
        if result is None:
            continue
        sw, vals = result
        summed_weight += sw
        for mask_name in get_masks():
            for key in ["pt", "sieie", "weight"]:
                values[mask_name][key].append(vals[mask_name][key])

    for mask_name in get_masks():
        for key in ["pt", "sieie", "weight"]:
            values[mask_name][key] = np.concatenate(values[mask_name][key])

    return summed_weight, values


def WriteHdf5(wf, pt_range, summed_weight, values):
    pt_key = f"pt_{pt_range[0]}to{pt_range[1]}"
    grp = wf.require_group(pt_key)
    grp.attrs["summed_weight"] = summed_weight
    for mask_name in get_masks():
        for key in ["pt", "sieie", "weight"]:
            grp[f"{mask_name}/{key}"] = values[mask_name][key]


def GetFiles(datasets, campaign):
    files = []
    for dataset in datasets:
        result = subprocess.run(
            ["xrdfs", "root://eosuser.cern.ch", "ls", "-u",
             f"/eos/user/s/shjeon/pyRATOutput/hinvisible__FullAnalysis/{campaign}/NanoAODv12/{dataset}"],
            capture_output=True, text=True
        )
        files += [f.strip() for f in result.stdout.splitlines() if f.endswith(".root")]
    return files

def Worker(args):
    file, pt_range, hlt = args
    p = ProcessFiles()
    try:
        p.PreRun(file=file, pt_range=pt_range, hlt=hlt)
        p.Run()
        return p.Out()
    except:
        #print (f"{file} not processed")
        return None

def RunWorker(dataset_name, datasets, campaign, nproc=120):
    files = GetFiles(datasets, campaign)
    is_data = "EGamma" in dataset_name
    hlts = ["HLT_Photon200"] if is_data else ["MC"]

    for hlt in hlts:
        os.makedirs(f"{OUTPUT_PATH}/processed/{hlt}", exist_ok=True)
        hdf5_path = f"{OUTPUT_PATH}/processed/{hlt}/{dataset_name}.h5"
        with h5py.File(hdf5_path, "w") as wf:
            for pt_range in get_pt_ranges(hlt):
                tasks = [(file, pt_range, hlt) for file in files]
                with mp.Pool(processes=nproc) as pool:
                    results = list(tqdm(
                        pool.imap_unordered(Worker, tasks),
                        desc=f"{hlt}:{dataset_name}:{pt_range}",
                        total=len(tasks)
                    ))
                summed_weight, values = AggregateResults(results)
                WriteHdf5(wf, pt_range, summed_weight, values)



if __name__ == "__main__":
    configs = {
        "Run3Summer22": {
            "EGamma22": ["EGammaRun2022C", "EGammaRun2022D"],
            "GJets22": ["GJ-4Jets-2NLO2LO_PTG-120_BiasedPTG_TuneSherpaDef_13p6TeV_sherpaMEPS"]
        },
        "Run3Summer22EE": {
            "EGamma22EE": ["EGammaRun2022E", "EGammaRun2022F", "EGammaRun2022G"],
            "GJets22EE": ["GJ-4Jets-2NLO2LO_PTG-120_BiasedPTG_TuneSherpaDef_13p6TeV_sherpaMEPS"]
        },
        "Run3Summer23": {
            "EGamma23": ["EGamma0Run2023C", "EGamma1Run2023C"],
            "GJets23": ["GJ-4Jets-2NLO2LO_PTG-120_BiasedPTG_TuneSherpaDef_13p6TeV_sherpaMEPS"]
        },
        "Run3Summer23BPix": {
            "EGamma23BPix": ["EGamma0Run2023D", "EGamma1Run2023D"],
            "GJets23BPix": ["GJ-4Jets-2NLO2LO_PTG-120_BiasedPTG_TuneSherpaDef_13p6TeV_sherpaMEPS"]
        },
    }
    for campaign in configs:
        for dataset_name, datasets in configs[campaign].items():
            RunWorker(dataset_name=dataset_name, datasets=datasets, campaign=campaign)
