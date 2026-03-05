import os
import uproot
import subprocess
import ROOT as R
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import multiprocessing as mp
from functools import partial
from scipy.optimize import minimize
import mplhep as hep
hep.style.use("CMS")

def get_luminosity(sample):
    if sample.endswith("22"):
        campaign = "Run3Summer22"
    elif sample.endswith("22EE"):
        campaign = "Run3Summer22EE"
    elif sample.endswith("23"):
        campaign = "Run3Summer23"
    elif sample.endswith("23BPix"):
        campaign = "Run3Summer23BPix"
    luminosity = {
        "Run3Summer22": 7.98,
        "Run3Summer22EE": 26.67,
        "Run3Summer23": 17.96,
        "Run3Summer23BPix": 9.69
    }
    return luminosity[campaign]

class CreateHistograms:
    def __init__(self, config, nbins, xmin, xmax):
        self.config_name = config["name"]
        self.mask = config["mask"]
        self.samples = config["samples"]
        self.nbins = nbins
        self.xmin = xmin
        self.xmax = xmax
        os.makedirs(f"{OUTPUT_PATH}/histograms/{self.config_name}", exist_ok=True)
        self.hists = {}
        pass
    def CreateHist(self, tag):
        hist = R.TH1F(tag, tag, self.nbins, self.xmin, self.xmax)
        hist.Sumw2()
        hist.SetDirectory(0)
        return hist
    def Run(self):
        for _key, _samples in self.samples.items():
            for _sample in _samples:
                with h5py.File(f"{OUTPUT_PATH}/processed/{_key}/{_sample}.h5") as rf:
                    for gname in rf.keys():
                        g = rf[gname]
                        scale = get_luminosity(_sample)/g.attrs["summed_weight"] if "GJets" in _sample else 1.0
                        hist = self.hists[gname] if gname in self.hists else self.CreateHist(tag=gname)
                        sieie = np.asarray(g[self.mask]["sieie"][:], dtype=np.float64)
                        weight = np.asarray(g[self.mask]["weight"][:], dtype=np.float64)
                        hist.FillN(len(weight), sieie, weight * scale)
                        self.hists[gname] = hist
    def Store(self):
        wf = R.TFile(f"{OUTPUT_PATH}/histograms/{self.config_name}.root", "RECREATE")
        for _, hist in self.hists.items():
            hist.Write()
        wf.Close()


for config in [
    {"name": "realMC", "mask": "pass_alliso", "samples": {"MC": ["GJets22", "GJets22EE", "GJets23", "GJets23BPix"]}},
    {"name": "fake200", "mask": "fail_oneiso", "samples": {"HLT_Photon200": ["EGamma22", "EGamma22EE", "EGamma23", "EGamma23BPix"]}},
    {"name": "data200", "mask": "pass_alliso", "samples": {"HLT_Photon200": ["EGamma22", "EGamma22EE", "EGamma23", "EGamma23BPix"]}},
]:
    p = CreateHistograms(config=config, nbins = 2, xmin = 0.000, xmax = 0.020)
    p.Run()
    p.Store()
