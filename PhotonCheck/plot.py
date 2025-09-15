import os
import uproot
import matplotlib.pyplot as plt
import h5py
import numpy as np
import json
from hist import Hist
import mplhep as hep

hep.style.use("CMS")

def get_cross_sections(sample_name):
    with open("cross_sections.json") as rf:
        cross_sections = json.load(rf)
    return cross_sections[sample_name]

def get_colors(dataset):
    colors = {
        "Sherpa": "darkorange",
        "aMCatNLO": "silver",
        "MadGraph": "limegreen"
    }
    return colors[dataset]

def get_hists():
    bins = {
        "genphoton_pt": [230,250,270,290,310,330,350,400,450,500,600,700,800,1000],
        "monojet_pt": [100,120,140,160,180,200,220,250,300,350,400,450,500,700]
    }
    hists = {
        "ngenjets": {
            "hist": Hist.new.Reg(10, -0.5, 9.5, name="ngenjets", label="N(Gen.Jet)", overflow=True).Weight(),
            "xlabel": "N(Gen.Jet)"
        },
        "genphoton_pt": {
            "hist": Hist.new.Var(bins["genphoton_pt"], name="genphoton_pt", label="Photon PT [GeV]", overflow=True).Weight(),
            "xlabel": "Lead.Gen.Photon PT [GeV]"
        },
        "genphoton_eta": {
            "hist": Hist.new.Reg(20, -2.5, 2.5, name="genphoton_eta", label="Photon ETA", overflow=True).Weight(),
            "xlabel": "Lead.Gen.Photon ETA"
        },
        "monojet_pt": {
            "hist": Hist.new.Var(bins["monojet_pt"], name="monojet_pt", label="Jet PT [GeV]", overflow=True).Weight(),
            "xlabel": "Lead.Gen.Jet PT [GeV]"
        },
        "monojet_eta": {
            "hist": Hist.new.Reg(20, -2.5, 2.5, name="monojet_eta", label="Jet ETA", overflow=True).Weight(),
            "xlabel": "Lead.Gen.Jet ETA"
        },
    }
    return hists

class Histogrammer:
    def __init__(self, sample_name):
        self.sample_name = sample_name
        self._initialize()
    def _initialize(self):
        self.hists = get_hists()
        self.weight_sum_filtered = 0
        self.weight_sum_prefilter = 0
    def Process(self, file):
        with h5py.File(file, "r") as rf:
            self.weight_sum_prefilter += float(rf["weight_sum_prefilter"][()])
            self.weight_sum_filtered += float(rf["weight_sum_filtered"][()])
            weights = rf["weight"][:]
            weights_corr = rf["weight_corr"][:]
            for obj_name in ["genphoton", "monojet"]:
                obj = rf[obj_name][:]
                for val_name in ["pt", "eta"]:
                    val = obj[val_name]
                    self.hists[f"{obj_name}_{val_name}"]["hist"].fill(val, weight=weights*weights_corr)
            for var_name in ["ngenjets"]:
                val = rf[var_name][:]
                self.hists[var_name]["hist"].fill(val, weight=weights*weights_corr)
    def Save(self):
        with uproot.recreate(f"outputs/rootfiles/{self.sample_name}.root") as wf:
            for hist_name, info in self.hists.items():
                wf[hist_name] = info["hist"] / self.weight_sum_prefilter

class Plotter:
    def __init__(self, hist_name, hist_info):
        self.hist_name = hist_name
        self.hist_info = hist_info
        self.fig, (self.ax, self.rax) = plt.subplots(
            2, 1, sharex=True, 
            gridspec_kw={"height_ratios": [4, 1]}, figsize=(8, 8)
        )
        self.hists = {}
    def Stack(self, dataset_name, sample_names):
        counts_stacked = None
        for sample_name in sample_names:
            with uproot.open(f"outputs/rootfiles/{sample_name}.root") as rf:
                counts, edges = rf[self.hist_name].to_numpy()
                if not dataset_name in self.hists:
                    self.centers = 0.5 * (edges[:-1] + edges[1:])
                    self.edges = edges
                counts = counts * get_cross_sections(sample_name)
                if counts_stacked is None:
                    counts_stacked = counts
                else:
                    counts_stacked += counts
        self.hists[dataset_name] = counts_stacked
    def Plot(self, reference, variations):
        hist_ref = self.hists[reference]
        widths = np.diff(self.edges)

        for var in variations:
            hist_var = self.hists[var]
            self.ax.step(
                self.edges, np.append(hist_var/widths, (hist_var/widths)[-1]), where="post",
                linewidth=2, color=get_colors(var), label=var
            )
            self.rax.step(
                self.edges, np.append(hist_var/hist_ref, (hist_var/hist_ref)[-1]), where="post",
                linewidth=2, color=get_colors(var), label=var
            )

        self.ax.fill_between(
            self.edges, np.append(hist_ref/widths, (hist_ref/widths)[-1]),
            step="post", alpha=0.6, color=get_colors(reference), label=reference
        )
        self.rax.axhline(1.0, color="crimson", linestyle="--", linewidth=2)

        ymax = np.max(hist_ref/widths) * 50
        ymin = np.min(hist_ref/widths)
        ymin = ymax / 50 / 20 * 0.01 if ymin == 0 else ymin / 20

        os.system("mkdir -p outputs/hists")

        self.ax.legend(loc="upper right", fontsize=25)

        self.ax.set_ylabel("dσ/dX [pb per unit]")
        self.ax.set_yscale("log")
        self.ax.set_ylim(ymin, ymax)

        self.rax.set_ylim(0.5,1.5)
        self.rax.set_xlabel(self.hist_info["xlabel"])
        plt.tight_layout()
        plt.savefig(f"outputs/hists/{self.hist_name}.pdf")
        plt.savefig(f"outputs/hists/{self.hist_name}.png")

def main(datasets):
    #for dataset_name, sample_names in datasets.items():
    #    for sample_name in sample_names:
    #        histogrammer = Histogrammer(sample_name=sample_name)
    #        data = [f"inputs/{sample_name}/{f}" for f in os.listdir(f"inputs/{sample_name}") if f.endswith(".h5")]
    #        for i, file in enumerate(data):
    #            histogrammer.Process(file=file)
    #        histogrammer.Save()

    for hist_name, hist_info, in get_hists().items():
        plotter = Plotter(hist_name=hist_name, hist_info=hist_info)
        for dataset_name, sample_names in datasets.items():
            plotter.Stack(dataset_name, sample_names)
        plotter.Plot("aMCatNLO", ["Sherpa", "MadGraph"])

if __name__ == "__main__":
    datasets = {
        "Sherpa" : ["Sherpa"],
        "aMCatNLO" : ["aMCatNLO_100to200", "aMCatNLO_200to400", "aMCatNLO_400to600", "aMCatNLO_600"],
        "MadGraph" : ["MadGraph_100to200_1000", "MadGraph_100to200_200to400", "MadGraph_100to200_400to600", "MadGraph_100to200_40to200",
                      "MadGraph_100to200_600to1000", "MadGraph_200_1000", "MadGraph_200_400to600", "MadGraph_200_40to400", "MadGraph_200_600to1000"]
    }

    main(datasets)
