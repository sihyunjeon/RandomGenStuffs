import os
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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
        "MadGraph": "limegreen"
    }
    return colors[dataset]

def get_hists():
    bins = {
        "2d_genjet_pt_1": [50, 100, 150, 200, 300, 400, 600, 1000],
        "2d_genjet_pt_2": [50, 60, 80, 100, 120, 150, 180, 210, 250, 300, 400, 500, 800],
        "2d_genphoton_pt_1": [150, 200, 300, 400, 600, 1000],
        "2d_genphoton_pt_2": [150, 170, 200, 250, 300, 350, 400, 450, 500, 600, 700],
    }
    hists = {f"{genphoton}_{genjet}": {
        "hist": Hist.new
                    .Var(bins[genphoton], name="genphoton_pt", overflow=True)
                    .Var(bins[genjet], name="genjet_pt", overflow=True)
                    .Weight(),
        "xlabel": "Gen.Photon pT",
        "ylabel": "Gen.Jet pT"
        } for genjet in ["2d_genjet_pt_1", "2d_genjet_pt_2"] for genphoton in ["2d_genphoton_pt_1", "2d_genphoton_pt_2"]
    }
    hists["genphoton_pt"] = {
        "hist": Hist.new
                    .Var([150, 160, 170, 180, 200, 220, 250, 300, 350, 400, 450, 500, 600, 700, 1000], name="genphoton_pt", overflow=True)
                    .Weight(),
        "xlabel": "Gen.Photon pT",
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
            genphoton_pt = ak.flatten(rf["genphoton"][:]["pt"])
            genjet_pt = ak.flatten(rf["genjet"][:]["pt"])
            for hist_name in list(self.hists.keys()):
                if "2d" in hist_name:
                    self.hists[hist_name]["hist"].fill(genphoton_pt, genjet_pt, weight=weights)
                else:
                    # fill only genphoton_pt
                    self.hists[hist_name]["hist"].fill(genphoton_pt, weight=weights)
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
        self.errors = {}

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

    def Stack2D(self, dataset_name, sample_names):
        counts_stacked = None
        errors_stacked = None
        for sample_name in sample_names:
            with uproot.open(f"outputs/rootfiles/{sample_name}.root") as rf:
                hist = rf[self.hist_name]
                counts, edges_x, edges_y = hist.to_numpy()
                variances = hist.variances()
                errors = np.sqrt(variances)
                if counts_stacked is None:
                    counts_stacked = np.zeros_like(counts)
                    errors_stacked = np.zeros_like(errors)
                    self.edges_x, self.edges_y = edges_x, edges_y
                counts_stacked += counts * get_cross_sections(sample_name)
                errors_stacked += (errors * get_cross_sections(sample_name)) ** 2
        self.hists[dataset_name] = counts_stacked
        self.errors[dataset_name] = np.sqrt(errors_stacked)

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

        result = {
            "edges_x": self.edges.tolist(),
            "ratio": (hist_var/hist_ref).tolist()
        }

        with open(f"{self.hist_name}.json", "w") as f:
            json.dump(result, f, indent=6)

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

    def Plot2D(self, dataset_name):
        counts = self.hists[dataset_name]

        X, Y = np.meshgrid(self.edges_x, self.edges_y, indexing="xy")
        fig, ax = plt.subplots(figsize=(10, 10))
        mesh = ax.pcolormesh(X, Y, counts.T, cmap="viridis", shading="auto")  

        fig.colorbar(mesh, ax=ax, label="dσ/dXdY [pb]")
        plt.tight_layout()
        plt.savefig(f"outputs/hists/{self.hist_name}_{dataset_name}.png")
        plt.savefig(f"outputs/hists/{self.hist_name}_{dataset_name}.pdf")
        plt.close()

    def Plot2DRatio(self, nominal, variation):
        counts_nom = self.hists[nominal]
        counts_var = self.hists[variation]
        errors_nom = self.errors[nominal]
        errors_var = self.errors[variation]
        ratio = np.divide(counts_nom, counts_var, out=np.zeros_like(counts_nom), where=counts_var!=0)
        rel_errors_nom = np.divide(errors_nom, counts_nom, out=np.zeros_like(errors_nom))
        rel_errors_var = np.divide(errors_var, counts_var, out=np.zeros_like(errors_var))
        ratio_errors = ratio * np.sqrt(rel_errors_nom**2 + rel_errors_var**2)

        X, Y = np.meshgrid(self.edges_x, self.edges_y, indexing="xy")
        fig, ax = plt.subplots(figsize=(10, 10))
        mesh = ax.pcolormesh(X, Y, ratio.T, cmap="rainbow", shading="auto")
        fig.colorbar(mesh, ax=ax, label="k-factor")

        with uproot.recreate(f"outputs/rootfiles/{self.hist_name}_ratio_{nominal}vs{variation}.root") as wf:
            wf["mgtosherpa_kfactor"] = (ratio, self.edges_x, self.edges_y)

        ax.set_xscale("log")
        ax.set_yscale("log")
        for i in range(len(self.edges_x) - 1):
            for j in range(len(self.edges_y) - 1):
                x_center = 0.5 * (self.edges_x[i] + self.edges_x[i+1])
                y_center = 0.5 * (self.edges_y[j] + self.edges_y[j+1])
                val = ratio[i, j]
                err = ratio_errors[i, j]
                if val != 0:  # only annotate non-zero
                    ax.text(
                        x_center, y_center, f"{val:.3f}\n$\pm${err:.3f}",
                        ha="center", va="center", color="black", fontsize=12,
                    )

        plt.tight_layout()
        plt.savefig(f"outputs/hists/{self.hist_name}_{nominal}vs{variation}.png")
        plt.savefig(f"outputs/hists/{self.hist_name}_{nominal}vs{variation}.pdf")
        plt.close()

    def Save2DRatio(self, nominal, variation):
        counts_nom = self.hists[nominal]
        counts_var = self.hists[variation]

        ratio = np.divide(counts_nom, counts_var, out=np.zeros_like(counts_nom), where=counts_var!=0)
        result = {
            "edges_x": self.edges_x.tolist(),
            "edges_y": self.edges_y.tolist(),
            "ratio": ratio.tolist()
        }

        with open(f"{self.hist_name}.json", "w") as f:
            json.dump(result, f, indent=6)


def main(datasets):
    '''
    for dataset_name, sample_names in datasets.items():
        for sample_name in sample_names:
            histogrammer = Histogrammer(sample_name=sample_name)
            data = [f"inputs/{sample_name}/{f}" for f in os.listdir(f"inputs/{sample_name}") if f.endswith(".h5")]
            for i, file in enumerate(data):
                histogrammer.Process(file=file)
            histogrammer.Save()
    return
    '''
    for hist_name, hist_info, in get_hists().items():
        plotter = Plotter(hist_name=hist_name, hist_info=hist_info)
        if "2d" in hist_name:
            for dataset_name, sample_names in datasets.items():
                plotter.Stack2D(dataset_name, sample_names)
            for dataset_name in list(datasets.keys()):
                plotter.Plot2D(dataset_name)
            plotter.Plot2DRatio(nominal="Sherpa", variation="MadGraph")
            plotter.Save2DRatio(nominal="Sherpa", variation="MadGraph")
        else:
            for dataset_name, sample_names in datasets.items():
                plotter.Stack(dataset_name, sample_names)
            plotter.Plot("Sherpa", ["MadGraph"])

if __name__ == "__main__":
    datasets = {
        "Sherpa" : ["Sherpa"],
        "MadGraph" : ["MadGraph_100to200_1000", "MadGraph_100to200_200to400", "MadGraph_100to200_400to600", "MadGraph_100to200_40to200", "MadGraph_100to200_600to1000", "MadGraph_200_1000", "MadGraph_200_400to600", "MadGraph_200_40to400", "MadGraph_200_600to1000"]
    }

    main(datasets)
