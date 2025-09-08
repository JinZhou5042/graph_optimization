import os

import itertools
from collections import OrderedDict

import multiprocessing as mp
import json

import pandas as pd
import numpy as np
from numba import jit
from scipy.stats import chi2

import dask_awkward as dak
import awkward as ak

import matplotlib.pyplot as plt

from analysis import variables as var
from analysis import calculations as calc
from analysis import plotting as pl

from analysis_tools import storage_config


### Selections
triggers = {
    "2016preVFP": "DoublePhoton60",
    "2016postVFP": "DoublePhoton60",
    "2017": "DoublePhoton70",
    "2018": "DoublePhoton70"
}


def preselection(ev, year):

    trigger = ev.HLT[triggers[year]]
    ev = ev[trigger]

    # Clean candidate constituents
    ev["Photon"] = ev.Photon[ev.Photon.pt > 60]

    ev["MLPhoton"] = ev.MLPhoton[
        (ev.MLPhoton.pt > 60) &
        (ev.MLPhoton.massEnergyRatio > 0) &
        (ev.MLPhoton.diphotonScore > 0)
    ]

    # Get candidates
    candidates = calc.candidates(ev)
    candidates = candidates[candidates.delta_r > 0.1]

    has_candidate = dak.num(candidates) > 0
    ev = ev[has_candidate]

    return ev

reference_triggers = {
    "2016": "Ele27_eta2p1_WPLoose_Gsf",
    "2016_HIPM": "Ele27_eta2p1_WPLoose_Gsf",
    "2017": "Ele40_WPTight_Gsf",
    "2018": "Ele40_WPTight_Gsf"
}

def offline_pt(ev):
    ev["Photon"] = ev.Photon[ev.Photon.pt > 90]
    ev["MLPhoton"] = ev.MLPhoton[ev.MLPhoton.pt > 90]

    ev["Candidates"] = calc.candidates(ev)
    ev = ev[ak.num(ev.Candidates) > 0]

    ev["Candidate"] = calc.candidate(ev.Candidates)

    return ev

def trigger_study(ev, year):

    if year.startswith('2016'):
        year = '2016'
    trigger = ev.HLT[reference_triggers[year]]
    ev = ev[trigger]

    # Clean candidate constituents
    photons = ev.Photon[ev.Photon.pt > 50]

    diphotons = ev.MLPhoton[
        (ev.MLPhoton.pt > 50) &
        (ev.MLPhoton.massEnergyRatio > 0) &
        (ev.MLPhoton.diphotonScore > 0)
    ]

    ev["Photon"] = photons
    ev["MLPhoton"] = diphotons

    # Get candidates
    candidates = calc.candidates(ev)
    candidates = candidates[candidates.delta_r > 0.1]

    has_candidate = dak.num(candidates) > 0
    ev = ev[has_candidate]

    return ev

def SR(ev, photon_pt_cut=90, diphoton_pt_cut=90):
    photonPt = ev.Candidates.photon.pt > photon_pt_cut
    photonID = ev.Candidates.photon.cutBased > 0
    pho_pass = photonID #& photonPt

    diphotonPt = ev.Candidates.diphoton.pt > diphoton_pt_cut
    diphotonScore = ev.Candidates.diphoton.diphotonScore > 0.90
    diphotonIso = ev.Candidates.diphoton.pfIsolation > 0.90
    dipho_pass = diphotonScore & diphotonIso #& diphotonPt

    # delta_r = ev.Candidates.delta_r > 1
    # ket_frac = ev.Candidates.ket_frac < 0.4
    # jet_energy_frac = ev.Candidates.jet_energy_frac < 0.90
    # ev_pass = delta_r & ket_frac & jet_energy_frac

    ev["Candidates"] = ev.Candidates[pho_pass & dipho_pass]
    ev = ev[ak.num(ev.Candidates) > 0]
    ev["Candidate"] = calc.candidate(ev.Candidates)
    return ev

def CR(ev, photon_pt_cut=90, diphoton_pt_cut=90):
    photonPt = ev.Candidates.photon.pt > photon_pt_cut
    photonID = ev.Candidates.photon.cutBased > 0
    pho_pass = photonPt & photonID
    
    diphotonPt = ev.Candidates.diphoton.pt > diphoton_pt_cut
    diphotonIso = (ev.Candidates.diphoton.pfIsolation < 0.82) & (ev.Candidates.diphoton.pfIsolation > 0.4)
    diphotonScore = ev.Candidates.diphoton.diphotonScore > 0.9
    dipho_pass = diphotonPt & diphotonScore & diphotonIso

    # ket_frac = ev.Candidates.ket_frac < 0.4
    # delta_r = ev.Candidates.delta_r > 1
    # jet_energy_frac = ev.Candidates.jet_energy_frac < 0.90
    # ev_pass = delta_r & ket_frac & jet_energy_frac

    ev["Candidates"] = ev.Candidates[pho_pass & dipho_pass]# & ev_pass]
    ev = ev[ak.num(ev.Candidates) > 0]
    ev["Candidate"] = calc.candidate(ev.Candidates)
    return ev

selections = {
    "preselection": preselection,
    "SR": SR,
    "CR": CR,
    "offline_pt": offline_pt,
}

def select(ev, selection):
    if selection not in selections:
        raise ValueError(f"Selection {selection} not found.")
    return selections[selection](ev)


cache_dir = f"{storage_config.cache_dir}/selections"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

vars_and_cuts = OrderedDict({
    'ket_frac': {
        "op" : "<",
        "cuts": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # "cuts": [0.4]
    },
    # 'energy_ratio': {
    #   "op" : ">",
    #   "cuts": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    #   },
    'diphoton_isolation': {
        "op" : ">",
        "cuts": [0.6, 0.7, 0.75, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.95] 
        # "cuts": [0.9]
        },
    'diphoton_score': {
        "op" : ">",
        "cuts": [0.0, 0.1, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999]
        # "cuts": [0.9]
        },
    'jet_energy_frac': {
        "op" : "<",
        "cuts": [0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 1.0]
        # "cuts": [0.90]
        },
    'photon_id' : {
        "op" : ">=",
        "cuts": [0.0, 1.0, 2.0, 3.0]
        # "cuts": [1.0]
      },
    # 'photon_mvaID' : {
    #   "op" : ">",
    #   "cuts": [-0.25, -0.2, -0.175, -0.15, -0.125, -0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1]
    #   },
    # 'photon_mvaID' : {
    #     "op" : ">",
    #     "cuts": [-0.02]
    #     },
    'delta_r': {
        "op" : ">",
        "cuts": [0.4, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
        },
    # 'delta_eta': {
    #   "op" : ">",
    #   "cuts": [ 0.025, 0.075, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0]
    #   },
})

def plot_cut_table():
    headers = ["variable", "direction", "cut grid"]
    rows = []

    for var_name, var_dict in vars_and_cuts.items():
        if var_name == "photon_id":
            cut_names = "none, loose, medium, tight"
            direction = ""
        else:
            cut_names = ", ".join([str(c) for c in var_dict["cuts"]])
            direction = var_dict["op"]

        rows.append([
            var.variables[var_name]["label"],
            direction,
            cut_names
        ])
    
    fig, ax = plt.subplots(figsize=(12,len(rows)*0.3))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.auto_set_column_width(col=list(range(len(headers))))
    for i in range(len(rows)+1):
        for j in range(len(headers)):
            # table[i+1, j].set_fontsize(16)
            table[i, j].set_height(0.4)



@jit
def cut_and_count(vals, cut_combinations, ops, weights):
    # Loop over every possible combination of cuts and count the number of ev that pass
    counts = {}
    for cuts in cut_combinations:
        mask = np.ones(vals.shape[0], dtype=np.int64)
        for j in range(len(cuts)):
            if ops[j] == "<":
                mask = mask & (vals[:,j] < cuts[j])
            elif ops[j] == ">":
                mask = mask & (vals[:,j] > cuts[j])
            elif ops[j] == "<=":
                mask = mask & (vals[:,j] <= cuts[j])
            elif ops[j] == ">=":
                mask = mask & (vals[:,j] >= cuts[j])
        if weights is not None:
            mask = mask.astype(np.float64)
            counts[cuts] = np.sum(mask*weights)
        else:
            counts[cuts] = np.sum(mask)
    return counts

@jit
def calculate_significance(signal_counts, data_counts, xs_factor):
    sigs = {}
    for combo in signal_counts.keys():
        signal = signal_counts[combo]*xs_factor
        data = data_counts[combo]
        denom = np.sqrt(signal+data)
        if denom == 0:
            sigs[combo] = 0.0
        else:
            sigs[combo] = signal / denom
    return sigs

def search_grid_point(
        signal_skim, data_skims, signal_xs_factors,
        print_time=False,
        print_corr=False,
        ):
    if print_time:
        import time
        t1 = time.time()
    
    # print(f"Optimizing cuts for signal point: {signal_skim.sample_name}")

    signal_skim.scale_mc()

    # Get the points in the signal ellipse
    variables = list(vars_and_cuts.keys()) + ['diphoton_mass', 'triphoton_mass']
    signal_df = pd.DataFrame({var_name: signal_skim[var_name] for var_name in variables})
    signal_weights = signal_skim.get('weight').to_numpy()
    data_df = pd.DataFrame({var_name: data_skims[var_name] for var_name in variables})

    # Preliminary selections to remove outliers
    M_BKK = signal_skim.signal_point.M_BKK
    M_R = signal_skim.signal_point.M_R
    signal_mask1 = (signal_df['triphoton_mass'] > M_BKK*0.8) & (signal_df['triphoton_mass'] < M_BKK*1.2) \
        & (signal_df['diphoton_mass'] > M_R*0.5) & (signal_df['diphoton_mass'] < M_R*1.5)
    
    data_mask1 = (data_df['triphoton_mass'] > M_BKK*0.8) & (data_df['triphoton_mass'] < M_BKK*1.2) \
        & (data_df['diphoton_mass'] > M_R*0.5) & (data_df['diphoton_mass'] < M_R*1.5)
    
    signal_df = signal_df[signal_mask1]
    signal_weights = signal_weights[signal_mask1]
    data_df = data_df[data_mask1]

    # Fit an ellipse to the signal points
    ellipse_pars = confidence_ellipse(signal_df['triphoton_mass'], signal_df['diphoton_mass'])

    signal_in_signal_window = in_ellipse(signal_df['triphoton_mass'], signal_df['diphoton_mass'], ellipse_pars)
    data_in_signal_window = in_ellipse(data_df['triphoton_mass'], data_df['diphoton_mass'], ellipse_pars)

    signal_df = signal_df[signal_in_signal_window]
    signal_weights = signal_weights[signal_in_signal_window]
    data_df = data_df[data_in_signal_window]

    if print_corr:
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 500)
        print("Signal correlations")
        print(signal_df.corr())
        print("Data correlations")
        print(data_df.corr())

    # Get the variables and cuts
    var_names = list(vars_and_cuts.keys())
    var_cuts = [vars_and_cuts[var_name]["cuts"] for var_name in var_names]
    ops = [vars_and_cuts[var_name]["op"] for var_name in var_names]

    signal_vals = signal_df[var_names].values
    data_vals = data_df[var_names].values

    cut_combinations = list(map(tuple, itertools.product(*var_cuts)))

    # Calculate the number of ev that pass the cuts
    signal_counts = cut_and_count(signal_vals, cut_combinations, ops, signal_weights)
    data_counts = cut_and_count(data_vals, cut_combinations, ops, np.ones(data_vals.shape[0]))

    # Calculate the significance
    significance = {}
    for xs_factor in signal_xs_factors:
        significance[xs_factor] = calculate_significance(signal_counts, data_counts, xs_factor)

    if print_time:
        t2 = time.time()
        print(f"Time taken: {t2-t1}")
    return significance

def best_cut_table(significance, signal_dataset, save=True, plot=True):
    optimal_cuts = []
    for xs_factor, sig_dict in significance.items():
        vals = np.array(list(sig_dict.values()))

        #Print the cut values for the maximum significance
        max_cuts = max(sig_dict, key=sig_dict.get) 
        max_cut_dict = dict(zip(vars_and_cuts.keys(), max_cuts))
        # print(f"Max significance for xs_factor = {xs_factor}: {sig_dict[max_cuts]}")
        optimal_cuts.append({"xs_factor": xs_factor, "significance": np.round(sig_dict[max_cuts],3), **max_cut_dict})


    df = pd.DataFrame(optimal_cuts)


    # Convert the photon id column to an int and then to a string [0.0, 1.0] -> ['none', 'loose', 'medium', 'tight']
    df['photon_id'] = df['photon_id'].astype(int).map({0: 'none', 1: 'loose', 2: 'medium', 3: 'tight'}).astype(str)


    # replace the column names with their labels as found in var.variables
    col_names = df.columns.to_list()
    for i, col in enumerate(col_names):
        if col in var.variables:
            col_names[i] = var.variables[col]['label']
    df.columns = col_names


    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10,1))

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    if save:
        if not os.path.exists(f"{cache_dir}/optimal_cuts"):
            os.makedirs(f"{cache_dir}/optimal_cuts")

        fig.savefig(f"{cache_dir}/optimal_cuts/{signal_dataset.name}_optimal_cuts.png")

    if plot:
        plt.show()
    else:
        plt.close(fig)
    
    return df

def plot_significance_variation(
    significance,
    signal_dataset,
    vars_to_vary=list(vars_and_cuts.keys()),
    save=True,
    plot=True
    ):

    fig, axs = plt.subplots(len(vars_to_vary)//2, 2, figsize=(12, 6*len(vars_to_vary)//2 + 1))
    axs = axs.flatten()

    cache = f"{cache_dir}/significance_variation"
    if not os.path.exists(cache):
        os.makedirs(cache)

    icolor=0
    colors = pl.colors
    for signal_wgt, sig in significance.items():
        vals = np.array(list(sig.values()))

        #Print the cut values for the maximum significance
        max_cuts = max(sig, key=sig.get) 
        max_cut_dict = dict(zip(vars_and_cuts.keys(), max_cuts))

        for i, var_name in enumerate(vars_to_vary):
            var_cuts = vars_and_cuts[var_name]["cuts"]
            var_sig = []
            for cut_val in var_cuts:
                new_cuts = max_cut_dict.copy()
                new_cuts[var_name] = cut_val
                var_sig.append(sig[tuple(new_cuts.values())])

            axs[i].plot(var_cuts, var_sig, label=f"{signal_wgt}x $\\sigma_{{ \\text{{signal}} }}$", color=colors[icolor])
            axs[i].set_xlabel(var.variables[var_name]['label'])
            axs[i].set_ylabel("Significance")
            axs[i].legend()

        icolor += 1

    # Don't plot the last axis if it's empty
    if len(vars_to_vary) % 2 != 0:
        axs[-1].axis('off')

    plt.tight_layout()

    signal_name = signal_point_tag.replace("_", ", ").replace("-", "=")
    plt.suptitle(f'Significance variation for {signal_name}', y=1.02)

    if save:
        fig.savefig(f"{cache}/{signal_dataset.name}_significance_variation.png")
    
    if plot:
        plt.show()
    else:
        plt.close(fig)

def confidence_ellipse(x, y, CL=0.95):
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    c2 = chi2.ppf(CL, 2)
    l1 = c2 * eigenvalues[0]
    l2 = c2 * eigenvalues[1]
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    return (np.mean(x), np.mean(y)), 2*np.sqrt(l1), 2*np.sqrt(l2), theta*180/np.pi

def in_ellipse(x, y, ellipse_pars):
    center, width, height, theta = ellipse_pars
    x0, y0 = center
    x_centered, y_centered = x - x0, y - y0
    theta = theta*np.pi/180
    x_rotated = np.cos(theta)*x_centered + np.sin(theta)*y_centered
    y_rotated = -np.sin(theta)*x_centered + np.cos(theta)*y_centered
    return (x_rotated/(width/2))**2 + (y_rotated/(height/2))**2 < 1
