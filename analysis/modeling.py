import os
import importlib

import multiprocessing

from typing import Union

import json

import multiprocessing as mp

import ROOT
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from analysis import skim as sk
from analysis import variables as var
from analysis import plotting as pl
from analysis import systematics as sys

from analysis_tools import storage_config
from analysis_tools import conversion
from analysis_tools import signal_info as si

from scipy.optimize import curve_fit

top_dir = storage_config.top_dir
cache_base = f"{storage_config.cache_dir}/modeling"
if not os.path.exists(cache_base):
    os.makedirs(cache_base, exist_ok=True)

fit_values_dir = f"{cache_base}/fit_pars"
if not os.path.exists(fit_values_dir):
    os.makedirs(fit_values_dir, exist_ok=True)

plots_dir = f"{cache_base}/fit_plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

root_cache = f"{cache_base}/root_files"
if not os.path.exists(root_cache):
    os.makedirs(root_cache)

import random
random_string = lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))

def plot_correlation(
    x1_values,
    x2_values,
    x1_spec = (4, 150, 1000), # (nbins, min, max)
    x2_spec = (4, 0.001, 0.04),
    x1_name = "triphoton_mass",
    x2_name = "alpha"
    ):


    mask = (x1_values > x1_spec[1]) & (x1_values < x1_spec[2]) & (x2_values > x2_spec[1]) & (x2_values < x2_spec[2])
    tm_values = x1_values[mask]
    a_values = x2_values[mask]

    # Print the overall correlation
    print(f"Overall correlation between {x1_name} and {x2_name}: {np.corrcoef(tm_values, a_values)[0, 1]}")

    tm_bin_edges = np.percentile(tm_values, np.linspace(0, 100, x1_spec[0]+1))
    a_bin_edges = np.percentile(a_values, np.linspace(0, 100, x2_spec[0]+1))

    corr_in_tm_bins = []
    for i in range(x1_spec[0]):
        mask = (tm_values >= tm_bin_edges[i]) & (tm_values < tm_bin_edges[i+1])
        corr_in_tm_bins.append(np.corrcoef(tm_values[mask], a_values[mask])[0, 1])

    corr_in_a_bins = []
    for i in range(x2_spec[0]):
        mask = (a_values >= a_bin_edges[i]) & (a_values < a_bin_edges[i+1])
        corr_in_a_bins.append(np.corrcoef(tm_values[mask], a_values[mask])[0, 1])

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    twin_ax = [ax[0].twinx(), ax[1].twinx()]
    ax[0].step(tm_bin_edges, np.append(corr_in_tm_bins, corr_in_tm_bins[-1]), where='post', label='Correlation')
    ax[0].set_xlabel(var.get_label(x1_name))
    ax[0].set_ylabel('Correlation')
    tm_hist = var.get_hist('triphoton_mass', bounds=x1_spec[1:], nbins=16).fill(tm_values)
    tm_hist.plot(ax=twin_ax[0], color='red', density=True, label='Density')
    ax[0].legend()

    ax[1].step(a_bin_edges, np.append(corr_in_a_bins, corr_in_a_bins[-1]), where='post', label='Correlation')
    ax[1].set_xlabel(var.get_label(x2_name))
    ax[1].set_ylabel('Correlation')
    ax[1].legend()
    a_hist = var.get_hist('alpha', bounds=x2_spec[1:], nbins=16).fill(a_values)
    a_hist.plot(ax=twin_ax[1], color='red', density=True, label='Density')

    fig.tight_layout()
    plt.show()

def uniform(low, high):
    return np.random.uniform(low, high), low, high

class ExponentialMixtureModel2D:
    def __init__(
        self,
        tm_var, a_var,
        n_exp_tm, n_exp_a,
        prefix="bkg",
    ):

        # Exponential Mixture Model
        self.tm_ws = [ROOT.RooRealVar(f"bkg_tm_w_{i}", f"Background mixture weight {i}", 1/n_exp_tm, 0, 1) for i in range(n_exp_tm-1)]
        self.tm_raw_rates = [ROOT.RooRealVar(f"bkg_tm_raw_rate_{i}", f"Unordered rate {i}", -0.01, -1, -0.001) for i in range(n_exp_tm)]
        self.tm_ordered_rates = [ROOT.RooAddition(f"bkg_tm_rate_{i}", f"Ordered Rate for exponential {i}", ROOT.RooArgList(*self.tm_raw_rates[:i])) for i in range(n_exp_tm)]
        self.tm_exps = [ROOT.RooExponential(f"bkg_tm_exp_{i}", f"exp_{i}", tm_var, self.tm_ordered_rates[i]) for i in range(n_exp_tm)]

        self.a_ws = [ROOT.RooRealVar(f"bkg_a_w_{i}", f"Background mixture weight {i}", 1/n_exp_a, 0, 1) for i in range(n_exp_a-1)]
        self.a_raw_rates = [ROOT.RooRealVar(f"bkg_a_raw_rate_{i}", f"Unordered rate {i}", -0.01, -1000, -0.001) for i in range(n_exp_a)]
        self.a_ordered_rates = [ROOT.RooAddition(f"bkg_a_rate_{i}", f"Ordered Rate for exponential {i}", ROOT.RooArgList(*self.a_raw_rates[:i])) for i in range(n_exp_a)]
        self.a_exps = [ROOT.RooExponential(f"bkg_a_exp_{i}", f"exp_{i}", a_var, self.a_ordered_rates[i]) for i in range(n_exp_a)]

        self.tm_pdf = ROOT.RooAddPdf("model_bkg_tm", "Background model for triphoton mass", ROOT.RooArgList(*self.tm_exps), ROOT.RooArgList(*self.tm_ws), True)
        self.a_pdf = ROOT.RooAddPdf("model_bkg_a", "Background model for a", ROOT.RooArgList(*self.a_exps), ROOT.RooArgList(*self.a_ws), True)
        self.tail_pdf = ROOT.RooProdPdf("tail_pdf", "tail_pdf", self.tm_pdf, self.a_pdf)

        # Normal CDF Efficiency
        self.tm_mean = ROOT.RooRealVar("tm_mean", "trig_m_mean", 220, -260, 300)
        self.tm_sigma = ROOT.RooRealVar("tm_sigma", "trig_m_sigma", 20, 0, 50)

        self.a_mean = ROOT.RooRealVar("a_mean", "a_mean", 0.004, 0.002, 0.006)
        self.a_sigma = ROOT.RooRealVar("a_sigma", "a_sigma", 0.002, 0, 0.005)

        self.tm_eff = ROOT.RooFormulaVar(
            "tm_eff",
            "0.5*(1 + erf((@0-@1)/(sqrt(2)*@2)))",
            ROOT.RooArgList(tm_var, self.tm_mean, self.tm_sigma)
        )
        self.a_eff = ROOT.RooFormulaVar(
            "a_eff",
            "0.5*(1 + erf((@0-@1)/(sqrt(2)*@2)))",
            ROOT.RooArgList(a_var, self.a_mean, self.a_sigma)
        )

        self.eff = ROOT.RooFormulaVar("eff", "@0*@1", ROOT.RooArgList(self.tm_eff, self.a_eff))
        self.pdf = ROOT.RooEffProd("model", "model", self.tail_pdf, self.eff)

class JohnsonSUModel2D:
    def __init__(
        self,
        triphoton_mass_var, alpha_var,
        M_BKK, Mass_Ratio,
        prefix="sig",
        **par_specs
    ):

        fitted_par_specs = {
            'dM_BKK': uniform(-0.1*M_BKK, 0.1*M_BKK),
            'triphoton_mass_lambda': uniform(1,50),
            'triphoton_mass_gamma': (0, -5, 5),
            'triphoton_mass_delta': uniform(0.1,5),
            'dMass_Ratio': uniform(-0.2*Mass_Ratio, 0.2*Mass_Ratio),
            'alpha_lambda': uniform(1e-5, 20),
            'alpha_gamma': (0, -5, 5),
            'alpha_delta': uniform(0.1,5)
        }

        for key, spec in par_specs.items():
            if isinstance(spec, tuple):
                fitted_par_specs[key] = spec
            elif isinstance(spec, float):
                _spec = fitted_par_specs[key]
                fitted_par_specs[key] = (spec, _spec[1], _spec[2])

        fitted = {}
        other = {}

        ## Constants
        other['M_BKK'] = ROOT.RooRealVar("M_BKK", "M_BKK", M_BKK, 0.99 * M_BKK, 1.01 * M_BKK)
        other['M_BKK'].setConstant(True)

        other['Mass_Ratio'] = ROOT.RooRealVar("Mass_Ratio", "Mass_Ratio", Mass_Ratio, 0.99 * Mass_Ratio, 1.01 * Mass_Ratio)
        other['Mass_Ratio'].setConstant(True)

        ## Fitted Parameters
        for key, spec in fitted_par_specs.items():
            fitted[key] = ROOT.RooRealVar(key, key, *spec)

        other['triphoton_mass_mean'] = ROOT.RooFormulaVar("triphoton_mass_mean", "mean", "@0+@1", ROOT.RooArgList(other['M_BKK'], fitted['dM_BKK']))
        other['triphoton_mass_pdf'] = ROOT.RooJohnson(
            "triphoton_mass_pdf", "triphoton_mass_pdf", triphoton_mass_var,
            other['triphoton_mass_mean'],
            fitted['triphoton_mass_lambda'],
            fitted['triphoton_mass_gamma'],
            fitted['triphoton_mass_delta']
            )

        other['alpha_mean'] = ROOT.RooFormulaVar("alpha_mean", "mean", "@0+@1", ROOT.RooArgList(other['Mass_Ratio'], fitted['dMass_Ratio']))
        other['alpha_pdf'] = ROOT.RooJohnson(
            "alpha_pdf", "alpha_pdf", alpha_var,
            other['alpha_mean'],
            fitted['alpha_lambda'],
            fitted['alpha_gamma'],
            fitted['alpha_delta']
            )
        
        other['pdf'] = ROOT.RooProdPdf("model_sig", "model_sig", ROOT.RooArgList(other["triphoton_mass_pdf"], other["alpha_pdf"]))
        self.pdf = other['pdf']

        self.fitted = fitted
        self.other = other

## Signal Fit Parameters
def get_root_filename(d, selection="SR"):
    return f"{root_cache}/{d.name}_{selection}.root"

def make_root_files(
    selection="SR",
    multiprocess = False,
    ):

    if multiprocess:
        signal_skims = sk.load_skims("signal", "preselection")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//4) as pool:
            results = pool.map(make_root_file, signal_skims)
    else:
        signal_skims = sk.load_skims("signal", "preselection")
        [make_root_file(a_skim, selection=selection, remake=True) for a_skim in signal_skims]

def make_root_file(
    d: sk.SkimmedDataset,
    selection="SR",
    remake=False
    ):
    from analysis import selections as sel

    root_file = get_root_filename(d, selection=selection)
    if not os.path.exists(root_file) or remake:
        print(f"Creating root file named {root_file}")
        events = d.events

        ev_nom = sel.select(events, selection)

        f = ROOT.TFile(root_file, "RECREATE")

        trees = {}
        trees['signal_nominal'] = conversion.to_root_tree(
            [
                ev_nom.Candidate.triphoton.mass,
                ev_nom.Candidate.alpha
            ],
            "signal_nominal",
            ["triphoton_mass", "alpha"]
        )

        for systematic, func in sys.shape_systematics.items():
            ev_up, ev_down = func(events, d.year)

            ev_up = sel.select(ev_up, selection)
            trees[f"signal_{systematic}_up"] = conversion.to_root_tree(
                [
                    ev_up.Candidate.triphoton.mass,
                    ev_up.Candidate.alpha
                ],
                f"signal_{systematic}_up",
                ["triphoton_mass", "alpha"]
            )

            ev_down = sel.select(ev_down, selection)
            trees[f"signal_{systematic}_down"] = conversion.to_root_tree(
                [
                    ev_down.Candidate.triphoton.mass,
                    ev_down.Candidate.alpha
                ],
                f"signal_{systematic}_down",
                ["triphoton_mass", "alpha"]
            )

        for tree_name, tree in trees.items():
            tree.Write()

        f.Close()
    
    return root_file

# Signal Efficiency
def process_skim(a_skim):
    sp = a_skim.signal_point

    n_SR = len(a_skim.events)
    n_total = a_skim.get_n_nanoaod_events()

    result = {
        "era": a_skim.era,
        "M_BKK": sp.M_BKK,
        "M_R": sp.M_R,
        "Mass_Ratio": sp.M_R/sp.M_BKK,
        "efficiency": n_SR/n_total,
        "efficiency_error": np.sqrt(n_SR)/n_total
    }
    return result

def get_signal_efficiency(remake=False, per_year=False, selection="SR"):
    signal_efficiency_cache = f"{cache_base}/signal_efficiencies.json"
    if per_year:
        signal_efficiency_cache = f"{cache_base}/signal_efficiencies_per_year.json"

    if os.path.exists(signal_efficiency_cache) and not remake:
        signal_efficiencies = json.load(open(signal_efficiency_cache))
    else:
        signal_skims = sk.load_skims("signal", "preselection", apply_selection=selection)
        if not per_year: 
            signal_skims = [signal_skims[sp] for sp in signal_skims.signal_points]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as pool:
            results = pool.map(process_skim, signal_skims)

        signal_efficiencies = []
        for result in results:
            signal_efficiencies.append(result)

        json.dump(signal_efficiencies, open(signal_efficiency_cache, "w"), indent=4)

    # Convert to dataframe
    df = pd.DataFrame(signal_efficiencies)
    return df

def regress_signal_efficiency(M_BKK, M_R):
    df = get_signal_efficiency()
    reg = SignalEfficiencyRegressionGP(df, ['efficiency'])
    return reg.predict(M_BKK, M_R)['efficiency']

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern, RationalQuadratic
class SignalEfficiencyRegressionGP:
    def __init__(self, df, outcome_names):
        self.df = df
        self.outcome_names = outcome_names
        self.models = {}

        # Normailize X
        self.X_raw = df[['M_BKK', 'Mass_Ratio']]
        self.X_min = self.X_raw.min()
        self.X_max = self.X_raw.max()
        self.X = (self.X_raw - self.X_min) / (self.X_max - self.X_min)

        # self.X = df[['M_BKK', 'Mass_Ratio']]
        self.Y_raw = df[outcome_names]
        self.Y_raw_errs = df[[f"{outcome}_error" for outcome in outcome_names]]
        self.Y_raw_errs.columns = outcome_names

        self.Y_min = self.Y_raw.min()
        self.Y_max = self.Y_raw.max()
    
        self.Y = (self.Y_raw - self.Y_min) / (self.Y_max - self.Y_min)
        self.Y_errs = self.Y_raw_errs / (self.Y_max - self.Y_min)

        self.fit()
    
    # def mean_fn(self, x, a, b, c, d):
    #     return a - b*np.exp(c*x[0])

    # def mean_fn(self, x, a, b, c, d):
    #     return a - b*np.log(x[0]+c) + d*x[1]

    def mean_fn(self, x, a, b, c):
        return a - b*np.log(np.abs(x[0]+c))

    # def mean_fn(self, x, a, b, c):
    #     return a - b/(x[0]+c) 

    # def mean_fn(self, x, a, b):
    #     return a*x[0] + b*x[1]

    # def mean_fn(self, x, a, b, c, d):
    #     return a*x[0] + b*x[1] + c*x[0]*x[1] + d

    def fit(self):
        # Use a Gaussian Process

        self.mean_model = {}
        self.gp_model = {}

        for outcome in self.outcome_names:
            mean_reg = curve_fit(self.mean_fn, self.X.values.T, self.Y[outcome].values)
            residuals = self.Y[outcome].values - self.mean_fn(self.X.values.T, *mean_reg[0])

            # kernel = ConstantKernel()*RBF()
            kernel = RBF([1,1])
            # kernel = RationalQuadratic()
            # kernel = ConstantKernel()*RationalQuadratic()
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.Y_errs[outcome].values,
                n_restarts_optimizer=10
            ).fit(self.X, residuals)

            self.mean_model[outcome] = mean_reg
            self.gp_model[outcome] = gp

    
    def predict(self, M_BKK, Mass_Ratio):
        X_raw = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        X = (X_raw - self.X_min) / (self.X_max - self.X_min)
        predictions = {}
        for outcome in self.outcome_names:
            mean_prediction = self.mean_fn(X.values.T, *self.mean_model[outcome][0])[0]
            gp_prediction = self.gp_model[outcome].predict(X)[0]
            predictions[outcome] = (mean_prediction + gp_prediction) * (self.Y_max[outcome] - self.Y_min[outcome]) + self.Y_min[outcome]
        return predictions

    def predict_error(self, M_BKK, Mass_Ratio):
        X_raw = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        X = (X_raw - self.X_min) / (self.X_max - self.X_min)
        predictions = {}
        for outcome in self.outcome_names:
            gp_prediction = self.gp_model[outcome].predict(X, return_std=True)[1][0]
            predictions[outcome] = gp_prediction * (self.Y_max[outcome] - self.Y_min[outcome])
        return predictions

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
class SignalEfficiencyRegressionKNN:
    def __init__(self, df, outcome_names):
        self.df = df
        self.outcome_names = outcome_names
        self.models = {}
        self.X = df[['M_BKK', 'Mass_Ratio']]
        self.Y = df[outcome_names]
    
        self.linear_outcomes = []
        
        self.fit()
    
    def fit(self):

        for outcome in self.outcome_names:
            if outcome in self.linear_outcomes:
                reg = LinearRegression().fit(self.X, self.Y[outcome])
                self.models[outcome] = reg
            else:
                reg = KNeighborsRegressor(n_neighbors=2).fit(self.X, self.Y[outcome])
                self.models[outcome] = reg
    
    def predict(self, M_BKK, Mass_Ratio):

        predictions = {}
        for outcome in self.outcome_names:
            X = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
            prediction = self.models[outcome].predict(X)[0]
            predictions[outcome] = prediction
        return predictions

# Signal Cross Section
def get_signal_xs():

    signal_skims = sk.load_skims("signal", "preselection")
    
    df = []
    for proc in signal_skims.signal_processes:
        some_skims = signal_skims[proc]
        signal_points = some_skims.signal_points
        
        for sp in signal_points:
            xs_info = si.get_signal_xs_pb(proc, sp.M_BKK, sp.M_R)
            df.append({
                "Process": proc,
                "M_BKK": sp.M_BKK,
                "Mass_Ratio": sp.M_R/sp.M_BKK,
                "xs": xs_info['xs'],
                "xs_error": xs_info['error']
            })

    df = pd.DataFrame(df)

    # Multiply by k factor for Process = BkkToGRadionToGGG
    k = get_k_factor()
    df.loc[df['Process'] == 'BkkToGRadionToGGG', 'xs'] *= k

    return df

class SignalXSRegression:
    def __init__(self, df, outcome_names):
        self.df = df
        self.outcome_names = outcome_names
        self.models = {}

        self.X = df[['M_BKK', 'Mass_Ratio']]
        # self.X_raw = df[['M_BKK', 'Mass_Ratio']]
        # self.X_max = self.X_raw.max()
        # self.X = self.X_raw/self.X_max
        self.Y_raw = df[outcome_names]
        self.Y = np.log(self.Y_raw)
        # self.logY = np.log(self.Y)

        self.fit()


    def fit_poly(self, x, a, b, c, d, e, f, g):
        return a + b*x[0] + c*x[0]**2 + d*x[0]**3 + e*x[0]**4 + f*x[0]**5 + g*x[0]**6
    def fit(self):

        self.fit_functions = {
            'xs': self.fit_poly,
        }

        for outcome in self.outcome_names:
            self.models[outcome] = curve_fit(self.fit_functions[outcome], self.X.values.T, self.Y[outcome].values)

    def predict(self, M_BKK, Mass_Ratio):
        
        X = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        # X_raw = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        # X = X_raw/self.X_max

        predictions = {}
        for outcome in self.outcome_names:
            prediction = self.fit_functions[outcome](X.values.T, *self.models[outcome][0])
            predictions[outcome] = np.exp(prediction)

        return predictions

def get_k_factor():
    signal_skims = sk.load_skims('signal', 'preselection')

    signal_skims_by_process = {process: signal_skims[process] for process in signal_skims.signal_processes}
    signal_points_by_process = {process: skim.signal_points for process, skim in signal_skims_by_process.items()}
    signal_points = set.intersection(*signal_points_by_process.values())

    xs_dict = {}
    for process in signal_skims.signal_processes:
        xs_dict[process] = {sp.M_BKK: si.get_signal_xs_pb(process, sp.M_BKK, sp.M_R) for sp in signal_points}

    ks = []
    for sp in signal_points:
        if sp.M_BKK > 600:
            continue
        p1 = "BkkToGRadionJetsToGGGJets"
        p2 = "BkkToGRadionToGGG"
        xs1 = xs_dict[p1][sp.M_BKK]['xs']
        xs2 = xs_dict[p2][sp.M_BKK]['xs']
        ks.append(xs1/xs2)
    k = sum(ks)/len(ks)

    return k

def regress_signal_xs(M_BKK, M_R):
    df = get_signal_xs()
    reg = SignalXSRegression(df, ['xs'])
    return reg.predict(M_BKK, M_R)['xs']

def r_scaling(M_BKK):
    if M_BKK < 300: return 10000
    if M_BKK < 600: return 5000
    if M_BKK < 800: return 1_000
    if M_BKK < 1000: return 500
    if M_BKK < 1200: return 200
    if M_BKK < 1400: return 100
    if M_BKK < 1600: return 50
    if M_BKK < 1800: return 20
    if M_BKK < 2000: return 10
    if M_BKK < 2200: return 6
    if M_BKK < 2400: return 4
    if M_BKK < 2600: return 2
    return 1

# Signal Fit Parameters
def get_fit_parameters(
    remake=False, 
    optimize_tries=10,
    per_year=False,
    remake_root_files=False,
    selection="SR",
    systematic="nominal",
    plot_fits=False,
    reg=None
    ):

    par_cache = f"{cache_base}/fit_pars"
    if per_year:
        signal_par_cache = f"{par_cache}/signal_pars_per_year.json"
        signal_skims = sk.load_skims("signal", "preselection")
    else:
        signal_par_cache = f"{par_cache}/signal_pars.json"
        signal_skims = sk.load_skims("signal", "preselection")
        signal_skims = [signal_skims[sp] for sp in signal_skims.signal_points]

    if systematic != "nominal":
        signal_par_cache = signal_par_cache.replace(".json", f"_{systematic}.json")

    if not os.path.exists(signal_par_cache) or remake:        

        if remake_root_files:
            make_root_files(selection=selection, multiprocess=True)

        pars = []
        for a_skim in signal_skims:
            pars.append((
                a_skim,
                selection,
                plot_fits,
                False,
                optimize_tries,
                f"signal_{systematic}",
                False,
                True,
                reg
            ))
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//4) as pool:
            results = pool.starmap(fit_signal, pars)
        
        signal_pars = []
        for result in results:
            signal_pars.append(result)

        json.dump(signal_pars, open(signal_par_cache, "w"), indent=4)
    else:
        signal_pars = json.load(open(signal_par_cache))

    # Convert to dataframe
    df = pd.DataFrame(signal_pars)
    return df

def fit_signal(
    a_skim,
    selection="SR",
    save_plot=True,
    display_plot=False, 
    n_tries=1,
    tree_name="signal_nominal",
    return_plottables=False,
    return_regression_results=False,
    regression=None,
    ):

    M_BKK = a_skim.signal_point.M_BKK
    Mass_Ratio = a_skim.signal_point.M_R / a_skim.signal_point.M_BKK

    par_specs = {}
    if regression is not None:
        nominal = regression.predict(M_BKK, Mass_Ratio)
        error = regression.predict_error(M_BKK, Mass_Ratio)
        for key in nominal.keys():
            par_specs[key] = nominal[key]

    if isinstance(a_skim, sk.SkimmedDataset):
        root_files = [get_root_filename(a_skim)]
    else:
        root_files = [get_root_filename(_, selection=selection) for _ in a_skim]

    triphoton_mass = ROOT.RooRealVar("triphoton_mass", "Triphoton Mass", 0.9*M_BKK, 1.1*M_BKK)

    alpha_min = 0.6*Mass_Ratio
    alpha_max = 1.4*Mass_Ratio
    if Mass_Ratio <= 0.01:
        alpha_max = 2*Mass_Ratio
    if Mass_Ratio <= 0.005:
        alpha_max = 3*Mass_Ratio
    if Mass_Ratio < 0.003 and M_BKK <= 300:
        alpha_max = 4.5*Mass_Ratio

    alpha = ROOT.RooRealVar("alpha", "alpha", alpha_min, alpha_max)
    # weight = ROOT.RooRealVar("weight", "weight", 0, 0, 1)

    # f = ROOT.TFile(root_file)
    t = ROOT.TChain(tree_name)
    for root_file in root_files:
        t.Add(root_file)

    # mc = ROOT.RooDataSet("signal", "signal", t, ROOT.RooArgSet(triphoton_mass, alpha, weight), "", "weight")
    mc = ROOT.RooDataSet("signal", "signal", t, ROOT.RooArgSet(triphoton_mass, alpha))
    # f.Close()

    nll = None
    fit_pars = {'fitted:': {}, 'other': {}, 'nll': None}
    for i in range(n_tries):

        model = JohnsonSUModel2D(triphoton_mass, alpha, M_BKK, Mass_Ratio, **par_specs)
        model.pdf.fitTo(mc, ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
        nll = model.pdf.createNLL(mc)

        if fit_pars['nll'] is None or nll.getVal() < fit_pars['nll']:
            fit_pars['nll'] = nll.getVal()
            fit_pars['fitted'] = model.fitted
            fit_pars['other'] = model.other

    fit_values = {}
    fit_values.update({key: val.getVal() for key, val in fit_pars['fitted'].items()})
    fit_values.update({f"{key}_error": val.getError() for key, val in fit_pars['fitted'].items()})

    if save_plot:
        from IPython.display import display, Image
        fit_image_file = f"{plots_dir}/{a_skim.name}_{selection}_{tree_name}.png"

        pdf = fit_pars['other']['pdf']

        can, lines = pl.plot_1D_marginals(pdf, mc, triphoton_mass, alpha)
        can.SaveAs(fit_image_file)

        if display_plot:
            return pl.plot_1D_marginals(pdf, mc, triphoton_mass, alpha), pl.plot_2D_pull(pdf, mc, triphoton_mass, alpha, (32, 0.8*M_BKK, 1.2*M_BKK), (32, 0.3*Mass_Ratio, 4*Mass_Ratio))

    if return_regression_results:
        result = {
            "process": a_skim.signal_process,
            "era": a_skim.era,
            "M_BKK": M_BKK,
            "M_R": a_skim.signal_point.M_R,
            "Mass_Ratio": Mass_Ratio,
            }
        result.update(fit_values)
        return result

    return a_skim, fit_values

class SignalRegression:
    def __init__(self, df, outcome_names):
        from sklearn.linear_model import LinearRegression

        self.df = df
        self.outcome_names = outcome_names
        self.models = {}
        self.X = df[['M_BKK', 'Mass_Ratio']]
        self.Y = df[outcome_names]

        for outcome in outcome_names:
            reg = LinearRegression().fit(self.X, self.Y[outcome])
            self.models[outcome] = reg
    
    def linear(self, x, a, b, c):
        return a + b*x[0] + c*x[1]
    
    def quadratic(self, x, a, b, c, d, e, f):
        return a + b*x[0] + c*x[1] + d*x[0]**2 + e*x[1]**2 + f*x[0]*x[1]

    def predict(self, M_BKK, Mass_Ratio, outcome_names=None):
        if outcome_names is None:
            outcome_names = self.outcome_names
        predictions = {}
        for outcome in outcome_names:
            X = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
            prediction = self.models[outcome].predict(X)[0]
            predictions[outcome] = prediction
        return predictions

class SignalRegressionKNN:
    def __init__(self, df, outcome_names):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.linear_model import LinearRegression
        self.df = df
        self.outcome_names = outcome_names
        self.models = {}
        self.X = df[['M_BKK', 'Mass_Ratio']]
        self.Y = df[outcome_names]
    
        self.linear_outcomes = [
            "dM_BKK",
            "triphoton_mass_lambda",
            "triphoton_mass_gamma",
            "triphoton_mass_delta"
            ]
        
        self.fit()
    
    def fit(self):

        for outcome in self.outcome_names:
            if outcome in self.linear_outcomes:
                reg = LinearRegression().fit(self.X, self.Y[outcome])
                self.models[outcome] = reg
            else:
                reg = KNeighborsRegressor(n_neighbors=2).fit(self.X, self.Y[outcome])
                self.models[outcome] = reg
    
    def predict(self, M_BKK, Mass_Ratio):

        predictions = {}
        for outcome in self.outcome_names:
            X = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
            prediction = self.models[outcome].predict(X)[0]
            predictions[outcome] = prediction
        return predictions

class SignalRegressionPoly:
    def __init__(self, df, outcome_names):

        self.df = df
        self.outcome_names = outcome_names
        self.models = {}

        # Normailize X
        self.X_raw = df[['M_BKK', 'Mass_Ratio']]
        self.X_min = self.X_raw.min()
        self.X_max = self.X_raw.max()
        self.X = (self.X_raw - self.X_min) / (self.X_max - self.X_min)

        # Normailize Y
        self.Y_raw = df[outcome_names]
        self.Y_min = self.Y_raw.min()
        self.Y_max = self.Y_raw.max()
        self.Y = (self.Y_raw - self.Y_min) / (self.Y_max - self.Y_min)

        self.fit()


    def fit_1st_degree_polynomial(self, x, a, b, c, d):
        return a + b*x[0] + c*x[1] + d*x[0]*x[1]
    
    def fit_2nd_degree_polynomial(self, x, a, b, c, d, e, f, g, h, i):
        return a + b*x[0] + c*x[1] + d*x[0]*x[1] + e*x[0]**2 + f*x[1]**2 + g*x[0]**2*x[1] + h*x[0]*x[1]**2 + i*x[0]**2*x[1]**2
    
    def fit_3rd_degree_polynomial(self, x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
        return a + b*x[0] + c*x[1] + d*x[0]*x[1] + e*x[0]**2 + f*x[1]**2 + g*x[0]**2*x[1] + h*x[0]*x[1]**2 + i*x[0]**2*x[1]**2 + j*x[0]**3 + k*x[1]**3 + l*x[0]**3*x[1] + m*x[0]*x[1]**3 + n*x[0]**3*x[1]**2 + o*x[0]**2*x[1]**3 + p*x[0]**3*x[1]**3

    def fit_4th_degree_polynomial(self, x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, y, z):
        return a + b*x[0] + c*x[1] + d*x[0]*x[1] + e*x[0]**2 + f*x[1]**2 + g*x[0]**2*x[1] + h*x[0]*x[1]**2 + i*x[0]**2*x[1]**2 + j*x[0]**3 + k*x[1]**3 + l*x[0]**3*x[1] + m*x[0]*x[1]**3 + n*x[0]**3*x[1]**2 + o*x[0]**2*x[1]**3 + p*x[0]**3*x[1]**3 + q*x[0]**4 + r*x[1]**4 + s*x[0]**4*x[1] + t*x[0]*x[1]**4 + u*x[0]**4*x[1]**2 + v*x[0]**2*x[1]**4 + w*x[0]**4*x[1]**3 + y*x[0]**3*x[1]**4 + z*x[0]**4*x[1]**4

    def fit(self):

        self.fit_functions = {
            'dM_BKK': self.fit_1st_degree_polynomial,
            'dMass_Ratio': self.fit_2nd_degree_polynomial,
            'triphoton_mass_lambda': self.fit_1st_degree_polynomial,
            'triphoton_mass_gamma': self.fit_1st_degree_polynomial,
            'triphoton_mass_delta': self.fit_2nd_degree_polynomial,
            'alpha_lambda': self.fit_2nd_degree_polynomial,
            'alpha_gamma': self.fit_4th_degree_polynomial,
            'alpha_delta': self.fit_3rd_degree_polynomial
        }

        for outcome in self.outcome_names:
            self.models[outcome] = curve_fit(self.fit_functions[outcome], self.X.values.T, self.Y[outcome].values)

    def predict(self, M_BKK, Mass_Ratio):
        
        X_raw = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        X = (X_raw - self.X_min) / (self.X_max - self.X_min)

        predictions = {}
        for outcome in self.outcome_names:
            prediction = self.fit_functions[outcome](X.values.T, *self.models[outcome][0])
            predictions[outcome] = (prediction * (self.Y_max[outcome] - self.Y_min[outcome])) + self.Y_min[outcome]

        return predictions

class SignalRegressionGP:
    def __init__(self, df, outcome_names):


        self.df = df
        self.outcome_names = outcome_names
        self.models = {}

        # Normailize X
        self.X_raw = df[['M_BKK', 'Mass_Ratio']]
        self.X_min = self.X_raw.min()
        self.X_max = self.X_raw.max()
        self.X = (self.X_raw - self.X_min) / (self.X_max - self.X_min)

        # self.X = df[['M_BKK', 'Mass_Ratio']]
        self.Y_raw = df[outcome_names]
        self.Y_raw_errs = df[[f"{outcome}_error" for outcome in outcome_names]]
        self.Y_raw_errs.columns = outcome_names

        self.Y_min = self.Y_raw.min()
        self.Y_max = self.Y_raw.max()
    
        self.Y = (self.Y_raw - self.Y_min) / (self.Y_max - self.Y_min)
        self.Y_errs = self.Y_raw_errs / (self.Y_max - self.Y_min)

        self.fit()
    
    def fit(self):
        # Use a Gaussian Process
        outputscale_bounds = {'alpha_gamma': (1e-9, 1), 'alpha_delta':(1e-9, 1)}
        lengthscale_bounds = {}
        for outcome in self.outcome_names:

            # const_kernel = ConstantKernel(1e-9, outputscale_bounds.get(outcome, outputscale_bound))
            const_kernel = ConstantKernel(1e-1)

            # other_kernel = RBF([1.0, 1.0], lengthscale_bounds.get(outcome, lengthscale_bound))
            # other_kernel = Matern([1.0, 1.0], length_scale_bounds=(0.1, 10), nu=nus.get(outcome, nu))
            # other_kernel = RationalQuadratic(length_scale_bounds=(0.1, 10))
            # other_kernel = RBF([1.0, 1.0], length_scale_bounds=(0.1, 10))

            # kernel = const_kernel * other_kernel
            # kernel = RBF([1.0,1.0])
            kernel = RationalQuadratic()
            reg = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.Y_errs[outcome].to_numpy()**2,
                n_restarts_optimizer=10,
                ).fit(self.X, self.Y[outcome])
            self.models[outcome] = reg
    
    def predict(self, M_BKK, Mass_Ratio):
        X_raw = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        X = (X_raw - self.X_min) / (self.X_max - self.X_min)
        predictions = {}
        for outcome in self.outcome_names:
            # X = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
            prediction = self.models[outcome].predict(X)[0]
            predictions[outcome] = prediction*(self.Y_max[outcome] - self.Y_min[outcome]) + self.Y_min[outcome]
        return predictions

    def predict_error(self, M_BKK, Mass_Ratio):
        X_raw = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        X = (X_raw - self.X_min) / (self.X_max - self.X_min)
        predictions = {}
        for outcome in self.outcome_names:
            # print(self.models[outcome].predict(X, return_std=True))
            prediction = self.models[outcome].predict(X, return_std=True)[1][0]
            predictions[outcome] = prediction*(self.Y_max[outcome] - self.Y_min[outcome])

        return predictions


def fit_outcome(X, Y, Y_err, outcome):
    linear_model = LinearRegression().fit(X, Y)
    res = Y - linear_model.predict(X)

    # const_kernel = ConstantKernel(1e-9, outputscale_bounds.get(outcome, outputscale_bound))
    const_kernel = ConstantKernel(1e-1)

    # other_kernel = RBF([1.0, 1.0], lengthscale_bounds.get(outcome, lengthscale_bound))
    # other_kernel = RBF([1.0, 1.0], length_scale_bounds=(0.5, 10))
    other_kernel = RBF([1.0,1.0])

    # kernel = const_kernel * other_kernel
    kernel = other_kernel
    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=Y_err.to_numpy()**2,
        n_restarts_optimizer=10,
        ).fit(X, res)
    return linear_model, gp_model, outcome

class SignalRegressionGPLinearMean:
    def __init__(self, df, outcome_names):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        from sklearn.linear_model import LinearRegression

        self.df = df
        self.outcome_names = outcome_names

        # Normailize X
        self.X_raw = df[['M_BKK', 'Mass_Ratio']]
        self.X_min = self.X_raw.min()
        self.X_max = self.X_raw.max()
        self.X = (self.X_raw - self.X_min) / (self.X_max - self.X_min)

        # self.X = df[['M_BKK', 'Mass_Ratio']]
        self.Y_raw = df[outcome_names]
        self.Y_raw_errs = df[[f"{outcome}_error" for outcome in outcome_names]]
        self.Y_raw_errs.columns = outcome_names

        self.Y_min = self.Y_raw.min()
        self.Y_max = self.Y_raw.max()
    
        self.Y = (self.Y_raw - self.Y_min) / (self.Y_max - self.Y_min)
        self.Y_errs = self.Y_raw_errs / (self.Y_max - self.Y_min)

        self.fit()
    
    def fit(self):
        # Use a Gaussian Process

        self.linear_model = {}
        self.gp_model = {}

        for outcome in self.outcome_names:
            # linear_model = LinearRegression().fit(self.X, self.Y[outcome])
            # res = self.Y[outcome] - linear_model.predict(self.X)
            res = self.Y[outcome]

            # const_kernel = ConstantKernel(1e-9, outputscale_bounds.get(outcome, outputscale_bound))
            # const_kernel = ConstantKernel(1e-1)

            # other_kernel = RBF([1.0, 1.0], lengthscale_bounds.get(outcome, lengthscale_bound))
            # other_kernel = RBF([1.0, 1.0], length_scale_bounds=(0.5, 10))
            other_kernel = RBF([1.0,1.0])

            # kernel = const_kernel * other_kernel
            kernel = other_kernel
            gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.Y_errs[outcome].to_numpy()**2,
                n_restarts_optimizer=10,
                ).fit(self.X, res)
            
            # self.linear_model[outcome] = linear_model
            self.linear_model[outcome] = None
            self.gp_model[outcome] = gp_model

        # args = [(self.X, self.Y[outcome], self.Y_errs[outcome], outcome) for outcome in self.outcome_names]
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        #     results = pool.starmap(fit_outcome, args)
        
        # for linear_model, gp_model, outcome in results:
        #     self.linear_model[outcome] = linear_model
        #     self.gp_model[outcome] = gp_model

    
    def predict(self, M_BKK, Mass_Ratio):
        X_raw = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        X = (X_raw - self.X_min) / (self.X_max - self.X_min)
        predictions = {}
        for outcome in self.outcome_names:
            # X = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])

            # linear_prediction = self.linear_model[outcome].predict(X)[0]
            linear_prediction = 0
            gp_prediction = self.gp_model[outcome].predict(X)[0]
            predictions[outcome] = (linear_prediction + gp_prediction) * (self.Y_max[outcome] - self.Y_min[outcome]) + self.Y_min[outcome]
        return predictions

    def predict_error(self, M_BKK, Mass_Ratio):
        X_raw = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])
        X = (X_raw - self.X_min) / (self.X_max - self.X_min)
        predictions = {}
        for outcome in self.outcome_names:
            # X = pd.DataFrame([[M_BKK, Mass_Ratio]], columns=['M_BKK', 'Mass_Ratio'])

            # linear_prediction = self.linear_model[outcome].predict(X)[0]
            gp_prediction = self.gp_model[outcome].predict(X, return_std=True)[1][0]
            predictions[outcome] = gp_prediction * (self.Y_max[outcome] - self.Y_min[outcome])
        return predictions

def regress_fit_parameters(
    M_BKK, Mass_Ratio,
    model = SignalRegressionKNN,
    drop_point= [],
    drop_M_BKK=[200],
    drop_Mass_Ratio=[],
    systematic="nominal"):

    df = get_fit_parameters(
        systematic=systematic
    )

    # Drop columns #TODO this is too agressive as a test, it would be better to pair the M_BKK and Mass_Ratios instead of dropping all of each
    if drop_point:
        mask = (df.M_BKK == drop_point[0]) & (df.Mass_Ratio == drop_point[1])
        df = df[~mask]
    if drop_M_BKK:
        df = df[~df.M_BKK.isin(drop_M_BKK)]
    if drop_Mass_Ratio:
        df = df[~df.Mass_Ratio.isin(drop_Mass_Ratio)]

    # Sort the parameters
    fitted_pars = [par for par in df.columns if par not in ['M_BKK', 'Mass_Ratio', 'M_R'] and not "error" in par]

    # Do regression (KNN for now)
    reg = model(df, fitted_pars)

    return reg.predict(M_BKK, Mass_Ratio)

def optimize_fit_parameters_in_loop(
    initial_tries=25,
    n_loops=4,
    drop_fn=None,
    reg_cls=SignalRegressionGP
    ):
    """
    Fits the signal with <initial_tries> random tries to start with,
    then uses the signal regression to predict the signal parameters
    """

    df = get_fit_parameters(remake=True, optimize_tries=initial_tries)
    if drop_fn is not None: df = drop_fn(df)
    fitted_pars = fitted_pars = [par for par in df.columns if par not in ['M_BKK', "M_R", 'Mass_Ratio', 'process', 'era'] and not "error" in par]
    for i in range(n_loops):
        print(f"Loop {i+1}")
        reg = reg_cls(df, fitted_pars)
        df = get_fit_parameters(remake=True, optimize_tries=1, reg=reg)
        if drop_fn is not None: df = drop_fn(df)
    
    return df, reg, fitted_pars

# Background Model Selection
def get_nll(data, triphoton_mass, alpha, n_exp_triphoton_mass, n_exp_alpha):
    model = ExponentialMixtureModel2D(triphoton_mass, alpha, n_exp_triphoton_mass, n_exp_alpha)
    model.pdf.fitTo(data, ROOT.RooFit.PrintLevel(-1))
    nll = model.pdf.createNLL(data)
    return nll.getVal()

def AIC(nll, n_pars):
    return 2*n_pars + 2*nll

def BIC(nll, n_pars, n):
    return n_pars*ROOT.TMath.Log(n) + 2*nll

def multi_nll(data, tm_var, a_var, n_exp_tm, n_exp_a):
    nll = get_nll(data, tm_var, a_var, n_exp_tm, n_exp_a)
    n_pars = (2*n_exp_tm - 1) + (2*n_exp_a - 1) + 4
    return nll, n_exp_tm, n_exp_a, AIC(nll, n_pars), BIC(nll, n_pars, data.numEntries())

def plot_AIC_BIC(data, tm_var, a_var, max_k, min_k=2):
    n_points = data.numEntries()

    args = [(data, tm_var, a_var, ki, kj) for ki in range(min_k, max_k+1) for kj in range(min_k, max_k+1)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(multi_nll, args)

    # Sort the results
    entries = []
    for nll, ki, kj, aic, bic in results:
        entries.append({
            "n_exp_tm": ki,
            "n_exp_a": kj,
            "AIC": aic,
            "BIC": bic
        })

    df = pd.DataFrame(entries)

    # Plot the AIC and BIC for different numbers of exponentials
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    for ki in sorted(df.n_exp_tm.unique()):
        # AIC
        mask = df.n_exp_tm == ki
        ax[0].plot(df[mask].n_exp_a, df[mask].AIC, label=f"$k_{{\\gamma\\gamma\\gamma}} = {ki}$", marker='o', color=pl.colors[ki-min_k])

        # BIC
        mask = df.n_exp_tm == ki
        ax[1].plot(df[mask].n_exp_a, df[mask].BIC, label=f"$k_{{\\gamma\\gamma\\gamma}} = {ki}$", marker='o', color=pl.colors[ki-min_k])
    
    ax[0].set_xlabel("$k_{m_{\\gamma\\gamma}/m_{\\gamma\\gamma\\gamma}}$")
    ax[0].set_ylabel("AIC")
    ax[0].legend()

    ax[1].set_xlabel("$k_{m_{\\gamma\\gamma}/m_{\\gamma\\gamma\\gamma}}$")
    ax[1].set_ylabel("BIC")
    ax[1].legend()

    fig.tight_layout()
    plt.show()

    fig.savefig(f"{plots_dir}/AIC_BIC.png")
