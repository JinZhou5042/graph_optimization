import os

from analysis import modeling as md
from analysis import skim

from analysis_tools import signal_info
from analysis_tools import dataset_info
from analysis_tools import storage_config as stor

import ROOT
import textwrap


def create_background_workspace(workspace_dir):
    # Create variables
    tm_bounds = (150, 1000)
    a_bounds = (0.001, 0.04)

    tm_var = ROOT.RooRealVar("triphoton_mass", "Triphoton Mass [GeV]", *tm_bounds)
    a_var = ROOT.RooRealVar("alpha", "Alpha", *a_bounds)

    f = ROOT.TFile(f"{md.root_cache}/data_CR.root")
    t = f.Get("data_CR")

    data = ROOT.RooDataSet("data_CR", "data_CR", t, ROOT.RooArgSet(tm_var, a_var))

    f.Close()

    n_exp_tm = 2
    n_exp_a = 2

    tm_ws = [ROOT.RooRealVar(f"tm_w_{i}", f"Mixture weight {i}", 1/n_exp_tm, 0, 1) for i in range(n_exp_tm-1)]
    tm_raw_rates = [ROOT.RooRealVar(f"tm_raw_rate_{i}", f"Unordered rate {i}", -0.01, -1000, -0.001) for i in range(n_exp_tm)]
    tm_ordered_rates = [ROOT.RooAddition(f"tm_rate_{i}", f"Ordered Rate for exponential {i}", ROOT.RooArgList(*tm_raw_rates[:i])) for i in range(n_exp_tm)]
    tm_exps = [ROOT.RooExponential(f"tm_exp_{i}", f"exp_{i}", tm_var, tm_ordered_rates[i]) for i in range(n_exp_tm)]

    tm_pdf = ROOT.RooAddPdf(f"tm_pdf", f"tm model", ROOT.RooArgList(*tm_exps), ROOT.RooArgList(*tm_ws), True)

    a_ws = [ROOT.RooRealVar(f"a_w_{i}", f"Mixture weight {i}", 1/n_exp_a, 0, 1) for i in range(n_exp_a-1)]
    a_raw_rates = [ROOT.RooRealVar(f"a_raw_rate_{i}", f"Unordered rate {i}", -0.01, -1000, -0.001) for i in range(n_exp_a)]
    a_ordered_rates = [ROOT.RooAddition(f"a_rate_{i}", f"Ordered Rate for exponential {i}", ROOT.RooArgList(*a_raw_rates[:i])) for i in range(n_exp_a)]
    a_exps = [ROOT.RooExponential(f"a_exp_{i}", f"exp_{i}", a_var, a_ordered_rates[i]) for i in range(n_exp_a)]

    a_pdf = ROOT.RooAddPdf(f"a_pdf", f"a model", ROOT.RooArgList(*a_exps), ROOT.RooArgList(*a_ws), True)

    tail_pdf = ROOT.RooProdPdf("tail_pdf", "tail_pdf", tm_pdf, a_pdf)

    tm_mean = ROOT.RooRealVar("tm_mean", "trig_m_mean", 220, -260, 300)
    tm_sigma = ROOT.RooRealVar("tm_sigma", "trig_m_sigma", 20, 0, 50)

    a_mean = ROOT.RooRealVar("a_mean", "a_mean", 0.004, 0.002, 0.006)
    a_sigma = ROOT.RooRealVar("a_sigma", "a_sigma", 0.002, 0, 0.005)

    # Normal CDF
    tm_eff = ROOT.RooFormulaVar(
        "tm_eff",
        "0.5*(1 + erf((@0-@1)/(sqrt(2)*@2)))",
        ROOT.RooArgList(tm_var, tm_mean, tm_sigma)
    )
    a_eff = ROOT.RooFormulaVar(
        "a_eff",
        "0.5*(1 + erf((@0-@1)/(sqrt(2)*@2)))",
        ROOT.RooArgList(a_var, a_mean, a_sigma)
    )

    eff = ROOT.RooFormulaVar("eff", "@0*@1", ROOT.RooArgList(tm_eff, a_eff))
    model = ROOT.RooEffProd("model_bkg", "Background Model", tail_pdf, eff)

    fit_result = model.fitTo(data, ROOT.RooFit.PrintLevel(-1))

    # Background normalization
    norm = ROOT.RooRealVar("model_bkg_norm", "Number of background events in CR", data.numEntries(), 0.9*data.numEntries(), 3*data.numEntries())
    for r in tm_raw_rates:
        r.setConstant(False)
    for r in a_raw_rates:
        r.setConstant(False)
    for r in [tm_mean, tm_sigma, a_mean, a_sigma]:
        r.setConstant(False)

    # Save to workspace
    fout = ROOT.TFile(f"{workspace_dir}/workspace_bkg.root", "RECREATE")
    ws = ROOT.RooWorkspace(f"workspace_bkg", f"workspace_bkg")
    getattr(ws, "import")(data)
    getattr(ws, "import")(model)
    getattr(ws, "import")(norm)
    ws.Write()
    fout.Close()

import pickle
def get_systematic(M_BKK, Mass_Ratio, par, systematic):
    reg_nominal_file = f"{md.cache_base}/interpolation_models/nominal_regression.pickle"
    with open(reg_nominal_file, 'rb') as f: reg_nominal = pickle.load(f)

    reg_up_file = f"{md.cache_base}/interpolation_models/{systematic}_up_regression.pickle"
    with open(reg_up_file, 'rb') as f: reg_up = pickle.load(f)

    reg_down_file = f"{md.cache_base}/interpolation_models/{systematic}_down_regression.pickle"
    with open(reg_down_file, 'rb') as f: reg_down = pickle.load(f)

    nominal = reg_nominal.predict(M_BKK, Mass_Ratio)[par]
    up = reg_up.predict(M_BKK, Mass_Ratio)[par]
    down = reg_down.predict(M_BKK, Mass_Ratio)[par]

    return max(abs(up-nominal), abs(down-nominal))

def create_signal_workspace(M_BKK, Mass_Ratio, workspace_dir):

    M_BKK = float(M_BKK)
    Mass_Ratio = float(Mass_Ratio)

    M_R = M_BKK*Mass_Ratio

    # Create variables
    triphoton_mass = ROOT.RooRealVar("triphoton_mass", "Triphoton Mass", 350, 600)
    alpha = ROOT.RooRealVar("alpha", "alpha", 0.01, 0.03)
    weight = ROOT.RooRealVar("weight", "weight", 0, 0, 1)

    # Regress fit parameters
    regression = md.regress_fit_parameters(
        M_BKK, Mass_Ratio,
        drop_M_BKK=[200, M_BKK],
        drop_Mass_Ratio=[Mass_Ratio]
    )

    # Build the signal model
    nuisances = {}
    pars = {}

    systematics = ["Photon_Scale", "Photon_Smearing", "MLPhoton_Scale"]
    for systematic in systematics:
        nuisances[systematic] = ROOT.RooRealVar(f"{systematic}_nuisance", f"{systematic}_nuisance", 0, -5, 5)
        nuisances[systematic].setConstant(True)

    for par in regression:
        sum_sys_quad = '+'.join([f"@{i}**2 *{get_systematic(M_BKK, Mass_Ratio, par, systematic)}**2" for i, systematic in enumerate(systematics)])
        pars[par] = ROOT.RooFormulaVar(
            par, par,
            f"{regression[par]}*(1 + sqrt({sum_sys_quad}))",
            ROOT.RooArgList(*[nuisances[systematic] for systematic in systematics])
        )

    ## Triphoton Mass Mean
    pars['triphoton_mass_mean'] = ROOT.RooFormulaVar(
        "triphoton_mass_mean", "triphoton_mass_mean",
        f"{M_BKK}+@0",
        ROOT.RooArgList(pars['dM_BKK'])
    )

    ## Alpha Mean
    pars['alpha_mean'] = ROOT.RooFormulaVar(
        "alpha_mean", "alpha_mean",
        f"{Mass_Ratio}+@0",
        ROOT.RooArgList(pars['dMass_Ratio'])
    )


    ## PDFs
    triphoton_mass_pdf = ROOT.RooJohnson(
        "triphoton_mass_pdf", "triphoton_mass_pdf", triphoton_mass,
        pars['triphoton_mass_mean'], pars['triphoton_mass_lambda'], pars['triphoton_mass_gamma'], pars['triphoton_mass_delta']
    )

    alpha_pdf = ROOT.RooJohnson(
        "alpha_pdf", "alpha_pdf", alpha,
        pars['alpha_mean'], pars['alpha_lambda'], pars['alpha_gamma'], pars['alpha_delta']
    )

    model_sig = ROOT.RooProdPdf("model_sig", "pdf", ROOT.RooArgList(triphoton_mass_pdf, alpha_pdf))

    # Normalizations
    signal_xs = md.regress_signal_xs(M_BKK, M_R)
    r_scaling = md.r_scaling(M_BKK)
    signal_xs = ROOT.RooRealVar(
        "xs_signal", f"Cross section of signal in fb",
        signal_xs 
    )
    xs_scaling = ROOT.RooRealVar(
        "xs_scaling", f"Cross section scaling factor",
        r_scaling
    )
    signal_xs.setConstant(True)
    xs_scaling.setConstant(True)

    scaled_signal_xs = ROOT.RooFormulaVar(
        "scaled_xs_signal", "Scaled cross section of signal in pb",
        "@0/@1", ROOT.RooArgList(signal_xs, xs_scaling)
    )

    signal_eff = md.regress_signal_efficiency(M_BKK, M_R)
    eff_signal = ROOT.RooRealVar(
        "eff_signal", "Efficiency of signal",
        signal_eff,
    )
    eff_signal.setConstant(True)

    norm_sig = ROOT.RooProduct(
        "model_sig_norm", "Normalization term for signal",
        ROOT.RooArgList(scaled_signal_xs, eff_signal)
    )

    # Save to workspace
    fout = ROOT.TFile(f"{workspace_dir}/workspace_sig.root", "RECREATE")
    ws = ROOT.RooWorkspace(f"workspace_sig", f"workspace_sig")
    getattr(ws, "import")(model_sig)
    getattr(ws, "import")(norm_sig)
    ws.Write()
    fout.Close()

def create_data_card(workspace_dir):
    lumi = sum(dataset_info.lumis_in_fb.values())
    lumi = round(lumi, 2)
    data_card = textwrap.dedent(f"""
        ---------------------------------------------
        imax 1
        jmax 1
        kmax *
        ---------------------------------------------

        shapes      signal       Tag0      workspace_sig.root      workspace_sig:model_sig
        shapes      bkg_mass     Tag0      workspace_bkg.root      workspace_bkg:model_bkg
        shapes      data_obs     Tag0      workspace_bkg.root      workspace_bkg:data_CR

        ---------------------------------------------
        bin             Tag0
        observation     -1
        ---------------------------------------------
        bin             Tag0         Tag0
        process         signal       bkg_mass
        process         0            1
        rate            {lumi}       1.0
        ---------------------------------------------
        lumi_13TeV              lnN           1.06       -
        CMS_dipho_eff           lnN           1.1         -
        ---------------------------------------------
        Photon_Scale_nuisance        param   0   1
        Photon_Smearing_nuisance     param   0   1
        MLPhoton_Scale_nuisance      param   0   1
        ---------------------------------------------
        """
    )
    with open(f"{workspace_dir}/datacard.txt", "w") as f:
        f.write(data_card)
    
def run_combine(M_BKK, Mass_Ratio, method, output_dir):

    # Create workspace
    workspace_dir = f"{output_dir}/{M_BKK}_{Mass_Ratio}"
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    
    if not os.path.exists(f"{workspace_dir}/workspace_bkg.root"):
        create_background_workspace(workspace_dir)
    if not os.path.exists(f"{workspace_dir}/workspace_sig.root"):
        create_signal_workspace(M_BKK, Mass_Ratio, workspace_dir)
    if not os.path.exists(f"{workspace_dir}/datacard.txt"):
        create_data_card(workspace_dir)

    # Run combine
    os.system(f"cd {workspace_dir}; combine -M {method} datacard.txt")


if __name__ == '__main__':
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--M_BKK', type=str, help='M_BKK')
    parser.add_argument('--Mass_Ratio', type=str, help='Mass Ratio')
    parser.add_argument('--method', '-M', type=str, help='Method', default='AsymptoticLimits')
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory', default=f'{stor.cache_dir}/combine/scan/points')

    args = parser.parse_args()

    run_combine(args.M_BKK, args.Mass_Ratio, args.method, args.output_dir)
