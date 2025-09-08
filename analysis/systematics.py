import numpy as np
import awkward as ak
import correctionlib
import os
from copy import deepcopy

from analysis import calculations as calc
from analysis import selections as sel

import logging
logger = logging.getLogger(__name__)

systematics_dir = os.path.join(os.path.dirname(__file__), 'systematics')
if not os.path.exists(systematics_dir):
    os.makedirs(systematics_dir)


def MLPhoton_Scale(events, year="", uncertainty=0.02):
    # MLPhoton systematics
    MLPhoton = events.MLPhoton

    energy = MLPhoton.mass/MLPhoton.massEnergyRatio
    energy_up = 1.02*energy
    energy_down = 0.98*energy

    mass_up = MLPhoton.massEnergyRatio*energy_up
    mass_down = MLPhoton.massEnergyRatio*energy_down
    pt_up = np.sqrt(energy_up**2 - mass_up**2)/np.cosh(MLPhoton.eta)
    pt_down = np.sqrt(energy_down**2 - mass_down**2)/np.cosh(MLPhoton.eta)

    MLPhoton_energy_up = ak.copy(MLPhoton)
    MLPhoton_energy_up['mass'] = mass_up
    MLPhoton_energy_up['pt'] = pt_up

    MLPhoton_energy_down = ak.copy(MLPhoton)
    MLPhoton_energy_down['mass'] = mass_down
    MLPhoton_energy_down['pt'] = pt_down

    ev_up = ak.copy(events)
    ev_up['MLPhoton'] = MLPhoton_energy_up
    ev_up['Candidates'] = calc.candidates(ev_up)
    ev_up['Candidate'] = calc.candidate(ev_up.Candidates)

    ev_down = ak.copy(events)
    ev_down['MLPhoton'] = MLPhoton_energy_down
    ev_down['Candidates'] = calc.candidates(ev_down)
    ev_down['Candidate'] = calc.candidate(ev_down.Candidates)

    return ev_up, ev_down


def Photon_ID(stuff):
    pass


# first dummy, keeping it at this point as reference for even simpler implementations
def photon_pt_scale_dummy(pt, **kwargs):
    return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * pt[:, None]


def Photon_Scale(events, year):
    """
    Applies the corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py
    """
    
    assert hasattr(events, "GenPart"), "This function is only meant for MC!"
    assert year in ["2016preVFP", "2016postVFP", "2017", "2018"], "Only years 2016preVFP, 2016postVFP, 2017, 2018 are supported!"

    # Add SC eta
    if not hasattr(events.Photon, 'ScEta'):
        events["Photon"] = calc.add_photon_SC_eta(events.Photon, events.PV)

    # for later unflattening:
    counts = ak.num(events.Photon.pt)

    run = ak.broadcast_arrays(events.run, events.Photon.pt)[0]
    gain = events.Photon.seedGain
    eta = events.Photon.ScEta
    r9 = events.Photon.r9
    pt = events.Photon.pt

    path_json = f'{systematics_dir}/scaleAndSmearing/EGM_ScaleUnc_{year}.json'
    evaluator = correctionlib.CorrectionSet.from_file(path_json)["UL-EGM_ScaleUnc"]

    # the uncertainty is applied in reverse because the correction is meant for data as I understand fro EGM instructions here: https://cms-talk.web.cern.ch/t/pnoton-energy-corrections-in-nanoaod-v11/34327/2
    scale_up = evaluator.evaluate(year, "scaleup", eta, gain)
    scale_down = evaluator.evaluate(year, "scaledown", eta, gain)

    # return corr_up_variation, corr_down_variation

    # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
    # return np.concatenate((corr_up_variation.reshape(-1,1), corr_down_variation.reshape(-1,1)), axis=1) * _pt[:, None]

    # pt_up = pt * (1 + corr_up_variation)
    # pt_down = pt * (1 - corr_down_variation)

    photon_up = ak.copy(events.Photon)
    photon_up['pt'] = pt*scale_up

    photon_down = ak.copy(events.Photon)
    photon_down['pt'] = pt*scale_down


    ev_up = ak.copy(events)
    ev_up['Photon'] = photon_up
    ev_up['Candidates'] = calc.candidates(ev_up)
    ev_up['Candidate'] = calc.candidate(ev_up.Candidates)

    ev_down = ak.copy(events)
    ev_down['Photon'] = photon_down
    ev_down['Candidates'] = calc.candidates(ev_down)
    ev_down['Candidate'] = calc.candidate(ev_down.Candidates)

    return ev_up, ev_down


def Photon_Smearing(events, year=""):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    """

    assert hasattr(events, "GenPart"), "This function is only meant for MC!"

    if not hasattr(events.Photon, 'ScEta'):
        events["Photon"] = calc.add_photon_SC_eta(events.Photon, events.PV)

    pt = events.Photon.pt
    eta = events.Photon.ScEta

    dEsigmaUp = events.Photon.dEsigmaUp
    dEsigmaDown = events.Photon.dEsigmaDown

    photon_up = ak.copy(events.Photon)
    photon_up['pt'] = pt + abs(dEsigmaUp) / np.cosh(eta)

    photon_down = ak.copy(events.Photon)
    photon_down['pt'] = pt - abs(dEsigmaDown) / np.cosh(eta)

    ev_up = ak.copy(events)
    ev_up['Photon'] = photon_up
    ev_up['Candidates'] = calc.candidates(ev_up)
    ev_up['Candidate'] = calc.candidate(ev_up.Candidates)

    ev_down = ak.copy(events)
    ev_down['Photon'] = photon_down
    ev_down['Candidates'] = calc.candidates(ev_down)
    ev_down['Candidate'] = calc.candidate(ev_down.Candidates)

    return ev_up, ev_down

shape_systematics = {
    "Photon_Scale": Photon_Scale,
    "Photon_Smearing": Photon_Smearing,
    "MLPhoton_Scale": MLPhoton_Scale,
}

systematics = {
    "MLPhoton_Scale": MLPhoton_Scale,
    "Photon_Scale": Photon_Scale,
    "Photon_Smearing": Photon_Smearing,
    "Photon_ID": Photon_ID
}