import numpy as np

import hist
import awkward as ak

from analysis import calculations as calc

# Constants
radion_pdgid = 9000025
photon_pdgid = 22
bkk_pdgid = 9000121

photon_mva_cut = {'barrel': {80: 0.42, 90: -0.02},
                  'endcap': {80: 0.14, 90: -0.26}}

PhotonID_tags = {1: 'loose',
                 2: 'medium',
                 3: 'tight',
                 0: 'no'}


def get(events, var_name):

    # If alias is defined, use that (note: this assumes the object is the events object)
    if var_name in variables and 'get' in variables[var_name]:
        return variables[var_name]['get'](events)

    # Otherwise access the variable using the dot notation
    keys = var_name.split('.')
    v = ak.copy(events)
    for key in keys:
        v = getattr(v, key)
    return v

def get_label(var_name):
    if var_name in variables:
        return variables[var_name]['label']
    return var_name

# Data structure for variable information/access
variables = {
    # Diphoton variables (Candidate)
    'diphoton_energy' : {
        'get' : lambda ev: ev.Candidate.diphoton.energy,
        'bounds': (0, 4000), 'nbins': 256,
        'label': 'Diphoton Energy [GeV]'
        },
    'diphoton_mass' : {
        'get' : lambda ev: ev.Candidate.diphoton.mass,
        'bounds': (0, 4), 'nbins': 128,
        'label': '$m_{\\gamma \\gamma}$ [GeV]'
        },
    'diphoton_moe' : {
        'get' : lambda ev: ev.Candidate.diphoton.massEnergyRatio,
        'bounds': (0, 0.05), 'nbins': 128,
        'label': 'Diphoton $m/E$'
        },
    'diphoton_pt' : {
        'get' : lambda ev: ev.Candidate.diphoton.pt,
        'bounds': (0, 4000), 'nbins': 256,
        'label': '$\\Gamma_{p_T}$ [GeV]'
        },
    'diphoton_eta' : {
        'get' : lambda ev: ev.Candidate.diphoton.eta,
        'bounds': (-1.5, 1.5), 'nbins': 32,
        'label': '$\\Gamma_{\\eta}$'
        },
    'diphoton_score' : {
        'get' : lambda ev: ev.Candidate.diphoton.diphotonScore,
        'bounds': (0, 1), 'nbins': 32,
        'label': '$\\Gamma_{\\text{score}}$'
        },
    'diphoton_isolation' : {
        'get' : lambda ev: ev.Candidate.diphoton.pfIsolation,
        'bounds': (0, 1), 'nbins': 32,
        'label': '$\\Gamma_{\\text{iso}}$'
        },

    # Photon variables (Candidate)
    'photon' : {
        'get' : lambda ev: ev.Candidate.photon,
    },
    'photon_energy' : {
        'get' : lambda ev: ev.Candidate.photon.energy,
        'bounds': (0, 4000), 'nbins': 256,
        'label': 'photon energy [GeV]'
        },
    'photon_pt' : {
        'get' : lambda ev: ev.Candidate.photon.pt,
        'bounds': (0, 4000), 'nbins': 256,
        'label': 'photon $p_T$ [GeV]'
        },
    'photon_eta' : {
        'get' : lambda ev: ev.Candidate.photon.eta,
        'bounds': (-4, 4), 'nbins': 32,
        'label': 'photon $\\eta$'
        },
    'photon_sieie' : {
        'get' : lambda ev: ev.Candidate.photon.sieie,
        'bounds': (0, 0.05), 'nbins': 128,
        'label': '$\\sigma_{i\\eta i\\eta}$'
        },
    'photon_id' : {
        'get' : lambda ev: ev.Candidate.photon.cutBased,
        'bin_edges' : [0, 1, 2, 3, 4],
        'label': 'Photon ID'
        },
    'photon_mvaID' : {
        'get' : lambda ev: ev.Candidate.photon.mvaID,
        'bounds': (-1, 1), 'nbins': 64,
        'label': 'Photon MVA ID'
        }, 

    # Candidate variables
    'delta_r' : {
        'get' : lambda ev: ev.Candidate.delta_r,
        'bounds': (0, 5), 'nbins': 32,
        'label': '$\\Delta R$'
        },
    'delta_eta' : {
        'get' : lambda ev: ev.Candidate.delta_eta,
        'bounds': (0, 4), 'nbins': 32,
        'label': '$| \\Delta \\eta |$'
        },
    'ket_frac' : {
        'get': lambda ev: ev.Candidate.ket_frac,
        'bounds': (0, 1), 'nbins': 32,
        'label': '$| \\vec{p_{T_{\\gamma}}} + \\vec{p_{T_{\\gamma \\gamma}}}| / \\sum_\\gamma{E}$'
        },
    'energy_ratio' : {
        'get': lambda ev: ev.Candidate.diphoton.energy/ev.Candidate.photon.energy,
        'bounds': (0, 2), 'nbins': 32,
        'label': '$E_{diphoton}/E_{\\gamma}$'
        },
    'triphoton_mass' : {
        'get' : lambda ev: ev.Candidate.triphoton.mass,
        'bounds': (0, 800), 'nbins': 64,
        'label': '$m_{\\gamma \\gamma \\gamma}$ [GeV]'
        },
    'alpha' : {
        'get' : lambda ev: ev.Candidate.alpha,
        'bounds': (0, 0.05), 'nbins': 64,
        'label': '$m_{\\gamma \\gamma}/m_{\\gamma \\gamma \\gamma}$'
        },

    # Event variables
    'pf_met' : {
        'get': lambda ev: ev.MET.pt,
        'bounds': (0, 500), 'nbins': 128,
        'label': 'PF MET [GeV]'
        },
    'jet_energy_frac' : {
        'get' : lambda ev: ev.Candidate.jet_energy_frac,
        'bounds': (0.5, 1), 'nbins': 32,
        'label': '$\\sum_{jets}{E}/\\sum_{jets+\\gamma}{E}$'
        },
    'jet_eta' : {
        'get' : lambda ev: ev.Jet.eta,
        'bounds': (-4, 4), 'nbins': 32,
        'label': 'jet $\\eta$'
        },

    'significance': {
        'bounds': (0, 500), 'nbins': 64,
        'label': 'Significance'
        },

    'weight': {
        'get': lambda ev: ev['weight'],
    },
}

defaults = {
    'underflow': False, 'overflow': False,
}

class BinnedVariable:
    def __init__(
            self,
            name,
            fill_name=None,
            **var_params,
            ):
        
        self.name = name
        self.fill_name = fill_name

        self.info = defaults.copy()
        if name in variables:
            self.info.update(variables[name])

        self.info.update(var_params)

        if 'label' not in self.info:
            self.info['label'] = name

        if 'bin_edges' in self.info:
            assert self.info['bin_edges'] is not None, "Must provide bin edges"
        elif 'bounds' in self.info and 'nbins' in self.info:
            assert self.info['bounds'] is not None, "Must provide bounds"
            assert self.info['nbins'] is not None, "Must provide nbins"
        else:
            raise ValueError("Must provide either bin edges or bounds/nbins")
    
    @property
    def bin_edges(self):
        if 'bin_edges' in self.info:
            return self.info['bin_edges']
        return np.linspace(*self.info['bounds'], self.info['nbins']+1)

    @property
    def bounds(self):
        if 'bounds' in self.info:
            return self.info['bounds']
        return (self.info['bin_edges'][0], self.info['bin_edges'][-1])
    
    @property
    def nbins(self):
        if 'nbins' in self.info:
            return self.info['nbins']
        return len(self.info['bin_edges'])-1

    @property
    def label(self):
        return self.info['label']

    @property
    def axis(self):
        return hist.axis.Variable(
            self.bin_edges,
            name=self.fill_name if self.fill_name else self.name,
            label=self.label,
            underflow=self.info['underflow'], overflow=self.info['overflow'],
        )


def get_var_params(var_name, **kwargs):
    params = {}
    for k,v in kwargs.items():
        if k.startswith(f'{var_name}_'):
            _k = k.replace(f'{var_name}_', '')
            params[_k] = v
    return params

def get_hist(*var_names, **var_params):
    _variables = []
    for var_name in var_names:
        if len(var_names) == 1:
            params = var_params
        else:
            params = get_var_params(var_name, **var_params)
        _variables.append(BinnedVariable(var_name, **params))

    return hist.Hist(*[_var.axis for _var in _variables], storage=hist.storage.Weight())