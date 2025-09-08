import numpy as np

import awkward as ak
import dask_awkward as dak



# Functions to caclulate variables
def jet_energy_frac(candidates, jets):
    sum_jet_e = ak.sum(jets.energy, axis=1)
    candidates_e = candidates.triphoton.energy
    jet_energy_frac = sum_jet_e/(sum_jet_e + candidates_e)
    return jet_energy_frac

def candidates(events):
    """
    Returns all combinations of photons and diphotons for every event.
    """
    # Check if the objects are delayed
    photons = events.Photon
    diphotons = events.MLPhoton

    if isinstance(photons, dak.Array) or isinstance(diphotons, dak.Array):
        _ak = dak
    else:
        _ak = ak

    with np.errstate(divide='ignore', invalid='ignore'):
        diphotons = diphotons[(diphotons.mass > 0) & (diphotons.energy > 0)]
        candidates = _ak.cartesian({'diphoton': diphotons, 'photon': photons})

        candidates['delta_r'] = candidates.photon.delta_r(candidates.diphoton)
        candidates['triphoton'] = candidates.photon + candidates.diphoton
        candidates['delta_eta'] = abs(candidates.photon.eta - candidates.diphoton.eta)
        candidates['ket_frac'] = candidates.triphoton.pt/candidates.triphoton.energy
        candidates['alpha'] = candidates.diphoton.mass/candidates.triphoton.mass
        candidates['jet_energy_frac'] = jet_energy_frac(candidates, events.Jet)

    # Filter candidates
    candidates = candidates[(candidates.delta_r > 0.4)]

    return candidates

def candidate(candidates):
    # Return one candidate per event
    candidates = candidates[ak.argsort(candidates.diphoton.diphotonScore, axis=1, ascending=False)]
    return candidates[:,0]


def gen_variables(events):
    from analysis import variables as var
    gp = events.GenPart

    mother_is_bkk = gp[gp.genPartIdxMother].pdgId == var.bkk_pdgid
    mother_is_radion = gp[gp.genPartIdxMother].pdgId == var.radion_pdgid

    bkks = gp[gp.pdgId == var.bkk_pdgid]
    prompt_photons = gp[(gp.pdgId == var.photon_pdgid) & mother_is_bkk]
    radions = gp[(gp.pdgId == var.radion_pdgid) & mother_is_bkk]
    secondary_photons = gp[(gp.pdgId == var.photon_pdgid) & mother_is_radion]

    has_the_things = (ak.num(bkks) >= 1) & (ak.num(prompt_photons) >= 1) & (ak.num(radions) >= 1) & (ak.num(secondary_photons) > 1)

    bkks = bkks[has_the_things]
    prompt_photons = prompt_photons[has_the_things]
    radions = radions[has_the_things]
    secondary_photons = secondary_photons[has_the_things]

    p1 = prompt_photons[:, 0]
    p2 = secondary_photons[:, 0]
    p3 = secondary_photons[:, 1]
    p4 = radions[:, 0]

    diphoton_mass = np.sqrt(2*p2.pt*p3.pt*(np.cosh(p2.eta - p3.eta) - np.cos(p2.phi - p3.phi)))

    energy = lambda p: p.pt * np.cosh(p.eta)
    px = lambda p: p.pt * np.cos(p.phi)
    py = lambda p: p.pt * np.sin(p.phi)
    pz = lambda p: p.pt * np.sinh(p.eta)

    _px = px(p1) + px(p2) + px(p3)
    _py = py(p1) + py(p2) + py(p3)
    _pz = pz(p1) + pz(p2) + pz(p3)
    _p = np.sqrt(_px**2 + _py**2 + _pz**2)
    _energy = energy(p1) + energy(p2) + energy(p3)

    # _px = px(p1) + px(p4)
    # _py = py(p1) + py(p4)
    # _pz = pz(p1) + pz(p4)
    # _p = np.sqrt(_px**2 + _py**2 + _pz**2)
    # _energy = energy(p1) + energy(p4)

    triphoton_mass = np.sqrt((_energy)**2 - _p**2)

    return triphoton_mass, diphoton_mass

def deep_matching(obj0, obj1, dr_max=0.1):
    """Returns a boolean array with same shape as obj0 where 
    it is true if there is some obj1 within dr_max of obj0. Also
    returns the array of obj1, sorted by the percent difference in pt,
    that matches the shape of obj0[has_match]."""

    matching = dak.cartesian([obj0, obj1], nested=True)
    matching = matching[matching['0'].delta_r(matching['1'])<0.1]
    has_match = dak.num(matching['1'], axis=2) > 0

    # Sort axis 2 by the percent difference in pt
    matching = matching[dak.argsort(abs(matching['0'].pt - matching['1'].pt)/matching['0'].pt, axis=2, ascending=True)]
    matched_obj1 = matching['1'][has_match,0]

    return has_match, matched_obj1

def add_photon_SC_eta(photons: ak.Array, PV: ak.Array) -> ak.Array:
    """
    Add supercluster eta to photon object, following the implementation from https://github.com/bartokm/GbbMET/blob/026dac6fde5a1d449b2cfcaef037f704e34d2678/analyzer/Analyzer.h#L2487
    In the current NanoAODv11, there is only the photon eta which is the SC eta corrected by the PV position.
    The SC eta is needed to correctly apply a number of corrections and systematics.
    """

    if "superclusterEta" in photons.fields:
        photons["ScEta"] = photons.superclusterEta
        return photons

    PV_x = PV.x.to_numpy()
    PV_y = PV.y.to_numpy()
    PV_z = PV.z.to_numpy()

    mask_barrel = photons.isScEtaEB
    mask_endcap = photons.isScEtaEE

    tg_theta_over_2 = np.exp(-photons.eta)
    # avoid dividion by zero
    tg_theta_over_2 = np.where(tg_theta_over_2 == 1., 1 - 1e-10, tg_theta_over_2)
    tg_theta = 2 * tg_theta_over_2 / (1 - tg_theta_over_2 * tg_theta_over_2)  # tg(a+b) = tg(a)+tg(b) / (1-tg(a)*tg(b))

    # calculations for EB
    R = 130.
    angle_x0_y0 = np.zeros_like(PV_x)

    angle_x0_y0[PV_x > 0] = np.arctan(PV_y[PV_x > 0] / PV_x[PV_x > 0])
    angle_x0_y0[PV_x < 0] = np.pi + np.arctan(PV_y[PV_x < 0] / PV_x[PV_x < 0])
    angle_x0_y0[((PV_x == 0) & (PV_y >= 0))] = np.pi / 2
    angle_x0_y0[((PV_x == 0) & (PV_y < 0))] = -np.pi / 2

    alpha = angle_x0_y0 + (np.pi - photons.phi)
    sin_beta = np.sqrt(PV_x**2 + PV_y**2) / R * np.sin(alpha)
    beta = np.abs(np.arcsin(sin_beta))
    gamma = np.pi / 2 - alpha - beta
    length = np.sqrt(R**2 + PV_x**2 + PV_y**2 - 2 * R * np.sqrt(PV_x**2 + PV_y**2) * np.cos(gamma))
    z0_zSC = length / tg_theta

    tg_sctheta = np.copy(tg_theta)
    # correct values for EB
    tg_sctheta = ak.where(mask_barrel, R / (PV_z + z0_zSC), tg_sctheta)

    # calculations for EE
    intersection_z = np.where(photons.eta > 0, 310., -310.)
    base = intersection_z - PV_z
    r = base * tg_theta
    crystalX = PV_x + r * np.cos(photons.phi)
    crystalY = PV_y + r * np.sin(photons.phi)
    # correct values for EE
    tg_sctheta = ak.where(
        mask_endcap, np.sqrt(crystalX**2 + crystalY**2) / intersection_z, tg_sctheta
    )

    sctheta = np.arctan(tg_sctheta)
    sctheta = ak.where(
        sctheta < 0, np.pi + sctheta, sctheta
    )
    ScEta = -np.log(
        np.tan(sctheta / 2)
    )

    photons["ScEta"] = ScEta

    return photons
