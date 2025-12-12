import numpy as np

FLUX_SCALE = 1e18  # flux is provided in units of (x 1e18 n/m²/s)

def get_default_geometry():
    return {
        "shield_factor": 0.3,
        "geom_factor": 0.8,
        "k_dpa": 1e-27,        # dpa per (n/m2)
        "exposure_time": 1e7,  # s
    }


def get_geometry_uncertainties():
    return {
        "shield_factor": 0.05,
        "geom_factor": 0.05,
        "k_dpa": 0.1,
        "exposure_time": 0.02,
    }


def get_epistemic_fraction():
    # Multiplicative epistemic uncertainty applied uniformly across each sample
    return 0.1

def evaluate(phi, geom=None):
    if geom is None:
        geom = get_default_geometry()
    # phi is expected in units of "x 1e18 n/m²/s"
    phi = np.asarray(phi, dtype=float)
    f_shield = geom["shield_factor"]
    f_geom = geom["geom_factor"]
    k_dpa = geom["k_dpa"]
    t_exp = geom["exposure_time"]

    flux_physical = phi * FLUX_SCALE
    flux_local = flux_physical * f_shield * f_geom
    dpa_peak = k_dpa * flux_local * t_exp
    return dpa_peak


def _perturb_geom(geom, uncertainties, n_points, rng):
    perturbed = {}
    for key, val in geom.items():
        if key in uncertainties:
            perturbed[key] = val * (1.0 + rng.normal(0.0, uncertainties[key], size=n_points))
        else:
            perturbed[key] = val
    return perturbed

def sample_model(inputs, n_samples=1, geom=None, rng=None):
    if geom is None:
        geom = get_default_geometry()
    if rng is None:
        rng = np.random.default_rng()

    phi = np.asarray(inputs["phi"], dtype=float)
    phi = np.atleast_1d(phi)
    n_points = phi.shape[0]
    uncertainties = get_geometry_uncertainties()
    epistemic_frac = get_epistemic_fraction()

    all_samples = np.empty((n_points, n_samples))
    for i in range(n_samples):
        perturbed = _perturb_geom(geom, uncertainties, n_points, rng)
        epistemic_factor = 1.0 + rng.normal(0.0, epistemic_frac)
        all_samples[:, i] = evaluate(phi, geom=perturbed) * epistemic_factor
    return all_samples
