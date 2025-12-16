import numpy as np

FLUX_SCALE = 1e18  # flux is provided in units of (x 1e18 n/m²/s)


def get_default_geometry():
    return {
        "shield_factor": 0.3,
        "geom_factor": 0.8,
        "k_dpa": 1e-27,         # dpa per (n/m2)
        "exposure_time": 1e7,   # s

        # Replace exp(beta*phi) with a saturating response:
        # nonlinear(phi) = 1 + a * (1 - exp(-phi/phi0))
        "nl_amp": 2.0,          # dimensionless amplitude of nonlinear boost
        "phi0": 2.0,            # in "x 1e18 n/m²/s" units; sets where it saturates
    }


def get_geometry_uncertainties():
    return {
        "shield_factor": 0.05,
        "geom_factor": 0.05,
        "k_dpa": 0.10,
        "exposure_time": 0.02,
        "nl_amp": 0.20,
        "phi0": 0.25,
    }


def get_epistemic_std():
    # Additive epistemic uncertainty (dpa), constant-ish not scaling with signal
    # (think: model discrepancy)
    return 0.0015


def get_aleatoric_std():
    # Additive measurement / process noise (dpa), constant
    return 0.002


def evaluate(phi, geom=None):
    if geom is None:
        geom = get_default_geometry()

    phi = np.asarray(phi, dtype=float)

    f_shield = float(geom["shield_factor"])
    f_geom   = float(geom["geom_factor"])
    k_dpa    = float(geom["k_dpa"])
    t_exp    = float(geom["exposure_time"])
    nl_amp   = float(geom["nl_amp"])
    phi0     = max(float(geom["phi0"]), 1e-12)

    flux_physical = phi * FLUX_SCALE
    flux_local = flux_physical * f_shield * f_geom

    # Saturating nonlinear response (prevents explosive growth)
    nonlinear = 1.0 + nl_amp * (1.0 - np.exp(-np.clip(phi, 0.0, None) / phi0))

    dpa_peak = k_dpa * flux_local * t_exp * nonlinear
    return dpa_peak


def _perturb_geom_global(geom, uncertainties, rng):
    """Perturb parameters once per sample (global), not per-point."""
    perturbed = {}
    for key, val in geom.items():
        if key in uncertainties:
            perturbed[key] = float(val) * (1.0 + rng.normal(0.0, uncertainties[key]))
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
    n_points = phi.size

    uncertainties = get_geometry_uncertainties()
    epistemic_std = get_epistemic_std()
    aleatoric_std = get_aleatoric_std()

    all_samples = np.empty((n_points, n_samples), dtype=float)

    for i in range(n_samples):
        perturbed = _perturb_geom_global(geom, uncertainties, rng)
        base = evaluate(phi, geom=perturbed)

        # Additive epistemic + aleatoric noise -> roughly constant scatter
        eps_model = rng.normal(0.0, epistemic_std, size=n_points)
        eps_obs   = rng.normal(0.0, aleatoric_std, size=n_points)

        all_samples[:, i] = base + eps_model + eps_obs

    return all_samples
