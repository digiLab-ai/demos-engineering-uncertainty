import numpy as np

def get_default_geometry():
    return {
        "shield_factor": 0.3,
        "geom_factor": 0.8,
        "k_dpa": 1e-27,        # dpa per (n/m2)
        "exposure_time": 1e7,  # s
        "noise_std": 0.005     # dpa
    }

def evaluate(flux_avg, geom=None):
    if geom is None:
        geom = get_default_geometry()
    flux_avg = np.asarray(flux_avg, dtype=float)
    f_shield = geom["shield_factor"]
    f_geom = geom["geom_factor"]
    k_dpa = geom["k_dpa"]
    t_exp = geom["exposure_time"]

    flux_local = flux_avg * f_shield * f_geom
    dpa_peak = k_dpa * flux_local * t_exp
    return dpa_peak

def sample_model(inputs, n_samples=1, geom=None, rng=None):
    if geom is None:
        geom = get_default_geometry()
    if rng is None:
        rng = np.random.default_rng()

    flux_avg = np.asarray(inputs["flux_avg"], dtype=float)
    base = evaluate(flux_avg, geom=geom)
    noise_std = geom["noise_std"]

    base = np.atleast_1d(base)
    n_points = base.shape[0]
    noise = rng.normal(loc=0.0, scale=noise_std, size=(n_points, n_samples))
    samples = base[:, None] + noise
    return samples
