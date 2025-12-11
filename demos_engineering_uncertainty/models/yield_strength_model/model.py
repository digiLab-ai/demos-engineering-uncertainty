import numpy as np

def get_default_geometry():
    return {
        "material": "EUROFER first-wall backing",
        "strain_rate": 1e-3,    # 1/s
        "sigma_0": 550.0,       # MPa
        "T_ref": 300.0,         # K
        "a_T": 1.2e-3,          # per K
        "a_dpa": 0.06,          # per dpa
    }


def get_geometry_uncertainties():
    return {
        "sigma_0": 0.1,
        "a_T": 0.05,
        "a_dpa": 0.1,
        "strain_rate": 0.1,
        "T_ref": 0.02,
    }

def evaluate(temperature, dpa, geom=None):
    if geom is None:
        geom = get_default_geometry()
    temperature = np.asarray(temperature, dtype=float)
    dpa = np.asarray(dpa, dtype=float)

    sigma_0 = geom["sigma_0"]
    T_ref = geom["T_ref"]
    a_T = geom["a_T"]
    a_dpa = geom["a_dpa"]

    temp_factor = 1.0 - a_T * (temperature - T_ref)
    damage_factor = 1.0 - a_dpa * dpa

    sigma_base = sigma_0 * temp_factor * damage_factor
    sigma_base = np.maximum(sigma_base, 0.0)
    return sigma_base


def _perturb_value(val, frac, rng):
    return val * (1.0 + rng.normal(0.0, frac))

def sample_model(inputs, n_samples=1, geom=None, rng=None):
    if geom is None:
        geom = get_default_geometry()
    if rng is None:
        rng = np.random.default_rng()

    T = np.asarray(inputs["temperature"], dtype=float)
    dpa = np.asarray(inputs["dpa"], dtype=float)
    T = np.atleast_1d(T)
    dpa = np.atleast_1d(dpa)
    n_points = T.shape[0]
    uncertainties = get_geometry_uncertainties()

    all_samples = np.empty((n_points, n_samples))
    for i in range(n_samples):
        perturbed = {}
        for key, val in geom.items():
            if key in uncertainties:
                perturbed[key] = val * (1.0 + rng.normal(0.0, uncertainties[key], size=n_points))
            else:
                perturbed[key] = val
        all_samples[:, i] = evaluate(T, dpa, geom=perturbed)
    return all_samples
