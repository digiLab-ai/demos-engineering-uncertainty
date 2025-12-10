import numpy as np

def get_default_geometry():
    return {
        "material": "Generic ferritic steel",
        "strain_rate": 1e-3,    # 1/s
        "sigma_0": 600.0,       # MPa
        "T_ref": 300.0,         # K
        "a_T": 1e-3,            # per K
        "a_dpa": 0.05,          # per dpa
        "noise_std": 10.0       # MPa
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

def sample_model(inputs, n_samples=1, geom=None, rng=None):
    if geom is None:
        geom = get_default_geometry()
    if rng is None:
        rng = np.random.default_rng()

    T = np.asarray(inputs["temperature"], dtype=float)
    dpa = np.asarray(inputs["dpa"], dtype=float)
    base = evaluate(T, dpa, geom=geom)
    noise_std = geom["noise_std"]

    base = np.atleast_1d(base)
    n_points = base.shape[0]
    noise = rng.normal(loc=0.0, scale=noise_std, size=(n_points, n_samples))
    samples = base[:, None] + noise
    return samples
