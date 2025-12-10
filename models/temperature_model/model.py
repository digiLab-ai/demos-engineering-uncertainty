import numpy as np

def get_default_geometry():
    """Return fixed geometric / material factors for the temperature model."""
    return {
        "cooling_coefficient": 20.0,   # arbitrary units
        "sink_temperature": 300.0,     # K
        "thermal_conductivity": 15.0,  # W/(m·K)
        "density": 7800.0,             # kg/m3
        "heat_capacity": 500.0,        # J/(kg·K)
        "thickness": 0.02,             # m
        "noise_std": 5.0               # K
    }

def evaluate(q_avg, geom=None):
    if geom is None:
        geom = get_default_geometry()
    q_avg = np.asarray(q_avg, dtype=float)
    h = geom["cooling_coefficient"]
    T_cool = geom["sink_temperature"]
    T_peak = T_cool + q_avg / h
    return T_peak

def sample_model(inputs, n_samples=1, geom=None, rng=None):
    if geom is None:
        geom = get_default_geometry()
    if rng is None:
        rng = np.random.default_rng()

    q_avg = np.asarray(inputs["q_avg"], dtype=float)
    base = evaluate(q_avg, geom=geom)
    noise_std = geom["noise_std"]

    base = np.atleast_1d(base)
    n_points = base.shape[0]
    noise = rng.normal(loc=0.0, scale=noise_std, size=(n_points, n_samples))
    samples = base[:, None] + noise
    return samples
