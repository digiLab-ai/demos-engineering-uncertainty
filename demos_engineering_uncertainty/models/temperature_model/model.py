import numpy as np

def get_default_geometry():
    """Return fixed geometric / material factors for the temperature model."""
    return {
        "cooling_coefficient": 1., # 20.0,   # arbitrary units
        "sink_temperature": 300.0,     # K
        "thermal_conductivity": 15.0,  # W/(m·K)
        "density": 7800.0,             # kg/m3
        "heat_capacity": 500.0,        # J/(kg·K)
        "thickness": 0.02,             # m
        "q_buffer_threshold": 3.0,     # arbitrary units; below this, coolant buffers T to sink_temperature
    }


def get_geometry_uncertainties():
    """Fractional 1-sigma uncertainties on fixed parameters."""
    return {
        "cooling_coefficient": 0.005,
        "sink_temperature": 0.0001,
        "thermal_conductivity": 0.0005,
        "density": 0.0002,
        "heat_capacity": 0.0002,
        "thickness": 0.00005,
        "q_buffer_threshold": 0.00,
    }

def evaluate(q, geom=None):
    if geom is None:
        geom = get_default_geometry()
    q = np.asarray(q, dtype=float)
    h = geom["cooling_coefficient"]
    T_cool = geom["sink_temperature"]
    q_excess = np.maximum(q - geom["q_buffer_threshold"], 0.)
    q_step = 10. * (q > geom["q_buffer_threshold"])
    T = T_cool + q_step + q_excess / h
    return T


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

    q = np.asarray(inputs["q"], dtype=float)
    q = np.atleast_1d(q)
    n_points = q.shape[0]
    uncertainties = get_geometry_uncertainties()

    all_samples = np.empty((n_points, n_samples))
    for i in range(n_samples):
        perturbed = _perturb_geom(geom, uncertainties, n_points, rng)
        all_samples[:, i] = evaluate(q, geom=perturbed)
    return all_samples
