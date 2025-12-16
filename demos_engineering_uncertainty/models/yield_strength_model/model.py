import numpy as np


def get_default_geometry():
    return {
        "material": "EUROFER first-wall backing",
        "strain_rate": 1e-3,    # 1/s

        # Baseline (unirradiated) UTS-like stress at reference temperature
        "sigma_0": 550.0,       # MPa

        # Temperature dependence (still simple linear softening around T_ref)
        # NOTE: your original uses Kelvin; keep that, but see evaluate() docstring.
        "T_ref": 300.0,         # K
        "a_T": 1.2e-3,          # per K

        # --- Updated DPA effect: thresholded + saturating *increase* (UTS hardening) ---
        "dpa_thr": 0.02,        # DPA threshold below which effect is negligible
        "dpa_sat": 2.0,         # DPA scale for saturation (above threshold)
        "h_dpa_max": 0.35,      # max fractional UTS increase (e.g. +35% at saturation)

        # Optional: small residual linear term if you want (set 0.0 to disable)
        "h_dpa_lin": 0.00,      # additional fractional increase per DPA above threshold

        # Safety clamp
        "sigma_floor": 0.0,     # MPa
    }


def get_geometry_uncertainties():
    return {
        "sigma_0": 0.10,
        "a_T": 0.05,
        "strain_rate": 0.10,
        "T_ref": 0.02,

        # Uncertainties for new DPA parameters
        "dpa_thr": 0.30,
        "dpa_sat": 0.40,
        "h_dpa_max": 0.30,
        "h_dpa_lin": 0.50,
        "sigma_floor": 0.50,
    }


def get_epistemic_fraction():
    # Multiplicative epistemic uncertainty per sample (e.g., model bias)
    return 0.1


def evaluate(T, dpa, geom=None):
    """
    Evaluate a simple EUROFER UTS proxy with:
      - linear temperature softening about T_ref (in Kelvin),
      - thresholded, saturating irradiation hardening vs DPA.

    Parameters
    ----------
    T : array-like
        Temperature in Kelvin (consistent with your original T_ref/a_T usage).
    dpa : array-like
        DPA value(s).
    """
    if geom is None:
        geom = get_default_geometry()

    T = np.asarray(T, dtype=float)
    dpa = np.asarray(dpa, dtype=float)
    T, dpa = np.broadcast_arrays(T, dpa)

    sigma_0 = float(geom["sigma_0"])
    T_ref = float(geom["T_ref"])
    a_T = float(geom["a_T"])

    # --- Temperature factor (unchanged from your original, but clamped) ---
    temp_factor = 1.0 - a_T * (T - T_ref)
    temp_factor = np.clip(temp_factor, 0.0, None)

    # --- NEW: DPA hardening factor (UTS increase), thresholded & saturating ---
    dpa_thr = max(float(geom["dpa_thr"]), 0.0)
    dpa_sat = max(float(geom["dpa_sat"]), 1e-12)
    h_max = float(geom["h_dpa_max"])
    h_lin = float(geom.get("h_dpa_lin", 0.0))

    dpa_eff = np.clip(dpa - dpa_thr, 0.0, None)

    # saturating component: 0 at threshold, -> h_max as dpa_eff >> dpa_sat
    harden_sat = h_max * (1.0 - np.exp(-dpa_eff / dpa_sat))
    # optional extra linear tail (kept small or 0)
    harden_lin = h_lin * dpa_eff

    dpa_factor = 1.0 + harden_sat + harden_lin

    sigma = sigma_0 * temp_factor * dpa_factor

    sigma_floor = float(geom.get("sigma_floor", 0.0))
    sigma = np.maximum(sigma, sigma_floor)
    return sigma


def sample_model(inputs, n_samples=1, geom=None, rng=None):
    if geom is None:
        geom = get_default_geometry()
    if rng is None:
        rng = np.random.default_rng()

    T = np.asarray(inputs["T"], dtype=float)
    dpa = np.asarray(inputs["dpa"], dtype=float)
    T = np.atleast_1d(T)
    dpa = np.atleast_1d(dpa)

    # Broadcast to common shape
    T_b, dpa_b = np.broadcast_arrays(T, dpa)
    out_shape = T_b.shape
    T_1d = T_b.reshape(-1)
    dpa_1d = dpa_b.reshape(-1)
    n_points = T_1d.size

    uncertainties = get_geometry_uncertainties()
    epistemic_frac = get_epistemic_fraction()

    all_samples = np.empty((n_points, n_samples), dtype=float)

    for i in range(n_samples):
        perturbed = {}
        for key, val in geom.items():
            if key in uncertainties:
                # global-per-sample parameter perturbation
                perturbed[key] = float(val) * (1.0 + rng.normal(0.0, uncertainties[key]))
            else:
                perturbed[key] = val

        epistemic_factor = 1.0 + rng.normal(0.0, epistemic_frac)
        sigma = evaluate(T_1d, dpa_1d, geom=perturbed) * epistemic_factor
        all_samples[:, i] = sigma

    return all_samples.reshape((*out_shape, n_samples))

