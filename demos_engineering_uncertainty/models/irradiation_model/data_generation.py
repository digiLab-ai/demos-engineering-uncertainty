import numpy as np
import pandas as pd
from pathlib import Path
from . import model

def generate_datasets(output_root: str = "data/irradiation", n_train: int = 200, n_val: int = 50, seed: int = 123):
    rng = np.random.default_rng(seed)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Flux expressed in units of "x 1e18 n/m²/s" to keep values near unity
    flux_train = rng.uniform(0.0, 5.0, size=n_train)
    flux_val = rng.uniform(0.0, 5.0, size=n_val)

    train_inputs = pd.DataFrame({"phi": flux_train})
    val_inputs = pd.DataFrame({"phi": flux_val})

    dpa_train = model.sample_model({"phi": flux_train}, n_samples=1, rng=rng)[:, 0]
    dpa_val = model.sample_model({"phi": flux_val}, n_samples=1, rng=rng)[:, 0]

    train_outputs = pd.DataFrame({"dpa_peak": dpa_train})
    val_outputs = pd.DataFrame({"dpa_peak": dpa_val})

    train_inputs.to_csv(output_root / "train_inputs.csv", index=False)
    val_inputs.to_csv(output_root / "val_inputs.csv", index=False)
    train_outputs.to_csv(output_root / "train_outputs.csv", index=False)
    val_outputs.to_csv(output_root / "val_outputs.csv", index=False)
