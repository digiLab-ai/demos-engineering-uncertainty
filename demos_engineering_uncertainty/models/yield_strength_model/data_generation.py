import numpy as np
import pandas as pd
from pathlib import Path
from . import model


def generate_datasets(output_root: str = "data/yield_strength", n_train: int = 60, n_val: int = 15, seed: int = 999):
    rng = np.random.default_rng(seed)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    T_train = rng.uniform(300.0, 1200.0, size=n_train)
    dpa_train = rng.uniform(0.0, 5.0, size=n_train)

    T_val = rng.uniform(300.0, 1200.0, size=n_val)
    dpa_val = rng.uniform(0.0, 5.0, size=n_val)

    train_inputs = pd.DataFrame({
        "T": T_train,
        "dpa": dpa_train
    })
    val_inputs = pd.DataFrame({
        "T": T_val,
        "dpa": dpa_val
    })

    failure_train = model.sample_model({"T": T_train, "dpa": dpa_train}, n_samples=1, rng=rng)[:, 0]
    failure_val = model.sample_model({"T": T_val, "dpa": dpa_val}, n_samples=1, rng=rng)[:, 0]

    train_outputs = pd.DataFrame({"failure": failure_train})
    val_outputs = pd.DataFrame({"failure": failure_val})

    train_inputs.to_csv(output_root / "train_inputs.csv", index=False)
    val_inputs.to_csv(output_root / "val_inputs.csv", index=False)
    train_outputs.to_csv(output_root / "train_outputs.csv", index=False)
    val_outputs.to_csv(output_root / "val_outputs.csv", index=False)
