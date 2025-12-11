import numpy as np
import pandas as pd
from pathlib import Path
from . import model

def generate_datasets(output_root: str = "data/temperature", n_train: int = 200, n_val: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    def sample_with_gap(size):
        # Sample uniformly from [0,2) and [6,10] with equal probability
        mask = rng.random(size) < 0.5
        lows = rng.uniform(0.0, 2.0, size=size)
        highs = rng.uniform(6.0, 10.0, size=size)
        return np.where(mask, lows, highs)

    q_train = sample_with_gap(n_train)
    q_val = sample_with_gap(n_val)

    train_inputs = pd.DataFrame({"q_peak": q_train})
    val_inputs = pd.DataFrame({"q_peak": q_val})

    T_train = model.sample_model({"q_peak": q_train}, n_samples=1, rng=rng)[:, 0]
    T_val = model.sample_model({"q_peak": q_val}, n_samples=1, rng=rng)[:, 0]

    train_outputs = pd.DataFrame({"T_peak": T_train})
    val_outputs = pd.DataFrame({"T_peak": T_val})

    train_inputs.to_csv(output_root / "train_inputs.csv", index=False)
    val_inputs.to_csv(output_root / "val_inputs.csv", index=False)
    train_outputs.to_csv(output_root / "train_outputs.csv", index=False)
    val_outputs.to_csv(output_root / "val_outputs.csv", index=False)
