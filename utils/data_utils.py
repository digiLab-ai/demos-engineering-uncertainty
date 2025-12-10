import pandas as pd
from pathlib import Path

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV relative to the project root."""
    p = Path(path)
    return pd.read_csv(p)

def save_csv(df, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
