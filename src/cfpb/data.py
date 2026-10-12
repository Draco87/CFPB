"""Loading the combined dataset and making time-based splits."""
import pandas as pd

from cfpb import config


def part_files():
    files = sorted(config.COMBINED_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parts in {config.COMBINED_DIR} - run: python scripts/00_combine_archive.py")
    return files


def load_combined(columns=None) -> pd.DataFrame:
    """Load every part as one DataFrame. Pass `columns` to save RAM,
    e.g. load_combined(["date", "product", "company_response"]) skips the narratives."""
    return pd.concat((pd.read_parquet(f, columns=columns) for f in part_files()),
                     ignore_index=True)


def load_sample(n_per_part: int, columns=None, seed: int = config.SEED) -> pd.DataFrame:
    """Random sample of up to n rows from each part (each part = ~2-18 months of data),
    so every period is represented and RAM stays low."""
    parts = []
    for f in part_files():
        d = pd.read_parquet(f, columns=columns)
        parts.append(d.sample(min(len(d), n_per_part), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def time_split(df: pd.DataFrame):
    """Train < VAL_START <= val < TEST_START <= test. Never split randomly."""
    train = df[df["date"] < config.VAL_START]
    val = df[(df["date"] >= config.VAL_START) & (df["date"] < config.TEST_START)]
    test = df[df["date"] >= config.TEST_START]
    return train, val, test
