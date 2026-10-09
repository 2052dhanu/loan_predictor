"""Synthetic data generation and loading of your own (categorical) loan data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .credit_model import build_true_network
from .sampling import forward_sample_batch


def generate_dataset(n, seed=0, bn=None) -> pd.DataFrame:
    """Simulate n loan applicants by ancestral sampling from the ground-truth network."""
    bn = bn or build_true_network()
    s = forward_sample_batch(bn, n, np.random.default_rng(seed))
    return pd.DataFrame({v: np.asarray(bn.states[v], dtype=object)[s[v]] for v in bn.nodes})


def load_dataset(path, bn) -> pd.DataFrame:
    """Read a CSV whose columns are the network variables and whose values are state names."""
    df = pd.read_csv(path)
    missing = [v for v in bn.nodes if v not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df = df[bn.nodes]
    bn.encode_dataframe(df)          # raises with a clear message on unknown labels
    return df


def discretize_quantiles(series: pd.Series, labels) -> pd.Series:
    """Turn a numeric column (income, loan amount, ...) into equal-frequency categories, e.g.
    discretize_quantiles(df['income'], ['Low', 'Medium', 'High']). Use this to map a real
    dataset (German Credit, Lending Club, Kaggle loan data) onto the network's states."""
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=len(labels), labels=list(labels)).astype(object)
