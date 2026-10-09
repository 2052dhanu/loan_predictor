"""Experiment 2: parameter estimation -- MLE vs Bayesian (Dirichlet / Laplace) estimation.

For growing training-set sizes we learn the CPTs and measure KL(true || learned). With little
data MLE assigns probability 0 to unseen combinations (huge KL), the Bayesian estimate does not.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesnet import build_structure, generate_dataset, kl_divergence, learn_parameters


def compute_learning_curve(true_bn, sizes=(20, 50, 100, 200, 500, 1000, 5000, 20000),
                           reps=20, alpha=1.0, seed=0):
    structure = build_structure()
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        for _ in range(reps):
            df = generate_dataset(n, seed=int(rng.integers(1 << 31)), bn=true_bn)
            rows.append(dict(n=n, method="MLE",
                             kl=kl_divergence(true_bn, learn_parameters(structure, df, "mle"))))
            rows.append(dict(n=n, method=f"Bayesian (alpha={alpha:g})",
                             kl=kl_divergence(true_bn, learn_parameters(structure, df, "bayes", alpha))))
    return pd.DataFrame(rows)


def plot_learning_curve(df):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for (method, g), c in zip(df.groupby("method"), ["tab:red", "tab:blue"]):
        q = g.groupby("n")["kl"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.plot(q.index, q[0.5], marker="o", color=c, label=method + " (median)")
        ax.fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training set size")
    ax.set_ylabel("KL(true || learned)")
    ax.set_title("Parameter learning error (band = inter-quartile range, 20 repetitions)")
    ax.legend()
    fig.tight_layout()
    return fig


def run(true_bn, out_dir="results", sizes=(20, 50, 100, 200, 500, 1000, 5000, 20000), reps=20, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    df = compute_learning_curve(true_bn, sizes, reps, seed=seed)
    summary = df.groupby(["method", "n"])["kl"].median().unstack(0)
    print("\nMedian KL(true || learned):")
    print(summary.to_string(float_format=lambda x: f"{x:.5f}"))
    path = os.path.join(out_dir, "learning_curve.png")
    fig = plot_learning_curve(df)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    df.to_csv(os.path.join(out_dir, "learning_curve.csv"), index=False)
    print(f"saved {path}")
