"""Parameter estimation for a Bayesian network with known structure (complete data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import BayesianNetwork


def count_table(bn: BayesianNetwork, enc: dict, v) -> np.ndarray:
    """N[pa_1, .., pa_k, x] = number of rows with those parent values and X = x."""
    counts = np.zeros(bn.cpt_shape(v))
    np.add.at(counts, tuple(enc[p] for p in bn.parents[v]) + (enc[v],), 1.0)
    return counts


def learn_parameters(structure: BayesianNetwork, data: pd.DataFrame,
                     method: str = "bayes", alpha: float = 1.0) -> BayesianNetwork:
    """Learn all CPTs from a dataframe (one column per variable, state names as values).

    method="mle"   : theta = N(x, pa) / N(pa)                         (maximum likelihood)
    method="bayes" : theta = (N(x, pa) + alpha) / (N(pa) + alpha*|X|)  (posterior mean under a
                     Dirichlet(alpha) prior; alpha = 1 is Laplace smoothing)
    """
    if method not in ("mle", "bayes"):
        raise ValueError("method must be 'mle' or 'bayes'")
    bn = structure.copy()
    enc = bn.encode_dataframe(data)
    for v in bn.nodes:
        counts = count_table(bn, enc, v)
        if method == "bayes":
            counts = counts + alpha
        total = counts.sum(axis=-1, keepdims=True)
        # parent configurations never seen in the data fall back to a uniform distribution
        table = np.where(total > 0, counts / np.where(total > 0, total, 1.0), 1.0 / bn.card(v))
        bn.cpt[v] = table
    return bn


def log_likelihood(bn: BayesianNetwork, data: pd.DataFrame) -> float:
    """Average log-likelihood per row, log P(row) = sum_v log P(x_v | pa_v)."""
    enc = bn.encode_dataframe(data)
    total = np.zeros(len(data))
    with np.errstate(divide="ignore"):
        for v in bn.nodes:
            idx = tuple(enc[p] for p in bn.parents[v]) + (enc[v],)
            total += np.log(bn.cpt[v][idx])
    return float(total.mean())


def kl_divergence(p: BayesianNetwork, q: BayesianNetwork, eps: float = 1e-10) -> float:
    """KL(P || Q) between the two joint distributions (exact; fine for small networks).
    Q's entries are floored at eps so that zero probabilities give a large finite penalty."""
    pj = p.joint_factor()
    qj = q.joint_factor().reorder(pj.variables)
    pv, qv = pj.values.ravel(), np.maximum(qj.values.ravel(), eps)
    mask = pv > 0
    return float(np.sum(pv[mask] * np.log(pv[mask] / qv[mask])))
