"""Approximate inference by sampling: rejection sampling, likelihood weighting, Gibbs sampling.

All three samplers return a `SampleStream`: for every run a sequence of (query value, weight)
pairs. The posterior estimate after k samples is the weighted frequency of the first k pairs,
so ONE stream gives the whole convergence curve (estimate vs number of samples).

  rejection sampling    weight = 1 if the sample agrees with the evidence, else 0
  likelihood weighting  evidence variables are clamped; weight = prod P(e_i | parents)
  Gibbs sampling        a Markov chain; weight = 1 for every post-burn-in sweep
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .network import BayesianNetwork


def _as_list(q):
    return [q] if isinstance(q, str) else list(q)


def _categorical(probs, u):
    """Vectorised inverse-CDF sampling. probs: (n, k) rows summing to 1; u: (n,) uniforms."""
    cum = np.cumsum(probs, axis=1)
    idx = (u[:, None] >= cum).sum(axis=1)
    return np.minimum(idx, probs.shape[1] - 1)


def _parent_index(bn, v, samples, n):
    ps = bn.parents[v]
    if not ps:
        return np.zeros(n, dtype=int)
    return np.ravel_multi_index(tuple(samples[p] for p in ps), tuple(bn.card(p) for p in ps))


def _query_index(bn, query, samples):
    """Encode the (joint) value of the query variables as one integer per sample."""
    q = _as_list(query)
    cards = tuple(bn.card(v) for v in q)
    if len(q) == 1:
        return np.asarray(samples[q[0]]), cards[0]
    return np.ravel_multi_index(tuple(samples[v] for v in q), cards), int(np.prod(cards))


@dataclass
class SampleStream:
    qidx: np.ndarray       # (runs, n) query value of every sample
    weights: np.ndarray    # (runs, n) sample weights
    n_states: int

    def running(self, checkpoints):
        """Posterior estimates after the first k samples, k in checkpoints.
        Returns an array (runs, len(checkpoints), n_states). If no sample has positive weight yet
        (e.g. every sample was rejected) the uninformative uniform guess is returned."""
        cps = np.asarray(checkpoints, dtype=int)
        runs = self.qidx.shape[0]
        num = np.empty((runs, len(cps), self.n_states))
        for s in range(self.n_states):
            num[:, :, s] = np.cumsum(self.weights * (self.qidx == s), axis=1)[:, cps - 1]
        den = num.sum(axis=2, keepdims=True)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0 / self.n_states)

    def final(self):
        return self.running([self.qidx.shape[1]])[:, 0, :]


# ------------------------------------------------------------------ forward sampling
def forward_sample_batch(bn: BayesianNetwork, n, rng=None):
    """Ancestral sampling: draw every node given its already-sampled parents. Returns
    {var: int array of length n}."""
    rng = np.random.default_rng(rng)
    out = {}
    for v in bn.topological_order():
        pidx = _parent_index(bn, v, out, n)
        probs = bn.cpt[v].reshape(-1, bn.card(v))[pidx]
        out[v] = _categorical(probs, rng.random(n))
    return out


# ------------------------------------------------------------------------ samplers
def rejection_stream(bn, query, evidence, n_samples, n_runs=1, rng=None) -> SampleStream:
    """Sample from the prior, discard every sample that contradicts the evidence."""
    rng = np.random.default_rng(rng)
    ev = bn.encode_evidence(evidence)
    total = n_runs * n_samples
    s = forward_sample_batch(bn, total, rng)
    ok = np.ones(total, dtype=bool)
    for v, i in ev.items():
        ok &= s[v] == i
    q, n_states = _query_index(bn, query, s)
    return SampleStream(q.reshape(n_runs, n_samples),
                        ok.astype(float).reshape(n_runs, n_samples), n_states)


def likelihood_weighting_stream(bn, query, evidence, n_samples, n_runs=1, rng=None) -> SampleStream:
    """Clamp the evidence, sample the rest, weight each sample by P(evidence | parents)."""
    rng = np.random.default_rng(rng)
    ev = bn.encode_evidence(evidence)
    total = n_runs * n_samples
    samples, w = {}, np.ones(total)
    for v in bn.topological_order():
        pidx = _parent_index(bn, v, samples, total)
        flat = bn.cpt[v].reshape(-1, bn.card(v))
        if v in ev:
            samples[v] = np.full(total, ev[v], dtype=int)
            w *= flat[pidx, ev[v]]
        else:
            samples[v] = _categorical(flat[pidx], rng.random(total))
    q, n_states = _query_index(bn, query, samples)
    return SampleStream(q.reshape(n_runs, n_samples), w.reshape(n_runs, n_samples), n_states)


def _blanket_tables(bn, variables):
    """For each variable X the table  P(X | Pa(X)) * prod_{c in children(X)} P(c | Pa(c)),
    which is proportional to P(X | Markov blanket of X). The last axis is X itself."""
    tables = {}
    for X in variables:
        f = bn.factor(X)
        for c in bn.children[X]:
            f = f.product(bn.factor(c))
        others = [v for v in f.variables if v != X]
        tables[X] = (others, f.reorder(others + [X]).values)
    return tables


def gibbs_stream(bn, query, evidence, n_samples, burn_in=500, n_chains=1, rng=None) -> SampleStream:
    """Gibbs sampling. Evidence stays clamped; each sweep resamples every other variable from
    P(X | Markov blanket). `n_chains` independent chains are advanced in parallel (vectorised);
    the first `burn_in` sweeps of each chain are discarded."""
    rng = np.random.default_rng(rng)
    ev = bn.encode_evidence(evidence)
    state = forward_sample_batch(bn, n_chains, rng)          # arbitrary starting point
    for v, i in ev.items():
        state[v] = np.full(n_chains, i, dtype=int)
    free = [v for v in bn.topological_order() if v not in ev]
    tables = _blanket_tables(bn, free)
    qvars = _as_list(query)
    cards = tuple(bn.card(v) for v in qvars)
    n_states = int(np.prod(cards))
    out = np.empty((n_chains, n_samples), dtype=int)

    for sweep in range(burn_in + n_samples):
        for X in free:
            others, table = tables[X]
            if others:
                vals = table[tuple(state[o] for o in others)]            # (chains, card)
            else:
                vals = np.broadcast_to(table, (n_chains, table.shape[-1]))
            probs = vals / vals.sum(axis=1, keepdims=True)
            state[X] = _categorical(probs, rng.random(n_chains))
        if sweep >= burn_in:
            q = state[qvars[0]] if len(qvars) == 1 else np.ravel_multi_index(
                tuple(state[v] for v in qvars), cards)
            out[:, sweep - burn_in] = q
    return SampleStream(out, np.ones_like(out, dtype=float), n_states)


# ------------------------------------------------------------------ convenience API
METHODS = ("rejection", "likelihood_weighting", "gibbs")


def approximate_posterior(bn, var, evidence, method, n_samples, rng=None, burn_in=500):
    """Estimate P(var | evidence). Returns (probability array over var's states, info dict)."""
    t0 = time.perf_counter()
    if method == "rejection":
        st = rejection_stream(bn, var, evidence, n_samples, rng=rng)
        info = {"accepted": int(st.weights.sum())}
    elif method == "likelihood_weighting":
        st = likelihood_weighting_stream(bn, var, evidence, n_samples, rng=rng)
        w = st.weights[0]
        info = {"effective_sample_size": float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else 0.0}
    elif method == "gibbs":
        st = gibbs_stream(bn, var, evidence, n_samples, burn_in=burn_in, rng=rng)
        info = {"burn_in": burn_in}
    else:
        raise ValueError(f"method must be one of {METHODS}")
    probs = st.final()[0]
    info["seconds"] = time.perf_counter() - t0
    return probs, info


def gelman_rubin(chains):
    """Potential scale reduction factor R-hat for a (n_chains, n_samples) array.
    Values close to 1 (say < 1.05) indicate the chains have converged to the same distribution."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    if m < 2 or n < 2:
        return float("nan")
    W = chains.var(axis=1, ddof=1).mean()
    if W == 0:
        return float("nan")
    B = n * chains.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))
