"""Experiment 1: how fast do the approximate methods converge to the exact (VE) answer?

For a query P(var = target | evidence) we compute the exact value with Variable Elimination and
then run rejection sampling, likelihood weighting and Gibbs sampling many times. The result is
(a) one example run per method and (b) the mean absolute error versus number of samples.
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesnet import (variable_elimination, rejection_stream, likelihood_weighting_stream,
                      gibbs_stream)

SCENARIOS = {
    "moderate evidence": dict(
        var="DefaultRisk", target="Yes",
        evidence={"CreditHistory": "Poor", "ExistingDebt": "High"}),
    "rare evidence": dict(
        var="Employment", target="Unstable",
        evidence={"Income": "Low", "CreditHistory": "Good", "ExistingDebt": "High",
                  "DefaultRisk": "No"}),
}
COLORS = {"Rejection sampling": "tab:red", "Likelihood weighting": "tab:green",
          "Gibbs sampling": "tab:blue"}


def make_checkpoints(n, n_points=25, lo=50):
    cps = np.unique(np.append(np.round(np.logspace(np.log10(min(lo, n)), np.log10(n), n_points)).astype(int), n))
    return cps[cps >= 1]


def compute_convergence(bn, var, target, evidence, max_samples=50_000, gibbs_samples=20_000,
                        runs=20, burn_in=500, seed=0):
    t_idx = bn.states[var].index(target)
    exact = float(variable_elimination(bn, var, evidence).factor.values[t_idx])
    rng = np.random.default_rng(seed)
    jobs = {
        "Rejection sampling": lambda: rejection_stream(bn, var, evidence, max_samples, runs, rng),
        "Likelihood weighting": lambda: likelihood_weighting_stream(bn, var, evidence, max_samples, runs, rng),
        "Gibbs sampling": lambda: gibbs_stream(bn, var, evidence, gibbs_samples, burn_in, runs, rng),
    }
    out = {"exact": exact, "var": var, "target": target, "evidence": evidence, "methods": {}}
    for name, job in jobs.items():
        t0 = time.perf_counter()
        stream = job()
        seconds = time.perf_counter() - t0
        cps = make_checkpoints(stream.qidx.shape[1])
        est = stream.running(cps)[:, :, t_idx]                  # (runs, checkpoints)
        err = np.abs(est - exact)
        info = {}
        if name == "Rejection sampling":
            info["acceptance_rate"] = float(stream.weights.mean())
        out["methods"][name] = dict(samples=cps, trace=est[0], mean_err=err.mean(0),
                                    std_err=err.std(0), seconds=seconds, **info)
    return out


def plot_convergence(res, title=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, m in res["methods"].items():
        c = COLORS[name]
        a1.plot(m["samples"], m["trace"], color=c, label=name)
        a2.plot(m["samples"], m["mean_err"], color=c, marker="o", ms=3, label=name)
        lo = np.maximum(m["mean_err"] - m["std_err"], 1e-6)
        a2.fill_between(m["samples"], lo, m["mean_err"] + m["std_err"], color=c, alpha=0.12)
    a1.axhline(res["exact"], color="black", ls="--", label=f"exact (VE) = {res['exact']:.4f}")
    a1.set_xscale("log")
    a1.set_xlabel("number of samples")
    a1.set_ylabel(f"estimate of P({res['var']}={res['target']} | e)")
    a1.set_title("One run per method")
    a1.legend()
    # reference slope 1/sqrt(n), anchored at likelihood weighting's first error
    lw = res["methods"]["Likelihood weighting"]
    ref = lw["mean_err"][0] * np.sqrt(lw["samples"][0] / lw["samples"])
    a2.plot(lw["samples"], ref, color="gray", ls=":", label=r"$\propto 1/\sqrt{n}$")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("number of samples")
    a2.set_ylabel("mean absolute error (band = 1 std)")
    a2.set_title("Mean error over independent runs")
    a2.legend()
    ev = ", ".join(f"{k}={v}" for k, v in res["evidence"].items())
    fig.suptitle(title or f"P({res['var']}={res['target']} | {ev})", fontsize=11)
    fig.tight_layout()
    return fig


def run(bn, out_dir="results", max_samples=50_000, gibbs_samples=20_000, runs=20, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, sc in SCENARIOS.items():
        res = compute_convergence(bn, sc["var"], sc["target"], sc["evidence"], max_samples,
                                  gibbs_samples, runs, seed=seed)
        fig = plot_convergence(res, f"Scenario: {name} -- P({sc['var']}={sc['target']} | "
                               + ", ".join(f"{k}={v}" for k, v in sc["evidence"].items()) + ")")
        path = os.path.join(out_dir, f"convergence_{name.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        for method, m in res["methods"].items():
            for n, e, s in zip(m["samples"], m["mean_err"], m["std_err"]):
                rows.append(dict(scenario=name, method=method, n_samples=int(n),
                                 mean_abs_error=e, std_abs_error=s, exact=res["exact"]))
        print(f"\n[{name}]  exact P = {res['exact']:.4f}")
        for method, m in res["methods"].items():
            extra = f", acceptance rate {m['acceptance_rate']:.3%}" if "acceptance_rate" in m else ""
            print(f"  {method:<22} final mean |error| = {m['mean_err'][-1]:.5f} "
                  f"({int(m['samples'][-1])} samples, {m['seconds']:.2f}s for {runs} runs{extra})")
        print(f"  saved {path}")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "convergence_summary.csv"), index=False)
