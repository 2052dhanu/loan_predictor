"""End-to-end demo of the whole project.   python main.py [--quick] [--skip-experiments]"""
import argparse
import os

import numpy as np
import pandas as pd

from bayesnet import (build_structure, build_true_network, generate_dataset, learn_parameters,
                      log_likelihood, kl_divergence, variable_elimination, enumeration_query,
                      posterior_dict, map_query, most_probable_explanation, approximate_posterior,
                      gibbs_stream, gelman_rubin)
from experiments import convergence, learning_curve

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "loan_data.csv")
RESULTS = os.path.join(ROOT, "results")


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def fmt(d):
    return ", ".join(f"{k}={v:.4f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-data", type=int, default=5000, help="rows of synthetic training data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="smaller experiments (fast)")
    ap.add_argument("--skip-experiments", action="store_true")
    a = ap.parse_args()

    # 1 ------------------------------------------------------------------ data
    banner("1. DATA  (simulated from the ground-truth network)")
    true_bn = build_true_network()
    df = generate_dataset(a.n_data, seed=a.seed, bn=true_bn)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    df.to_csv(DATA, index=False)
    print(f"{len(df)} applicants saved to {DATA}\n{df.head()}")
    print("\nDefault rate in data:", f"{(df['DefaultRisk'] == 'Yes').mean():.3f}")

    # 2 ------------------------------------------------------------ structure
    banner("2. STRUCTURE: independence, I-map, Markov blanket")
    bn0 = build_structure()
    print("Edges:", [(p, v) for v in bn0.nodes for p in bn0.parents[v]])
    print("Free parameters:", bn0.n_parameters(), "(a full joint table would need",
          int(np.prod([bn0.card(v) for v in bn0.nodes])) - 1, ")")
    print("Markov blanket of Income:", sorted(bn0.markov_blanket("Income")))
    print("\nLocal Markov independencies (X _|_ non-descendants | parents):")
    for v, nd, ps in bn0.local_independencies():
        print(f"  {v} _|_ {nd} | {ps}")
    print("\nd-separation queries (True => conditionally independent in every distribution of this I-map):")
    tests = [("Employment", "DefaultRisk", ["Income"]),
             ("Employment", "DefaultRisk", []),
             ("CreditHistory", "ExistingDebt", ["Income"]),
             ("CreditHistory", "ExistingDebt", []),
             ("CreditHistory", "ExistingDebt", ["Income", "DefaultRisk"]),   # collider opened
             ("Employment", "CreditHistory", ["Income"])]
    for x, y, z in tests:
        print(f"  {x:<14} _|_ {y:<13} | {str(z):<28} -> {bn0.is_d_separated(x, y, z)}")

    # 3 ------------------------------------------------------------ learning
    banner("3. PARAMETER LEARNING (Bayesian estimate, Dirichlet alpha=1)")
    bn = learn_parameters(bn0, df, "bayes", alpha=1.0)
    mle = learn_parameters(bn0, df, "mle")
    print("Learned CPT P(DefaultRisk | Income, CreditHistory, ExistingDebt):")
    print(bn.cpt_dataframe("DefaultRisk").round(3).to_string())
    print(f"\nKL(true || learned Bayes) = {kl_divergence(true_bn, bn):.5f}   "
          f"KL(true || MLE) = {kl_divergence(true_bn, mle):.5f}")
    print(f"avg log-likelihood on training data: {log_likelihood(bn, df):.4f}")

    # 4 --------------------------------------------------- exact inference
    banner("4. EXACT INFERENCE (Variable Elimination)")
    queries = [
        ("DefaultRisk", {}),
        ("DefaultRisk", {"CreditHistory": "Poor", "ExistingDebt": "High"}),
        ("DefaultRisk", {"Employment": "Stable", "Income": "High", "CreditHistory": "Good"}),
        ("Employment", {"DefaultRisk": "Yes"}),
    ]
    for var, ev in queries:
        res = variable_elimination(bn, var, ev)
        brute = enumeration_query(bn, var, ev)
        ok = np.allclose(res.factor.values, brute.values)
        print(f"P({var} | {ev or 'no evidence'})\n   {fmt(posterior_dict(bn, var, ev))}\n"
              f"   P(evidence)={res.evidence_prob:.4f}  order={res.order}  "
              f"largest factor={res.max_factor_size}  matches enumeration: {ok}")
    print("\nExplaining away (collider DefaultRisk <- CreditHistory, ExistingDebt), given Income=Medium, DefaultRisk=Yes:")
    base = {"Income": "Medium", "DefaultRisk": "Yes"}
    p1 = posterior_dict(bn, "CreditHistory", base)["Poor"]
    p2 = posterior_dict(bn, "CreditHistory", {**base, "ExistingDebt": "High"})["Poor"]
    print(f"   P(CreditHistory=Poor | ...) = {p1:.4f}  ->  {p2:.4f} after also observing ExistingDebt=High")
    print("   (high debt already explains the default, so a poor credit history becomes less likely)")

    # 5 ------------------------------------------------------------------ MAP
    banner("5. MAP / MPE")
    ev = {"DefaultRisk": "Yes"}
    print("MAP  argmax P(Employment, Income | DefaultRisk=Yes):", map_query(bn, ["Employment", "Income"], ev))
    print("MPE  most probable explanation of DefaultRisk=Yes:  ", most_probable_explanation(bn, ev))

    # 6 ----------------------------------------------- approximate inference
    banner("6. EXACT vs APPROXIMATE INFERENCE")
    n = 5000 if a.quick else 20000
    var, ev = "DefaultRisk", {"CreditHistory": "Poor", "ExistingDebt": "High"}
    exact = posterior_dict(bn, var, ev)
    print(f"Query P({var} | {ev}), {n} samples each\n   exact (VE)             {fmt(exact)}")
    for m in ("rejection", "likelihood_weighting", "gibbs"):
        p, info = approximate_posterior(bn, var, ev, m, n, rng=a.seed)
        d = dict(zip(bn.states[var], p))
        extra = ", ".join(f"{k}={v:.3g}" for k, v in info.items())
        print(f"   {m:<22} {fmt(d)}   [{extra}]")
    chains = gibbs_stream(bn, var, ev, 2000 if a.quick else 5000, burn_in=200, n_chains=4, rng=a.seed)
    print("   Gelman-Rubin R-hat over 4 Gibbs chains:", f"{gelman_rubin(chains.qidx == 1):.4f}", "(close to 1 = converged)")

    # 7 ------------------------------------------------------- experiments
    if not a.skip_experiments:
        banner("7. EXPERIMENTS (plots are written to ./results)")
        if a.quick:
            convergence.run(bn, RESULTS, max_samples=10_000, gibbs_samples=3_000, runs=8, seed=a.seed)
            learning_curve.run(true_bn, RESULTS, sizes=(20, 100, 500, 5000), reps=8, seed=a.seed)
        else:
            convergence.run(bn, RESULTS, seed=a.seed)
            learning_curve.run(true_bn, RESULTS, seed=a.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()
