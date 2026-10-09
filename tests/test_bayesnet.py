import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesnet import (BayesianNetwork, Factor, approximate_posterior, build_structure,
                      build_true_network, enumeration_query, generate_dataset, gelman_rubin,
                      gibbs_stream, kl_divergence, learn_parameters, map_query,
                      variable_elimination)

BN = build_true_network()


# ------------------------------------------------------------------ factors
def test_factor_product_and_marginalize():
    a = Factor(["A", "B"], [2, 2], [[0.1, 0.9], [0.5, 0.5]])
    b = Factor(["B", "C"], [2, 2], [[0.2, 0.8], [0.6, 0.4]])
    p = a.product(b)
    assert set(p.variables) == {"A", "B", "C"}
    expected = np.einsum("ab,bc->abc", a.values, b.values)
    assert np.allclose(p.reorder(["A", "B", "C"]).values, expected)
    assert np.allclose(p.marginalize(["C"]).reorder(["A", "B"]).values, a.values * b.values.sum(1))


def test_factor_reduce():
    f = Factor(["A", "B"], [2, 3], np.arange(6.0))
    r = f.reduce({"B": 2})
    assert r.variables == ["A"] and np.allclose(r.values, [2, 5])


# ------------------------------------------------------------------ network
def test_joint_sums_to_one_and_chain_rule():
    assert BN.joint_factor().values.sum() == pytest.approx(1.0)
    a = {"Employment": "Stable", "Income": "High", "CreditHistory": "Good",
         "ExistingDebt": "Low", "DefaultRisk": "No"}
    j = BN.joint_factor()
    idx = tuple(BN.states[v].index(a[v]) for v in j.variables)
    assert BN.joint_probability(a) == pytest.approx(j.values[idx])


def test_cycle_rejected():
    with pytest.raises(ValueError):
        BayesianNetwork({"A": ["0", "1"], "B": ["0", "1"]}, {"A": ["B"], "B": ["A"]})


def test_bad_cpt_rejected():
    with pytest.raises(ValueError):
        BN.copy().set_cpt("Employment", [0.5, 0.6])


def test_d_separation():
    assert BN.is_d_separated("Employment", "DefaultRisk", ["Income"])
    assert not BN.is_d_separated("Employment", "DefaultRisk")
    assert BN.is_d_separated("CreditHistory", "ExistingDebt", ["Income"])
    assert not BN.is_d_separated("CreditHistory", "ExistingDebt")
    # conditioning on the collider re-connects the two
    assert not BN.is_d_separated("CreditHistory", "ExistingDebt", ["Income", "DefaultRisk"])


def test_d_separation_matches_numeric_independence():
    """If the graph says d-separated, the exact distribution must factorize accordingly."""
    ev = {"Income": "Medium"}
    a = variable_elimination(BN, "CreditHistory", ev).factor.values
    b = variable_elimination(BN, "CreditHistory", {**ev, "ExistingDebt": "High"}).factor.values
    assert np.allclose(a, b)


def test_markov_blanket():
    assert BN.markov_blanket("CreditHistory") == {"Income", "DefaultRisk", "ExistingDebt"}


# ------------------------------------------------------------ exact inference
def all_queries():
    vars_ = BN.nodes
    for q in vars_:
        others = [v for v in vars_ if v != q]
        for k in range(0, 4):
            for ev_vars in itertools.combinations(others, k):
                for states in itertools.product(*[BN.states[v] for v in ev_vars]):
                    yield q, dict(zip(ev_vars, states))


def test_ve_equals_enumeration_on_many_queries():
    qs = list(all_queries())[::7]                      # a spread of ~100+ queries
    assert len(qs) > 50
    for q, ev in qs:
        ve = variable_elimination(BN, q, ev).factor.values
        brute = enumeration_query(BN, q, ev).values
        assert np.allclose(ve, brute), (q, ev)


def test_ve_any_order_same_answer():
    ev = {"DefaultRisk": "Yes"}
    base = variable_elimination(BN, "Employment", ev)
    for perm in itertools.permutations(base.order):
        r = variable_elimination(BN, "Employment", ev, order=list(perm))
        assert np.allclose(r.factor.values, base.factor.values)


def test_ve_prunes_barren_nodes():
    r = variable_elimination(BN, "CreditHistory", {})
    assert set(r.pruned) == {"ExistingDebt", "DefaultRisk"}


def test_ve_multi_variable_query_and_map():
    f = variable_elimination(BN, ["Employment", "Income"], {"DefaultRisk": "Yes"}).factor
    assert f.values.sum() == pytest.approx(1.0) and f.values.shape == (2, 3)
    assign, p = map_query(BN, ["Employment", "Income"], {"DefaultRisk": "Yes"})
    assert p == pytest.approx(f.values.max())


def test_zero_probability_evidence_raises():
    bn = BN.copy()
    bn.set_cpt("Employment", [1.0, 0.0])
    with pytest.raises(ValueError):
        variable_elimination(bn, "Income", {"Employment": "Stable"})


# ------------------------------------------------------- approximate inference
EV = {"CreditHistory": "Poor", "ExistingDebt": "High"}
EXACT = variable_elimination(BN, "DefaultRisk", EV).factor.values


@pytest.mark.parametrize("method,n,tol", [("rejection", 200_000, 0.01),
                                          ("likelihood_weighting", 200_000, 0.01),
                                          ("gibbs", 30_000, 0.02)])
def test_samplers_converge_to_exact(method, n, tol):
    p, _ = approximate_posterior(BN, "DefaultRisk", EV, method, n, rng=0)
    assert np.allclose(p, EXACT, atol=tol)


def test_gibbs_with_query_on_upstream_variable():
    ev = {"DefaultRisk": "Yes", "CreditHistory": "Poor"}
    exact = variable_elimination(BN, "Employment", ev).factor.values
    p, _ = approximate_posterior(BN, "Employment", ev, "gibbs", 30_000, rng=1)
    assert np.allclose(p, exact, atol=0.02)


def test_rhat_close_to_one_for_converged_chains():
    st = gibbs_stream(BN, "DefaultRisk", EV, 4000, burn_in=200, n_chains=4, rng=3)
    assert gelman_rubin(st.qidx == 1) < 1.05


# ------------------------------------------------------------------ learning
def test_learning_recovers_parameters():
    df = generate_dataset(100_000, seed=0, bn=BN)
    learned = learn_parameters(build_structure(), df, "mle")
    for v in BN.nodes:
        assert np.allclose(learned.cpt[v], BN.cpt[v], atol=0.02)
    assert kl_divergence(BN, learned) < 1e-3


def test_bayesian_estimate_beats_mle_on_tiny_data():
    df = generate_dataset(30, seed=5, bn=BN)
    s = build_structure()
    assert kl_divergence(BN, learn_parameters(s, df, "bayes")) < kl_divergence(BN, learn_parameters(s, df, "mle"))


def test_learned_cpts_are_valid():
    df = generate_dataset(50, seed=2, bn=BN)
    for method in ("mle", "bayes"):
        learned = learn_parameters(build_structure(), df, method)
        for v in learned.nodes:
            assert np.allclose(learned.cpt[v].sum(-1), 1.0)
