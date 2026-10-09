"""Loan Default Risk -- Bayesian Network explorer.      streamlit run app.py"""
import os

import numpy as np
import pandas as pd
import streamlit as st

from bayesnet import (approximate_posterior, build_structure, build_true_network,
                      generate_dataset, kl_divergence, learn_parameters, load_dataset,
                      map_query, most_probable_explanation, variable_elimination)
from bayesnet.sampling import METHODS
from experiments.convergence import compute_convergence, plot_convergence
from experiments.learning_curve import compute_learning_curve, plot_learning_curve

st.set_page_config(page_title="Loan Default Bayesian Network", layout="wide")
TRUE_BN = build_true_network()
NOT_OBSERVED = "(not observed)"


# ----------------------------------------------------------------------- sidebar: model
st.sidebar.header("1. Training data")
source = st.sidebar.radio("Data source", ["Synthetic (simulated)", "Upload CSV"])
if source == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with columns " + ", ".join(TRUE_BN.nodes), type="csv")
    if up is None:
        st.info("Upload a CSV (or switch to synthetic data) to continue.")
        st.stop()
    try:
        data = load_dataset(up, build_structure())
    except Exception as e:  # noqa
        st.error(f"Could not use this file: {e}")
        st.stop()
else:
    n_rows = st.sidebar.slider("Number of applicants", 50, 50_000, 5_000, step=50)
    seed = st.sidebar.number_input("Random seed", 0, 10_000, 42)
    data = generate_dataset(n_rows, seed=int(seed), bn=TRUE_BN)

method = st.sidebar.radio("Parameter estimation", ["Bayesian (Dirichlet prior)", "Maximum likelihood"])
alpha = st.sidebar.slider("Dirichlet alpha (pseudo-count)", 0.1, 5.0, 1.0, 0.1,
                          disabled=method.startswith("Maximum"))
bn = learn_parameters(build_structure(), data, "mle" if method.startswith("Maximum") else "bayes", alpha)

st.sidebar.header("2. Applicant / evidence")
query = st.sidebar.selectbox("Query variable", bn.nodes, index=bn.nodes.index("DefaultRisk"))
evidence = {}
for v in bn.nodes:
    if v == query:
        continue
    choice = st.sidebar.selectbox(v, [NOT_OBSERVED] + bn.states[v], key=f"ev_{v}")
    if choice != NOT_OBSERVED:
        evidence[v] = choice

st.title("Loan Default Risk -- Bayesian Network")
st.caption("Exact inference (variable elimination), approximate inference (sampling) and "
           "parameter learning on a discrete Bayesian network.")
tab_inf, tab_cmp, tab_conv, tab_net, tab_learn = st.tabs(
    ["Inference", "Exact vs approximate", "Convergence", "Network & independence", "Learning"])

# ---------------------------------------------------------------------------- inference
with tab_inf:
    try:
        res = variable_elimination(bn, query, evidence)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    post = pd.Series(res.factor.values, index=bn.states[query], name="probability")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(f"P({query} | evidence)")
        st.bar_chart(post)
        st.dataframe(post.to_frame().style.format("{:.4f}"))
    with c2:
        if query == "DefaultRisk":
            p = float(post["Yes"])
            st.metric("Probability of default", f"{p:.1%}")
            band = "LOW" if p < 0.15 else "MEDIUM" if p < 0.40 else "HIGH"
            st.markdown(f"**Risk band: {band}**")
        st.write(f"P(evidence) = **{res.evidence_prob:.4f}**")
        st.write(f"Elimination order: `{res.order}`")
        st.write(f"Largest intermediate factor: {res.max_factor_size} entries")
        st.write(f"Pruned (irrelevant) nodes: `{res.pruned}`")
    st.subheader("MAP / MPE")
    mpe, p_mpe = most_probable_explanation(bn, evidence)
    st.write(f"Most probable explanation of the evidence: `{mpe}` (posterior {p_mpe:.4f})")

# ------------------------------------------------------------------------- comparison
with tab_cmp:
    st.subheader("Same query, three samplers")
    n = st.slider("Samples", 100, 200_000, 20_000, step=100)
    burn = st.slider("Gibbs burn-in", 0, 5_000, 500, step=100)
    rs = st.number_input("Sampler seed", 0, 10_000, 0)
    rows = [dict(method="Exact (variable elimination)", **{s: p for s, p in zip(bn.states[query], post)})]
    notes = {}
    for m in METHODS:
        p, info = approximate_posterior(bn, query, evidence, m, n, rng=int(rs), burn_in=burn)
        rows.append(dict(method=m, **dict(zip(bn.states[query], p))))
        notes[m] = info
    table = pd.DataFrame(rows).set_index("method")
    table["max abs error"] = (table[bn.states[query]] - post.values).abs().max(axis=1)
    st.dataframe(table.style.format("{:.4f}"))
    st.bar_chart(table[bn.states[query]].T)
    st.write(f"Rejection sampling kept **{notes['rejection']['accepted']}** of {n} samples "
             f"({notes['rejection']['accepted'] / n:.2%}).  "
             f"Likelihood weighting effective sample size: **{notes['likelihood_weighting']['effective_sample_size']:.0f}**.")
    if notes["rejection"]["accepted"] == 0:
        st.warning("No sample matched the evidence -- rejection sampling returns an uninformative guess.")

# ------------------------------------------------------------------------- convergence
with tab_conv:
    st.subheader("Error versus number of samples")
    if not evidence:
        st.info("Select some evidence in the sidebar to see the difference between the samplers.")
    target = st.selectbox("State whose probability is tracked", bn.states[query])
    cc1, cc2, cc3 = st.columns(3)
    max_s = cc1.select_slider("Max samples (rejection / LW)", [5_000, 10_000, 50_000, 100_000], 50_000)
    g_s = cc2.select_slider("Gibbs sweeps", [1_000, 3_000, 10_000, 20_000], 3_000)
    runs = cc3.select_slider("Independent runs", [5, 10, 20, 50], 10)
    if st.button("Run convergence experiment"):
        with st.spinner("Sampling..."):
            r = compute_convergence(bn, query, target, evidence, max_s, g_s, runs)
        st.pyplot(plot_convergence(r))
        st.write(f"Exact value: **{r['exact']:.4f}**")
        st.dataframe(pd.DataFrame({k: dict(final_error=v["mean_err"][-1], seconds=v["seconds"])
                                   for k, v in r["methods"].items()}).T.style.format("{:.5f}"))

# ----------------------------------------------------------------------------- network
with tab_net:
    left, right = st.columns(2)
    with left:
        st.subheader("Structure")
        st.graphviz_chart(bn.to_dot())
        st.write(f"Free parameters: **{bn.n_parameters()}** "
                 f"(full joint table: {int(np.prod([bn.card(v) for v in bn.nodes])) - 1})")
        v = st.selectbox("Markov blanket of", bn.nodes)
        st.write(sorted(bn.markov_blanket(v)))
        st.subheader("Conditional probability tables")
        which = st.selectbox("Show CPT of", bn.nodes, index=bn.nodes.index("DefaultRisk"))
        st.dataframe(bn.cpt_dataframe(which).style.format("{:.3f}"))
    with right:
        st.subheader("d-separation checker")
        x = st.selectbox("X", bn.nodes, index=0)
        y = st.selectbox("Y", [n for n in bn.nodes if n != x], index=len(bn.nodes) - 2)
        z = st.multiselect("Observed (Z)", [n for n in bn.nodes if n not in (x, y)])
        sep = bn.is_d_separated(x, y, z)
        st.markdown(f"**{x} ⫫ {y} | {z or '{}'}** is **{'TRUE' if sep else 'FALSE'}** in the graph")
        st.caption("True means the graph guarantees independence. False means the graph allows "
                   "dependence (active trail), e.g. conditioning on a collider or its descendant.")
        st.subheader("Local Markov independencies")
        for v, nd, ps in bn.local_independencies():
            st.write(f"{v} ⫫ {nd} | {ps}")

# ----------------------------------------------------------------------------- learning
with tab_learn:
    st.subheader("Training data")
    st.dataframe(data.head(20))
    st.write(f"{len(data)} rows. Default rate: {(data['DefaultRisk'] == 'Yes').mean():.1%}")
    if source.startswith("Synthetic"):
        st.write(f"KL(true || learned) = **{kl_divergence(TRUE_BN, bn):.5f}**")
        which = st.selectbox("Compare CPT", bn.nodes, index=bn.nodes.index("Income"), key="cmp_cpt")
        st.write("Learned"); st.dataframe(bn.cpt_dataframe(which).style.format("{:.3f}"))
        st.write("True"); st.dataframe(TRUE_BN.cpt_dataframe(which).style.format("{:.3f}"))
        if st.button("Run learning-curve experiment (MLE vs Bayesian)"):
            with st.spinner("Learning on many dataset sizes..."):
                lc = compute_learning_curve(TRUE_BN, sizes=(20, 50, 100, 500, 2000, 10000), reps=10)
            st.pyplot(plot_learning_curve(lc))
    else:
        st.info("The true distribution is unknown for uploaded data, so only the learned CPTs are shown.")
