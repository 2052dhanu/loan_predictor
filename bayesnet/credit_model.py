"""The loan-default domain: variables, graph structure and the ground-truth distribution that
is used to generate the synthetic dataset.

Graph:   Employment --> Income --> CreditHistory --> DefaultRisk
                           |  \\--> ExistingDebt --^      ^
                           \\______________________________|
"""
from __future__ import annotations

import numpy as np

from .network import BayesianNetwork

STATES = {
    "Employment": ["Unstable", "Stable"],
    "Income": ["Low", "Medium", "High"],
    "CreditHistory": ["Poor", "Fair", "Good"],
    "ExistingDebt": ["Low", "High"],
    "DefaultRisk": ["No", "Yes"],
}

PARENTS = {
    "Employment": [],
    "Income": ["Employment"],
    "CreditHistory": ["Income"],
    "ExistingDebt": ["Income"],
    "DefaultRisk": ["Income", "CreditHistory", "ExistingDebt"],
}


def build_structure() -> BayesianNetwork:
    """Graph with uniform (untrained) CPTs -- this is what parameter learning starts from."""
    return BayesianNetwork(STATES, PARENTS)


def build_true_network() -> BayesianNetwork:
    """Network with the 'real world' CPTs used to simulate loan applicants."""
    bn = build_structure()
    bn.set_cpt("Employment", [0.30, 0.70])
    bn.set_cpt("Income", [[0.55, 0.35, 0.10],      # Employment = Unstable
                          [0.15, 0.45, 0.40]])     # Employment = Stable
    bn.set_cpt("CreditHistory", [[0.45, 0.35, 0.20],    # Income = Low
                                 [0.20, 0.45, 0.35],    # Medium
                                 [0.08, 0.30, 0.62]])   # High
    bn.set_cpt("ExistingDebt", [[0.35, 0.65],
                                [0.55, 0.45],
                                [0.70, 0.30]])
    # P(Default | Income, Credit, Debt): a logistic score keeps the 18-row table consistent
    income_eff = [0.6, 0.0, -0.6]
    credit_eff = [1.2, 0.0, -1.0]
    debt_eff = [-0.5, 0.7]
    table = np.zeros((3, 3, 2, 2))
    for i in range(3):
        for c in range(3):
            for d in range(2):
                logit = -1.2 + income_eff[i] + credit_eff[c] + debt_eff[d]
                p = 1.0 / (1.0 + np.exp(-logit))
                table[i, c, d] = [1 - p, p]
    bn.set_cpt("DefaultRisk", table)
    return bn
