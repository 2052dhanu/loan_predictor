"""Exact inference: Variable Elimination (with pruning + min-fill ordering), brute-force
enumeration (as a correctness baseline) and MAP / MPE queries."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import combinations

import numpy as np

from .factor import Factor
from .network import BayesianNetwork


def _as_list(q):
    return [q] if isinstance(q, str) else list(q)


@dataclass
class VEResult:
    factor: Factor                 # normalized posterior P(query | evidence)
    order: list                    # elimination order that was used
    max_factor_size: int           # largest intermediate table (number of entries)
    evidence_prob: float           # P(evidence)
    pruned: list = field(default_factory=list)   # barren nodes removed before inference


def elimination_order(scopes, hidden, heuristic="min_fill"):
    """Greedy elimination ordering on the interaction (moral) graph of the factor scopes."""
    adj = {}
    for scope in scopes:
        for v in scope:
            adj.setdefault(v, set())
        for a, b in combinations(scope, 2):
            adj[a].add(b)
            adj[b].add(a)
    for v in hidden:
        adj.setdefault(v, set())
    remaining, order = set(hidden), []

    def cost(v):
        nb = adj[v]
        if heuristic == "min_neighbors":
            return (len(nb), 0)
        fill = sum(1 for a, b in combinations(nb, 2) if b not in adj[a])
        return (fill, len(nb))

    while remaining:
        v = min(remaining, key=lambda x: (cost(x), x))
        nb = adj[v]
        for a, b in combinations(nb, 2):          # connect v's neighbours (fill-in edges)
            adj[a].add(b)
            adj[b].add(a)
        for n in nb:
            adj[n].discard(v)
        del adj[v]
        remaining.discard(v)
        order.append(v)
    return order


def variable_elimination(bn: BayesianNetwork, query, evidence=None, order=None,
                         heuristic="min_fill", prune=True) -> VEResult:
    """P(query | evidence) by sum-product variable elimination.

    1. keep only the ancestors of query u evidence (the other nodes are barren: they sum to 1)
    2. reduce every CPT by the evidence
    3. for each hidden variable: multiply the factors that mention it, then sum it out
    4. multiply what is left, normalize
    """
    query = _as_list(query)
    ev = bn.encode_evidence(evidence)
    if set(query) & set(ev):
        raise ValueError("A query variable cannot also be evidence.")

    relevant = bn.ancestors(set(query) | set(ev)) if prune else set(bn.nodes)
    pruned = [v for v in bn.topological_order() if v not in relevant]
    factors = [bn.factor(v).reduce(ev) for v in bn.topological_order() if v in relevant]
    hidden = [v for v in bn.topological_order() if v in relevant and v not in query and v not in ev]

    if order is None:
        order = elimination_order([f.variables for f in factors], hidden, heuristic)
    elif set(order) != set(hidden):
        raise ValueError(f"order must contain exactly the hidden variables {hidden}")

    max_size = 0
    for var in order:
        involved = [f for f in factors if var in f.cards]
        rest = [f for f in factors if var not in f.cards]
        prod = reduce(lambda a, b: a.product(b), involved)
        max_size = max(max_size, prod.values.size)
        factors = rest + [prod.marginalize([var])]

    final = reduce(lambda a, b: a.product(b), factors)
    p_evidence = float(final.values.sum())
    posterior = final.normalize().reorder(query)
    return VEResult(posterior, list(order), max_size, p_evidence, pruned)


def enumeration_query(bn: BayesianNetwork, query, evidence=None) -> Factor:
    """Brute force: build the whole joint, condition, marginalize. Exponential in #variables."""
    query = _as_list(query)
    joint = bn.joint_factor().reduce(bn.encode_evidence(evidence))
    others = [v for v in joint.variables if v not in query]
    return joint.marginalize(others).normalize().reorder(query)


def posterior_dict(bn, query, evidence=None, **kw):
    """Convenience: single variable -> {state: prob}; several -> {(s1, s2, ..): prob}."""
    f = variable_elimination(bn, query, evidence, **kw).factor
    return factor_to_dict(bn, f)


def factor_to_dict(bn, f: Factor):
    if len(f.variables) == 1:
        v = f.variables[0]
        return {s: float(p) for s, p in zip(bn.states[v], f.values)}
    out = {}
    for idx in np.ndindex(*f.values.shape):
        out[tuple(bn.states[v][i] for v, i in zip(f.variables, idx))] = float(f.values[idx])
    return out


def map_query(bn, query, evidence=None):
    """MAP: argmax_q P(q | e). Returns ({var: state}, probability)."""
    f = variable_elimination(bn, query, evidence).factor
    idx = np.unravel_index(np.argmax(f.values), f.values.shape)
    return {v: bn.states[v][i] for v, i in zip(f.variables, idx)}, float(f.values[idx])


def most_probable_explanation(bn, evidence=None):
    """MPE: best joint assignment of ALL unobserved variables given the evidence."""
    ev = evidence or {}
    return map_query(bn, [v for v in bn.nodes if v not in ev], ev)
