"""Bayesian network over discrete variables: structure, CPTs and graph-theoretic queries."""
from __future__ import annotations

from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd

from .factor import Factor


def _as_set(x):
    return {x} if isinstance(x, str) else set(x)


class BayesianNetwork:
    """A DAG with a conditional probability table (CPT) per node.

    states  : {variable: [state names]}
    parents : {variable: [parent names]}   (order matters: it is the CPT axis order)

    cpt[v] has shape (card(parent_1), ..., card(parent_k), card(v)); the last axis sums to 1.
    """

    def __init__(self, states: dict, parents: dict):
        self.states = {v: list(s) for v, s in states.items()}
        self.nodes = list(self.states)
        unknown = set(parents) - set(self.states)
        if unknown:
            raise ValueError(f"parents given for unknown variables: {unknown}")
        self.parents = {v: list(parents.get(v, [])) for v in self.nodes}
        for v, ps in self.parents.items():
            for p in ps:
                if p not in self.states:
                    raise ValueError(f"{v} has unknown parent {p}")
        self.children = {v: [c for c in self.nodes if v in self.parents[c]] for v in self.nodes}
        self._order = self._toposort()
        self.cpt = {v: self._uniform_cpt(v) for v in self.nodes}

    # ------------------------------------------------------------- basic access
    def card(self, v):
        return len(self.states[v])

    def topological_order(self):
        return list(self._order)

    def cpt_shape(self, v):
        return tuple(self.card(p) for p in self.parents[v]) + (self.card(v),)

    def _uniform_cpt(self, v):
        return np.full(self.cpt_shape(v), 1.0 / self.card(v))

    def _toposort(self):
        indeg = {v: len(self.parents[v]) for v in self.nodes}
        ready = [v for v in self.nodes if indeg[v] == 0]
        order = []
        while ready:
            v = ready.pop(0)
            order.append(v)
            for c in self.children[v]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("The graph contains a cycle -- a Bayesian network must be a DAG.")
        return order

    def copy(self):
        new = BayesianNetwork(self.states, self.parents)
        new.cpt = {v: a.copy() for v, a in self.cpt.items()}
        return new

    # --------------------------------------------------------------------- CPTs
    def set_cpt(self, v, table):
        arr = np.asarray(table, dtype=float)
        if arr.shape != self.cpt_shape(v):
            raise ValueError(f"CPT of {v} must have shape {self.cpt_shape(v)}, got {arr.shape}")
        if (arr < 0).any() or not np.allclose(arr.sum(axis=-1), 1.0, atol=1e-8):
            raise ValueError(f"CPT of {v}: entries must be >= 0 and every row must sum to 1")
        self.cpt[v] = arr

    def factor(self, v) -> Factor:
        """CPT of v as a Factor over parents(v) + [v]."""
        scope = self.parents[v] + [v]
        return Factor(scope, [self.card(x) for x in scope], self.cpt[v])

    def joint_factor(self) -> Factor:
        """Full joint distribution via the chain rule P(X) = prod_i P(Xi | Pa(Xi))."""
        return reduce(lambda a, b: a.product(b), (self.factor(v) for v in self._order))

    def joint_probability(self, assignment: dict) -> float:
        """P(full assignment) given as {var: state name}."""
        p = 1.0
        for v in self.nodes:
            idx = tuple(self.states[q].index(assignment[q]) for q in self.parents[v] + [v])
            p *= self.cpt[v][idx]
        return float(p)

    def n_parameters(self) -> int:
        """Number of free parameters, sum_v (prod parent cards) * (card(v) - 1)."""
        return int(sum(np.prod(self.cpt_shape(v)[:-1]) * (self.card(v) - 1) for v in self.nodes))

    def cpt_dataframe(self, v) -> pd.DataFrame:
        cols = self.states[v]
        flat = self.cpt[v].reshape(-1, len(cols))
        ps = self.parents[v]
        if ps:
            index = pd.MultiIndex.from_product([self.states[p] for p in ps], names=ps)
        else:
            index = pd.Index(["(prior)"])
        return pd.DataFrame(flat, index=index, columns=[f"{v}={s}" for s in cols])

    # ----------------------------------------------------------------- encoding
    def encode_evidence(self, evidence) -> dict:
        """{var: state name} -> {var: state index}."""
        out = {}
        for v, s in (evidence or {}).items():
            if v not in self.states:
                raise KeyError(f"unknown variable '{v}'")
            if s not in self.states[v]:
                raise KeyError(f"'{s}' is not a state of {v}; choose from {self.states[v]}")
            out[v] = self.states[v].index(s)
        return out

    def encode_dataframe(self, df: pd.DataFrame) -> dict:
        out = {}
        for v in self.nodes:
            codes = df[v].map({s: i for i, s in enumerate(self.states[v])})
            if codes.isna().any():
                bad = df.loc[codes.isna(), v].unique()[:5]
                raise ValueError(f"column {v} has values outside {self.states[v]}: {bad}")
            out[v] = codes.to_numpy(dtype=int)
        return out

    # ------------------------------------------------------------ graph queries
    def ancestors(self, nodes, include_self=True):
        nodes = _as_set(nodes)
        seen, stack = set(), [p for n in nodes for p in self.parents[n]]
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(self.parents[v])
        return seen | nodes if include_self else seen

    def descendants(self, node):
        seen, stack = set(), list(self.children[node])
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(self.children[v])
        return seen

    def markov_blanket(self, v):
        """Parents, children and the children's other parents."""
        mb = set(self.parents[v]) | set(self.children[v])
        for c in self.children[v]:
            mb |= set(self.parents[c])
        return mb - {v}

    def is_d_separated(self, x, y, z=()) -> bool:
        """True iff X and Y are d-separated given Z (so X _|_ Y | Z holds in every
        distribution that factorizes over this graph).

        Method: restrict to the ancestral set of X u Y u Z, moralize it (marry parents,
        drop directions), delete Z, and test whether X can still reach Y.
        """
        X, Y, Z = _as_set(x), _as_set(y), _as_set(z)
        if X & Y or (X | Y) & Z:
            raise ValueError("X, Y and Z must be disjoint")
        keep = self.ancestors(X | Y | Z)
        adj = {v: set() for v in keep}
        for v in keep:
            ps = self.parents[v]
            for p in ps:
                adj[v].add(p)
                adj[p].add(v)
            for a, b in combinations(ps, 2):
                adj[a].add(b)
                adj[b].add(a)
        seen, stack = set(), list(X)
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            if v in Y:
                return False
            stack.extend(n for n in adj[v] if n not in Z and n not in seen)
        return True

    def local_independencies(self):
        """Local Markov property: X _|_ NonDescendants(X) \\ Parents(X) | Parents(X)."""
        out = []
        for v in self.nodes:
            nd = set(self.nodes) - {v} - self.descendants(v) - set(self.parents[v])
            if nd:
                out.append((v, sorted(nd), list(self.parents[v])))
        return out

    def to_dot(self) -> str:
        lines = ["digraph BN {", "  rankdir=LR;", "  node [shape=ellipse, style=filled, fillcolor=lightblue];"]
        lines += [f'  "{p}" -> "{v}";' for v in self.nodes for p in self.parents[v]]
        return "\n".join(lines + ["}"])
