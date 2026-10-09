"""Discrete factors (probability tables) -- the basic data structure for inference."""
from __future__ import annotations

import numpy as np


class Factor:
    """A table phi(X1..Xk). `values` has one axis per variable, in `variables` order."""

    def __init__(self, variables, cards, values):
        self.variables = list(variables)
        self.cards = {v: int(c) for v, c in zip(self.variables, cards)}
        self.values = np.asarray(values, dtype=float).reshape(
            [self.cards[v] for v in self.variables]
        )

    # ------------------------------------------------------------------ helpers
    def _aligned(self, order):
        """Return values broadcastable against a factor whose axes follow `order`."""
        present = [v for v in order if v in self.cards]
        arr = np.transpose(self.values, [self.variables.index(v) for v in present])
        shape = [self.cards[v] if v in self.cards else 1 for v in order]
        return arr.reshape(shape)

    def reorder(self, order):
        """Return the same factor with its axes arranged as `order`."""
        order = list(order)
        assert set(order) == set(self.variables), "reorder needs the same variables"
        perm = [self.variables.index(v) for v in order]
        return Factor(order, [self.cards[v] for v in order], np.transpose(self.values, perm))

    # --------------------------------------------------------------- operations
    def product(self, other: "Factor") -> "Factor":
        """Factor product: phi1(A,B) * phi2(B,C) -> phi(A,B,C)."""
        order = self.variables + [v for v in other.variables if v not in self.cards]
        cards = {**self.cards, **other.cards}
        for v in self.cards:
            assert self.cards[v] == other.cards.get(v, self.cards[v]), f"cardinality clash on {v}"
        vals = self._aligned(order) * other._aligned(order)
        return Factor(order, [cards[v] for v in order], vals)

    def marginalize(self, variables) -> "Factor":
        """Sum out `variables`."""
        drop = [v for v in variables if v in self.cards]
        axes = tuple(self.variables.index(v) for v in drop)
        keep = [v for v in self.variables if v not in drop]
        return Factor(keep, [self.cards[v] for v in keep], self.values.sum(axis=axes))

    def maximize(self, variables) -> "Factor":
        """Max out `variables` (max-product, used for MAP-style computations)."""
        drop = [v for v in variables if v in self.cards]
        axes = tuple(self.variables.index(v) for v in drop)
        keep = [v for v in self.variables if v not in drop]
        vals = self.values.max(axis=axes) if axes else self.values
        return Factor(keep, [self.cards[v] for v in keep], vals)

    def reduce(self, evidence: dict) -> "Factor":
        """Condition on evidence {var: state_index}; evidence vars disappear from the scope."""
        idx = tuple(evidence[v] if v in evidence else slice(None) for v in self.variables)
        keep = [v for v in self.variables if v not in evidence]
        return Factor(keep, [self.cards[v] for v in keep], self.values[idx])

    def normalize(self) -> "Factor":
        total = self.values.sum()
        if total <= 0:
            raise ValueError("Cannot normalize: the evidence has probability zero.")
        return Factor(self.variables, [self.cards[v] for v in self.variables], self.values / total)

    def __repr__(self):
        return f"Factor({self.variables}, shape={self.values.shape})"
