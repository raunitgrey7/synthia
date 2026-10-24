"""Bayesian network sampler over discretised columns, with an optional differentially private fit.

Numeric columns are binned (quantile bins normally, equal-width bins on the public bounds under DP) so every
column becomes categorical. Structure learning computes pairwise mutual information, takes the maximum spanning
tree (Chow-Liu) and, when ``max_parents`` is two, adds a second parent per node when its mutual information with
the node clears ``second_parent_threshold`` nats. Conditional probability tables are estimated from counts with a
pseudocount; sampling follows the topological order and draws numeric values uniformly inside the chosen bin.

Under DP three Gaussian-mechanism queries touch the data: the stack of all pairwise contingency tables (one
record adds one to one cell of each of the ``d(d-1)/2`` tables, so L2 sensitivity is ``sqrt(d(d-1)/2)``), the
stack of all conditional count tables (one cell per node, sensitivity ``sqrt(d)``) and the stack of fine
within-bin histograms of the numeric columns (one cell per numeric column, sensitivity ``sqrt(d_numeric)``).
Negative noisy counts are clipped at zero before normalisation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from synthia.generators.base import Generator
from synthia.privacy import Accountant, DPConfig, PrivacyReport, calibrate_noise_multiplier
from synthia.privacy.mechanisms import gaussian_mechanism
from synthia.schema import TableSchema
from synthia.transforms import Discretizer


def mutual_information(table: np.ndarray) -> float:
    """Mutual information in nats of a (possibly noisy, non-negative) contingency table."""
    total = table.sum()
    if total <= 0:
        return 0.0
    joint = table / total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    ratio = np.zeros_like(joint)
    ratio[mask] = joint[mask] / (px @ py)[mask]
    return float(np.sum(joint[mask] * np.log(ratio[mask])))


def maximum_spanning_tree(weights: np.ndarray) -> list[tuple[int, int]]:
    """Kruskal on a dense symmetric weight matrix; returns the chosen undirected edges."""
    d = weights.shape[0]
    parent = list(range(d))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = sorted(((weights[i, j], i, j) for i in range(d) for j in range(i + 1, d)), reverse=True)
    chosen: list[tuple[int, int]] = []
    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            chosen.append((i, j))
    return chosen


def orient_tree(edges: list[tuple[int, int]], d: int, root: int) -> tuple[list[int], list[int | None]]:
    """Breadth-first orientation from the root: returns the node order and each node's single parent."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(d)}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    order = [root]
    parents: list[int | None] = [None] * d
    seen = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nxt in sorted(adjacency[node]):
            if nxt not in seen:
                seen.add(nxt)
                parents[nxt] = node
                order.append(nxt)
                queue.append(nxt)
    for node in range(d):
        if node not in seen:
            order.append(node)
            seen.add(node)
    return order, parents


class BayesianNetworkGenerator(Generator):
    """Tree-or-two-parent Bayesian network on binned columns; see the module docstring for the DP plan."""

    name = "bayesnet"

    def __init__(
        self,
        schema: TableSchema,
        dp: DPConfig | None = None,
        seed: int = 0,
        n_bins: int | None = None,
        max_parents: int | None = None,
        second_parent_threshold: float = 0.02,
        pseudocount: float = 0.5,
    ) -> None:
        super().__init__(schema, dp=dp, seed=seed)
        self.n_bins = n_bins if n_bins is not None else (6 if dp is not None else 10)
        self.max_parents = max_parents if max_parents is not None else (1 if dp is not None else 2)
        if self.max_parents not in (1, 2):
            raise ValueError("max_parents must be 1 or 2")
        if pseudocount <= 0:
            raise ValueError("pseudocount must be positive")
        self.second_parent_threshold = second_parent_threshold
        self.pseudocount = pseudocount
        self.discretizer = Discretizer(schema, n_bins=self.n_bins, strategy="uniform" if dp else "quantile")
        self.order: list[int] = []
        self.parents: list[tuple[int, ...]] = []
        self.tables: list[np.ndarray] = []
        self.mutual_info = np.zeros((len(schema), len(schema)))

    def _accountant(self, sigma: float) -> Accountant:
        acc = Accountant()
        acc.add_gaussian(sigma, count=1, label="pairwise contingency tables")
        acc.add_gaussian(sigma, count=1, label="conditional count tables")
        if self.schema.numeric_names:
            acc.add_gaussian(sigma, count=1, label="within-bin numeric profiles")
        return acc

    def _fit(self, df: pd.DataFrame) -> None:
        self.discretizer.fit(df)
        codes = self.discretizer.transform(df)
        cards = self.discretizer.cardinalities
        d = len(cards)
        sigma = None
        if self.dp is not None:
            sigma = calibrate_noise_multiplier(self.dp.epsilon, self.dp.delta, self._accountant)
        self.mutual_info = self._pairwise_mi(codes, cards, sigma)
        self._learn_structure(d)
        self.tables = self._fit_tables(codes, cards, sigma)
        if sigma is not None:
            self._noise_profiles(df, sigma)
        if self.dp is not None and sigma is not None:
            acc = self._accountant(sigma)
            self.privacy_report = PrivacyReport(
                epsilon_target=self.dp.epsilon,
                epsilon_spent=acc.epsilon(self.dp.delta),
                delta=self.dp.delta,
                noise_multiplier=sigma,
                rdp_order=acc.best_order(self.dp.delta),
                queries=acc.queries,
                caveats=self._dp_caveats(),
            )

    def _pairwise_mi(self, codes: np.ndarray, cards: list[int], sigma: float | None) -> np.ndarray:
        d = len(cards)
        mi = np.zeros((d, d))
        pairs = list(combinations(range(d), 2))
        sensitivity = np.sqrt(len(pairs)) if pairs else 1.0
        for i, j in pairs:
            table = _contingency(codes[:, i], codes[:, j], cards[i], cards[j])
            if sigma is not None:
                table = np.clip(gaussian_mechanism(table, sensitivity, sigma, self._rng), 0.0, None)
            mi[i, j] = mi[j, i] = mutual_information(table)
        return mi

    def _learn_structure(self, d: int) -> None:
        edges = maximum_spanning_tree(self.mutual_info)
        root = int(np.argmax(self.mutual_info.sum(axis=1)))
        self.order, single = orient_tree(edges, d, root)
        position = {node: k for k, node in enumerate(self.order)}
        parents: list[tuple[int, ...]] = [() for _ in range(d)]
        for node in range(d):
            first = single[node]
            if first is None:
                continue
            chosen = [first]
            if self.max_parents == 2:
                second = self._second_parent(node, first, position)
                if second is not None:
                    chosen.append(second)
            parents[node] = tuple(chosen)
        self.parents = parents

    def _second_parent(self, node: int, first: int, position: dict[int, int]) -> int | None:
        candidates = [c for c in position if position[c] < position[node] and c not in (node, first)]
        if not candidates:
            return None
        best = max(candidates, key=lambda c: self.mutual_info[node, c])
        return best if self.mutual_info[node, best] >= self.second_parent_threshold else None

    def _fit_tables(self, codes: np.ndarray, cards: list[int], sigma: float | None) -> list[np.ndarray]:
        d = len(cards)
        tables = []
        for node in range(d):
            parent_cols = self.parents[node]
            config_card = int(np.prod([cards[p] for p in parent_cols])) if parent_cols else 1
            config = _config_index(codes, parent_cols, cards)
            counts = np.zeros((config_card, cards[node]))
            np.add.at(counts, (config, codes[:, node]), 1.0)
            if sigma is not None:
                counts = np.clip(gaussian_mechanism(counts, np.sqrt(d), sigma, self._rng), 0.0, None)
            tables.append(_normalise_rows(counts + self.pseudocount))
        return tables

    def _noise_profiles(self, df: pd.DataFrame, sigma: float) -> None:
        names = self.schema.numeric_names
        for name in names:
            exact = self.discretizer.profile_counts(name, df[name].to_numpy(dtype=np.float64))
            self.discretizer.set_profile(name, gaussian_mechanism(exact, np.sqrt(len(names)), sigma, self._rng))

    def _sample(self, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
        cards = self.discretizer.cardinalities
        codes = np.zeros((n_rows, len(cards)), dtype=np.int64)
        for node in self.order:
            config = _config_index(codes, self.parents[node], cards)
            probs = self.tables[node][config]
            cumulative = np.cumsum(probs, axis=1)
            draws = rng.random(n_rows)[:, None]
            codes[:, node] = np.minimum((draws > cumulative).sum(axis=1), cards[node] - 1)
        return self.discretizer.inverse(codes, rng)

    def structure(self) -> dict[str, list[str]]:
        """Learned parents by column name, for the report and for tests."""
        names = self.schema.names
        return {names[node]: [names[p] for p in self.parents[node]] for node in range(len(names))}


def _contingency(a: np.ndarray, b: np.ndarray, card_a: int, card_b: int) -> np.ndarray:
    table = np.zeros((card_a, card_b))
    np.add.at(table, (a, b), 1.0)
    return table


def _config_index(codes: np.ndarray, parent_cols: tuple[int, ...], cards: list[int]) -> np.ndarray:
    index = np.zeros(len(codes), dtype=np.int64)
    for p in parent_cols:
        index = index * cards[p] + codes[:, p]
    return index


def _normalise_rows(table: np.ndarray) -> np.ndarray:
    totals = table.sum(axis=1, keepdims=True)
    return table / totals
