"""Channel coding utilities for DCT-QIM steganography.

This module currently provides a deterministic, dependency-free LDPC code:

* progressive-edge-growth parity-check matrix construction
* systematic encoding after Gaussian elimination
* normalized min-sum belief propagation decoding
* LLR interface compatible with ``qim_utils.message_llrs``

The implementation is intentionally dependency-light so that it can run in the
existing remote Python 3.7 / PyTorch 1.13 environment.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _as_bits(value: Sequence[int]) -> np.ndarray:
    bits = np.asarray(value, dtype=np.int8).reshape(-1)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("input must be binary")
    return bits


class LDPCCode:
    """A small deterministic binary LDPC code.

    Parameters
    ----------
    n:
        Coded-block length. In the DCT-QIM runner this is the number of QIM
        coded bits, e.g. 283.
    k:
        Secret-message length. The parity-check length is ``n - k``.
    seed:
        Seed for deterministic variable-node ordering.
    column_weight:
        Number of check nodes connected to each variable node. Three is a
        common regular LDPC choice and works well at low channel BER.
    """

    def __init__(
        self,
        n: int,
        k: int,
        seed: int = 20260826,
        column_weight: int = 3,
    ) -> None:
        if n <= 0 or k <= 0 or k >= n:
            raise ValueError("require 0 < k < n")
        if column_weight < 2:
            raise ValueError("column_weight must be at least 2")

        self.n = int(n)
        self.k = int(k)
        self.m = self.n - self.k
        self.seed = int(seed)
        self.column_weight = int(column_weight)
        self.code_rate = self.k / self.n

        self.h = self._build_regular_matrix_with_retries()
        self.column_order, self.h_systematic, self.a_matrix = self._systematize(self.h)
        self.var_to_checks = [
            np.flatnonzero(self.h[:, v]).astype(np.int64).tolist()
            for v in range(self.n)
        ]
        self.check_to_vars = [
            np.flatnonzero(self.h[c, :]).astype(np.int64).tolist()
            for c in range(self.m)
        ]

    @classmethod
    def from_rate(
        cls,
        n: int,
        rate: float,
        seed: int = 20260826,
        column_weight: int = 3,
    ) -> "LDPCCode":
        if not 0.0 < rate < 1.0:
            raise ValueError("rate must be in (0, 1)")
        k = int(np.floor(n * rate))
        if k <= 0 or k >= n:
            raise ValueError(f"rate {rate} gives invalid k={k} for n={n}")
        return cls(n=n, k=k, seed=seed, column_weight=column_weight)

    def _build_regular_matrix_with_retries(self, attempts: int = 20) -> np.ndarray:
        """Build a full-rank regular matrix, retrying deterministic seeds."""
        last_error = None
        for attempt in range(attempts):
            try:
                h = self._build_peg_matrix(self.seed + attempt * 100003)
                self._rref(h)
                return h
            except RuntimeError:
                pass

            try:
                h = self._build_regular_matrix(self.seed + attempt * 100003)
                self._rref(h)
                return h
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(f"could not build full-rank LDPC matrix: {last_error}")

    def _build_peg_matrix(self, seed: int) -> np.ndarray:
        """Progressive-edge-growth construction with random tie breaking."""
        rng = np.random.default_rng(seed)
        variable_order = rng.permutation(self.n)
        var_to_checks = [[] for _ in range(self.n)]
        check_to_vars = [[] for _ in range(self.m)]
        used_columns = set()

        for v in variable_order:
            accepted = None

            for _ in range(100):
                trial_checks = []
                trial_check_to_vars = [
                    list(neighbors) for neighbors in check_to_vars
                ]
                ok = True

                for _ in range(self.column_weight):
                    candidates = [
                        c for c in range(self.m) if c not in trial_checks
                    ]
                    if not candidates:
                        ok = False
                        break

                    depths = self._bipartite_depths(
                        int(v), var_to_checks, trial_check_to_vars
                    )
                    max_depth = max(depths[c] for c in candidates)
                    farthest = [c for c in candidates if depths[c] == max_depth]

                    chosen = int(rng.choice(farthest))
                    trial_checks.append(chosen)
                    trial_check_to_vars[chosen].append(int(v))

                if not ok:
                    continue

                column = tuple(sorted(trial_checks))
                if column not in used_columns:
                    accepted = (trial_checks, trial_check_to_vars)
                    break

            if accepted is None:
                raise RuntimeError("PEG construction failed to find a unique column")

            trial_checks, trial_check_to_vars = accepted
            var_to_checks[int(v)] = trial_checks
            check_to_vars = trial_check_to_vars
            used_columns.add(tuple(sorted(trial_checks)))

        h = np.zeros((self.m, self.n), dtype=bool)
        for v, checks in enumerate(var_to_checks):
            for c in checks:
                h[c, v] = True
        return h

    def _build_regular_matrix(self, seed: int) -> np.ndarray:
        """Construct a balanced random regular parity-check matrix."""
        rng = np.random.default_rng(seed)
        total_edges = self.n * self.column_weight
        base_degree = total_edges // self.m
        remainder = total_edges % self.m

        target_degree = np.full(self.m, base_degree, dtype=np.int64)
        target_degree[:remainder] += 1

        endpoint_pool = []
        for check, degree in enumerate(target_degree):
            endpoint_pool.extend([check] * int(degree))
        rng.shuffle(endpoint_pool)

        h = np.zeros((self.m, self.n), dtype=bool)
        used_columns = set()

        for v in range(self.n):
            checks = []
            position = 0
            while len(checks) < self.column_weight and position < len(endpoint_pool):
                check = int(endpoint_pool[position])
                if check not in checks:
                    checks.append(check)
                    endpoint_pool.pop(position)
                else:
                    position += 1

            if len(checks) != self.column_weight:
                raise RuntimeError("could not assign distinct checks to variable")

            column_key = tuple(sorted(checks))
            if column_key in used_columns:
                raise RuntimeError("duplicate LDPC column")

            h[checks, v] = True
            used_columns.add(column_key)

        return h

    @staticmethod
    def _bipartite_depths(
        source_variable: int,
        var_to_checks: List[List[int]],
        check_to_vars: List[List[int]],
    ) -> np.ndarray:
        """Return BFS depth from a variable node to every check node."""
        depths = np.full(len(check_to_vars), -1, dtype=np.int64)
        current_vars = [source_variable]
        visited_vars = {source_variable}
        depth = 0

        while current_vars:
            current_checks = []
            for v in current_vars:
                for c in var_to_checks[v]:
                    if depths[c] == -1:
                        depths[c] = depth + 1
                        current_checks.append(c)

            next_vars = []
            for c in current_checks:
                for v in check_to_vars[c]:
                    if v not in visited_vars:
                        visited_vars.add(v)
                        next_vars.append(v)

            current_vars = next_vars
            depth += 1

        return depths

    @staticmethod
    def _rref(matrix: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Return reduced row echelon form and pivot columns."""
        h = matrix.copy()
        rows, cols = h.shape
        pivot_cols: List[int] = []
        row = 0

        for col in range(cols):
            if row >= rows:
                break
            pivot_rows = np.flatnonzero(h[row:, col])
            if len(pivot_rows) == 0:
                continue

            pivot = row + int(pivot_rows[0])
            if pivot != row:
                h[[row, pivot]] = h[[pivot, row]]

            rows_with_one = np.flatnonzero(h[:, col])
            for r in rows_with_one:
                if r != row:
                    h[r] ^= h[row]

            pivot_cols.append(col)
            row += 1

        if row != rows:
            raise RuntimeError("LDPC parity-check matrix is not full row rank")
        return h, pivot_cols

    def _systematize(
        self, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Permute columns so H becomes ``[A | I]``.

        Returns
        -------
        column_order:
            The permuted column order. ``column_order[j]`` gives the physical
            coded-bit index corresponding to systematic position ``j``.
        h_systematic:
            The parity-check matrix in ``[A | I]`` form.
        a_matrix:
            The ``m x k`` matrix ``A`` used for systematic encoding.
        """
        _, pivot_cols = self._rref(h)
        pivot_set = set(pivot_cols)
        information_cols = [c for c in range(self.n) if c not in pivot_set]
        column_order = np.asarray(
            information_cols + pivot_cols, dtype=np.int64
        )

        h_systematic = h[:, column_order].copy()

        # Force the last m columns to become the identity. A generic RREF can
        # choose earlier information columns as pivots, so elimination must be
        # restricted to the selected parity columns.
        for row, parity_col in enumerate(range(self.k, self.n)):
            pivot_candidates = np.flatnonzero(h_systematic[row:, parity_col])
            if len(pivot_candidates) == 0:
                raise RuntimeError("failed to construct systematic LDPC matrix")
            pivot = row + int(pivot_candidates[0])
            if pivot != row:
                h_systematic[[row, pivot]] = h_systematic[[pivot, row]]

            rows_with_one = np.flatnonzero(h_systematic[:, parity_col])
            for other_row in rows_with_one:
                if other_row != row:
                    h_systematic[other_row] ^= h_systematic[row]

        if not np.array_equal(
            h_systematic[:, self.k :], np.eye(self.m, dtype=bool)
        ):
            raise RuntimeError("systematic LDPC identity check failed")

        a_matrix = h_systematic[:, : self.k].astype(bool)
        return column_order, h_systematic, a_matrix

    def encode(self, secret_bits: Sequence[int]) -> np.ndarray:
        """Systematically encode secret bits into a length-n codeword."""
        secret = _as_bits(secret_bits)
        if len(secret) != self.k:
            raise ValueError(f"expected {self.k} secret bits, got {len(secret)}")

        parity = (secret @ self.a_matrix.T) % 2
        coded_systematic = np.concatenate((secret, parity)).astype(np.int8)

        physical = np.empty(self.n, dtype=np.int8)
        physical[self.column_order] = coded_systematic
        return physical

    def check(self, coded_bits: Sequence[int]) -> bool:
        bits = _as_bits(coded_bits)
        if len(bits) != self.n:
            raise ValueError(f"expected {self.n} coded bits, got {len(bits)}")
        syndrome = np.logical_and(self.h, bits.astype(bool)).sum(axis=1) % 2
        return not np.any(syndrome)

    def decode(
        self,
        received_bits: Sequence[int],
        channel_llrs: Sequence[float],
        max_iterations: int = 60,
        scaling: float = 0.80,
    ) -> Dict[str, np.ndarray]:
        """Decode with normalized min-sum belief propagation.

        ``channel_llrs`` uses the same sign convention as QIM extraction:
        positive values favour bit 0 and negative values favour bit 1.
        """
        received = _as_bits(received_bits)
        llr = np.asarray(channel_llrs, dtype=np.float64).reshape(-1)
        if len(received) != self.n or len(llr) != self.n:
            raise ValueError(
                f"expected {self.n} coded bits and LLRs, got "
                f"{len(received)} and {len(llr)}"
            )
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        # Belief propagation runs on the original sparse parity-check matrix.
        # Systematic column ordering is used only for encoding and extracting
        # the information bits after decoding.
        llr_sys = llr.copy()
        received_sys = received.copy()

        # q[v, c] is the message from variable v to check c.
        q = np.zeros((self.n, self.column_weight), dtype=np.float64)
        for v in range(self.n):
            q[v, :] = llr_sys[v]

        iterations_used = 0
        final_llr = llr_sys.copy()
        decoded_sys = received_sys.copy()
        checks_satisfied = False

        for iteration in range(1, max_iterations + 1):
            iterations_used = iteration

            # Check-to-variable messages using normalized min-sum.
            r = np.zeros_like(q)
            for c in range(self.m):
                variables = self.check_to_vars[c]
                if not variables:
                    continue
                if len(variables) == 1:
                    # A degree-one check provides no extrinsic information.
                    v = int(variables[0])
                    pos = int(self.var_to_checks[v].index(c))
                    r[v, pos] = 0.0
                    continue

                positions = {
                    int(v): int(self.var_to_checks[v].index(c))
                    for v in variables
                }
                local_indices = {int(v): j for j, v in enumerate(variables)}
                signs = np.asarray(
                    [np.sign(q[v, positions[v]]) for v in variables],
                    dtype=np.float64,
                )
                magnitudes = np.asarray(
                    [abs(float(q[v, positions[v]])) for v in variables],
                    dtype=np.float64,
                )

                for v in variables:
                    pos = positions[v]
                    local_index = local_indices[int(v)]
                    sign_product = float(
                        np.prod(signs) / signs[local_index]
                    )
                    min_magnitude = float(
                        np.min(np.delete(magnitudes, local_index))
                    )
                    r[v, pos] = scaling * sign_product * min_magnitude

            # Variable-to-check messages.
            for v in range(self.n):
                checks = self.var_to_checks[v]
                total = llr_sys[v] + float(np.sum(r[v, :]))
                final_llr[v] = total
                for local_pos, c in enumerate(checks):
                    q[v, local_pos] = np.clip(
                        total - r[v, local_pos], -100.0, 100.0
                    )

            decoded_sys = (final_llr < 0.0).astype(np.int8)
            syndrome = (
                np.logical_and(self.h, decoded_sys.astype(bool))
                .sum(axis=1)
                % 2
            )
            if not np.any(syndrome):
                checks_satisfied = True
                break

        decoded_physical = decoded_sys.copy()
        decoded_systematic = decoded_physical[self.column_order]
        decoded_secret = decoded_systematic[: self.k].copy()

        return {
            "secret_bits": decoded_secret,
            "coded_bits": decoded_physical,
            "decoded_llrs": final_llr,
            "iterations_used": iterations_used,
            "checks_satisfied": bool(checks_satisfied),
        }


def build_ldpc_code(
    coded_length: int,
    rate: float,
    seed: int = 20260826,
    column_weight: int = 3,
) -> LDPCCode:
    """ Convenience constructor used by experiment scripts."""
    return LDPCCode.from_rate(
        n=coded_length,
        rate=rate,
        seed=seed,
        column_weight=column_weight,
    )