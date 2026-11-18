"""
Optimal NCT synthesis for 3 and 4 lines, following Shende, Prasad, Markov, Hayes,
"Synthesis of Reversible Logic Circuits", IEEE TCAD 2003 (arXiv quant-ph/0207001), Section 4:

  * a library of optimal circuits is built by breadth-first search (property of optimality,
    Lemma 35: every prefix of an optimal circuit is optimal), one circuit per permutation;
  * a permutation beyond the library is split as P = y o x with x, y both in the library
    (Fig. 8 with one level of recursion = meet in the middle), cost = d(x) + d(y).

n = 3: the library is the whole group (40320 permutations, max 8 gates) -> always optimal.
n = 4: library depth 6 (26.5 M permutations, ~80 s to build) -> exact optimum whenever it is
       <= 12 gates; otherwise None (the caller keeps its own circuit). Odd permutations on
       4 lines are not NCT-constructible without an extra line (paper, Lemma 11 / Cor. 13),
       so they return None as well.

Gate format as in the framework: list over lines, 1 = (positive) control, 2 = unused,
3 = target; line 0 is the MSB. A circuit is applied left to right.
"""

import time

import numpy as np

try:
    from .nct_core import NCTOracle, nct_library, parity, pack, unpack
except ImportError:
    from nct_core import NCTOracle, nct_library, parity, pack, unpack


DEFAULT_DEPTH = {3: 20, 4: 6}   # n = 3: BFS stops by itself when the group is exhausted


class OptimalNCT:
    """The library is built lazily, on the first synthesize() call that needs it (an odd
    4-line permutation is rejected before that, so it never pays the build cost)."""

    def __init__(self, n, depth=None, verbose=False, log=None):
        if n not in DEFAULT_DEPTH:
            raise ValueError("optimal NCT synthesis is only supported for 3 or 4 lines")
        self.n = n
        self.size = 1 << n
        self.depth = depth or DEFAULT_DEPTH[n]
        self.verbose = verbose
        self.log = log
        self.lib = nct_library(n)
        self._oracle = None

    @property
    def oracle(self):
        if self._oracle is None:
            t0 = time.time()
            self._oracle = NCTOracle(self.n, self.depth, verbose=self.verbose)
            if self.log:
                self.log(f"[NCT] Optimal-synthesis library for {self.n} lines: "
                         f"{self._oracle.keys.size:,} permutations, built in {time.time() - t0:.1f}s")
        return self._oracle

    # circuit of a permutation inside the library: walk down one gate at a time
    def _descend(self, perm):
        d = self.oracle.lookup(perm)
        assert d is not None
        circ, p = [], perm
        while d > 0:
            for g, gp in zip(self.lib, self.oracle.lib):
                q = gp[p]                      # q = g o p, and p = g o q (g is an involution)
                if self.oracle.lookup(q) == d - 1:
                    circ.append(g)
                    p, d = q, d - 1
                    break
        return circ[::-1]                      # p = g1 o g2 o ... -> apply ..., g2, g1

    def synthesize(self, perm):
        """Optimal NCT circuit (list of gates) for perm, or None if not found / impossible."""
        perm = np.asarray(perm, dtype=np.uint8)
        if self.n >= 4 and parity(perm):
            return None
        if self.oracle.lookup(perm) is not None:
            return self._descend(perm)
        # meet in the middle: P = y o x, x in the ball, y = P o x^-1 looked up in the table
        o = self.oracle
        keys = o.keys
        best = None                           # (cost, x_key, y_key)
        chunk = 2_000_000
        for s in range(0, keys.size, chunk):
            x = unpack(keys[s:s + chunk], self.size)
            inv = np.empty_like(x)
            rows = np.arange(x.shape[0])[:, None]
            inv[rows, x] = np.arange(self.size, dtype=np.uint8)[None, :]
            y = perm[inv]
            ky = pack(y)
            pos = np.searchsorted(keys, ky)
            pos[pos >= keys.size] = 0
            hit = keys[pos] == ky
            if hit.any():
                idx = np.nonzero(hit)[0]
                tot = o.dist[s + idx].astype(int) + o.dist[pos[idx]].astype(int)
                j = int(np.argmin(tot))
                if best is None or tot[j] < best[0]:
                    best = (int(tot[j]), keys[s + idx[j]], ky[idx[j]])
        if best is None:
            return None
        xp = unpack(np.array([best[1]], dtype=np.uint64), self.size)[0]
        yp = unpack(np.array([best[2]], dtype=np.uint64), self.size)[0]
        return self._descend(xp) + self._descend(yp)   # x first, then y
