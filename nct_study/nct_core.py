"""
Core utilities for the MPMCT -> NCT conversion study.

Conventions (same as utils/topology.py):
  - A state is an integer in [0, 2^n); line i (0-based) is bit weight 2^(n-1-i), i.e.
    line 0 is the MSB, matching the MSB-first gate lists used in the framework.
  - An MPMCT gate from the framework is a length-n list: 0/1 = negative/positive control,
    3 = target. Here we also allow 2 = "not a control" (don't care), so the same format can
    describe NCT gates (NOT = one 3 and all 2s, CNOT = one control, Toffoli = two controls).
  - A circuit is applied left to right. Its permutation p maps input x -> output p[x].
    Appending gate g to circuit p gives g[p] (numpy fancy indexing).

Optimal NCT gate counts are obtained by exact breadth-first search over the permutation
group (n = 3: the full group, 40320 elements) or by meet-in-the-middle over BFS balls
(n = 4), so every number reported by the experiments is a true optimum, not a heuristic.
"""

import itertools
import numpy as np


# ---------------------------------------------------------------------------
# Gates and permutations
# ---------------------------------------------------------------------------

def gate_perm(gate):
    """Permutation (np.uint8 array of length 2^n) realised by a 0/1/2/3 gate list."""
    n = len(gate)
    size = 1 << n
    target = gate.index(3)
    tmask = 1 << (n - 1 - target)
    perm = np.arange(size, dtype=np.uint8)
    for x in range(size):
        ok = True
        for i, v in enumerate(gate):
            if v in (0, 1) and ((x >> (n - 1 - i)) & 1) != v:
                ok = False
                break
        if ok:
            perm[x] = x ^ tmask
    return perm


def circuit_perm(gates, n):
    perm = np.arange(1 << n, dtype=np.uint8)
    for g in gates:
        perm = gate_perm(g)[perm]
    return perm


def gate_str(gate):
    """Readable form, e.g. [1,0,3] -> 'T(x0,~x1 -> x2)'."""
    t = gate.index(3)
    ctrl = [("" if v == 1 else "~") + f"x{i}" for i, v in enumerate(gate) if v in (0, 1)]
    name = {0: "NOT", 1: "CNOT", 2: "TOF"}.get(len(ctrl), f"C{len(ctrl)}NOT")
    return f"{name}({','.join(ctrl)}{' -> ' if ctrl else ''}x{t})"


def nct_library(n):
    """All NCT gates on n lines: positive controls only, at most 2 controls."""
    lib = []
    for t in range(n):
        others = [i for i in range(n) if i != t]
        for k in range(3):
            for ctrl in itertools.combinations(others, k):
                g = [2] * n
                g[t] = 3
                for c in ctrl:
                    g[c] = 1
                lib.append(g)
    return lib


def full_control_gates(n):
    """All full-control MPMCT gates (hypercube edges): every non-target line is a 0/1 control.
    This is exactly the gate set the stepwise-encoding framework emits."""
    gates = []
    for t in range(n):
        for bits in itertools.product((0, 1), repeat=n - 1):
            g = list(bits)
            g.insert(t, 3)
            gates.append(g)
    return gates


def parity(perm):
    """0 = even permutation, 1 = odd."""
    perm = list(perm)
    seen = [False] * len(perm)
    transpositions = 0
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        transpositions += length - 1
    return transpositions & 1


def with_dirty_ancilla(perm, n):
    """Embed an n-bit permutation on n+1 lines, the extra line being the new LSB whose value
    is passed through unchanged (a 'dirty'/borrowed ancilla: must be restored for both 0 and 1)."""
    size = 1 << n
    out = np.empty(2 * size, dtype=np.uint8)
    for x in range(size):
        out[2 * x] = 2 * perm[x]
        out[2 * x + 1] = 2 * perm[x] + 1
    return out


# ---------------------------------------------------------------------------
# Packing permutations into integer keys (4 bits per entry, n <= 4)
# ---------------------------------------------------------------------------

def pack(perms):
    """(k, 2^n) uint8 -> (k,) uint64 keys."""
    perms = np.asarray(perms)
    if perms.ndim == 1:
        perms = perms[None, :]
    keys = np.zeros(perms.shape[0], dtype=np.uint64)
    for i in range(perms.shape[1]):
        keys |= perms[:, i].astype(np.uint64) << np.uint64(4 * i)
    return keys


def unpack(keys, size):
    keys = np.asarray(keys, dtype=np.uint64)
    out = np.empty((keys.shape[0], size), dtype=np.uint8)
    for i in range(size):
        out[:, i] = ((keys >> np.uint64(4 * i)) & np.uint64(15)).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Exact optimal NCT cost
# ---------------------------------------------------------------------------

class NCTOracle:
    """
    Exact minimum NCT gate count.

    BFS from the identity up to `depth` builds a sorted table (key -> distance). Because every
    NCT gate is an involution, dist(p) == dist(p^-1), and a permutation P of cost <= 2*depth
    is found by meet in the middle:  P = y o x  with x, y in the ball, cost = d(x) + d(y).
    For n = 3 the ball is the whole group, so lookups are exact directly.
    """

    def __init__(self, n, depth, chunk=400_000, verbose=True):
        self.n = n
        self.size = 1 << n
        self.depth = depth
        lib = [gate_perm(g) for g in nct_library(n)]
        self.lib = np.stack(lib)

        ident = np.arange(self.size, dtype=np.uint8)
        seen = pack(ident)
        all_keys = [seen.copy()]
        all_dist = [np.zeros(1, dtype=np.uint8)]
        frontier = ident[None, :]
        self.level_sizes = [1]
        for d in range(1, depth + 1):
            new_keys = []
            for s in range(0, frontier.shape[0], chunk):
                f = frontier[s:s + chunk]
                cand = self.lib[:, f].reshape(-1, self.size)  # every gate appended to every state
                k = np.unique(pack(cand))
                pos = np.searchsorted(seen, k)
                pos[pos >= seen.size] = 0
                k = k[seen[pos] != k]
                new_keys.append(k)
            k = np.unique(np.concatenate(new_keys)) if new_keys else np.zeros(0, np.uint64)
            if k.size == 0:
                break
            self.level_sizes.append(int(k.size))
            if verbose:
                print(f"  [BFS n={n}] depth {d}: {k.size:,} new permutations", flush=True)
            all_keys.append(k)
            all_dist.append(np.full(k.size, d, dtype=np.uint8))
            seen = np.union1d(seen, k)
            frontier = unpack(k, self.size)

        keys = np.concatenate(all_keys)
        dist = np.concatenate(all_dist)
        order = np.argsort(keys)
        self.keys = keys[order]
        self.dist = dist[order]
        self.complete = sum(self.level_sizes) == _group_order(self.size) // (1 if n <= 3 else 2)
        self._ball_perms = None

    def lookup(self, perm):
        """Exact distance if perm is inside the BFS ball, else None."""
        k = pack(perm)[0]
        i = np.searchsorted(self.keys, k)
        if i < self.keys.size and self.keys[i] == k:
            return int(self.dist[i])
        return None

    def cost(self, perm):
        """Exact NCT cost; None if unreachable (odd permutation for n >= 4) or > 2*depth."""
        perm = np.asarray(perm, dtype=np.uint8)
        if self.n >= 4 and parity(perm):
            return None
        d = self.lookup(perm)
        if d is not None:
            return d
        # meet in the middle: P = y o x  <=>  y = P o x^-1, i.e. y = P[x_inv]
        # left_depth < depth trades exactness range (left_depth + depth) for speed
        left = getattr(self, "left_depth", self.depth)
        if self._ball_perms != left:
            sel = self.dist <= left
            ball = unpack(self.keys[sel], self.size)
            inv = np.empty_like(ball)
            rows = np.arange(ball.shape[0])[:, None]
            inv[rows, ball] = np.arange(self.size, dtype=np.uint8)[None, :]
            self._ball_inv = inv
            self._ball_inv_dist = self.dist[sel]
            self._ball_perms = left
        best = None
        chunk = 2_000_000
        for s in range(0, self._ball_inv.shape[0], chunk):
            y = perm[self._ball_inv[s:s + chunk]]
            ky = pack(y)
            pos = np.searchsorted(self.keys, ky)
            pos[pos >= self.keys.size] = 0
            hit = self.keys[pos] == ky
            if hit.any():
                tot = self._ball_inv_dist[s:s + chunk][hit].astype(int) + self.dist[pos[hit]].astype(int)
                m = int(tot.min())
                best = m if best is None else min(best, m)
        return best


def _group_order(size):
    out = 1
    for i in range(2, size + 1):
        out *= i
    return out
