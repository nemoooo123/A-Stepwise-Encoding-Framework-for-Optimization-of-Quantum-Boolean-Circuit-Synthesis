"""
MPMCT -> strict NCT converter with local optimization (works for any n).

Pipeline (convert_to_nct):
  1. MPMCT-level merging: bring mergeable gates together with the commutation rule and merge
       C(S, x=0; t) . C(S, x=1; t) = C(S; t)          (drop a control)
       C(S; t)      . C(S, x=v; t) = C(S, x=~v; t)    (one gate fewer)
       A . A = I
  2. Gate-wise decomposition into NCT:
       negative controls -> NOT pairs
       k <= 2 controls   -> the gate itself
       k >= 3 controls   -> Barenco et al. 1995: Lemma 7.2 (V-chain, k-2 borrowed lines) if enough
                            idle lines exist, else Lemma 7.3 (split over one borrowed line)
       a gate with no idle line at all (n-1 controls, n >= 4) borrows ONE extra ancilla line,
       appended as the last line; it is used dirty, i.e. restored for both 0 and 1.
  3. NCT-level optimization, repeated until nothing changes:
       - deletion + moving rule (peephole)
       - window resynthesis: collect gates that act inside a set of <= K lines (skipping gates
         that commute with everything collected), and replace the window by an optimal NCT
         circuit for that K-line permutation if it is shorter
           K = 3: exact table of all 40320 permutations
           K = 4: optional exact table of all permutations with optimum <= 6 (BFS ball)

Gate format (same as the framework): list over lines, 0/1 = negative/positive control,
2 = not involved, 3 = target. Line 0 is the MSB.
"""

import itertools

import numpy as np

try:  # imported as nct_study.nct_convert (e.g. from main.py)
    from .nct_core import NCTOracle, nct_library, gate_perm, pack
except ImportError:  # run / imported from inside nct_study/
    from nct_core import NCTOracle, nct_library, gate_perm, pack


# ---------------------------------------------------------------------------
# basic gate helpers
# ---------------------------------------------------------------------------

def mk(n, ctrls, target, polarity=None):
    g = [2] * n
    g[target] = 3
    for c in ctrls:
        g[c] = 1 if polarity is None else polarity[c]
    return g


def support(g):
    return [i for i, v in enumerate(g) if v != 2]


def commute(g, h):
    """Sufficient commutation test for (mixed-polarity) multi-control Toffoli gates."""
    tg, th = g.index(3), h.index(3)
    if tg == th:
        return True
    for i in range(len(g)):  # contradicting polarities on a shared control: never both fire
        if i != tg and i != th and g[i] in (0, 1) and h[i] in (0, 1) and g[i] != h[i]:
            return True
    return h[tg] == 2 and g[th] == 2


def quantum_cost(gates):
    """NOT = CNOT = 1, Toffoli = 5 (the convention used by Saeedi et al. 2010)."""
    cost = 0
    for g in gates:
        k = sum(1 for v in g if v in (0, 1))
        cost += 5 if k == 2 else 1
    return cost


def mct_ncv_cost(m, n_lines):
    """
    Quantum cost (NCV elementary gates) of a positive-control C^m NOT on n_lines lines, using
    the formulas Saeedi et al. 2010 (JETC, p.4) take from Barenco 1995 / Maslov 2008:
      C^m NOT, enough idle lines (m <= ceil(n/2), i.e. >= m-2 idle) : 12m - 22
      C^(n-2) NOT, exactly one idle line                            : 24n - 88
      C^(n-1) NOT, no idle line                                     : 2^n - 3
    plus NOT = CNOT = 1, Toffoli = 5. The cheapest applicable formula is taken (a gate may
    always ignore idle lines).
    """
    if m <= 1:
        return 1
    if m == 2:
        return 5
    free = n_lines - m - 1
    options = [2 ** (m + 1) - 3]                 # treat as C^(k-1)NOT on its own k = m+1 lines
    if free >= m - 2:
        options.append(12 * m - 22)
    if free >= 1:
        options.append(24 * (m + 2) - 88)        # C^(n'-2)NOT on n' = m+2 lines
    return min(options)


def ncv_quantum_cost(mpmct, n_lines):
    """Saeedi-comparable quantum cost of an MPMCT circuit: negative controls become NOT pairs
    (cancelled by deletion + MR where possible), every gate is then costed with mct_ncv_cost."""
    flat = []
    for g in mpmct:
        nots = [mk(n_lines, [], i) for i, v in enumerate(g) if v == 0]
        flat += nots + [[1 if v == 0 else v for v in g]] + nots
    flat = peephole(flat)
    return sum(mct_ncv_cost(sum(1 for v in g if v in (0, 1)), n_lines) for g in flat)


def simulate(gates, n_lines):
    """Permutation of all 2^n_lines states (vectorised)."""
    x = np.arange(1 << n_lines, dtype=np.int64)
    for g in gates:
        t = g.index(3)
        mask = np.ones(x.shape[0], dtype=bool)
        for i, v in enumerate(g):
            if v in (0, 1):
                mask &= ((x >> (n_lines - 1 - i)) & 1) == v
        x = np.where(mask, x ^ (1 << (n_lines - 1 - t)), x)
    return x


# ---------------------------------------------------------------------------
# 1. MPMCT-level merging
# ---------------------------------------------------------------------------

def _merge(a, b):
    """Return ('cancel', None), ('merge', gate) or None."""
    if a.index(3) != b.index(3):
        return None
    if a == b:
        return ("cancel", None)
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diff) != 1:
        return None
    d = diff[0]
    pair = {a[d], b[d]}
    out = list(a)
    if pair == {0, 1}:
        out[d] = 2
    elif 2 in pair:  # C(S;t) . C(S,x=v;t) = C(S,x=~v;t)
        v = a[d] if a[d] != 2 else b[d]
        out[d] = 1 - v
    else:
        return None
    return ("merge", out)


def merge_mpmct(gates, window=80):
    """Forward passes (no restart after each merge, so it scales to thousands of gates),
    repeated until a full pass changes nothing."""
    g = [list(x) for x in gates]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(g):
            hit = False
            for j in range(i + 1, min(len(g), i + 1 + window)):
                r = _merge(g[i], g[j])
                if r is None:
                    continue
                between = g[i + 1:j]
                if all(commute(h, g[j]) for h in between):      # slide g[j] left to g[i]
                    pos = i
                elif all(commute(g[i], h) for h in between):    # slide g[i] right to g[j]
                    pos = j
                else:
                    continue
                kind, new = r
                if kind == "cancel":
                    del g[j]
                    del g[i]
                else:
                    g[pos] = new
                    del g[j if pos == i else i]
                changed = hit = True
                break
            if not hit:
                i += 1
    return g


# ---------------------------------------------------------------------------
# 2. gate-wise decomposition (Barenco et al. 1995, Lemmas 7.2 / 7.3)
# ---------------------------------------------------------------------------

class NeedsAncilla(Exception):
    pass


def _vchain(L, C, t, A):
    """Lemma 7.2: C^m NOT with m-2 borrowed (dirty) lines A -> 4(m-2) Toffolis."""
    m = len(C)
    top = mk(L, [C[m - 1], A[m - 3]], t)
    desc = [mk(L, [C[j + 1], A[j - 1]], A[j]) for j in range(m - 3, 0, -1)]
    bottom = mk(L, [C[0], C[1]], A[0])
    asc = desc[::-1]
    half = [top] + desc + [bottom] + asc
    return half + half


def _mct(L, C, t, free):
    m = len(C)
    if m <= 2:
        return [mk(L, C, t)]
    if len(free) >= m - 2:
        return _vchain(L, C, t, free[:m - 2])
    if not free:
        raise NeedsAncilla
    # Lemma 7.3: split over one borrowed line a
    a, others = free[0], free[1:]
    m1 = (m + 1) // 2
    C1, C2 = C[:m1], C[m1:]
    P = _mct(L, C1, a, C2 + [t] + others)
    G = _mct(L, C2 + [a], t, C1 + others)
    return P + G + P + G


def decompose(g):
    L = len(g)
    t = g.index(3)
    C = [i for i, v in enumerate(g) if v in (0, 1)]
    free = [i for i in range(L) if g[i] == 2]
    core = _mct(L, C, t, free)
    nots = [mk(L, [], i) for i in C if g[i] == 0]
    return nots + core + nots


# ---------------------------------------------------------------------------
# 3. NCT-level optimization
# ---------------------------------------------------------------------------

def peephole(gates):
    out = []
    for g in gates:
        j = len(out) - 1
        while j >= 0:
            if out[j] == g:
                del out[j]
                break
            if not commute(out[j], g):
                j = -1
                break
            j -= 1
        if j < 0:
            out.append(g)
    return out


class WindowTable:
    """Optimal NCT circuits for k-line permutations (k = 3 full, k = 4 within the BFS ball)."""

    def __init__(self, k, depth):
        self.k = k
        self.oracle = NCTOracle(k, depth, verbose=False)
        self.lib = nct_library(k)
        self.cache = {}

    def best(self, perm):
        key = perm.tobytes()
        if key in self.cache:
            return self.cache[key]
        d = self.oracle.lookup(perm)
        circ = None
        if d is not None:
            circ, p = [], perm
            while d > 0:
                for g, gp in zip(self.lib, self.oracle.lib):
                    q = gp[p]
                    if self.oracle.lookup(q) == d - 1:
                        circ.append(g)
                        p, d = q, d - 1
                        break
            circ = circ[::-1]
        self.cache[key] = circ
        return circ


def _local_perm(gates, lines):
    k = len(lines)
    pos = {l: i for i, l in enumerate(lines)}
    perm = np.arange(1 << k, dtype=np.uint8)
    for g in gates:
        lg = [2] * k
        for l, v in enumerate(g):
            if v != 2:
                lg[pos[l]] = v
        perm = gate_perm(lg)[perm]
    return perm


def window_resynthesis(gates, table):
    """One left-to-right sweep; returns (new_gates, improved?)."""
    K = table.k
    L = len(gates[0]) if gates else 0
    g = list(gates)
    improved = False
    i = 0
    while i < len(g):
        lines = set(support(g[i]))
        if len(lines) > K:
            i += 1
            continue
        taken, skipped = [i], []
        for j in range(i + 1, min(len(g), i + 200)):
            s = set(support(g[j]))
            if (s | lines) <= set(range(L)) and len(s | lines) <= K and \
                    all(commute(g[k2], g[j]) for k2 in skipped):
                lines |= s
                taken.append(j)
            elif all(commute(g[k2], g[j]) for k2 in taken):
                skipped.append(j)
            else:
                break
        if len(taken) >= 2:
            # pad the line set up to K lines so the table of size K can be used
            ls = sorted(lines)
            for extra in range(L):
                if len(ls) >= K:
                    break
                if extra not in ls:
                    ls.append(extra)
            ls = sorted(ls)
            if len(ls) == K:
                circ = table.best(_local_perm([g[j] for j in taken], ls))
                if circ is not None and len(circ) < len(taken):
                    new = []
                    for lg in circ:
                        x = [2] * L
                        for p, v in enumerate(lg):
                            x[ls[p]] = v
                        new.append(x)
                    keep = [g[j] for j in range(len(g)) if j not in set(taken)]
                    # taken gates commute past the skipped ones, so the window can sit at i
                    g = keep[:i] + new + keep[i:]
                    improved = True
        i += 1
    return g, improved


def optimize_nct(gates, tables):
    g = peephole(gates)
    while True:
        any_imp = False
        for t in tables:
            g, imp = window_resynthesis(g, t)
            g = peephole(g)
            any_imp |= imp
        if not any_imp:
            return g


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------

def convert_to_nct(mpmct, n, tables=(), merge=True):
    """
    Returns dict with the NCT circuit and statistics. If any gate still has n-1 controls after
    merging (n >= 4), one dirty ancilla line is appended (line index n) for the whole circuit.
    """
    g = merge_mpmct(mpmct) if merge else [list(x) for x in mpmct]
    merged_count = len(g)
    need_anc = n >= 4 and any(sum(1 for v in x if v in (0, 1)) == n - 1 for x in g)
    L = n + 1 if need_anc else n
    if need_anc:
        g = [x + [2] for x in g]
    nct = []
    for x in g:
        nct += decompose(x)
    raw = len(nct)
    nct = optimize_nct(nct, tables)
    return {"nct": nct, "lines": L, "ancilla": int(need_anc), "mpmct_after_merge": merged_count,
            "nct_raw": raw, "nct_gates": len(nct), "qc": quantum_cost(nct)}


def check(mpmct, n, result):
    """Simulated equivalence (ancilla line, if any, must be passed through for both values)."""
    ref = simulate(mpmct, n)
    got = simulate(result["nct"], result["lines"])
    if result["lines"] == n:
        return bool(np.array_equal(ref, got))
    exp = np.empty(2 << n, dtype=np.int64)
    exp[0::2] = 2 * ref
    exp[1::2] = 2 * ref + 1
    return bool(np.array_equal(exp, got))
