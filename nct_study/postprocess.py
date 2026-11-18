"""
NCT post-processing for main.py: convert the best MPMCT circuit of every trial into a strict
NCT circuit (NOT / CNOT / Toffoli, positive controls only) with nct_convert.convert_to_nct:

    MPMCT merging -> Barenco decomposition -> deletion/MR + window resynthesis

The search itself is unchanged (its fitness is still the MPMCT gate count); this runs once per
trial after the search has finished.

If a merged circuit still contains a gate that uses every line (n-1 controls, n >= 4), the NCT
circuit gets ONE extra dirty ancilla line, appended as the last line: it may start in any value
and is restored afterwards.
"""

import time

import numpy as np

try:
    from .nct_convert import convert_to_nct, check, simulate, quantum_cost, WindowTable
    from .nct_optimal import OptimalNCT
except ImportError:
    from nct_convert import convert_to_nct, check, simulate, quantum_cost, WindowTable
    from nct_optimal import OptimalNCT


def build_tables(window4=False):
    """Exact optimal-NCT tables for window resynthesis: 3 lines (all 40320 permutations, fast)
    and optionally 4 lines (permutations with optimum <= 6; takes a while to build)."""
    tables = [WindowTable(3, 20)]
    if window4:
        tables.append(WindowTable(4, 6))
    return tables


def build_optimal(num_bits):
    """Optimal synthesizer (Shende et al. 2003, Sec. 4) for 3 / 4 lines, else None.
    n = 4 builds a 26.5 M-entry library (~80-100 s, a few hundred MB of RAM)."""
    if num_bits not in (3, 4):
        return None
    return OptimalNCT(num_bits, log=print)  # library is built on first use (skipped for odd functions)


def convert_trials(circuits, num_bits, tables, verify_max_lines=12, optimal=None, log=print):
    """
    Convert every trial's best MPMCT circuit. Returns a list of dicts (one per trial) with the
    convert_to_nct statistics plus 'method' ('convert' or 'optimal'), 'converted_gates' (the
    NCT count of the conversion pipeline), 'verified' (True / False, or None when the circuit has
    more than verify_max_lines lines and simulation was skipped) and 'time'.

    With `optimal` (an OptimalNCT for num_bits lines), the permutation computed by the MPMCT
    circuit is synthesized optimally and replaces the converted circuit when it is found.
    All trials compute the same function (the target or its inverse), so this is done once per
    distinct permutation; an inverse reuses the reversed circuit.
    """
    results = []
    opt_cache = {}
    for k, circ in enumerate(circuits):
        t0 = time.time()
        r = convert_to_nct([list(g) for g in circ], num_bits, tables=tables)
        r["method"] = "convert"
        r["converted_gates"] = r["nct_gates"]
        if optimal is not None:
            perm = simulate(circ, num_bits).astype(np.uint8)
            key = perm.tobytes()
            if key not in opt_cache:
                inv_key = np.argsort(perm).astype(np.uint8).tobytes()
                if inv_key in opt_cache:
                    c = opt_cache[inv_key]
                    opt_cache[key] = None if c is None else c[::-1]
                else:
                    opt_cache[key] = optimal.synthesize(perm)
            best = opt_cache[key]
            if best is not None and len(best) <= r["nct_gates"]:
                r.update(nct=[list(g) for g in best], lines=num_bits, ancilla=0,
                         nct_gates=len(best), qc=quantum_cost(best), method="optimal")
        r["verified"] = check(circ, num_bits, r) if r["lines"] <= verify_max_lines else None
        r["time"] = time.time() - t0
        results.append(r)
        log(f"--- NCT {k + 1}/{len(circuits)} | MPMCT {len(circ)} -> NCT {r['nct_gates']} "
            f"[{r['method']}] (QC {r['qc']}, ancilla {r['ancilla']}, verified {r['verified']}) "
            f"| {r['time']:.2f}s ---")
    return results


def best_index(results):
    """Trial with the fewest NCT gates, ties broken by quantum cost."""
    return min(range(len(results)), key=lambda i: (results[i]["nct_gates"], results[i]["qc"]))


def to_tfc(circ, lines, ancilla, title):
    """RevLib .tfc (t1 = NOT, t2 = CNOT, t3 = Toffoli). Line a = MSB; the dirty ancilla, if any,
    is the last line and is named 'anc'."""
    names = [chr(ord("a") + i) if i < 26 else f"x{i}" for i in range(lines - ancilla)]
    names += ["anc"] * ancilla
    out = [f"# {title}",
           "# strict NCT; line a = MSB" + ("; 'anc' = dirty ancilla (any input value, restored)"
                                           if ancilla else ""),
           f".v {','.join(names)}", f".i {','.join(names)}", f".o {','.join(names)}", "BEGIN"]
    for g in circ:
        ctrl = [names[i] for i, v in enumerate(g) if v == 1]
        out.append(f"t{len(ctrl) + 1} {','.join(ctrl + [names[g.index(3)]])}")
    out.append("END")
    return "\n".join(out) + "\n"
