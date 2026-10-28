"""Branch E: exact check of the antisymmetric epsilon-family claims (rational arithmetic)."""
from fractions import Fraction as F
from itertools import product
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('ex', pathlib.Path(__file__).with_name('00_exact_construction.py'))
ex = importlib.util.module_from_spec(spec)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(ex)

def joint(cells, key):   # joint distribution of (key, Y)
    out = {}
    for c in cells:
        k = c[key] if isinstance(key, str) else key(c)
        a = out.setdefault(k, [F(0), F(0)]); a[0] += c['m'] * c['eta']; a[1] += c['m'] * (1 - c['eta'])
    return out

def check(pi, d, eps_grid, budgets):
    base_cells, _ = ex.build(pi, d, F(0))
    ref = {k: joint(base_cells, k) for k in ('p1', 'p2', 'S')}
    ref_sym = joint(base_cells, lambda c: tuple(sorted((c['p1'], c['p2']))))
    print(f"\n=== pi={pi}, d={d} ===")
    for eps in eps_grid:
        try: cells, _ = ex.build(pi, d, eps)
        except AssertionError: print(f"eps={eps}: infeasible (eta outside [0,1])"); continue
        inv = all(joint(cells, k) == ref[k] for k in ('p1', 'p2', 'S'))
        sym = joint(cells, lambda c: tuple(sorted((c['p1'], c['p2'])))) == ref_sym
        aucs = {k: ex.auc(cells, k) for k in ('p1', 'p2', 'S')}
        vals = []
        for b in budgets:
            vid = ex.oracle_value(cells, lambda c: (c['S'], c['p1'], c['p2']), b)
            vsym = ex.oracle_value(cells, lambda c: (c['S'], tuple(sorted((c['p1'], c['p2'])))), b)
            vals.append(f"b={b}: Vid-Vsym={vid - vsym} (b|eps|={b * abs(eps)}) {'OK' if vid - vsym == b * abs(eps) else 'DIFF'}")
        print(f"eps={str(eps):>6}: marginal (score,label) joints unchanged={inv}; symmetric joint unchanged={sym}; "
              f"AUC p1/p2/S={aucs['p1']}/{aucs['p2']}/{aucs['S']}")
        for v in vals: print("      ", v)

check(F(1, 2), F(1, 10), [F(k, 20) for k in (-6, -4, -2, -1, 0, 1, 2, 4, 6)], [F(1, 20), F(1, 10), F(1, 9), F(3, 20)])
check(F(1, 25), F(1, 100), [F(k, 100) for k in (-3, -2, -1, 0, 1, 2, 3, 4)], [F(1, 20), F(1, 10), F(1, 9), F(3, 20)])
