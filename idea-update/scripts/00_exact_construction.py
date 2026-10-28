"""Independent exact (rational) check of GPT Pro's identity-information construction.

eta(i,j) = p_i + p_j - pi + eps*A_ij with antisymmetric, zero row/column-sum A.
Checks: candidate calibration, reference calibration, AUC equality, Brier,
and oracle audit value at budget b under (S,C), (S,C,D), symmetric and ordered information.
"""
from fractions import Fraction as F
from itertools import product

A = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]


def build(pi, d, eps):
    p = [pi - d, pi, pi + d]
    cells = []
    for i, j in product(range(3), range(3)):
        eta = p[i] + p[j] - pi + eps * A[i][j]
        assert 0 <= eta <= 1, eta
        cells.append(dict(i=i, j=j, p1=p[i], p2=p[j], eta=eta, m=F(1, 9)))
    hi = [c for c in cells if (c['p1'] + c['p2']) / 2 >= pi]
    lo = [c for c in cells if (c['p1'] + c['p2']) / 2 < pi]
    s_hi = sum(c['eta'] * c['m'] for c in hi) / sum(c['m'] for c in hi)
    s_lo = sum(c['eta'] * c['m'] for c in lo) / sum(c['m'] for c in lo)
    for c in cells:
        c['S'] = s_hi if (c['p1'] + c['p2']) / 2 >= pi else s_lo
        c['C'] = (c['p1'] + c['p2']) / 2
        c['D'] = abs(c['p1'] - c['p2'])
        c['yhat'] = c['S'] >= (s_hi + s_lo) / 2          # reference decision
        c['err'] = (1 - c['eta']) if c['yhat'] else c['eta']
    return cells, p


def calibrated(cells, key):
    groups = {}
    for c in cells:
        groups.setdefault(c[key], []).append(c)
    return all(sum(c['eta'] * c['m'] for c in g) / sum(c['m'] for c in g) == v for v, g in groups.items())


def auc(cells, key):
    pos = [(c[key], c['eta'] * c['m']) for c in cells]
    neg = [(c[key], (1 - c['eta']) * c['m']) for c in cells]
    num = sum(wp * wn * (1 if sp > sn else F(1, 2) if sp == sn else 0) for sp, wp in pos for sn, wn in neg)
    return num / (sum(w for _, w in pos) * sum(w for _, w in neg))


def brier(cells, key):
    return sum(c['m'] * (c['eta'] * (1 - c[key]) ** 2 + (1 - c['eta']) * c[key] ** 2) for c in cells)


def oracle_value(cells, keyfun, b):
    """Max expected errors captured with mass b, ranking by E[err | information]."""
    groups = {}
    for c in cells:
        groups.setdefault(keyfun(c), []).append(c)
    rates = sorted(((sum(c['err'] * c['m'] for c in g) / sum(c['m'] for c in g), sum(c['m'] for c in g))
                    for g in groups.values()), reverse=True)
    left, val = b, F(0)
    for r, m in rates:
        take = min(left, m); val += take * r; left -= take
        if left == 0: break
    return val


def report(pi, d, eps, b):
    cells, p = build(pi, d, eps)
    info = {
        'S,C': lambda c: (c['S'], c['C']),
        'S,C,D': lambda c: (c['S'], c['C'], c['D']),
        'symmetric (S, sorted P)': lambda c: (c['S'], tuple(sorted((c['p1'], c['p2'])))),
        'ordered (S, P1, P2)': lambda c: (c['S'], c['p1'], c['p2']),
    }
    print(f"\n=== pi={pi}, d={d}, eps={eps}, b={b} ===")
    print("calibrated: P1", calibrated(cells, 'p1'), " P2", calibrated(cells, 'p2'), " S", calibrated(cells, 'S'))
    for k in ('p1', 'p2', 'S'):
        print(f"AUC({k}) = {auc(cells, k)} = {float(auc(cells, k)):.6f}   Brier = {float(brier(cells, k)):.6f}")
    for name, f in info.items():
        v = oracle_value(cells, f, b)
        print(f"oracle expected errors per 10,000 at b={b}: {name:26s} {v * 10000} = {float(v * 10000):.1f}")


report(F(1, 2), F(1, 10), F(1, 10), F(1, 10))      # GPT Pro's 3x3 example
report(F(1, 25), F(1, 100), F(1, 100), F(1, 10))   # low-prevalence version: pi=.04, d=.01, b=.10
