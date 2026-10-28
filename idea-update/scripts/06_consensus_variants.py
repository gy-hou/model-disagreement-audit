"""Branch C: consensus definitions (mean, logit-mean, median, family-balanced); matched G1_v / G2_v, additive."""
import numpy as np, pandas as pd
from scipy.special import expit
from common import ar, Learner, lg, cohort, global_family_masks, compare, OUT, INSTS, YEARS

def consensus(c, fam, v):
    if v == 'mean': return c.mean(1)
    if v == 'logitmean': return expit(lg(c).mean(1))
    if v == 'median': return np.median(c, 1)
    if v == 'family': return np.column_stack([c[:, fam == f].mean(1) for f in np.unique(fam)]).mean(1)
    raise KeyError(v)

def feats(p, c, fam, v, with_d):
    cols = [lg(p), lg(consensus(c, fam, v))]
    if with_d: cols.append(np.ptp(c, 1))
    return np.column_stack(cols)

V = ('mean', 'logitmean', 'median', 'family')
rows = []
for inst in INSTS:
    tau = ar.TAUS[f'{inst}_2018']; members, masks = global_family_masks(inst); d = cohort(inst, 2018)
    for variant in ('global', 'family'):
        mk = masks[variant]; fam = members.model_id.to_numpy()[mk]
        models = {}
        for v in V:
            models[f'G1_{v}'] = Learner('additive').fit(feats(d['p_dep'], d['cand'][:, mk], fam, v, False), d['y'])
            models[f'G2_{v}'] = Learner('additive').fit(feats(d['p_dep'], d['cand'][:, mk], fam, v, True), d['y'])
        for yr in YEARS:
            z = cohort(inst, yr); c = z['cand'][:, mk]
            q = {k: m.predict(feats(z['p_dep'], c, fam, k[3:], k.startswith('G2'))) for k, m in models.items()}
            pairs = [(f'G2_{v}', f'G1_{v}') for v in V] + [(f'G1_{v}', 'G1_mean') for v in V[1:]]
            rr, _ = compare(z, tau, c, q, pairs, 1999, ar.SEED + 61 * yr + len(inst) + (0 if variant == 'global' else 5))
            for r in rr: r.update(inst=inst, year=yr, variant=variant, families=len(np.unique(fam)))
            rows += rr
df = pd.DataFrame(rows); df.to_csv(OUT / '06_consensus_variants.csv', index=False)
pd.set_option('display.width', 250)
piv = df.pivot_table(index=['inst', 'variant', 'year', 'target'], columns='new', values='d_hits')
print(piv.to_string())
inc = df[df.new.str.startswith('G2')]
print('\nG2_v - G1_v: rows', len(inc), ' +/0/-', int((inc.d_hits > 0).sum()), int((inc.d_hits == 0).sum()), int((inc.d_hits < 0).sum()),
      ' range', inc.d_hits.min(), inc.d_hits.max(), ' meaningful', int(inc.meaningful_gain.sum()), ' lower>0', int((inc.lo_hits > 0).sum()))
g1 = df[df.new.str.startswith('G1')]
print('G1_v - G1_mean: rows', len(g1), ' range', g1.d_hits.min(), g1.d_hits.max(),
      ' |dH|>=max(5,10% ceiling):', int((g1.d_hits.abs() >= np.maximum(5, .1 * g1.ceiling_ref)).sum()), ' lower>0', int((g1.lo_hits > 0).sum()))
print(df[(df.meaningful_gain) | (df.lo_hits > 0)][['inst', 'variant', 'year', 'new', 'ref', 'target', 'hits_ref', 'hits_new', 'd_hits', 'ceiling_ref', 'lo_hits', 'hi_hits']].to_string(index=False))
