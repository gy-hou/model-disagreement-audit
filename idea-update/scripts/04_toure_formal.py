"""Branch A: Toure & Stephens per-class contribution C_k = Var(p_k)/(2 mu_k) added to G1, with bootstrap."""
import numpy as np, pandas as pd
from common import ar, Learner, lg, cohort, global_family_masks, compare, logloss, OUT, INSTS, YEARS

def feats(p, c, kind):
    mu = np.clip(c.mean(1), 1e-6, 1 - 1e-6); var = c.var(1)
    cols = [lg(p), lg(c.mean(1))]
    if kind == 'G2':  cols.append(np.ptp(c, 1))
    if kind == 'G2T': cols.append(var / (2 * mu))
    if kind == 'G2Ttot': cols.append(var / (2 * mu) + var / (2 * (1 - mu)))
    return np.column_stack(cols)

rows = []
for inst in INSTS:
    tau = ar.TAUS[f'{inst}_2018']; _, masks = global_family_masks(inst); d = cohort(inst, 2018)
    for variant in ('global', 'family'):
        mk = masks[variant]
        for learner in ('additive', 'interaction'):
            models = {k: Learner(learner).fit(feats(d['p_dep'], d['cand'][:, mk], k), d['y']) for k in ('G1', 'G2', 'G2T', 'G2Ttot')}
            primary = variant == 'global' and learner == 'additive'
            for yr in YEARS:
                z = cohort(inst, yr); c = z['cand'][:, mk]
                q = {k: m.predict(feats(z['p_dep'], c, k)) for k, m in models.items()}
                if primary:
                    r1, _ = compare(z, tau, c, q, [('G2T', 'G1')], 9999, ar.SEED + 31 * yr + len(inst), family_size=12)
                    r2, _ = compare(z, tau, c, q, [('G2Ttot', 'G1'), ('G2', 'G1')], 1999, ar.SEED + 37 * yr + len(inst))
                    rr = r1 + r2
                else:
                    rr, _ = compare(z, tau, c, q, [('G2T', 'G1'), ('G2Ttot', 'G1'), ('G2', 'G1')], 1999, ar.SEED + 41 * yr + len(inst))
                for r in rr:
                    r.update(inst=inst, year=yr, variant=variant, learner=learner,
                             logloss_new=logloss(q[r['new']], z['y']), logloss_ref=logloss(q[r['ref']], z['y']))
                rows.append(pd.DataFrame(rr))
df = pd.concat(rows); df.to_csv(OUT / '04_toure_formal.csv', index=False)
pd.set_option('display.width', 250)
cols = ['inst', 'year', 'variant', 'learner', 'new', 'target', 'T', 'hits_ref', 'hits_new', 'd_hits', 'ceiling_ref', 'lo_hits', 'hi_hits', 'meaningful_gain', 'excluded', 'interval']
print(df[(df.variant == 'global') & (df.learner == 'additive') & (df.new == 'G2T')][cols].round(1).to_string(index=False))
s = df[df.new.isin(['G2T', 'G2Ttot'])]
print('\nAll Toure rows:', len(s), ' positive/zero/negative:', int((s.d_hits > 0).sum()), int((s.d_hits == 0).sum()), int((s.d_hits < 0).sum()),
      ' range', s.d_hits.min(), s.d_hits.max(), ' meaningful gains:', int(s.meaningful_gain.sum()), ' lower>0:', int((s.lo_hits > 0).sum()))
p = df[(df.variant == 'global') & (df.learner == 'additive') & (df.new == 'G2T')]
print('Primary exclusion (ceiling>=50):', int(p.excluded.sum()), 'of', int((p.ceiling_ref >= 50).sum()))
