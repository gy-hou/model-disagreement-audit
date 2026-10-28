"""Branch G (descriptive): queue-membership stability across the three v1.9 candidate reconstructions."""
import itertools, math, numpy as np, pandas as pd
from common import ar, Learner, lg, cohort, global_family_masks, OUT, INSTS, YEARS, ROOT

RB = ROOT / 'results/tables/v1_9_candidate_rebuild/local_cache'
SEEDS = (2026101901, 2026101902, 2026101903)
J = lambda a, b: len(a & b) / len(a | b)
rows = []
for inst in INSTS:
    tau = ar.TAUS[f'{inst}_2018']; _, masks = global_family_masks(inst); mk = masks['global']; d = cohort(inst, 2018)
    def feats(p, c, lvl): return np.column_stack([lg(p), lg(c.mean(1))] + ([np.ptp(c, 1)] if lvl == 2 else []))
    G = {k: Learner('additive').fit(feats(d['p_dep'], d['cand'][:, mk], lvl), d['y']) for k, lvl in (('G1', 1), ('G2', 2))}
    for yr in YEARS:
        z = cohort(inst, yr); p, ids, c = z['p_dep'], z['row_id'], z['cand'][:, mk]; b = math.ceil(.10 * len(ids))
        orig = {'Range': set(ar.order(np.ptp(c, 1), ids)[:b])}
        for k in ('G1', 'G2'):
            qk = G[k].predict(feats(p, c, 1 if k == 'G1' else 2))
            for t in ('AllErr', 'FN'):
                orig[f'{t}_{k}'] = set(ar.order(ar.target_probability(qk, p, tau, t), ids)[:b])
        reb = {}
        for s in SEEDS:
            e = np.load(RB / f'{inst}_2018__s{s}' / f'{inst}_{yr}_evaluation.npz')
            assert np.array_equal(e['row_id'], ids)
            reb[s] = {'Range': set(e['range_queue'].tolist()), **{f'{t}_{k}': set(e[f'{t}_{k}_queue'].tolist())
                      for t in ('AllErr', 'FN') for k in ('G1', 'G2')}}
        for pol in orig:
            pair = [J(reb[a][pol], reb[b_][pol]) for a, b_ in itertools.combinations(SEEDS, 2)]
            vs_orig = [J(reb[s][pol], orig[pol]) for s in SEEDS]
            rows.append(dict(inst=inst, year=yr, policy=pol, pairwise_J_mean=np.mean(pair), pairwise_J_min=np.min(pair),
                             vs_original_J_mean=np.mean(vs_orig), vs_original_J_min=np.min(vs_orig)))
df = pd.DataFrame(rows); df.to_csv(OUT / '10_queue_stability.csv', index=False)
pd.set_option('display.width', 200)
print(df.pivot_table(index=['inst', 'year'], columns='policy', values='pairwise_J_mean').round(3).to_string())
print(); print(df.pivot_table(index=['inst', 'year'], columns='policy', values='vs_original_J_mean').round(3).to_string())
