"""Audit discovery evaluation (7 discovery datasets), frozen protocol: idea-update/protocol_audit_discovery_v1.json.

Five-fold cross-fitting of the audit outcome head ONLY (base models, candidates and tau are upstream artifacts).
Queues are formed inside each held-out fold with pre-allocated budgets B_k; hits/targets/budgets are summed over folds.
Paired bootstrap: multinomial row weights drawn within each fold (fold sizes fixed), shared by all methods/targets.
Usage: python 14_discovery_eval.py [--smoke DATASET DRAWS OUTDIR]
"""
import json, math, sys, time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import audit_revision_v1_9 as ar  # noqa: E402
from src.audit_budget_routing_v0_6 import DATASET_CONFIGS, find_deployed_m2, resolve, load_frame  # noqa: E402

P = json.loads((ROOT / 'idea-update/protocol_audit_discovery_v1.json').read_text())
H = P['learners']['HGB']; A = P['learners']['additive_G1']
METHODS = ('G1', 'G2', 'SYM', 'ID', 'GX', 'G1add')
TARGETS = ('AllErr', 'FN')


def lg(x): return logit(np.clip(x, 1e-6, 1 - 1e-6))


def rep_features(p, c, rep):
    cols = [lg(p)[:, None], lg(c.mean(1))[:, None]]
    if rep in ('G2', 'SYM', 'ID'): cols.append(np.ptp(c, 1)[:, None])
    if rep in ('SYM', 'ID'): cols.append(lg(np.sort(c, 1)))
    if rep == 'ID': cols.append(lg(c))
    return np.hstack(cols)


class Encoder:
    """Fitted on the training fold only. <=120 levels -> native categorical; otherwise numeric parse or ordinal code."""
    def __init__(self, cols, cats): self.cols, self.cats = cols, set(cats)
    def fit(self, f):
        self.kind, self.levels = {}, {}
        for c in self.cols:
            num = pd.to_numeric(f[c], errors='coerce')
            is_cat = c in self.cats or (f[c].dtype == object and num.notna().mean() < .95)
            if not is_cat: self.kind[c] = 'num'; continue
            lv = sorted(f[c].dropna().astype(str).unique())
            if len(lv) <= 120: self.kind[c] = 'cat'
            elif num.notna().mean() >= .95: self.kind[c] = 'num'; continue
            else: self.kind[c] = 'ord'
            self.levels[c] = {v: i for i, v in enumerate(lv)}
        return self
    def transform(self, f):
        out = []
        for c in self.cols:
            if self.kind[c] == 'num': out.append(pd.to_numeric(f[c], errors='coerce').to_numpy(float))
            else: out.append(f[c].astype(str).map(self.levels[c]).astype(float).to_numpy())
        return np.column_stack(out)
    def catmask(self, prefix): return np.array([False] * prefix + [self.kind[c] == 'cat' for c in self.cols])


def fit_predict(method, Xtr, ytr, Xte, catmask=None):
    if method == 'G1add':
        sp = SplineTransformer(n_knots=A['n_knots'], degree=A['degree'], knots='quantile', include_bias=False); sc = StandardScaler()
        m = LogisticRegression(C=A['C'], solver='lbfgs', max_iter=A['max_iter'], tol=A['tol'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning); m.fit(sc.fit_transform(sp.fit_transform(Xtr)), ytr)
        return m.predict_proba(sc.transform(sp.transform(Xte)))[:, 1]
    m = HistGradientBoostingClassifier(max_iter=H['max_iter'], max_leaf_nodes=H['max_leaf_nodes'], min_samples_leaf=H['min_samples_leaf'],
                                       learning_rate=H['learning_rate'], l2_regularization=H['l2_regularization'], max_bins=H['max_bins'],
                                       early_stopping=False, random_state=H['random_state'],
                                       categorical_features=catmask if catmask is not None and catmask.any() else None)
    m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]


def allocate(n_k, B):
    raw = np.array(n_k) * B / sum(n_k); base = np.floor(raw).astype(int); r = B - base.sum()
    order = sorted(range(len(n_k)), key=lambda k: (-(raw[k] - base[k]), k)); base[order[:r]] += 1
    return base


def boot(order_by_queue, folds_idx, Bk, z_by_target, loss_by_method, draws, seed, chunk=250):
    """Returns per-queue resampled hits, per-target resampled totals, per-method resampled mean loss."""
    rng = np.random.default_rng(seed)
    hits = {q: np.zeros(draws) for q in order_by_queue}; tot = {t: np.zeros(draws) for t in z_by_target}
    loss = {m: np.zeros(draws) for m in loss_by_method}; N = sum(len(ix) for ix in folds_idx)
    done = 0
    while done < draws:
        m_ = min(chunk, draws - done)
        for k, ix in enumerate(folds_idx):
            n = len(ix)
            draw = rng.integers(0, n, size=(m_, n)) + (np.arange(m_) * n)[:, None]
            w = np.bincount(draw.ravel(), minlength=m_ * n).reshape(m_, n).astype(float)   # multinomial(n, 1/n) per row
            for t, z in z_by_target.items(): tot[t][done:done + m_] += w @ z[ix]
            for mth, l in loss_by_method.items(): loss[mth][done:done + m_] += w @ l[ix]
            for q, ords in order_by_queue.items():
                o = ords[k]; b = Bk[k]; W = min(n, b + int(8 * math.sqrt(max(b, 1))) + 50)
                wo = w[:, o[:W]]; cum = np.cumsum(wo, 1)
                short = cum[:, -1] < b
                sel = np.clip(b - (cum - wo), 0, wo); h = sel @ z_by_target[q[1]][ix][o[:W]]
                if short.any():                                    # rare: window too small for some draws
                    wf = w[short][:, o]; cf = np.cumsum(wf, 1)
                    h[short] = np.clip(b - (cf - wf), 0, wf) @ z_by_target[q[1]][ix][o]
                hits[q][done:done + m_] += h
        done += m_
    for mth in loss: loss[mth] /= N
    return hits, tot, loss


def run_dataset(name, draws, seed):
    cfg = DATASET_CONFIGS[name]; t0 = time.time(); status = []
    z = np.load(ar.OLD / f'{name}.npz'); ids, y, p, cand_all = z['row_id'], z['y'].astype(int), z['p_dep'], z['cand']
    mem = pd.read_csv(ar.OLD / f'{name}_members.csv')
    keep = mem.validation_auc.to_numpy() + P['candidates']['tolerance'] >= mem.validation_auc.max() - P['candidates']['auc_gap']
    cand = cand_all[:, keep]; M = cand.shape[1]
    tau = float(load_frame(cfg.validation_path)[cfg.target].mean())
    test = load_frame(cfg.test_path).set_index(cfg.id_column).loc[ids]
    if not (test[cfg.target].to_numpy() == y).all(): raise ValueError(f'{name}: label mismatch after ID alignment')
    dep = find_deployed_m2(json.loads(resolve(cfg.voters_path).read_text())['selected_voters'])
    art = joblib.load(resolve(Path(dep['artifact_path'])))
    raw_cols = list(art.get('raw_feature_columns') or art.get('feature_columns'))
    pred = p >= tau
    zt = {'AllErr': (pred != (y == 1)).astype(float), 'FN': ((~pred) & (y == 1)).astype(float)}
    methods = [m for m in METHODS if not (M < 2 and m in ('G2', 'SYM', 'ID'))]
    if M < 2: status.append(dict(dataset=name, issue='M_global<2: G2/SYM/ID not defined'))
    N = len(y); B = math.ceil(P['budget']['fraction'] * N)
    skf = StratifiedKFold(P['folds']['k'], shuffle=True, random_state=P['folds']['random_state'])
    folds = [va for _, va in skf.split(np.zeros(N), y)]
    Bk = allocate([len(f) for f in folds], B)
    q = {m: np.full(N, np.nan) for m in methods}
    for k, (tr, va) in enumerate(skf.split(np.zeros(N), y)):
        for m in methods:
            try:
                if m == 'GX':
                    enc = Encoder(raw_cols, cfg.categorical_features).fit(test.iloc[tr])
                    Xtr = np.hstack([rep_features(p[tr], cand[tr], 'G1'), enc.transform(test.iloc[tr])])
                    Xva = np.hstack([rep_features(p[va], cand[va], 'G1'), enc.transform(test.iloc[va])])
                    q[m][va] = fit_predict(m, Xtr, y[tr], Xva, enc.catmask(2))
                else:
                    rep = 'G1' if m == 'G1add' else m
                    q[m][va] = fit_predict(m, rep_features(p[tr], cand[tr], rep), y[tr], rep_features(p[va], cand[va], rep))
            except Exception as e:
                status.append(dict(dataset=name, issue=f'{m} fold {k} failed: {type(e).__name__}: {e}'))
    methods = [m for m in methods if np.isfinite(q[m]).all()]
    orders, obs = {}, []
    for m in methods:
        for t in TARGETS:
            risk = ar.target_probability(q[m], p, tau, t); ords = []
            for k, ix in enumerate(folds):
                o = np.lexsort((ids[ix], -np.round(risk[ix], 12))); ords.append(o)
            orders[(m, t)] = ords
            h = sum(zt[t][ix][o[:Bk[k]]].sum() for k, (ix, o) in enumerate(zip(folds, ords)))
            Tt = zt[t].sum(); cap_bound = sum(min(Bk[k], zt[t][ix].sum()) for k, ix in enumerate(folds))
            obs.append(dict(dataset=name, method=m, target=t, N=N, B=B, T=int(Tt), hits=int(h), capture=h / Tt if Tt else np.nan,
                            precision=h / B, fold_capacity=int(cap_bound), M_global=M))
    ll = {m: -(y * np.log(np.clip(q[m], 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - q[m], 1e-12, 1))) for m in methods}
    bh, bt, bl = boot(orders, folds, Bk, zt, ll, draws, seed)
    obs_df = pd.DataFrame(obs).set_index(['method', 'target'])
    rows = []
    for c in P['contrasts']['primary'] + P['contrasts']['secondary']:
        new, ref, t = c['new'], c['ref'], c['target']
        if new not in methods or ref not in methods:
            rows.append(dict(dataset=name, contrast=f'{new}-{ref}', target=t, role=c['role'], defined=False)); continue
        Hn, Hr = obs_df.loc[(new, t), 'hits'], obs_df.loc[(ref, t), 'hits']; d = Hn - Hr
        R = obs_df.loc[(ref, t), 'fold_capacity'] - Hr; dd = bh[(new, t)] - bh[(ref, t)]
        alpha = P['bootstrap']['alpha'] / (P['contrasts']['primary_family_size'] if c['role'] == 'primary' else 1)
        lo, hi = np.quantile(dd, [alpha / 2, 1 - alpha / 2]); plo, phi = np.quantile(dd, [.025, .975])
        thr = max(5, .10 * R)
        rows.append(dict(dataset=name, contrast=f'{new}-{ref}', target=t, role=c['role'], defined=True,
                         N=N, B=B, T=obs_df.loc[(ref, t), 'T'], M_global=M, hits_ref=int(Hr), hits_new=int(Hn), d_hits=int(d),
                         d_hits_per10k=1e4 * d / N, d_hits_per_B=d / B, d_precision_pp=100 * d / B,
                         remaining_capacity_ref=int(R), lo_pointwise=plo, hi_pointwise=phi,
                         lo_adjusted=lo if c['role'] == 'primary' else np.nan, hi_adjusted=hi if c['role'] == 'primary' else np.nan,
                         meaningful_gain=bool(d >= thr and lo > 0), meaningful_loss=bool(d <= -thr and hi < 0),
                         excluded=bool(R >= 50 and hi < .10 * R), materiality_threshold=thr,
                         d_logloss=float(ll[new].mean() - ll[ref].mean()),
                         d_logloss_lo=float(np.quantile(bl[new] - bl[ref], .025)), d_logloss_hi=float(np.quantile(bl[new] - bl[ref], .975))))
    lrows = [dict(dataset=name, method=m, oof_logloss=float(ll[m].mean())) for m in methods]
    return pd.DataFrame(obs), pd.DataFrame(rows), pd.DataFrame(lrows), pd.DataFrame(status), dict(dataset=name, seconds=round(time.time() - t0, 1), Bk=Bk.tolist(), M_global=M)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--smoke':
        datasets, draws, out = [sys.argv[2]], int(sys.argv[3]), Path(sys.argv[4])
    else:
        datasets, draws, out = P['datasets'], P['bootstrap']['draws'], ROOT / 'idea-update/results'
    out.mkdir(parents=True, exist_ok=True); parts = {k: [] for k in ('methods', 'contrasts', 'logloss', 'status')}; info = []
    for i, name in enumerate(datasets):
        try:
            o, c, l, s, inf = run_dataset(name, draws, P['bootstrap']['seed'] + i)
            parts['methods'].append(o); parts['contrasts'].append(c); parts['logloss'].append(l); parts['status'].append(s); info.append(inf)
            print(f'done {name} in {inf["seconds"]}s (M_global={inf["M_global"]}, B_k={inf["Bk"]})', flush=True)
        except Exception as e:
            parts['status'].append(pd.DataFrame([dict(dataset=name, issue=f'dataset failed: {type(e).__name__}: {e}')])); print('FAILED', name, e, flush=True)
    prefix = 'smoke_' if out != ROOT / 'idea-update/results' else '14_'
    for k, v in parts.items():
        (pd.concat(v) if v else pd.DataFrame()).to_csv(out / f'{prefix}{k}.csv', index=False)
    (out / f'{prefix}run_info.json').write_text(json.dumps(dict(draws=draws, datasets=datasets, info=info), indent=1))


if __name__ == '__main__':
    main()
