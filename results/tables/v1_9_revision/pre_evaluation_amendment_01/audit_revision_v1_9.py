"""Frozen target-aligned audit development and evaluation for v1.9.

Historical evaluation is explicitly exploratory. No raw loan data are published.
The independent entry point refuses missing exposure/isolation evidence.
Run freeze -> develop -> evaluate-historical; use --help for individual stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logit, xlogy
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
import warnings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/tables/v1_9_revision'
OLD = ROOT / 'results/tables/v1_4_mechanism/local_cache'
BASES = ('freddie_mac_2018', 'fannie_mae_2018')
TAUS = {'freddie_mac_2018': .0391, 'fannie_mae_2018': .0414}
SEED = 20261019
TARGETS = ('AllErr', 'FN', 'FP', 'Pos', 'EM_fixed', 'EM_variant')
POLICY_ORDER = ('ThresholdProb', 'RiskTop', 'AboveTauLow', 'BelowTauTop', 'G0', 'G1', 'Range', 'G2')
POLICY_SETS = {'P_score': POLICY_ORDER[:5], 'P_base': POLICY_ORDER[:6], 'P_full': POLICY_ORDER}
CONTRASTS = (('G2', 'G1'), ('P_full', 'P_base'), ('G1', 'G0'),
             ('P_base', 'P_score'), ('P_full', 'ThresholdProb'),
             ('P_full', 'Range'), ('Range', 'ThresholdProb'))
sys.modules.setdefault('audit_revision_v1_9', sys.modules[__name__])


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def validate_cohort(z):
    required = ('row_id', 'p_dep', 'cand', 'y')
    if any(k not in z for k in required):
        raise ValueError('Cohort needs row_id, p_dep, cand, y')
    n = len(z['row_id'])
    if not n or z['cand'].ndim != 2 or z['cand'].shape[1] < 1:
        raise ValueError('Empty cohort or candidate set')
    if any(len(z[k]) != n for k in required):
        raise ValueError('Unaligned cohort arrays')
    if len(np.unique(z['row_id'])) != n:
        raise ValueError('Real audit queues require unique row IDs')
    for key in ('p_dep', 'cand'):
        if not np.isfinite(z[key]).all() or np.any((z[key] < 0) | (z[key] > 1)):
            raise ValueError('Nonfinite or out-of-range probability')
    if not np.isin(z['y'], (0, 1)).all():
        raise ValueError('Outcome must be binary and observed under the contract')


def order(score, ids):
    score = np.asarray(score)
    if len(score) != len(ids) or not np.isfinite(score).all():
        raise ValueError('Ranking score must be finite and aligned')
    return np.lexsort((ids, -score))


def queue(score, ids, b):
    if not 0 < b <= 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid budget or repeated real case ID')
    k = int(math.ceil(b * len(ids)))
    return order(score, ids)[:k]


def target_vectors(z, tau, cand):
    y = z['y'].astype(bool)
    pred = z['p_dep'] >= tau
    err = pred != y
    fixed_flip = (z['cand'] >= tau).any(1) & (z['cand'] < tau).any(1)
    variant_flip = (cand >= tau).any(1) & (cand < tau).any(1)
    return {'AllErr': err, 'FN': (~pred) & y, 'FP': pred & (~y), 'Pos': y,
            'EM_fixed': err & fixed_flip, 'EM_variant': err & variant_flip}


def target_probability(q, p, tau, target, fixed_flip=None, variant_flip=None):
    pred = p >= tau
    if target == 'Pos': return q
    if target == 'FN': return np.where(pred, 0., q)
    if target == 'FP': return np.where(pred, 1. - q, 0.)
    err = np.where(pred, 1. - q, q)
    if target == 'AllErr': return err
    if target == 'EM_fixed': return err * fixed_flip
    if target == 'EM_variant': return err * variant_flip
    raise KeyError(target)


def raw_features(p, cand, level):
    # Unique consensus contract: logit of mean probabilities, NOT mean logits.
    cols = [logit(np.clip(p, 1e-6, 1 - 1e-6))]
    if level >= 1:
        cols.append(logit(np.clip(cand.mean(1), 1e-6, 1 - 1e-6)))
    if level >= 2:
        cols.append(np.ptp(cand, axis=1))
    return np.column_stack(cols)


class AuditRiskModel:
    """Same family, capacity parameters and folds for G1 and G2."""
    def __init__(self, level, learner):
        self.level = level
        self.learner = learner

    def fit(self, p, cand, y):
        x = raw_features(p, cand, self.level)
        self.constant = float(np.mean(y)) if len(np.unique(y)) == 1 else None
        if self.constant is not None: return self
        if self.learner == 'additive':
            self.spline = SplineTransformer(n_knots=5, degree=3, knots='quantile', include_bias=False)
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(self.spline.fit_transform(x))
            self.model = LogisticRegression(C=1., solver='lbfgs', max_iter=4000, tol=1e-7)
        elif self.learner == 'interaction':
            self.model = HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=7, min_samples_leaf=100,
                learning_rate=.05, l2_regularization=1., max_bins=128,
                early_stopping=False, random_state=SEED)
        else: raise ValueError('Unknown learner')
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.model.fit(x, y)
        return self

    def predict(self, p, cand):
        if self.constant is not None: return np.full(len(p), self.constant)
        x = raw_features(p, cand, self.level)
        if self.learner == 'additive':
            x = self.scaler.transform(self.spline.transform(x))
        return self.model.predict_proba(x)[:, 1]


AuditRiskModel.__module__ = 'audit_revision_v1_9'


def candidate_variants(base):
    members = pd.read_csv(OLD / f'{base}_members.csv')
    full = pd.read_csv(ROOT / f'results/tables/v0_7/candidate_pool_metrics_{base}.csv')
    best = float(full.validation_auc.max())
    global_ids = set(full.loc[full.validation_auc + 1e-6 >= best - .010, 'candidate_id'])
    if not global_ids.issubset(set(members.candidate_id)):
        raise ValueError('Family-retained cache misses globally eligible candidates')
    return members, {'global': np.array([x in global_ids for x in members.candidate_id]),
                     'family': np.ones(len(members), dtype=bool)}


def simple_scores(p, cand, ids, tau):
    mean = np.clip(cand.mean(1), 1e-6, 1-1e-6)
    prob = np.clip(cand, 1e-6, 1-1e-6)
    votes = (cand >= tau).mean(1)
    entropy = -(xlogy(votes, votes) + xlogy(1-votes, 1-votes))
    kl = (prob * np.log(prob / mean[:, None]) +
          (1-prob) * np.log((1-prob)/(1-mean[:, None]))).mean(1)
    scores = {'ThresholdProb': -abs(p-tau), 'RiskTop': p,
              'AboveTauLow': np.where(p >= tau, -p, p-2),
              'BelowTauTop': np.where(p < tau, p, -p-2),
              'VoteEntropy': entropy, 'MeanBernoulliKL': kl}
    if cand.shape[1] >= 2 and np.ptp(np.ptp(cand, axis=1)) > 0:
        scores['Range'] = np.ptp(cand, axis=1)
    # A reproducible one-off random queue; expectation is also reported separately.
    rng = np.random.default_rng(SEED)
    canonical = np.argsort(ids, kind='stable')
    random = np.empty(len(ids)); random[canonical] = rng.random(len(ids))
    scores['Random'] = random
    return scores


def all_orders(z, cand, tau, predictions):
    p, ids = z['p_dep'], z['row_id']
    fixed_flip = (z['cand'] >= tau).any(1) & (z['cand'] < tau).any(1)
    variant_flip = (cand >= tau).any(1) & (cand < tau).any(1)
    common = simple_scores(p, cand, ids, tau)
    ans = {}
    for target in TARGETS:
        d = {key: order(value, ids) for key, value in common.items()}
        for key, q in predictions.items():
            d[key] = order(target_probability(q, p, tau, target, fixed_flip, variant_flip), ids)
        ans[target] = d
    return ans


def choose_policies(orders, targets, budgets):
    result = {}
    for target, t in targets.items():
        for b in budgets:
            k = int(math.ceil(b * len(t)))
            for group, names in POLICY_SETS.items():
                available = [name for name in names if name in orders[target]]
                # Empty target has no estimable argmax: deterministic score fallback.
                chosen = (max(available, key=lambda name: (int(t[orders[target][name][:k]].sum()),
                          -POLICY_ORDER.index(name))) if t.sum() else 'ThresholdProb')
                result[f'{target}|{b}|{group}'] = chosen
    return result


def fixed_bin_edges(p, tau):
    edges = []
    for side in (p < tau, p >= tau):
        edges.append(np.unique(np.quantile(p[side], np.linspace(0, 1, 26)))[1:-1].tolist()
                     if side.any() else [])
    return edges


def assign_bins(p, tau, edges):
    ans = np.zeros(len(p), dtype=int)
    for side in (False, True):
        ix = (p >= tau) == side
        ans[ix] = int(side) * 25 + np.searchsorted(edges[int(side)], p[ix], side='right')
    return ans


def weighted_selected(order_, weights, sizes):
    """Bootstrap occurrences are weights, never repeat a real-world audit ID.

    Returns selected ORIGINAL indices and occurrence weights for each budget.
    Partial boundary multiplicity exactly fills the empirical-sample budget.
    """
    maximum = max(sizes)
    stop = min(len(order_), int(maximum * 1.5) + 50)
    sub = order_[:stop]
    w = weights[sub]
    cw = np.cumsum(w)
    if cw[-1] < maximum:
        sub = order_; w = weights[sub]; cw = np.cumsum(w)
    if cw[-1] < maximum: raise ValueError('Insufficient bootstrap mass for budget')
    out = []
    for size in sizes:
        j = int(np.searchsorted(cw, size, side='left'))
        selected = w[:j+1].astype(float, copy=True)
        selected[-1] = size - (cw[j-1] if j else 0)
        out.append((sub[:j+1], selected))
    return out


def _interval(values, alpha=.05):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite): return np.nan, np.nan, 0
    lo, hi = np.quantile(finite, [alpha/2, 1-alpha/2])
    return float(lo), float(hi), len(finite)


def evaluation_tables(z, cand, tau, predictions, selection, edges, budgets, boots, seed, scope,
                      simultaneous_family=8,policy_groups=True,contrasts=CONTRASTS):
    """Paired evaluation conditional on fixed fitted models and selected policy.

    All strategies/targets/budgets share multinomial row draws. Score bins are
    FIXED development edges, avoiding selection of evaluation-bin resolution.
    """
    validate_cohort(z)
    n = len(z['y']); sizes = [int(math.ceil(b*n)) for b in budgets]
    tt = target_vectors(z, tau, cand)
    oo = all_orders(z, cand, tau, predictions)
    bins = assign_bins(z['p_dep'], tau, edges)
    rows, differences, decompositions = [], [], []
    points = {}; samples = {}; pos_samples = {}
    # Deduplicate repeated queue orderings, including target-specific G queues.
    unique = {}; index = {}; lookup = {}
    for target, strategies in oo.items():
        for strategy, ranked in strategies.items():
            key = hashlib.sha256(ranked.tobytes()).hexdigest()
            if key not in unique: unique[key] = ranked
            lookup[target, strategy] = key
    for target in TARGETS:
        for bi, b in enumerate(budgets):
            for strategy in list(oo[target]) + (list(POLICY_SETS) if policy_groups else []):
                selected = selection[f'{target}|{b}|{strategy}'] if strategy in POLICY_SETS else strategy
                index[target, bi, strategy] = lookup[target, selected]
                samples[target, bi, strategy] = np.empty(boots)
                pos_samples[target, bi, strategy] = np.empty(boots)
    target_matrix=np.column_stack([tt[target] for target in TARGETS]).astype(float)
    target_column={target:i for i,target in enumerate(TARGETS)}
    qloss = {key: -(z['y'] * np.log(np.clip(q,1e-12,1-1e-12)) +
                        (1-z['y']) * np.log1p(-np.clip(q,1e-12,1-1e-12)))
             for key, q in predictions.items()}
    loss_samples = {key: np.empty(boots) for key in predictions}

    def draw(weights, replicate=None):
        totals=np.array([float(weights[t].sum()) for t in tt.values()])
        bn = np.bincount(bins, weights=weights, minlength=50)
        bt=np.column_stack([np.bincount(bins,weights=weights*t,minlength=50) for t in tt.values()])
        rates=np.divide(bt,bn[:,None],out=np.zeros((50,len(TARGETS))),where=bn[:,None]>0)
        chosen = {key: weighted_selected(ranked, weights, sizes) for key, ranked in unique.items()}
        cache = {}
        for (target, bi, strategy), qkey in index.items():
            evalkey = (bi, qkey)
            if evalkey not in cache:
                ix, sw = chosen[qkey][bi]
                hits=sw@target_matrix[ix]
                position_hits=sw@rates[bins[ix]]
                cache[evalkey]=(hits,np.divide(hits,totals,out=np.full(len(TARGETS),np.nan),where=totals>0),
                               np.divide(position_hits,totals,out=np.full(len(TARGETS),np.nan),where=totals>0))
            col=target_column[target]
            hits,capture,position=(v[col] for v in cache[evalkey])
            if replicate is None: points[target, bi, strategy] = (hits,capture,position)
            else:
                samples[target,bi,strategy][replicate] = capture
                pos_samples[target,bi,strategy][replicate] = position
        if replicate is not None:
            for key, loss in qloss.items(): loss_samples[key][replicate] = float(np.dot(weights,loss)/n)

    draw(np.ones(n, dtype=int))
    rng = np.random.default_rng(seed)
    for r in range(boots):
        weights = np.bincount(rng.integers(0,n,n), minlength=n)
        draw(weights, r)
    for (target,bi,strategy), (hits,capture,position) in points.items():
        b=budgets[bi]; k=sizes[bi]; den=int(tt[target].sum())
        lo,hi,valid=_interval(samples[target,bi,strategy])
        rows.append(dict(target=target,budget=b,N=n,B=k,effective_budget=k/n,target_n=den,
                         strategy=strategy,selected_policy=selection[f'{target}|{b}|{strategy}'] if strategy in POLICY_SETS else strategy,
                         hits=int(hits),capture=capture,yield_=hits/k,hits_per_1000_reviews=1000*hits/k,
                         lower=lo,upper=hi,bootstrap_valid=valid,bootstrap_empty_target=boots-valid,
                         expected_random_capture=k/n if den else np.nan,bootstrap=boots,
                         status='ok' if den else 'undefined_empty_target'))
    for target in TARGETS:
        for bi,b in enumerate(budgets):
            for a,s in contrasts:
                if (target,bi,a) not in points or (target,bi,s) not in points: continue
                ha,ca,pa=points[target,bi,a]; hs,cs,ps=points[target,bi,s]
                values=samples[target,bi,a]-samples[target,bi,s]
                lo,hi,valid=_interval(values)
                primary=target in ('AllErr','FN') and b==.1 and (a,s) in CONTRASTS[:2]
                alo,ahi,_=_interval(values,.05/simultaneous_family) if primary else (np.nan,np.nan,0)
                differences.append(dict(target=target,budget=b,N=n,B=sizes[bi],target_n=int(tt[target].sum()),
                    contrast=f'{a}-{s}',hits_difference=int(ha-hs),capture_difference=ca-cs,
                    yield_difference=(ha-hs)/sizes[bi],lower=lo,upper=hi,
                    family_adjusted_lower=alo,family_adjusted_upper=ahi,
                    family_size=simultaneous_family if primary else 0,
                    comparison_role='primary' if primary else 'secondary_diagnostic',
                    bootstrap_valid=valid,bootstrap_empty_target=boots-valid))
                if target in ('AllErr','FN') and b==.1 and (a,s) in (CONTRASTS[0],CONTRASTS[6]):
                    pv=pos_samples[target,bi,a]-pos_samples[target,bi,s]
                    wv=values-pv
                    pl,pu,_=_interval(pv); wl,wu,_=_interval(wv)
                    decompositions.append(dict(target=target,budget=b,contrast=f'{a}-{s}',
                        capture_difference=ca-cs,position_difference=pa-ps,
                        within_difference=(ca-pa)-(cs-ps),lower=lo,upper=hi,
                        position_lower=pl,position_upper=pu,within_lower=wl,within_upper=wu,
                        identity_residual=(ca-cs)-(pa-ps)-((ca-pa)-(cs-ps)),
                        bin_definition='50 maximum; fixed development quantiles separately on each threshold side'))
    predictive=[]
    for a,s in (('G2','G1'),('G1','G0')):
        if a not in loss_samples or s not in loss_samples: continue
        diff=loss_samples[a]-loss_samples[s];lo,hi,_=_interval(diff)
        predictive.append(dict(contrast=f'{a}-{s}',metric='log_loss',difference=float(np.mean(qloss[a]-qloss[s])),
                               lower=lo,upper=hi,interpretation='negative favors added information model'))
    # Membership changes are distinct from capture differences.
    swaps=[]
    for target in ('AllErr','FN'):
        if 'G2' not in oo[target] or 'G1' not in oo[target]: continue
        for bi,b in enumerate(budgets):
            a=oo[target]['G2'][:sizes[bi]];s=oo[target]['G1'][:sizes[bi]]
            enter=np.setdiff1d(a,s);leave=np.setdiff1d(s,a)
            swaps.append(dict(target=target,budget=b,entering_n=len(enter),leaving_n=len(leave),
                              entering_hits=int(tt[target][enter].sum()),leaving_hits=int(tt[target][leave].sum()),
                              queue_jaccard=len(np.intersect1d(a,s))/len(np.union1d(a,s))))
    return {'performance':rows,'paired_contrasts':differences,'decomposition':decompositions,
            'predictive':predictive,'budget_boundary':swaps}


def freeze():
    if (OUT/'analysis_plan.json').exists(): raise RuntimeError('Plan already frozen; do not overwrite')
    inputs={}
    paths=[ROOT/'src/audit_revision_v1_9.py',ROOT/'tests/test_audit_revision_v1_9.py',
           ROOT/'docs/revisions/kbs_v1_9/independent_evaluation_gate.json',
           ROOT/'docs/revisions/kbs_v1_9/DATA_AUDIT.md']
    for base in BASES:
        paths += [OLD/f'{base}.npz',OLD/f'{base}_members.csv',
                  ROOT/f'results/tables/v0_7/candidate_pool_metrics_{base}.csv']
        members,_=candidate_variants(base)
        paths += [ROOT/p for p in members.artifact_path]
        for year in (2021,2022,2023):
            name=base.replace('2018',str(year))
            paths += [OLD/f'{name}.npz',OLD/f'{name}_members.csv']
    paths += [OUT/'local_cache'/f'freddie_mac_{year}_strict_label_audit.parquet' for year in (2018,2021,2022,2023)]
    for path in paths: inputs[str(path.relative_to(ROOT))]=digest(path)
    plan=dict(version='v1.9-fixed-system-1',frozen_at_utc=now(),seed=SEED,
        status='exploratory_retrospective_only; no verified independent evaluation source',
        endpoint='historical recorded loan-age-based ever_60dpd_24m; not verified fixed-calendar follow-up; legacy targets preserved',
        exposure={'Freddie_2019_2020':'user confirmed previously used; not independent D_eval',
                  'Fannie_unsampled':'source labels used in historical prevalence/stratified sampling; not untouched'},
        development={base:'historical 2018 test, now explicitly audit development; old source artifacts preserved' for base in BASES},
        evaluation=[base.replace('2018',str(y)) for base in BASES for y in (2021,2022,2023)],
        historical_2018_baselines='Original family candidate set; descriptive Threshold/Range/score references, 1999 paired queue-rebuilding draws, target counts/hits/yield and Range-Threshold decomposition. Same cases now C_audit; never independent evaluation.',
        independent_evaluation='blocked until separately frozen cohort scope and verified exposure/isolation contract',
        data_roles='Original train/validation are A_base and eligibility/threshold sources; no new B_cal; historical test is C_audit. Original processing is not retrospectively a nested A/B/C design.',
        candidate_sets={'primary':'original 26-model grid global validation AUC floor .010, tolerance 1e-6',
                        'sensitivity':'original family-conditioned .010 set',
                        'no_new_brier_gate':True,'M_lt_2':'Range unavailable; no meaningful disagreement queue'},
        consensus='logit(clip(mean candidate probabilities,1e-6,1-1e-6)); never mean candidate logits',
        learners={'primary':{'name':'additive','features':'separate cubic splines, 5 quantile knots; StandardScaler; logistic C=1, max_iter=4000, tol=1e-7'},
                  'sensitivity':{'name':'interaction','features':'raw deployed logit; consensus logit; range when available',
                                 'algorithm':'HGBT leaves=7, iterations=100, min_leaf=100, rate=.05, l2=1, bins=128; no early stopping'}},
        folds=5,fold_seed=SEED,hyperparameter_search='none; identical fixed parameters and folds across information levels',
        policies={k:list(v) for k,v in POLICY_SETS.items()},
        policy_selection='maximize pooled 5-fold OOF hits per target/budget; ties use fixed POLICY_ORDER; empty target -> ThresholdProb',
        primary_targets=['AllErr','FN'],diagnostic_targets=['FP','Pos','EM_fixed','EM_variant'],
        budgets=[.05,.10,.20],primary_budget=.10,primary_contrasts=['G2-G1','P_full-P_base'],
        historical_primary_family_size=24,independent_future_two_cohort_primary_family_size=8,
        interval='paired multinomial row bootstrap, rebuild weighted top-B; percentile pointwise 95%; primary Bonferroni percentile intervals. Approximate conditional coverage, not exact null tests.',
        bootstrap_historical=1999,bootstrap_independent=9999,
        strict_endpoint_sensitivity=dict(cohorts=[f'freddie_mac_{y}' for y in (2021,2022,2023)],budget=.10,
            bootstrap=999,endpoint='strict_calendar_age0_24_60plus_or_reo',
            population='strict_known==True; eligible population and outcome BOTH change',
            model_and_policy='unchanged legacy-2018 development models and selected policy; no endpoint refit',
            interpretation='exploratory endpoint-plus-eligibility sensitivity, not a causal label correction or independent validation'),
        family='Historical primary: 6 cohort cells x 2 targets x 2 contrasts, global/additive only. Sensitivities descriptive; no confirmatory p-values.',
        scope='Conditional on fitted credit/audit models, chosen policy, threshold and candidate set. No training/selection uncertainty or macro-environment sampling; borrower cluster IDs unavailable.',
        decomposition='50 maximum fixed C_audit score bins, separately threshold sides; signed position/within components; not causal',
        empty_targets='capture and contrast undefined, hits=0, report excluded bootstrap draws',
        failures='raise for alignment, duplicate ID, probability, convergence or input hash errors; preserve failure record, never silently change models',
        retraining='candidate reconstruction/retraining not performed; fixed-system evaluation only',
        input_hashes=inputs,
        environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                     'scipy':scipy.__version__,'sklearn':sklearn.__version__})
    write_json(OUT/'analysis_plan.json',plan)
    write_json(OUT/'PLAN.sha256.json',{'analysis_plan.json':digest(OUT/'analysis_plan.json')})
    print('Plan frozen before fitting/evaluation; source and input hashes recorded.',flush=True)


def checked_plan():
    plan=read_json(OUT/'analysis_plan.json')
    receipt=read_json(OUT/'PLAN.sha256.json')
    if digest(OUT/'analysis_plan.json') != receipt['analysis_plan.json']: raise ValueError('Frozen plan changed')
    for path,expected in plan['input_hashes'].items():
        if digest(ROOT/path)!=expected: raise ValueError(f'Frozen input changed: {path}')
    return plan


def develop():
    plan=checked_plan()
    if (OUT/'FROZEN_DEVELOPMENT.json').exists(): raise RuntimeError('Development already frozen')
    registry=[]; metrics=[]; choices=[]; artifacts={}; failures=[]
    local=OUT/'local_cache';local.mkdir(parents=True,exist_ok=True)
    for base in BASES:
        z=load_npz(OLD/f'{base}.npz');validate_cohort(z)
        members,variants=candidate_variants(base)
        tau=TAUS[base]
        # Reuse exactly the same label-stratified partitions across all variants/models.
        if min(np.bincount(z['y'].astype(int),minlength=2))<5: raise ValueError('Insufficient outcome support for five folds')
        folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(z['p_dep'],z['y']))
        fold_id=np.empty(len(z['y']),dtype=int)
        for f,(_,valid) in enumerate(folds): fold_id[valid]=f
        np.savez_compressed(local/f'{base}_development_ids.npz',row_id=z['row_id'],fold_id=fold_id)
        artifacts[str((local/f'{base}_development_ids.npz').relative_to(ROOT))]=digest(local/f'{base}_development_ids.npz')
        for variant,keep in variants.items():
            cand=z['cand'][:,keep]
            for _,r in members.iterrows():
                j=int(r.name)
                registry.append(dict(base=base,variant=variant,candidate_id=r.candidate_id,family=r.axis,
                                     validation_auc=r.validation_auc,validation_brier=r.validation_brier,
                                     selected=bool(keep[j]),artifact_path=r.artifact_path,
                                     artifact_sha256=digest(ROOT/r.artifact_path)))
            for learner in ('additive','interaction'):
                key=f'{base}__{variant}__{learner}'
                pred={f'G{level}':np.empty(len(z['y'])) for level in range(3)}
                models={}
                for level in range(3):
                    for train,valid in folds:
                        model=AuditRiskModel(level,learner).fit(z['p_dep'][train],cand[train],z['y'][train])
                        pred[f'G{level}'][valid]=model.predict(z['p_dep'][valid],cand[valid])
                    models[f'G{level}']=AuditRiskModel(level,learner).fit(z['p_dep'],cand,z['y'])
                orders=all_orders(z,cand,tau,pred);tt=target_vectors(z,tau,cand)
                selection=choose_policies(orders,tt,plan['budgets'])
                record=dict(base=base,variant=variant,learner=learner,tau=tau,
                            candidate_ids=members.loc[keep,'candidate_id'].tolist(),
                            all_candidate_ids=members.candidate_id.tolist(),keep=keep.tolist(),
                            selection=selection,bin_edges=fixed_bin_edges(z['p_dep'],tau),
                            model_ids=list(models),development_role='C_audit historical 2018 test',
                            plan_sha256=digest(OUT/'analysis_plan.json'),frozen_at_utc=now())
                write_json(local/f'{key}.json',record)
                joblib.dump(models,local/f'{key}.joblib')
                np.savez_compressed(local/f'{key}_oof.npz',**pred)
                for suffix in ('.json','.joblib','_oof.npz'):
                    p=local/f'{key}{suffix}';artifacts[str(p.relative_to(ROOT))]=digest(p)
                for selection_key,policy in selection.items():
                    target,b,group=selection_key.split('|');k=int(math.ceil(float(b)*len(z['y'])))
                    hits=int(tt[target][orders[target][policy][:k]].sum());den=int(tt[target].sum())
                    choices.append(dict(base=base,variant=variant,learner=learner,target=target,budget=float(b),
                                        policy_set=group,selected_policy=policy,development_target_n=den,
                                        development_hits=hits,development_capture=hits/den if den else np.nan,
                                        selection_status='selected_oof_hits' if den else 'fallback_empty_target'))
                for level,q in pred.items():
                    metrics.append(dict(base=base,variant=variant,learner=learner,model=level,M=cand.shape[1],
                                        families=';'.join(map(str,sorted(members.loc[keep,'axis'].unique()))),
                                        oof_log_loss=float(np.mean(-(z['y']*np.log(np.clip(q,1e-12,1-1e-12))+(1-z['y'])*np.log1p(-np.clip(q,1e-12,1-1e-12))))),
                                        oof_brier=float(np.mean((z['y']-q)**2)),N=len(q),positive_n=int(z['y'].sum())))
                print(f'Development frozen: {key}; M={cand.shape[1]}',flush=True)
    pd.DataFrame(registry).to_csv(OUT/'candidate_registry.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'development_policy_choices.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'development_model_metrics.csv',index=False)
    for name in ('candidate_registry.csv','development_policy_choices.csv','development_model_metrics.csv'):
        artifacts[str((OUT/name).relative_to(ROOT))]=digest(OUT/name)
    write_json(OUT/'FROZEN_DEVELOPMENT.json',dict(frozen_at_utc=now(),plan_sha256=digest(OUT/'analysis_plan.json'),
                                               artifacts=artifacts,status='frozen before new evaluation; historical sources already exposed'))


def check_development():
    checked_plan();frozen=read_json(OUT/'FROZEN_DEVELOPMENT.json')
    if frozen['plan_sha256']!=digest(OUT/'analysis_plan.json'): raise ValueError('Development/plan mismatch')
    for path,expected in frozen['artifacts'].items():
        if digest(ROOT/path)!=expected: raise ValueError(f'Development artifact changed: {path}')
    return frozen


def evaluate_one(base,name,path,scope,boots,family,z_override=None,budgets_override=None,extra_metadata=None):
    z=load_npz(path) if z_override is None else z_override
    validate_cohort(z)
    supplied_ids=z.get('candidate_ids')
    for variant in ('global','family'):
        for learner in ('additive','interaction'):
            key=f'{base}__{variant}__{learner}'
            record=read_json(OUT/'local_cache'/f'{key}.json')
            if supplied_ids is not None and list(supplied_ids.astype(str))!=record['all_candidate_ids']:
                raise ValueError('New evaluation candidate columns do not match frozen registry')
            if z['cand'].shape[1]!=len(record['all_candidate_ids']): raise ValueError('Candidate column count mismatch')
            cand=z['cand'][:,record['keep']]
            models=joblib.load(OUT/'local_cache'/f'{key}.joblib')
            pred={model:m.predict(z['p_dep'],cand) for model,m in models.items()}
            result=evaluation_tables(z,cand,record['tau'],pred,record['selection'],record['bin_edges'],
                                     budgets_override or read_json(OUT/'analysis_plan.json')['budgets'],boots,
                                     SEED+int(hashlib.sha256(name.encode()).hexdigest()[:8],16),scope,family)
            dest=OUT/scope/name/f'{variant}__{learner}';dest.mkdir(parents=True,exist_ok=True)
            for table,rows in result.items():
                frame=pd.DataFrame(rows)
                for k,v in dict(cohort=name,base=base,variant=variant,learner=learner,scope=scope,
                                M=cand.shape[1],development='historical_2018_test').items(): frame[k]=v
                # Only global/additive is in the declared primary comparison family.
                if 'family_adjusted_lower' in frame and (variant!='global' or learner!='additive'):
                    frame[['family_adjusted_lower','family_adjusted_upper']]=np.nan
                    frame['comparison_role']='sensitivity_descriptive';frame['family_size']=0
                frame.to_csv(dest/f'{table}.csv',index=False)
            write_json(dest/'receipt.json',dict(completed_at_utc=now(),cohort=name,scope=scope,
                       dataset_sha256=digest(path),plan_sha256=digest(OUT/'analysis_plan.json'),
                       development_manifest_sha256=digest(OUT/'FROZEN_DEVELOPMENT.json'),
                       fixed_system_only=True,borrower_cluster_ids_available=False,bootstrap=boots,
                       endpoint_metadata=extra_metadata or {'endpoint':'historical recorded loan-age-based endpoint'},
                       bootstrap_scope='paired row occurrences; conditional on fixed models, candidate/policy selection, bins',
                       results={p.name:digest(p) for p in dest.glob('*.csv')}))
            print(f'Evaluated {name}/{variant}/{learner}, {boots} paired draws; {scope}',flush=True)


def evaluate_historical():
    check_development();plan=read_json(OUT/'analysis_plan.json')
    for name in plan['evaluation']:
        base=name.rsplit('_',1)[0]+'_2018'
        members=pd.read_csv(OLD/f'{name}_members.csv')
        original=pd.read_csv(OLD/f'{base}_members.csv')
        if members.candidate_id.tolist()!=original.candidate_id.tolist(): raise ValueError('Historical candidate order drift')
        evaluate_one(base,name,OLD/f'{name}.npz','exploratory_retrospective',plan['bootstrap_historical'],24)
    check_development()
    print('Historical evaluation completed; no independent validation claim.',flush=True)


def evaluate_strict():
    check_development();plan=read_json(OUT/'analysis_plan.json')
    spec=plan['strict_endpoint_sensitivity']
    for name in spec['cohorts']:
        source=OLD/f'{name}.npz';z=load_npz(source)
        path=OUT/'local_cache'/f'{name}_strict_label_audit.parquet'
        audit=pd.read_parquet(path)
        if not np.array_equal(audit.row_id,z['row_id']) or not np.array_equal(audit.legacy_y,z['y']):
            raise ValueError('Strict endpoint audit/source alignment mismatch')
        keep=audit.strict_known.to_numpy(dtype=bool)
        strict={k:v[keep] for k,v in z.items()}
        strict['y']=audit.strict_y.to_numpy(dtype=int)[keep]
        evaluate_one('freddie_mac_2018',name,path,'exploratory_strict_endpoint_eligible',
                     spec['bootstrap'],24,strict,[spec['budget']],
                     dict(endpoint=spec['endpoint'],eligible='strict_known',original_N=len(keep),strict_N=int(keep.sum()),
                          original_cache_sha256=digest(source),interpretation=spec['interpretation']))
    check_development()


def evaluate_development_baselines():
    check_development();plan=read_json(OUT/'analysis_plan.json')
    for base in BASES:
        z=load_npz(OLD/f'{base}.npz');tau=TAUS[base]
        contrasts=(('Range','ThresholdProb'),('AboveTauLow','Range'),
                   ('AboveTauLow','ThresholdProb'),('BelowTauTop','Range'),
                   ('BelowTauTop','ThresholdProb'))
        result=evaluation_tables(z,z['cand'],tau,{}, {},fixed_bin_edges(z['p_dep'],tau),[.1],
                                 plan['bootstrap_historical'],SEED,'historical_2018_baselines',
                                 policy_groups=False,contrasts=contrasts)
        dest=OUT/'historical_2018_baselines'/base;dest.mkdir(parents=True,exist_ok=True)
        for table,rows in result.items():
            if not rows:continue
            frame=pd.DataFrame(rows);frame['cohort']=base
            frame['scope']='historical_2018_exploratory_development';frame['variant']='family'
            frame.to_csv(dest/f'{table}.csv',index=False)
        write_json(dest/'receipt.json',dict(completed_at_utc=now(),scope='historical_2018_exploratory_development',
                   plan_sha256=digest(OUT/'analysis_plan.json'),dataset_sha256=digest(OLD/f'{base}.npz'),
                   bootstrap=plan['bootstrap_historical'],endpoint='historical recorded loan-age-based endpoint',
                   results={p.name:digest(p) for p in dest.glob('*.csv')}))
        print(f'Historical 2018 baselines, paired rebuilt queues: {base}',flush=True)


def validate_independent_gate(gate, dataset_path, frozen_digest):
    required=('user_external_use_confirmed_unseen','outcomes_unseen_before_development_freeze',
              'loan_identity_disjoint','outcome_contract_validated','label_time_audited',
              'candidate_and_deployment_frozen','cohort_scope_frozen_before_labels')
    if gate.get('status')!='independent_eligible' or not all(gate.get(k) is True for k in required):
        raise ValueError('Independent evaluation blocked: missing/negative exposure, isolation or time evidence')
    if gate.get('development_manifest_sha256')!=frozen_digest: raise ValueError('Gate references a different development freeze')
    if gate.get('dataset_sha256')!=digest(dataset_path): raise ValueError('Gate/dataset hash mismatch')
    audit=Path(gate.get('audit_document_path',''))
    if not audit.is_file() or gate.get('audit_document_sha256')!=digest(audit): raise ValueError('Missing verified audit document')
    scope=Path(gate.get('frozen_cohort_scope_path',''))
    if not scope.is_file() or gate.get('frozen_cohort_scope_sha256')!=digest(scope): raise ValueError('Missing independently frozen cohort scope')
    spec=read_json(scope)
    if gate.get('cohort') not in spec.get('cohorts',[]): raise ValueError('Cohort was not in frozen scope')
    if datetime.fromisoformat(spec['frozen_at_utc'])>=datetime.fromisoformat(gate['outcomes_unsealed_at_utc']):
        raise ValueError('Cohort scope was frozen after outcome exposure')
    if spec.get('primary_family_size')!=8 or len(spec.get('cohorts',[]))!=2:
        raise ValueError('This executable protocol requires two prespecified cohorts / eight primary contrasts')


def evaluate_independent(args):
    check_development()
    gate=read_json(args.gate)
    validate_independent_gate(gate,args.dataset,digest(OUT/'FROZEN_DEVELOPMENT.json'))
    if args.base not in BASES: raise ValueError('Unknown fixed reference system')
    # Only after gate validation may outcome arrays be opened.
    evaluate_one(args.base,gate['cohort'],Path(args.dataset),'independent_fixed_system_retrospective',9999,8)
    check_development()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','develop','evaluate-historical','evaluate-strict','evaluate-development-baselines','evaluate-independent'))
    parser.add_argument('--dataset');parser.add_argument('--gate');parser.add_argument('--base')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        try:
            if args.stage=='freeze': freeze()
            elif args.stage=='develop': develop()
            elif args.stage=='evaluate-historical': evaluate_historical()
            elif args.stage=='evaluate-strict': evaluate_strict()
            elif args.stage=='evaluate-development-baselines': evaluate_development_baselines()
            else:
                if not (args.dataset and args.gate and args.base): parser.error('Independent evaluation requires --dataset --gate --base')
                evaluate_independent(args)
        except Exception as exc:
            OUT.mkdir(parents=True,exist_ok=True)
            write_json(OUT/f'failure_{args.stage}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json',
                       dict(stage=args.stage,error_type=type(exc).__name__,message=str(exc),recorded_at_utc=now()))
            raise


if __name__=='__main__': main()
