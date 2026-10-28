"""Post-primary exploratory candidate reconstruction; never changes main artifacts.

Stages: freeze; run-base (one process per institution); freeze-development;
evaluate; summarize. All three reconstruction seeds are fixed before training.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import audit_revision_v1_9 as audit
import build_reference_ensemble_v0_7 as training
from v0_4_common import predict_artifact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/tables/v1_9_candidate_rebuild'
MODELS = ROOT / 'models/v1_9_candidate_rebuild'
SEEDS = (2026101901, 2026101902, 2026101903)
BASES = audit.BASES
TARGETS = ('AllErr', 'FN')
BOOTSTRAPS = 999
WORKER_THREADS = 5
SCOPE = 'post_primary_exploratory_candidate_reconstruction'


def relative(path): return str(Path(path).relative_to(ROOT))
def dump(path, obj): audit.write_json(path, obj)
def load(path): return audit.read_json(path)
def digest(path): return audit.digest(path)
def key(base, seed): return f'{base}__s{seed}'
def local(base, seed): return OUT / 'local_cache' / key(base, seed)


def input_paths():
    paths = [Path(__file__), ROOT/'src/audit_revision_v1_9.py',
             ROOT/'src/build_reference_ensemble_v0_7.py', ROOT/'src/train_hmda_phase_b.py',
             ROOT/'src/m1_feature_engineering.py', ROOT/'src/v0_4_common.py',
             ROOT/'src/preprocessing_freddie_mac.py', ROOT/'src/preprocessing_fannie_mae.py',
             ROOT/'tests/test_candidate_rebuild_v1_9.py',
             audit.OUT/'analysis_plan.json', audit.OUT/'FROZEN_DEVELOPMENT.json',
             audit.OUT/'main_historical_e2e3.csv']
    paths += [ROOT/p for p in training.candidate_trainer.CONFIG_FILES.values()]
    for base in BASES:
        spec=training.DATASET_REGISTRY[base]
        paths += [ROOT/spec.train_path, ROOT/spec.val_path,
                  ROOT/f'models/v0_7/deployment/{base}/m2_xgb.joblib',
                  ROOT/f'results/tables/v0_7/candidate_pool_metrics_{base}.csv',
                  audit.OLD/f'{base}_members.csv',
                  audit.OUT/'local_cache'/f'{base}_development_ids.npz']
        metrics=pd.read_csv(ROOT/f'results/tables/v0_7/candidate_pool_metrics_{base}.csv')
        paths += [ROOT/p for p in metrics.artifact_path]
        for year in (2018,2021,2022,2023):
            name=base.replace('2018',str(year))
            paths += [ROOT/f'data/processed/{name}_ever_60dpd_24m/test.parquet',
                      audit.OLD/f'{name}.npz', audit.OLD/f'{name}_members.csv']
    return sorted(set(paths))


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise RuntimeError('Protocol already frozen')
    protocol=dict(scope=SCOPE,independent_validation=False,main_results_already_seen=True,
        frozen_at_utc=audit.now(),bases=list(BASES),seeds=list(SEEDS),
        candidate_grid=dict(M1=6,M2=7,M6=9,M7=4),candidate_fits=156,
        training='original 2018 train; original 2018 validation for global eligibility',
        eligibility=dict(validation_auc_gap=.010,numerical_tolerance=1e-6,brier_gate=False),
        fixed_deployment='exact old p_dep arrays; no deployment refit or new threshold',taus=audit.TAUS,
        development='historical former 2018 test; five stratified folds identical to main',
        audit_models='G1/G2 additive AuditRiskModel, same parameters as main; OOF plus full-C refit',
        targets=list(TARGETS),budget=.10,bootstrap=BOOTSTRAPS,
        bootstrap_scope='paired row occurrences; fixed candidate fit and outcome learners; pointwise only',
        bootstrap_seed_rule='20261019 + first 8 SHA256 hex digits of cohort name; shared across candidate seeds',
        selection='no new policy argmax; only G2-G1 and queue membership stability',
        failure_rule='record all 26 candidates; no replacement; any fit failure invalidates that seed; M<2 invalidates disagreement evaluation',
        stability='Range versus original global Jaccard; per-case 0..3 membership frequencies kept private; only histograms public; no seed-as-cohort inference',
        outputs_private=['row IDs','candidate predictions','OOF predictions','case membership counts','fitted models'],
        resources=dict(max_concurrent_institutions=2,threads_per_worker=WORKER_THREADS),
        software=dict(python=platform.python_version(),**{p:importlib.metadata.version(p) for p in
                         ['numpy','pandas','scikit-learn','xgboost','interpret','joblib']}),
        protected_inputs={relative(p):digest(p) for p in input_paths()})
    dump(OUT/'protocol.json',protocol)
    (OUT/'protocol.sha256').write_text(digest(OUT/'protocol.json')+'\n')
    (OUT/'.gitignore').write_text('local_cache/\n')
    print('Frozen post-primary exploratory protocol; no independent claim.',flush=True)


def check_protocol():
    p=OUT/'protocol.json'
    if digest(p)!=(OUT/'protocol.sha256').read_text().strip(): raise ValueError('Protocol changed')
    obj=load(p)
    if obj['seeds']!=list(SEEDS) or obj['candidate_fits']!=156: raise ValueError('Frozen design mismatch')
    for path,sha in obj['protected_inputs'].items():
        if digest(ROOT/path)!=sha: raise ValueError(f'Protected input changed: {path}')
    return obj


@contextmanager
def limited_training_threads():
    """Limit explicit job counts as well as BLAS; no old configs are edited."""
    t=training.candidate_trainer
    old_rf,old_ebm,old_xgb=t.RandomForestClassifier,t.ExplainableBoostingClassifier,t.build_xgb
    def rf(**kw): kw['n_jobs']=WORKER_THREADS; return old_rf(**kw)
    def ebm(**kw): kw['n_jobs']=WORKER_THREADS; return old_ebm(**kw)
    def xgb(params,random_state):
        model=old_xgb(params,random_state=random_state);model.set_params(n_jobs=WORKER_THREADS);return model
    t.RandomForestClassifier,t.ExplainableBoostingClassifier,t.build_xgb=rf,ebm,xgb
    try:
        with threadpool_limits(limits=1): yield
    finally: t.RandomForestClassifier,t.ExplainableBoostingClassifier,t.build_xgb=old_rf,old_ebm,old_xgb


def actual_seed(model):
    if hasattr(model,'pipeline_'): model=model.pipeline_.named_steps['model']
    return int(model.get_params()['random_state'])


def eligible(metrics):
    if len(metrics)!=26 or metrics.candidate_id.nunique()!=26 or not (metrics.status=='ok').all():
        return np.zeros(len(metrics),bool),'incomplete_grid'
    if not np.isfinite(metrics.validation_auc).all(): return np.zeros(len(metrics),bool),'invalid_validation_auc'
    mask=(metrics.validation_auc.to_numpy()+1e-6>=metrics.validation_auc.max()-.010)
    return mask,'ok' if mask.sum()>=2 else 'degenerate_M_below_2'


def aligned_cohort(name):
    z=audit.load_npz(audit.OLD/f'{name}.npz');audit.validate_cohort(z)
    frame=pd.read_parquet(ROOT/f'data/processed/{name}_ever_60dpd_24m/test.parquet')
    if not np.array_equal(frame.row_id.to_numpy(),z['row_id']): raise ValueError('Processed/cache row identity mismatch')
    if not np.array_equal(frame.ever_60dpd_24m.to_numpy(),z['y']): raise ValueError('Processed/cache outcome mismatch')
    return frame,z


def predict_candidates(paths,frame):
    preds=[]
    for p in paths:
        artifact=joblib.load(ROOT/p)
        preds.append(predict_artifact(artifact,frame))
    ans=np.column_stack(preds)
    if not np.isfinite(ans).all() or np.any((ans<0)|(ans>1)): raise ValueError('Invalid candidate probability')
    return ans


def develop(base,seed,metrics):
    kept=metrics.loc[metrics.selected]
    frame,z=aligned_cohort(base)
    cand=predict_candidates(kept.artifact_path.tolist(),frame)
    if not np.ptp(cand,axis=1).any(): raise ValueError('Degenerate zero range despite multiple candidates')
    folds=list(StratifiedKFold(5,shuffle=True,random_state=audit.SEED).split(z['p_dep'],z['y']))
    fold_id=np.empty(len(z['y']),dtype=int)
    for f,(_,valid) in enumerate(folds):fold_id[valid]=f
    reference=audit.load_npz(audit.OUT/'local_cache'/f'{base}_development_ids.npz')
    if not np.array_equal(z['row_id'],reference['row_id']) or not np.array_equal(fold_id,reference['fold_id']):
        raise ValueError('Development folds differ from main')
    models={};oof={};rows=[]
    with threadpool_limits(limits=1):
        for level in (1,2):
            name=f'G{level}';q=np.empty(len(z['y']))
            for tr,va in folds:
                fit=audit.AuditRiskModel(level,'additive').fit(z['p_dep'][tr],cand[tr],z['y'][tr])
                q[va]=fit.predict(z['p_dep'][va],cand[va])
            models[name]=audit.AuditRiskModel(level,'additive').fit(z['p_dep'],cand,z['y'])
            oof[name]=q
            clipped=np.clip(q,1e-12,1-1e-12)
            rows.append(dict(base=base,seed=seed,model=name,M=len(kept),N=len(q),
                 oof_log_loss=float(np.mean(-z['y']*np.log(clipped)-(1-z['y'])*np.log1p(-clipped))),
                 oof_brier=float(np.mean((z['y']-q)**2))))
    d=local(base,seed);d.mkdir(parents=True,exist_ok=True)
    joblib.dump(models,d/'audit_models.joblib')
    np.savez_compressed(d/'development_predictions.npz',row_id=z['row_id'],fold_id=fold_id,cand=cand,**oof)
    pd.DataFrame(rows).to_csv(OUT/f'{key(base,seed)}_development.csv',index=False)
    return dict(candidate_ids=kept.candidate_id.tolist(),artifact_paths=kept.artifact_path.tolist(),
                audit_models_sha256=digest(d/'audit_models.joblib'),development_predictions_sha256=digest(d/'development_predictions.npz'))


def run_base(base):
    check_protocol()
    spec=training.DATASET_REGISTRY[base]
    train=training.load_split(spec.train_path);validation=training.load_split(spec.val_path)
    training._validate_phase1_frame(spec,train,'train');training._validate_phase1_frame(spec,validation,'validation')
    train_tree,val_tree=training._build_tree_frames(spec,train,validation)
    train.attrs[training._TREE_TRAIN_ATTR]=train_tree;validation.attrs[training._TREE_VAL_ATTR]=val_tree
    for seed in SEEDS:
        receipt=OUT/f'{key(base,seed)}_development.json'
        if receipt.exists(): raise RuntimeError('Completed seed must not be overwritten')
        rows=[];started=time.perf_counter()
        for axis in training.AXES:
            for template in training._axis_grid_specs(axis):
                candidate_id=template.get('name') or f"M6_rf_n{template['n_estimators']}_depth{template['max_depth'] if template['max_depth'] is not None else 'none'}"
                p=MODELS/key(base,seed)/f'{candidate_id}.joblib'
                if p.exists(): raise RuntimeError(f'Partial existing model prevents overwrite: {p.name}')
                row=dict(base=base,seed=seed,axis=axis,candidate_id=candidate_id,status='failed',
                         validation_auc=np.nan,validation_brier=np.nan,selected=False,artifact_path='',actual_seed=None)
                tic=time.perf_counter()
                try:
                    with limited_training_threads(),training._gse_training_context(spec),training._single_candidate_grid(axis,template,seed):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            result=training._dispatch_axis_training(spec,axis,train,validation,
                                  train[spec.target_col].to_numpy(int),validation[spec.target_col].to_numpy(int),seed)
                    if len(result)!=1: raise ValueError('Expected one candidate per seeded template')
                    fitted=result[0]
                    observed=actual_seed(fitted['model'])
                    if observed!=seed: raise ValueError(f'Actual model seed {observed} differs from {seed}')
                    artifact=training._artifact_from_row(spec,axis,candidate_id,fitted,seed=seed,train_tree_columns=list(train_tree))
                    training._atomic_joblib_dump(artifact,p)
                    row.update(status='ok',validation_auc=float(fitted['validation_auc']),
                        validation_brier=float(fitted['validation_brier']),actual_seed=observed,
                        artifact_path=relative(p),artifact_sha256=digest(p),
                        warnings=json.dumps(sorted(set(str(w.message) for w in caught))))
                except Exception as exc:
                    row.update(error=f'{type(exc).__name__}: {exc}')
                row['seconds']=time.perf_counter()-tic;rows.append(row)
                pd.DataFrame(rows).to_csv(OUT/f'{key(base,seed)}_candidates.csv',index=False)
                print(f'{key(base,seed)} {candidate_id}: {row["status"]}, {row["seconds"]:.1f}s',flush=True)
        metrics=pd.DataFrame(rows);mask,status=eligible(metrics);metrics['selected']=mask
        metrics.to_csv(OUT/f'{key(base,seed)}_candidates.csv',index=False)
        record=dict(base=base,seed=seed,status=status,M=int(mask.sum()),scope=SCOPE,
             protocol_sha256=digest(OUT/'protocol.json'),elapsed_seconds=time.perf_counter()-started,
             candidate_table_sha256=digest(OUT/f'{key(base,seed)}_candidates.csv'))
        if status=='ok':
            try: record.update(develop(base,seed,metrics))
            except Exception as exc:record.update(status='development_failed',error=f'{type(exc).__name__}: {exc}')
        record['completed_at_utc']=audit.now();dump(receipt,record)
        print(f'Development completed {key(base,seed)} status={record["status"]}, M={record["M"]}',flush=True)
    check_protocol()


def freeze_development():
    check_protocol()
    dest=OUT/'FROZEN_DEVELOPMENT.json'
    if dest.exists():raise RuntimeError('Development already frozen')
    paths=[];statuses=[]
    for base in BASES:
        for seed in SEEDS:
            p=OUT/f'{key(base,seed)}_development.json';r=load(p);statuses.append({k:r[k] for k in ('base','seed','status','M')})
            paths += [p,OUT/f'{key(base,seed)}_candidates.csv']
            if r['status']=='ok':
                paths += [local(base,seed)/'audit_models.joblib',local(base,seed)/'development_predictions.npz',
                          OUT/f'{key(base,seed)}_development.csv']
            for path in pd.read_csv(OUT/f'{key(base,seed)}_candidates.csv').artifact_path.dropna():paths.append(ROOT/path)
    dump(dest,dict(frozen_at_utc=audit.now(),protocol_sha256=digest(OUT/'protocol.json'),statuses=statuses,
                  artifacts={relative(p):digest(p) for p in paths}))


def check_development():
    check_protocol();f=load(OUT/'FROZEN_DEVELOPMENT.json')
    if f['protocol_sha256']!=digest(OUT/'protocol.json'):raise ValueError('Development protocol mismatch')
    for p,sha in f['artifacts'].items():
        if digest(ROOT/p)!=sha:raise ValueError(f'Frozen reconstruction changed: {p}')


def paired_results(z,cand,pred,tau,seed,boots=BOOTSTRAPS):
    n=len(z['y']);b=int(math.ceil(.1*n));ids=z['row_id'];t=audit.target_vectors(z,tau,cand)
    orders={target:{g:audit.order(audit.target_probability(q,z['p_dep'],tau,target),ids)
                    for g,q in pred.items()} for target in TARGETS}
    samples={target:[] for target in TARGETS};loss=[]
    ll={g:-(z['y']*np.log(np.clip(q,1e-12,1-1e-12))+(1-z['y'])*np.log1p(-np.clip(q,1e-12,1-1e-12))) for g,q in pred.items()}
    rng=np.random.default_rng(seed)
    for _ in range(boots):
        w=np.bincount(rng.integers(0,n,n),minlength=n)
        for target in TARGETS:
            den=float(w[t[target]].sum());h={}
            for g,o in orders[target].items():
                ix,sw=audit.weighted_selected(o,w,[b])[0];h[g]=float(sw@t[target][ix])
            samples[target].append((h['G2']-h['G1'])/den if den else np.nan)
        loss.append(float(w@(ll['G2']-ll['G1'])/n))
    rows=[];members={}
    for target in TARGETS:
        qa,qb=orders[target]['G2'][:b],orders[target]['G1'][:b]
        members[target]={'G1':qb,'G2':qa}
        enter=np.setdiff1d(qa,qb);leave=np.setdiff1d(qb,qa)
        den=int(t[target].sum());ha=int(t[target][qa].sum());hb=int(t[target][qb].sum())
        lo,hi,valid=audit._interval(samples[target])
        rows.append(dict(target=target,N=n,B=b,target_n=den,G1_hits=hb,G2_hits=ha,hits_difference=ha-hb,
             capture_difference=(ha-hb)/den if den else np.nan,lower=lo,upper=hi,
             entering_n=len(enter),leaving_n=len(leave),entering_hits=int(t[target][enter].sum()),
             leaving_hits=int(t[target][leave].sum()),queue_jaccard=len(np.intersect1d(qa,qb))/len(np.union1d(qa,qb)),
             bootstrap_valid=valid,bootstrap_empty_target=boots-valid,bootstrap=boots,family_size=0,
             comparison_role='post_primary_exploratory_sensitivity',status='ok' if den else 'undefined_empty_target'))
    lo,hi,_=audit._interval(loss)
    predictive=dict(metric='log_loss',contrast='G2-G1',difference=float(np.mean(ll['G2']-ll['G1'])),lower=lo,upper=hi)
    return rows,predictive,members


def evaluate():
    check_development()
    if (OUT/'EVALUATION_COMPLETE.json').exists():raise RuntimeError('Completed evaluation must not be overwritten')
    comparisons=[];predictive=[];stability=[];counts={};private_ids={};evaluated=[];skipped=[]
    for base in BASES:
        _,variant=audit.candidate_variants(base)
        for seed in SEEDS:
            r=load(OUT/f'{key(base,seed)}_development.json')
            if r['status']!='ok':skipped.append({k:r[k] for k in ('base','seed','status','M')});continue
            models=joblib.load(local(base,seed)/'audit_models.joblib')
            # Membership stability includes C and later cohorts; targets only evaluated later.
            for year in (2018,2021,2022,2023):
                name=base.replace('2018',str(year));frame,z=aligned_cohort(name)
                cand=predict_candidates(r['artifact_paths'],frame);n=len(z['y']);b=int(math.ceil(.1*n))
                old=z['cand'][:,variant['global']]
                if cand.shape[1]<2 or old.shape[1]<2:raise ValueError('Degenerate Range must not form a queue')
                qnew=audit.queue(np.ptp(cand,axis=1),z['row_id'],.1);qold=audit.queue(np.ptp(old,axis=1),z['row_id'],.1)
                stability.append(dict(base=base,cohort=name,seed=seed,M=cand.shape[1],N=n,B=b,
                     original_global_M=old.shape[1],range_original_jaccard=len(np.intersect1d(qnew,qold))/len(np.union1d(qnew,qold))))
                if name not in counts:counts[name]=np.zeros(n,dtype=np.int8);private_ids[name]=z['row_id']
                elif not np.array_equal(private_ids[name],z['row_id']):raise ValueError('Case identity changed across seed')
                counts[name][qnew]+=1
                pred={g:m.predict(z['p_dep'],cand) for g,m in models.items()}
                payload=dict(row_id=z['row_id'],cand=cand,range_queue=qnew,G1=pred['G1'],G2=pred['G2'])
                if year!=2018:
                    bootseed=audit.SEED+int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
                    rows,pp,member=paired_results(z,cand,pred,audit.TAUS[base],bootseed)
                    metadata=dict(base=base,cohort=name,seed=seed,M=cand.shape[1],scope=SCOPE)
                    comparisons.extend([dict(**row,**metadata) for row in rows]);predictive.append(dict(**pp,**metadata))
                    for target,d in member.items():
                        for g,ix in d.items():payload[f'{target}_{g}_queue']=ix
                    evaluated.append(dict(**metadata,N=n))
                np.savez_compressed(local(base,seed)/f'{name}_evaluation.npz',**payload)
                print(f'Evaluated {name} seed={seed} M={cand.shape[1]}',flush=True)
    freq=[]
    for name,cc in counts.items():
        trials=sum(x['cohort']==name for x in stability)
        np.savez_compressed(OUT/'local_cache'/f'{name}_selection_frequency.npz',row_id=private_ids[name],selected_count=cc,valid_seed_count=trials)
        for k in range(len(SEEDS)+1):freq.append(dict(cohort=name,selected_count=k,valid_seed_count=trials,N=len(cc),cases=int((cc==k).sum()),proportion=float((cc==k).mean())))
    for name,rows in [('comparisons',comparisons),('predictive',predictive),('range_stability',stability),('selection_frequency',freq)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    check_development()
    dump(OUT/'EVALUATION_COMPLETE.json',dict(completed_at_utc=audit.now(),protocol_sha256=digest(OUT/'protocol.json'),
         development_sha256=digest(OUT/'FROZEN_DEVELOPMENT.json'),evaluated=evaluated,skipped=skipped,
         public_results={p.name:digest(p) for p in OUT.glob('*.csv')},
         private_results={relative(p):digest(p) for p in (OUT/'local_cache').rglob('*.npz')}))


def summarize():
    check_development();receipt=load(OUT/'EVALUATION_COMPLETE.json')
    for p,sha in receipt['public_results'].items():
        if digest(OUT/p)!=sha:raise ValueError('Completed public output changed')
    comp=pd.read_csv(OUT/'comparisons.csv');stab=pd.read_csv(OUT/'range_stability.csv')
    summary=comp.groupby(['cohort','target'],sort=False).agg(valid_seeds=('seed','nunique'),
        hits_min=('hits_difference','min'),hits_max=('hits_difference','max'),
        capture_min=('capture_difference','min'),capture_max=('capture_difference','max'),
        pointwise_lower_min=('lower','min'),pointwise_upper_max=('upper','max')).reset_index()
    summary['interpretation']='descriptive span over three fixed reconstructions; not a confidence interval over training randomness'
    summary.to_csv(OUT/'across_seed_summary.csv',index=False)
    stab.groupby('cohort').agg(valid_seeds=('seed','nunique'),M_min=('M','min'),M_max=('M','max'),
         range_original_jaccard_min=('range_original_jaccard','min'),range_original_jaccard_max=('range_original_jaccard','max')).reset_index().to_csv(OUT/'range_stability_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run-base','freeze-development','evaluate','summarize'])
    parser.add_argument('--base',choices=BASES)
    args=parser.parse_args()
    if args.stage=='run-base':
        if not args.base:parser.error('--base is required')
        run_base(args.base)
    else:globals()[args.stage.replace('-','_')]()
