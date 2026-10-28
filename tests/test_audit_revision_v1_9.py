"""Scientific implementation risks: leakage boundaries, budgets, pairing, targets."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_revision_v1_9 import (AuditRiskModel,queue,raw_features,target_probability,
    target_vectors,weighted_selected,choose_policies,all_orders,validate_cohort,
    validate_independent_gate,evaluation_tables,fixed_bin_edges,qualify_comparisons)


def fixture_cohort(n=120):
    rng=np.random.default_rng(21)
    p=rng.uniform(.01,.4,n)
    return dict(row_id=np.arange(n),p_dep=p,cand=np.c_[p*.9,np.minimum(.99,p*1.1)],
                y=rng.binomial(1,p))


def test_real_queue_never_duplicates_and_refills_one_sided_scores():
    ids=np.arange(5);p=np.array([.01,.02,.03,.04,.8])
    above=np.where(p>=.1,-p,p-2)
    q=queue(above,ids,.8)
    assert len(q)==len(np.unique(ids[q]))==4
    assert set(q)=={1,2,3,4}
    with pytest.raises(ValueError):queue(above,np.array([0,0,1,2,3]),.8)
    with pytest.raises(ValueError):queue(above,ids,1.1)


def test_bootstrap_weighted_topB_equals_explicit_occurrence_ranking():
    score=np.array([.1,.8,.4,.8]);ids=np.arange(4);rank=np.lexsort((ids,-score))
    w=np.array([2,3,0,1]);target=np.array([1,0,1,1])
    expanded=np.repeat(rank,w[rank])
    for k,(ix,selected) in zip((1,4,6),weighted_selected(rank,w,[1,4,6])):
        assert selected.sum()==k
        assert np.dot(selected,target[ix])==target[expanded[:k]].sum()


def test_consensus_is_logit_of_mean_probability_not_mean_logit():
    p=np.array([.2,.3]);cand=np.array([[.01,.6],[.15,.7]])
    x=raw_features(p,cand,2)
    np.testing.assert_allclose(x[:,1],logit(cand.mean(1)))
    assert not np.allclose(x[:,1],logit(cand).mean(1))


def test_targets_partition_errors_and_em_fixed_is_preserved():
    z=fixture_cohort();tau=.15
    t=target_vectors(z,tau,z['cand'][:,:1])
    assert np.array_equal(t['AllErr'],t['FN']|t['FP'])
    assert not (t['FN']&t['FP']).any()
    assert not t['EM_variant'].any()
    assert np.array_equal(t['EM_fixed'],target_vectors(z,tau,z['cand'])['EM_fixed'])
    q=np.full(len(z['y']),.1)
    np.testing.assert_allclose(target_probability(q,z['p_dep'],tau,'AllErr'),
        target_probability(q,z['p_dep'],tau,'FN')+target_probability(q,z['p_dep'],tau,'FP'))


def test_policy_frozen_choice_does_not_use_evaluation_labels():
    z=fixture_cohort();pred={f'G{i}':z['p_dep'] for i in range(3)}
    orders=all_orders(z,z['cand'],.15,pred)
    selections=choose_policies(orders,target_vectors(z,.15,z['cand']),[.1])
    changed=dict(z,y=1-z['y'])
    orders2=all_orders(changed,changed['cand'],.15,pred)
    for t in orders:
        for q in orders[t]:np.testing.assert_array_equal(orders[t][q],orders2[t][q])
    assert all(k in selections for k in ('AllErr|0.1|P_full','FN|0.1|P_base'))


def test_empty_target_undefined_and_paired_identical_scores_zero_difference():
    z=fixture_cohort();z['y']=np.zeros(len(z['y']),dtype=int)
    pred={f'G{i}':z['p_dep'] for i in range(3)}
    selection=choose_policies(all_orders(z,z['cand'],.15,pred),target_vectors(z,.15,z['cand']),[.1])
    out=evaluation_tables(z,z['cand'],.15,pred,selection,fixed_bin_edges(z['p_dep'],.15),[.1],19,23,'unit_test')
    fn=[r for r in out['performance'] if r['target']=='FN']
    assert all(np.isnan(r['capture']) and r['target_n']==0 and r['bootstrap_empty_target']==19 for r in fn)
    err=next(r for r in out['paired_contrasts'] if r['target']=='AllErr' and r['contrast']=='G2-G1')
    assert err['capture_difference']==err['lower']==err['upper']==0
    assert all(abs(r['identity_residual'])<1e-12 for r in out['decomposition'] if np.isfinite(r['identity_residual']))


def test_independent_gate_refuses_exposed_or_missing_evidence_before_load(tmp_path):
    p=tmp_path/'never_read.npz'
    for gate in ({},{'status':'independent_eligible','user_external_use_confirmed_unseen':False}):
        with pytest.raises(ValueError,match='blocked'):
            validate_independent_gate(gate,p,'anything')
    assert not p.exists()


def test_same_learner_parameters_except_input_count_and_constant_outcome():
    z=fixture_cohort(300)
    for learner in ('additive','interaction'):
        a=AuditRiskModel(1,learner).fit(z['p_dep'],z['cand'],z['y'])
        b=AuditRiskModel(2,learner).fit(z['p_dep'],z['cand'],z['y'])
        assert a.model.get_params()==b.model.get_params()
        q=b.predict(z['p_dep'],z['cand'])
        assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        c=AuditRiskModel(2,learner).fit(z['p_dep'],z['cand'],np.zeros(300))
        assert np.array_equal(c.predict(z['p_dep'],z['cand']),np.zeros(300))


def test_invalid_outcomes_and_probability_rejected():
    z=fixture_cohort();validate_cohort(z)
    bad=dict(z,y=np.full(len(z['y']),-1))
    with pytest.raises(ValueError):validate_cohort(bad)
    bad=dict(z,p_dep=np.full(len(z['y']),np.nan))
    with pytest.raises(ValueError):validate_cohort(bad)


def test_independent_candidate_identity_is_mandatory():
    z=fixture_cohort()
    with pytest.raises(ValueError,match='candidate_ids'):validate_cohort(z,require_candidate_ids=True)
    z['candidate_ids']=np.array(['a','b']);validate_cohort(z,require_candidate_ids=True)
    z['candidate_ids']=np.array(['a','a'])
    with pytest.raises(ValueError,match='Repeated'):validate_cohort(z,require_candidate_ids=True)


def test_strict_endpoint_cannot_inherit_primary_family_or_pure_score_em_label():
    f=pd.DataFrame([dict(target='EM_fixed',family_adjusted_lower=.1,family_adjusted_upper=.2,
                         comparison_role='primary',family_size=24)])
    out=qualify_comparisons(f,'exploratory_strict_endpoint_eligible','global','additive')
    assert out.iloc[0].comparison_role=='sensitivity_descriptive' and out.iloc[0].family_size==0
    assert np.isnan(out.iloc[0].family_adjusted_lower)
    assert 'candidate-flip mask' in out.iloc[0].target_information_condition
