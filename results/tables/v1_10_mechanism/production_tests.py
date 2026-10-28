"""Qualified-design identities and guards; no fitted-outcome success criterion."""
import json
from hashlib import sha256
import numpy as np
import pytest
from src import mechanism_v1_10 as m


def test_population_candidate_global_gate_and_brier():
    t=m.exact_tables()["candidate_population_metrics"]
    assert t.passes_global_auc_010.all()
    assert t.auc.max()-t.auc.min()==pytest.approx(.0090003353267791)
    np.testing.assert_allclose(t.brier,[.05749615,.05749560,.05749515],atol=1e-13)
    assert m.AUC_GAP==.010


def test_consensus_does_not_reveal_u_but_range_does():
    s=m.design()
    np.testing.assert_allclose(s.candidates.mean(1),s.score,atol=1e-15)
    np.testing.assert_allclose(np.ptp(s.candidates,axis=1),s.spread,atol=1e-15)
    np.testing.assert_allclose(m.old.oracle_q(s),s.score,atol=1e-15)
    np.testing.assert_allclose(m.old.oracle_q(s,True),s.q,atol=1e-15)
    assert m.retained_oracles([True,True,True])["conditional_information_nats"]==pytest.approx(.0000636864162578965)


def test_oracle_budget_change_and_no_change_same_qualified_set():
    t=m.exact_tables()["population_comparisons"]
    a=t[(t.target=="Pos")&(t.budget==.1)].iloc[0]
    assert a.oracle_expected_extra_hits_10000==pytest.approx(1.5)
    assert a.exchange_mass==pytest.approx(.05)
    assert a.oracle_capture_gain==pytest.approx(.0024115755627009638)
    for target,hits in [("AllErr",1.5),("FN",2.5)]:
        assert t[(t.target==target)&(t.budget==.1)].iloc[0].oracle_expected_extra_hits_10000==pytest.approx(hits)
    b=t[(t.target=="Pos")&(t.budget==.2)].iloc[0]
    assert b.oracle_capture_gain==pytest.approx(0)
    assert b.exchange_mass==pytest.approx(0)
    np.testing.assert_allclose(t.oracle_expected_extra_hits_10000/10000,t.boundary_identity_rhs,atol=1e-12)


def test_actual_two_candidate_consensus_removes_added_information():
    for mask in ([True,True,False],[False,True,True]):
        r=m.retained_oracles(mask)
        assert r["M"]==2 and r["range_valid"]
        assert r["information_max_abs"]<1e-12
        assert abs(r["conditional_information_nats"])<1e-12
        assert abs(r["oracle_expected_extra_hits_10000"])<1e-12
    assert not m.retained_oracles([False,False,True])["range_valid"]


def test_validation_gate_is_inclusive_and_never_empty():
    np.testing.assert_array_equal(m.eligibility([.8,.7900000,.789998]),[True,True,False])
    assert m.eligibility([.7,.8,.9]).sum()==1
    with pytest.raises(ValueError):m.eligibility([.5,np.nan,.6])


def test_simulation_paired_identities_and_reproducible_streams():
    q,c,v=m.simulate(1100192)
    q2,c2,v2=m.simulate(1100192)
    assert q==q2 and c==c2 and v==v2
    for r in c:
        assert r["delta_capture"]==pytest.approx(r["delta_position"]+r["delta_within"])
        assert r["delta_hits"]==r["entering_hits"]-r["leaving_hits"]
        assert r["entering_n"]==r["leaving_n"]
    for r in q:assert r["B"]==int(np.ceil(r["N"]*r["budget"]))


def test_completed_run_refuses_overwrite(tmp_path,monkeypatch):
    (tmp_path/"run_manifest.json").write_text("{}")
    monkeypatch.setattr(m,"OUT",tmp_path)
    with pytest.raises(FileExistsError,match="refusing overwrite"):m.run()


def test_source_change_requires_explicit_amendment(tmp_path,monkeypatch):
    spec=m.specification();original_source=tmp_path/"source.py";original_source.write_text("# changed")
    (tmp_path/"protocol.json").write_text(json.dumps(dict(spec_sha256=sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
        source_sha256="original",test_sha256=m.digest(m.ROOT/"tests/test_mechanism_v1_10.py"))))
    monkeypatch.setattr(m,"OUT",tmp_path);monkeypatch.setattr(m,"__file__",str(original_source))
    with pytest.raises(ValueError,match="explicit amendment"):m.verify_protocol()


def test_numerical_recovery_preserves_objective_on_failed_seed():
    s=m.design();seed=261020109
    streams=np.random.SeedSequence([seed,110]).spawn(5)
    recovered=[]
    for k,n in enumerate(m.TRAIN_SIZES):
        count=m.old.exact_counts(s,n)
        success=np.random.default_rng(streams[k]).binomial(count,s.q)
        for aug in (False,True):
            probability,record=m.fit_additive_guarded(s,count,success,aug)
            assert np.all((probability>0)&(probability<1))
            if record:
                recovered.append(record)
                assert record['gradient_inf']<=1e-7
                assert record['reference_objective_gap']<=1e-10
                assert record['reference_prediction_max_gap']<=1e-6
    assert recovered
