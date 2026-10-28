"""Qualified finite-support extension; v1.9 inputs are immutable dependencies.

The deterministic design responds to a known manuscript gap. It is not a
preregistration. Freeze design/code/seeds before label repetitions; never
silently overwrite a completed run or refit a design after inspecting results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.special import xlogy, expit
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import mechanism_v1_9 as old

OUT = ROOT / "results/tables/v1_10_mechanism"
FIG = ROOT / "paper/v1_10/figures"
SEEDS = tuple(range(261020000, 261020300))
TRAIN_SIZES = (8000, 80000)
N_EVAL = N_VALIDATION = 10000
BUDGETS = (.05, .10, .15, .20)
TARGETS = ("Pos", "FN", "FP", "AllErr")
CANDIDATE_NAMES = ("lower", "center", "upper")
AUC_GAP, AUC_TOL = .010, 1e-6
HISTORICAL = (
    ROOT / "src/mechanism_v1_9.py",
    ROOT / "tests/test_mechanism_v1_9.py",
    *(ROOT / "results/tables/v1_9_mechanism" / name for name in (
        "protocol.json", "run_manifest.json", "population_atoms.csv",
        "candidate_population_metrics.csv", "mechanism_primary_summary.csv",
        "simulation_comparison_summary.csv")),
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def design():
    """All three candidates pass population global .010 without a relaxed gate."""
    u = np.tile([-1., 1.], 4)
    mass = np.repeat(np.array([.6, .2, .1, .1]) / 2, 2)
    s = np.repeat([.04, .08, .11, .112], 2)
    return old.Scenario("E_qualified", "E", mass, s, s,
                        .0004 + .0002 * u, s + .0025 * u, .100, u,
                        "Population-qualified three-candidate design; Pos boundary changes at .10, not .20")


def eligibility(aucs):
    aucs = np.asarray(aucs, float)
    if aucs.shape != (3,) or not np.all(np.isfinite(aucs)):
        raise ValueError("Three finite candidate AUCs are required")
    return aucs + AUC_TOL >= aucs.max() - AUC_GAP


def retained_oracles(mask):
    """Recompute W=(S, ACTUAL retained consensus) and its added retained range.

    Two retained candidates need not preserve C=S. In this construction their
    consensus reveals U, leaving no additional information for retained range.
    """
    s = design()
    selected = s.candidates[:, np.asarray(mask, bool)]
    if selected.shape[1] == 0:
        raise ValueError("At least the empirical best candidate must survive")
    c, d = selected.mean(1), np.ptp(selected, axis=1)
    actual = old.Scenario(s.name, s.kind, s.mass, s.score, c, d, s.q,
                          s.tau, s.u, "Actual validation-retained information")
    q0, q1 = old.oracle_q(actual), old.oracle_q(actual, True)
    pi0, _ = old.top_fraction(q0, s.mass, .10)
    pi1, _ = old.top_fraction(q1, s.mass, .10)
    gain = float(np.dot(s.mass * (pi1 - pi0), s.q))
    mi = old.logloss_population(q0, s.q, s.mass) - old.logloss_population(q1, s.q, s.mass)
    return dict(M=selected.shape[1], range_valid=selected.shape[1] >= 2 and np.ptp(d) > 0,
                conditional_information_nats=float(mi),
                oracle_expected_extra_hits_10000=10000 * gain,
                oracle_capture_gain=gain / float(s.mass @ s.q),
                information_max_abs=float(np.max(abs(q1-q0))))


def exact_tables():
    s = design()
    q0, q1 = old.oracle_q(s), old.oracle_q(s, True)
    aucs = np.array([old.auc_population(p, s.q, s.mass) for p in s.candidates.T])
    base_brier = old.brier_population(s.score, s.q, s.mass)
    candidates = pd.DataFrame([dict(candidate=name, auc=aucs[j],
        auc_gap_from_best=aucs.max()-aucs[j], passes_global_auc_010=bool(eligibility(aucs)[j]),
        brier=old.brier_population(s.candidates[:, j], s.q, s.mass),
        brier_excess_vs_deployment=old.brier_population(s.candidates[:, j],s.q,s.mass)-base_brier)
        for j, name in enumerate(CANDIDATE_NAMES)])
    atoms = pd.DataFrame(dict(atom=np.arange(8), mass=s.mass, score=s.score,
        consensus=s.consensus, range=s.spread, u=s.u, q=s.q,
        candidate_lower=s.candidates[:, 0], candidate_center=s.candidates[:, 1],
        candidate_upper=s.candidates[:, 2], oracle_W=q0, oracle_WD=q1))
    info = old.logloss_population(q0,s.q,s.mass)-old.logloss_population(q1,s.q,s.mass)
    rows=[]
    for target in TARGETS:
        eta0,eta1=old.target_probability(q0,s,target),old.target_probability(q1,s,target)
        total=float(s.mass@eta1)
        for b in BUDGETS:
            pi0,_=old.top_fraction(eta0,s.mass,b)
            pi1,cut=old.top_fraction(eta1,s.mass,b)
            gain=float((s.mass*(pi1-pi0))@eta1)
            rows.append(dict(target=target,budget=b,target_probability=total,
                oracle_capture_W=float((s.mass*pi0)@eta1/total) if total else np.nan,
                oracle_capture_WD=float((s.mass*pi1)@eta1/total) if total else np.nan,
                oracle_capture_gain=gain/total if total else np.nan,
                oracle_expected_extra_hits_10000=10000*gain,
                exchange_mass=.5*float(s.mass@abs(pi1-pi0)),
                boundary_identity_rhs=float((s.mass*(pi1-pi0))@(eta1-cut)),
                conditional_information_nats=float(info)))
    return {"population_atoms":atoms,"candidate_population_metrics":candidates,
            "population_comparisons":pd.DataFrame(rows)}


def specification():
    s=design()
    return dict(schema="qualified-mechanism-v1.10.1",design_name=s.name,
        design=dict(mass=s.mass.tolist(),score=s.score.tolist(),consensus=s.consensus.tolist(),
                    spread=s.spread.tolist(),q=s.q.tolist(),tau=s.tau,u=s.u.tolist()),
        design_history="Analytical construction responding to v1.9 qualification failure; first feasibility calculation used alpha=.0025 and top-score gap=.002; one nonproduction-seed implementation smoke test checks identities only; no stochastic design search or fitted-result tuning",
        seeds=list(SEEDS),repetitions=len(SEEDS),train_sizes=list(TRAIN_SIZES),
        primary_train_size=8000,n_eval=N_EVAL,n_validation=N_VALIDATION,
        budgets=list(BUDGETS),targets=list(TARGETS),primary="Pos at .10; .20 same-design no-boundary comparison",
        sampling="Fixed support masses. SeedSequence([seed,110]) spawns train8000, train80000, validation, evaluation, row-order independent streams. Evaluation labels/order shared across training sizes.",
        learners="Saturated Beta(.5,.5) estimates and additive logistic ridge1e-6/maxiter500, unchanged v1.9 helpers; same W/WD learner family, no tuning",
        qualification="All three designed candidates pass exact population global AUC best-.010 (+1e-6 tolerance). Main fitted repetitions keep this population-qualified set fixed.",
        validation_secondary="Independent 10000-label validation per seed applies actual global .010+1e-6 gate. Recompute actual retained mean/range and exact W/WD risks. Report M, M<2 and added-information/boundary frequencies. No fitted learner refit on retained subsets, no selected validation seeds.",
        uncertainty="Mean +/-1.96 Monte Carlo SE and central empirical 95% label-repetition quantiles; neither is a population/training-distribution confidence interval",
        no_claims="No real-loan result, no new primary comparison family, no new theoretical originality, no guarantee of finite-validation eligibility",
        historical_sha256={str(p.relative_to(ROOT)):digest(p) for p in HISTORICAL})


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"protocol.json").exists():
        raise FileExistsError("Protocol exists; refusing overwrite")
    spec=specification()
    dump(OUT/"protocol.json",dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        status="local pre-simulation freeze; not preregistration",spec=spec,
        spec_sha256=sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
        source_sha256=digest(__file__),test_sha256=digest(ROOT/"tests/test_mechanism_v1_10.py")))
    print("Frozen qualified design, five independent streams and all 300 seeds.")


def verify_protocol():
    p=json.loads((OUT/"protocol.json").read_text())
    if p["spec_sha256"]!=sha256(json.dumps(specification(),sort_keys=True).encode()).hexdigest():
        raise ValueError("Design or historical dependency differs from protocol")
    ap=OUT/"source_amendments.json"
    amendments=json.loads(ap.read_text()) if ap.exists() else []
    if p["test_sha256"]!=digest(ROOT/"tests/test_mechanism_v1_10.py"):
        if not any(a.get("test_sha256")==digest(ROOT/"tests/test_mechanism_v1_10.py") and
                   a.get("protocol_sha256")==digest(OUT/"protocol.json") and a.get("reason") for a in amendments):
            raise ValueError("Frozen test source changed without amendment")
    if p["source_sha256"]!=digest(__file__):
        if not any(a.get("source_sha256")==digest(__file__) and a.get("reason") and
                   a.get("protocol_sha256")==digest(OUT/"protocol.json") for a in amendments):
            raise ValueError("Source changed without explicit amendment")
    return p


def fit_additive_guarded(s,counts,successes,augmented=False):
    """Same objective/ridge; numerical recovery only after old solver failure.

    An increased line-search budget is checked against independent BFGS and
    gradient/objective tolerances. This is not learner or regularizer tuning.
    """
    try:
        return old.fit_additive(s,counts,successes,augmented),None
    except RuntimeError as error:
        original_error=str(error)
    x=old.basis(s,augmented);n=counts.sum();ridge=1e-6
    def objective(beta):
        eta=x@beta
        value=np.sum(counts*np.logaddexp(0,eta)-successes*eta)/n+ridge*np.dot(beta[1:],beta[1:])/2
        gradient=x.T@(counts*expit(eta)-successes)/n
        gradient[1:]+=ridge*beta[1:]
        return value,gradient
    initial=np.zeros(x.shape[1]);initial[0]=np.log((successes.sum()+.5)/(n-successes.sum()+.5))
    recovered=minimize(objective,initial,jac=True,method="L-BFGS-B",
        options={"maxiter":500,"maxls":100,"ftol":1e-12,"gtol":1e-9})
    reference=minimize(objective,initial,jac=True,method="BFGS",
        options={"maxiter":500,"gtol":1e-10})
    grad=float(np.max(abs(objective(recovered.x)[1])))
    refgrad=float(np.max(abs(objective(reference.x)[1])))
    gap=float(abs(recovered.fun-reference.fun))
    predgap=float(np.max(abs(expit(x@recovered.x)-expit(x@reference.x))))
    if not np.isfinite(recovered.fun) or grad>1e-7 or refgrad>1e-7 or gap>1e-10 or predgap>1e-6:
        raise RuntimeError("Numerical recovery did not meet the fixed gradient/objective checks")
    return expit(x@recovered.x),dict(original_error=original_error,solver="L-BFGS-B maxls100; BFGS cross-check",
        objective=float(recovered.fun),gradient_inf=grad,reference_gradient_inf=refgrad,
        reference_objective_gap=gap,reference_prediction_max_gap=predgap)


def simulate(seed):
    s=design()
    rngs=[np.random.default_rng(z) for z in np.random.SeedSequence([seed,110]).spawn(5)]
    val_counts=old.exact_counts(s,N_VALIDATION)
    val_success=rngs[2].binomial(val_counts,s.q)
    val_aucs=np.array([old.auc_population(p,val_success/val_counts,s.mass) for p in s.candidates.T])
    mask=eligibility(val_aucs)
    validation=dict(seed=seed,retained="|".join(np.array(CANDIDATE_NAMES)[mask]),
        validation_positive_n=int(val_success.sum()),**retained_oracles(mask))
    for j,name in enumerate(CANDIDATE_NAMES):
        validation[f"auc_{name}"]=val_aucs[j]
        validation[f"retained_{name}"]=bool(mask[j])
    atoms=rngs[4].permutation(np.repeat(np.arange(8),old.exact_counts(s,N_EVAL)))
    y=rngs[3].binomial(1,s.q[atoms]).astype(bool)
    pred=s.score[atoms]>=s.tau
    targets=dict(Pos=y,FN=~pred&y,FP=pred&~y,AllErr=pred!=y)
    bins=old.group_ids(s.score)[atoms]
    qrows, contrasts=[],[]
    solver_records=[]
    for k,n_train in enumerate(TRAIN_SIZES):
        counts=old.exact_counts(s,n_train)
        successes=rngs[k].binomial(counts,s.q)
        models={"Oracle_W":old.oracle_q(s),"Oracle_WD":old.oracle_q(s,True),
            "Saturated_W":old.fit_saturated(s,counts,successes),
            "Saturated_WD":old.fit_saturated(s,counts,successes,True)}
        for suffix,augmented in (("W",False),("WD",True)):
            models["Additive_"+suffix],record=fit_additive_guarded(s,counts,successes,augmented)
            if record:solver_records.append(dict(seed=seed,n_train=n_train,information=suffix,**record))
        losses={}
        for key,q in models.items():
            p=np.clip(q[atoms],1e-12,1-1e-12)
            losses[key]=float(np.mean(-xlogy(y,p)-xlogy(~y,1-p)))
        for target,z in targets.items():
            orders={key:np.argsort(-np.round(old.target_probability(q,s,target)[atoms],12),kind="stable") for key,q in models.items()}
            for b in BUDGETS:
                B=int(np.ceil(b*N_EVAL));queues={key:o[:B] for key,o in orders.items()};values={}
                for key,queue in queues.items():
                    cap,pos,within=old.empirical_decomposition(queue,z,bins);hits=int(z[queue].sum())
                    values[key]=(cap,pos,within,hits)
                    qrows.append(dict(seed=seed,n_train=n_train,target=target,budget=b,policy=key,
                        N=N_EVAL,B=B,target_n=int(z.sum()),hits=hits,capture=cap,yield_rate=hits/B,
                        position=pos,within=within,logloss=losses[key]))
                for learner in ("Oracle","Saturated","Additive"):
                    a,base=learner+"_WD",learner+"_W"
                    va,vb=values[a],values[base]
                    qa,qb=queues[a],queues[base]
                    entering=np.setdiff1d(qa,qb);leaving=np.setdiff1d(qb,qa)
                    eta=old.target_probability(s.q,s,target)[atoms]
                    contrasts.append(dict(seed=seed,n_train=n_train,target=target,budget=b,
                        learner=learner,delta_capture=va[0]-vb[0],delta_position=va[1]-vb[1],
                        delta_within=va[2]-vb[2],delta_hits=va[3]-vb[3],target_n=int(z.sum()),
                        entering_n=len(entering),leaving_n=len(leaving),
                        entering_hits=int(z[entering].sum()),leaving_hits=int(z[leaving].sum()),
                        conditional_expected_delta_hits=float(eta[entering].sum()-eta[leaving].sum()),
                        logloss_gain=losses[base]-losses[a]))
    validation["solver_recoveries"]=json.dumps(solver_records,sort_keys=True)
    return qrows,contrasts,validation


def validate_aggregates(q,c,v):
    assert len(q)==len(SEEDS)*len(TRAIN_SIZES)*len(TARGETS)*len(BUDGETS)*6
    assert len(c)==len(q)//2 and len(v)==len(SEEDS)
    assert np.all(q.B==np.ceil(q.N*q.budget))
    assert np.allclose(q.capture,q.hits/q.target_n,equal_nan=True)
    assert np.allclose(c.delta_capture,c.delta_position+c.delta_within,equal_nan=True,atol=1e-12)
    assert np.all(c.delta_hits==c.entering_hits-c.leaving_hits)
    assert np.all(c.entering_n==c.leaving_n)
    assert c.groupby(["n_train","target","budget","learner"]).size().eq(len(SEEDS)).all()


def run():
    if (OUT/"run_manifest.json").exists():
        raise FileExistsError("Completed production run; refusing overwrite")
    verify_protocol()
    tables=exact_tables()
    for name,t in tables.items():t.to_csv(OUT/f"{name}.csv",index=False)
    queues,comparisons,validation=[],[],[]
    for i,seed in enumerate(SEEDS):
        q,c,v=simulate(seed);queues.extend(q);comparisons.extend(c);validation.append(v)
        if (i+1)%50==0: print(f"Completed {i+1}/{len(SEEDS)} seeds",flush=True)
    q,c,v=pd.DataFrame(queues),pd.DataFrame(comparisons),pd.DataFrame(validation)
    validate_aggregates(q,c,v)
    q.to_csv(OUT/"queue_repetitions.csv",index=False)
    c.to_csv(OUT/"comparison_repetitions.csv",index=False)
    v.to_csv(OUT/"validation_repetitions.csv",index=False)
    solver_records=[row for cell in v.solver_recoveries for row in json.loads(cell)]
    pd.DataFrame(solver_records).to_csv(OUT/"solver_recoveries.csv",index=False)
    old.summarize(c,["n_train","target","budget","learner"],
        ["delta_capture","delta_hits","conditional_expected_delta_hits","logloss_gain","entering_n"]).to_csv(OUT/"comparison_summary.csv",index=False)
    rates=[]
    for name,flag in [("M_1",v.M==1),("M_2",v.M==2),("M_3",v.M==3),
                      ("range_valid",v.range_valid),
                      ("positive_conditional_information",v.conditional_information_nats>1e-12),
                      ("positive_oracle_boundary_gain",v.oracle_capture_gain>1e-12)]:
        p=float(flag.mean())
        rates.append(dict(event=name,count=int(flag.sum()),repetitions=len(v),fraction=p,
                          monte_carlo_standard_error=np.sqrt(p*(1-p)/len(v))))
    pd.DataFrame(rates).to_csv(OUT/"validation_summary.csv",index=False)
    verify_protocol()
    dump(OUT/"run_manifest.json",dict(status="complete",completed_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(OUT/"protocol.json"),execution_source_sha256=digest(__file__),
        source_amendment_sha256=digest(OUT/"source_amendments.json") if (OUT/"source_amendments.json").exists() else None,
        repeats=len(SEEDS),training_sizes=list(TRAIN_SIZES),queue_rows=len(q),comparison_rows=len(c),
        solver_recovery_count=len(solver_records),
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        outputs={p.name:digest(p) for p in OUT.glob("*.csv")}))
    print("Production complete; all numeric output hashes recorded.")


def figures():
    verify_protocol()
    receipt=json.loads((OUT/"run_manifest.json").read_text())
    for name,h in receipt["outputs"].items():
        if digest(OUT/name)!=h:raise ValueError(f"Numeric result changed: {name}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
        "axes.labelsize":9,"axes.titlesize":10,"xtick.labelsize":9,"ytick.labelsize":9,
        "legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "axes.spines.top":False,"axes.spines.right":False})
    hist=pd.read_csv(ROOT/"results/tables/v1_9_mechanism/mechanism_primary_summary.csv")
    cm=pd.read_csv(ROOT/"results/tables/v1_9_mechanism/candidate_population_metrics.csv")
    current=pd.read_csv(OUT/"comparison_summary.csv")
    pop=pd.read_csv(OUT/"population_comparisons.csv")
    exact=pop[(pop.target=="Pos")&(pop.budget==.1)].iloc[0]
    fig,axes=plt.subplots(2,2,figsize=(6.3,7.2))
    fig.subplots_adjust(left=.11,right=.98,bottom=.23,top=.93,wspace=.42,hspace=.64)
    blue,orange,green="#0072B2","#D55E00","#009E73"
    names=["A_exact","B_boundary_primary","C_information_no_boundary","D_control_approximation"]
    labels=["A: no information","B: boundary moves","C: boundary fixed","D: weak control","E: qualified (new)"]
    ax=axes[0,0];ax.set_title("A  Qualification and information",loc="left",weight="bold",pad=15);ax.axis("off")
    ax.text(0,1.01,"Designed probe",weight="bold",transform=ax.transAxes)
    ax.text(.64,1.01,"AUC",weight="bold",ha="center",transform=ax.transAxes)
    ax.text(.98,1.01,"Info.¹",weight="bold",ha="right",transform=ax.transAxes)
    for i,label in enumerate(labels):
        y=.87-i*.16
        if i<4:
            p=cm[(cm.scenario==names[i])&cm.model.str.startswith("candidate_")]
            count=int(p.passes_population_global_auc_010.sum())
            info=float(hist[hist.scenario==names[i]].oracle_population_logloss_gain.iloc[0])
        else:count,info=3,float(exact.conditional_information_nats)
        color=green if count==3 else orange
        ax.text(0,y,label,transform=ax.transAxes,weight="bold" if i==4 else "normal")
        ax.text(.64,y,f"{count}/3",color=color,weight="bold",ha="center",transform=ax.transAxes)
        ax.text(.98,y,"yes" if info>1e-12 else "no",ha="right",transform=ax.transAxes)
    ax.text(0,-.06,"Population AUC gap ≤ 0.010\nB/C/D: only one passes; range degenerates.\n¹Extra outcome information given (S, C).",fontsize=9,
            transform=ax.transAxes,va="top",linespacing=1.4)

    ax=axes[0,1];ax.set_title("B  E: value depends on budget",loc="left",weight="bold",pad=15)
    pp=pop[pop.target=="Pos"].sort_values("budget")
    ax.plot(pp.budget*100,pp.oracle_capture_gain*100,"D-",color=green,markersize=5,linewidth=1.5)
    ax.axhline(0,color=".6",linewidth=.8)
    ax.set_xticks([5,10,15,20]);ax.set_xlim(4,21);ax.set_ylim(-.02,.29)
    ax.set_xlabel("Review budget (%)");ax.set_ylabel("Oracle capture gain (pp)")
    ax.text(.03,.97,"All 3 candidates pass\nPopulation AUC gap: 0.00900",transform=ax.transAxes,va="top",fontsize=9)
    ax.annotate("+1.5 expected hits\nper 10,000 cases",(10,float(exact.oracle_capture_gain)*100),xytext=(7.5,.12),
                fontsize=9,arrowprops=dict(arrowstyle="-",color=".4",lw=.7))
    ax.annotate("Same queue",(20,0),xytext=(14.5,.045),fontsize=9,
                arrowprops=dict(arrowstyle="-",color=".4",lw=.7))

    ax=axes[1,0];ax.set_title("C  E: estimated target capture",loc="left",weight="bold",pad=15)
    for i,n in enumerate(TRAIN_SIZES):
        r=current[(current.n_train==n)&(current.target=="Pos")&(current.budget==.1)&
                  (current.learner=="Saturated")&(current.metric=="delta_capture")].iloc[0]
        ax.hlines(i,r.label_replication_q025*100,r.label_replication_q975*100,color=".65",lw=2)
        ax.errorbar(r["mean"]*100,i,xerr=np.array([[r["mean"]-r.mc_mean_lo],[r.mc_mean_hi-r["mean"]]])*100,
                    fmt="o",color=blue,markersize=6,elinewidth=4,capsize=4)
        ax.text(.02,.94-i*.14,f"n={n:,}: mean {r['mean']*100:+.3f} pp",transform=ax.transAxes,fontsize=9)
    ax.axvline(0,color=".6",lw=.8)
    ax.axvline(float(exact.oracle_capture_gain)*100,color=green,linestyle="--",lw=1.2)
    ax.set_yticks([0,1],["8,000","80,000"]);ax.set_ylim(-.6,1.8)
    ax.set_ylabel("Training labels");ax.set_xlabel("G2 − G1 capture (pp), 10%")
    handles=[Line2D([0],[0],color=blue,marker="o",lw=3,label="Mean ± 1.96 MCSE"),
        Line2D([0],[0],color=".65",lw=2,label="Central 95% repetitions"),
        Line2D([0],[0],color=green,ls="--",label="Population oracle")]
    ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(-.12,-.27),frameon=False,handlelength=1.8)

    ax=axes[1,1];ax.set_title("D  Oracle and fitted mechanisms",loc="left",weight="bold",pad=15)
    h=hist.set_index("scenario")
    for i,name in enumerate(names+["E_qualified"]):
        if i<4:
            r=h.loc[name];oracle=r.oracle_population_capture_gain*100
            values=[(r.saturated_mean*100,r.saturated_mc_mean_lo*100,r.saturated_mc_mean_hi*100),
                    (r.additive_mean*100,r.additive_mc_mean_lo*100,r.additive_mc_mean_hi*100)]
        else:
            oracle=float(exact.oracle_capture_gain)*100;values=[]
            for learner in ("Saturated","Additive"):
                r=current[(current.n_train==8000)&(current.target=="Pos")&(current.budget==.1)&(current.learner==learner)&(current.metric=="delta_capture")].iloc[0]
                values.append((r["mean"]*100,r.mc_mean_lo*100,r.mc_mean_hi*100))
        ax.plot(oracle,i-.19,"D",color="black",markersize=4)
        for j,(mu,lo,hi) in enumerate(values):
            ax.errorbar(mu,i+j*.19,xerr=[[mu-lo],[hi-mu]],fmt=["o","s"][j],
                        color=[blue,orange][j],markersize=4,elinewidth=1.1,capsize=2)
    ax.axvline(0,color=".65",lw=.8);ax.set_yticks(range(5),["A  3/3","B  1/3*","C  1/3*","D  1/3*","E  3/3"])
    ax.invert_yaxis();ax.set_xlabel("Capture gain (pp), 10%");ax.set_xlim(-.8,4)
    ax.legend(handles=[Line2D([0],[0],marker="D",color="black",ls="",label="Oracle"),
        Line2D([0],[0],marker="o",color=blue,ls="",label="Saturated"),
        Line2D([0],[0],marker="s",color=orange,ls="",label="Additive")],
        loc="upper left",bbox_to_anchor=(-.10,-.27),frameon=False,handlelength=1.3)
    fig.text(.11,.015,"300 label repetitions; fixed supports. *Fails multi-candidate AUC eligibility.\nFitted ensembles are fixed; Monte Carlo precision is not prospective uncertainty.",fontsize=9,va="bottom")
    FIG.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","svg","png"):
        fig.savefig(FIG/f"mechanisms_v1_10.{ext}",dpi=300,facecolor="white")
    plt.close(fig)
    dump(OUT/"figure_manifest.json",dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(__file__),run_manifest_sha256=digest(OUT/"run_manifest.json"),
        dimensions_inches=[6.3,7.2],minimum_font_pt=9,png_dpi=300,
        figures={str((FIG/f"mechanisms_v1_10.{e}").relative_to(ROOT)):digest(FIG/f"mechanisms_v1_10.{e}") for e in ("pdf","svg","png")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["design","freeze","run","figures"])
    args=parser.parse_args()
    if args.stage=="design":
        for name,table in exact_tables().items():print(name,"\n",table.to_string(index=False))
    else:globals()[args.stage]()
