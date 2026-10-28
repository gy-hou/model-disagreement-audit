"""Presentation-only v1.10 figures from frozen, aggregate v1.9 evidence.

No credit/audit fitting, queue construction, sampling, or bootstrap is performed.
Every plotted numeric cell is exported, with source and output hashes and checks.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.text import Text
from matplotlib.ticker import NullLocator, PercentFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper/v1_10/figures'
OUT = ROOT / 'results/tables/v1_10_presentation'
REV = ROOT / 'results/tables/v1_9_revision'
REBUILD = ROOT / 'results/tables/v1_9_candidate_rebuild'
BASES = ['freddie_mac_2018', 'fannie_mae_2018']
COHORTS = [f'{b[:-4]}{y}' for b in BASES for y in [2021, 2022, 2023]]
SEEDS = [2026101901, 2026101902, 2026101903]
BLUE, ORANGE, GREEN, PURPLE = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
INK, MUTED, GRID = '#203344', '#56616A', '#DCE2E7'
SOURCES, OUTPUTS, CHECKS = {}, {}, {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return path


def read(path):
    return pd.read_csv(source(path))


def export(frame, name):
    path = OUT / (name + '.csv')
    frame.to_csv(path, index=False, float_format='%.17g')
    OUTPUTS[str(path.relative_to(ROOT))] = sha(path)


def label(c):
    return ('Freddie' if c.startswith('freddie') else 'Fannie') + ' ' + c[-4:]


def style(ax, xgrid=True):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#9AA6AE')
    ax.tick_params(axis='y', length=0)
    if xgrid:
        ax.grid(axis='x', color=GRID, lw=.55)
    ax.set_axisbelow(True)


def save(fig, stem):
    # Fixed physical canvas: do not crop to a different final print width.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        if bounds.x0 < -.001 or bounds.y0 < -.001 or bounds.x1 > 1.001 or bounds.y1 > 1.001:
            clipped.append(artist.get_text())
    assert not clipped, f'Text outside canvas in {stem}: {clipped}'
    for ext in ['pdf', 'svg', 'png']:
        path = FIG / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, facecolor='white')
        OUTPUTS[str(path.relative_to(ROOT))] = sha(path)
    CHECKS[stem] = {'width_in': float(fig.get_figwidth()),
                    'height_in': float(fig.get_figheight()), 'png_dpi': 300,
                    'visible_text_inside_canvas': True}
    plt.close(fig)


def protocol():
    source(ROOT / 'paper/v1_9/design_details.tex')
    source(ROOT / 'docs/revisions/kbs_v1_9/data_roles.json')
    rows = [
        ('2018 training', 30000, 'Credit models and preprocessing'),
        ('2018 validation', 10000, 'Candidate eligibility, threshold, historical calibration'),
        ('2018 former test', 10000, 'Five-fold audit-model and policy development'),
        ('2021–2023 historical cohorts', None, 'Frozen-system retrospective evaluation')]
    export(pd.DataFrame(rows, columns=['data_role', 'cases_per_institution', 'actual_use']), 'figure1_data_roles')
    fig = plt.figure(figsize=(6.3, 3.25)); ax = fig.add_axes([.015, .02, .97, .96])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    def box(x,y,w,h,title,body,color=BLUE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.012',
                                  facecolor='#F4F7F9', edgecolor=color, lw=.9))
        ax.text(x+.014,y+h-.027,title,ha='left',va='top',fontsize=9.3,fontweight='bold',color=INK)
        ax.text(x+.014,y+h-.115,body,ha='left',va='top',fontsize=9,linespacing=1.5,color=INK)
    box(.015,.67,.285,.29,'Queue differences','Target and budget\nScore region\nQueue composition')
    box(.355,.67,.285,.29,'Mechanisms','Conditional information\nBudget boundary\nLearner approximation')
    box(.695,.67,.285,.29,'Incremental value','Matched G0 / G1 / G2\nFrozen policy choice\nCandidate reconstruction')
    for x in [.315,.655]:
        ax.annotate('',xy=(x+.028,.82),xytext=(x,.82),arrowprops={'arrowstyle':'->','color':MUTED,'lw':1})
    ax.text(.015,.59,'MORTGAGE DATA ROLES  ·  2018 counts per institution',fontsize=9,fontweight='bold',color=INK)
    box(.015,.195,.285,.335,'2018 credit models','Train: 30,000\nValidation: 10,000\nModels, candidates, τ')
    box(.355,.195,.285,.335,'2018 audit models','Former test: 10,000\nFive-fold choice\nFull-data refit')
    box(.695,.195,.285,.335,'2021–2023 transfer','Six historical cohorts\nFixed models / policies\nTarget hits / capture')
    for x in [.315,.655]:
        ax.annotate('',xy=(x+.028,.385),xytext=(x,.385),arrowprops={'arrowstyle':'->','color':MUTED,'lw':1})
    ax.text(.5,.075,'Three questions: different cases  |  additional information  |  extra target hits',
            ha='center',va='center',fontsize=9,color=INK)
    save(fig,'protocol_v1_10')


def historical():
    strategies = ['ThresholdProb', 'Range', 'RiskTop', 'AboveTauLow', 'BelowTauTop']
    names = ['Threshold', 'Range', 'RiskTop', 'AboveTauLow', 'BelowTauTop']
    colors = [BLUE, ORANGE, MUTED, GREEN, PURPLE]
    markers = ['o', 's', '^', 'D', 'v']
    data = pd.concat([read(REV/'historical_2018_baselines'/b/'performance.csv') for b in BASES])
    data = data[data.strategy.isin(strategies) & data.target.isin(['AllErr','FN'])].copy()
    assert len(data)==20 and (data.bootstrap==1999).all()
    assert np.allclose(data.capture, data.hits/data.target_n)
    assert (data.B==1000).all() and (data.N==10000).all()
    data['endpoint']='historical provider-age'; data['queue_definition']='fixed strategy, same queue across targets'
    export(data,'figure2_historical_targets')
    fig, axes = plt.subplots(2,2,figsize=(6.3,5.0))
    fig.subplots_adjust(left=.18,right=.975,bottom=.105,top=.86,wspace=.6,hspace=.85)
    for j,b in enumerate(BASES):
        for i,t in enumerate(['AllErr','FN']):
            ax=axes[i,j]; g=data[(data.cohort==b)&(data.target==t)].set_index('strategy').loc[strategies]
            for y,(_,r),c,m in zip(range(5),g.iterrows(),colors,markers):
                ax.errorbar(100*r.capture,y,xerr=[[100*(r.capture-r.lower)],[100*(r.upper-r.capture)]],fmt=m,color=c,ms=4.3,lw=1,capsize=2)
                ax.text(1.01,y,f'{int(r.hits):,}',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8.8,color=c)
            ax.set(yticks=range(5),yticklabels=names if j==0 else [],ylim=(4.5,-.65),xlim=(-1,62),xticks=[0,20,40])
            ax.set_title(f'{chr(65+i*2+j)}  {label(b)}',loc='left',fontsize=9.5,fontweight='bold',pad=23)
            ax.text(0,1.035,f'{"All errors" if t=="AllErr" else "FN"} (T = {int(g.target_n.iloc[0]):,})',transform=ax.transAxes,fontsize=8.8)
            ax.text(1.01,1.035,'Hits',transform=ax.transAxes,fontsize=8.8,ha='right')
            ax.set_xlabel('Target capture (%)',fontsize=9)
            style(ax)
    save(fig,'historical_targets_v1_10')
    return data


def geometry(historical_data):
    old=ROOT/'results/tables/v1_4_mechanism'
    bins=read(old/'mechanism_plot_bins.csv'); spans=read(old/'mechanism_plot_spans.csv'); comp=read(old/'mechanism_plot_composition.csv')
    dec=pd.concat([read(REV/'historical_2018_baselines'/b/'decomposition.csv') for b in BASES])
    assert len(dec)==4
    assert np.allclose(dec.capture_difference,dec.position_difference+dec.within_difference,atol=1e-13)
    queues=['ThresholdProb','Range','RiskTop','AboveTauLow','BelowTauTop']
    comp=comp[comp.queue.isin(queues)].copy()
    assert np.allclose(comp[['FP','TP','FN','TN']].sum(axis=1),1)
    for _, r in historical_data.iterrows():
        g=comp[(comp.cohort==r.cohort)&(comp.queue==r.strategy)].iloc[0]
        expected_hits=(g.FP+g.FN if r.target=='AllErr' else g.FN)*r.B
        assert np.isclose(expected_hits,r.hits,atol=1e-8)
    for frame,name in [(bins,'geometry_bins'),(spans,'geometry_spans'),(comp,'queue_composition'),(dec,'signed_decomposition')]:
        export(frame,'figure3_'+name)
    fig=plt.figure(figsize=(6.3,7.05))
    gs=fig.add_gridspec(3,2,left=.185,right=.95,bottom=.13,top=.90,wspace=.50,hspace=.83,height_ratios=[1.05,1.08,1])
    for j,b in enumerate(BASES):
        ax=fig.add_subplot(gs[0,j]); g=bins[bins.cohort==b]; s=spans[spans.cohort==b]
        ax.fill_between(g.p,g.q25,g.q75,color=BLUE,alpha=.16)
        ax.plot(g.p,g['median'],color=BLUE,lw=1.3)
        ax.axvline(s.tau.iloc[0],ls='--',color=INK,lw=.9)
        for k,q in enumerate(['ThresholdProb','Range']):
            r=s[s.queue==q].iloc[0]
            ax.plot([r.lo,r.hi],[.94-.1*k]*2,color=[BLUE,ORANGE][k],lw=2.5,
                    transform=ax.get_xaxis_transform(),marker=['o','s'][k],ms=3)
        ax.set_xscale('logit'); ax.set_xlim(.0025,.4)
        ax.set_xticks([.003,.01,.04,.1,.3],['.003','.01','.04','.10','.30']); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Deployment score',fontsize=9); ax.set_ylabel('Candidate range' if j==0 else '')
        ax.set_title(f'{chr(65+j)}  {label(b)}',loc='left',fontweight='bold',fontsize=9.6,pad=8)
        ax.yaxis.set_major_locator(MaxNLocator(4)); style(ax)
        ax=fig.add_subplot(gs[1,j]); g=comp[comp.cohort==b].set_index('queue').loc[queues]; left=np.zeros(5)
        for col,color,hatch in [('FP',ORANGE,''),('TP',GREEN,'///'),('FN',BLUE,'xx'),('TN','#D9E0E6','')]:
            ax.barh(range(5),g[col],left=left,color=color,height=.64,label=col,hatch=hatch,edgecolor='white',linewidth=.4)
            left+=g[col].to_numpy()
        ax.set(yticks=range(5),yticklabels=['Threshold','Range','RiskTop','AboveTauLow','BelowTauTop'] if j==0 else [],ylim=(4.55,-.55),xlim=(0,1),xticks=[0,.5,1])
        ax.xaxis.set_major_formatter(PercentFormatter(1,0))
        ax.set_title(f'{chr(67+j)}  Queue composition',loc='left',fontweight='bold',fontsize=9.2,pad=8); style(ax)
        ax=fig.add_subplot(gs[2,j]); g=dec[dec.cohort==b].set_index('target')
        for k,(key,lo,hi,color,marker,name) in enumerate([
            ('capture_difference','lower','upper',INK,'o','Total'),
            ('position_difference','position_lower','position_upper',ORANGE,'s','Position'),
            ('within_difference','within_lower','within_upper',BLUE,'D','Within')]):
            for i,t in enumerate(['AllErr','FN']):
                r=g.loc[t]; y=i+(k-1)*.19
                ax.errorbar(100*r[key],y,xerr=[[100*(r[key]-r[lo])],[100*(r[hi]-r[key])]],fmt=marker,color=color,ms=4.2,capsize=2,lw=1)
        ax.axvline(0,color=MUTED,lw=.9,ls='--')
        ax.set(yticks=[0,1],yticklabels=['All errors','FN'] if j==0 else [],ylim=(1.6,-.55),xlim=(-38,17),xticks=[-30,-15,0,15])
        ax.set_xlabel('Capture difference (pp)',fontsize=9)
        ax.set_title(f'{chr(69+j)}  Signed decomposition',loc='left',fontweight='bold',fontsize=9.3,pad=8); style(ax)
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',lw=2,label='Threshold span'),Line2D([],[],color=ORANGE,marker='s',lw=2,label='Range span'),Line2D([],[],color=INK,ls='--',lw=1,label='Threshold τ')],loc='upper center',bbox_to_anchor=(.55,.995),ncol=3,fontsize=8.6,frameon=False,handlelength=1.5,columnspacing=1.1)
    fig.legend(handles=[Patch(facecolor=c,hatch=h,label=l,edgecolor='white') for l,c,h in [('FP',ORANGE,''),('TP',GREEN,'///'),('FN',BLUE,'xx'),('TN','#D9E0E6','')]],loc='center',bbox_to_anchor=(.55,.366),ncol=4,fontsize=8.8,frameon=False,handlelength=1.5)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,label=l,lw=1) for c,m,l in [(INK,'o','Total'),(ORANGE,'s','Position'),(BLUE,'D','Within')]],loc='lower center',bbox_to_anchor=(.55,.005),ncol=3,fontsize=8.8,frameon=False)
    save(fig,'score_geometry_v1_10')


def main_comparisons():
    main=read(REV/'main_historical_e2e3.csv')
    assert len(main)==24 and (main.family_size==24).all() and (main.bootstrap_valid==9999).all()
    perfs={c:read(REV/'exploratory_retrospective'/c/'global__additive/performance.csv').query('budget == .1').set_index(['target','strategy']) for c in COHORTS}
    rows=[]
    for _,r in main.iterrows():
        p=perfs[r.cohort]; s1,s0=r.contrast.split('-'); h1=int(p.loc[(r.target,s1)].hits); h0=int(p.loc[(r.target,s0)].hits)
        d=r.to_dict(); d.update(reference_hits=h0,augmented_hits=h1,remaining_hit_ceiling=min(int(r.B),int(r.target_n))-h0,
                               endpoint='historical provider-age',queue_definition='target-aligned')
        assert h1-h0==r.hits_difference
        assert np.isclose(r.capture_difference,(h1-h0)/r.target_n)
        assert d['remaining_hit_ceiling']>=0 and r.hits_difference<=d['remaining_hit_ceiling']
        rows.append(d)
    joined=pd.DataFrame(rows); export(joined,'main_comparison_display_values')
    return joined


def forest_rows(fig,rect,frame,xlim,xticks,header_y,title,columns=True):
    ax=fig.add_axes(rect); g=frame.set_index('cohort')
    cohorts=[c for c in COHORTS if c in g.index]
    for y,c in enumerate(cohorts):
        r=g.loc[c];color=BLUE if c.startswith('freddie') else ORANGE; marker='o' if c.startswith('freddie') else 's'
        x=100*r.capture_difference; lo=100*r.family_adjusted_lower;hi=100*r.family_adjusted_upper
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,ms=4.8,lw=1.2,capsize=2.5)
        yy=rect[1]+rect[3]*(1-(y+.6)/(len(cohorts)+.2))
        # Use the axis transform, so all table cells share exact row baselines.
        for pos,txt in [(1.18,f'{int(r.hits_difference):+d}'),(1.52,f'{int(r.target_n):,}'),(1.87,f'{int(r.remaining_hit_ceiling):,}')]:
            if columns: ax.text(pos,y,txt,transform=ax.get_yaxis_transform(),ha='center',va='center',color=INK,fontsize=9)
    ax.set(yticks=range(len(cohorts)),yticklabels=[label(c) for c in cohorts],ylim=(len(cohorts)-.4,-.6),xlim=xlim,xticks=xticks)
    if len(cohorts)==6:ax.axhline(2.5,color=GRID,lw=.8)
    ax.axvline(0,color=MUTED,lw=.9,ls='--');style(ax)
    ax.set_xlabel('Capture difference (percentage points)',fontsize=9)
    fig.text(.035,header_y,title,fontsize=10.2,fontweight='bold',color=INK)
    if columns:
        for pos,txt in [(1.18,'Δ hits'),(1.52,'Target N'),(1.87,'G1 room')]:
            ax.text(pos,1.06,txt,transform=ax.transAxes,ha='center',fontsize=8.8,color=MUTED)
    return ax


def incremental(joined):
    data=joined[joined.contrast=='G2-G1'].copy();export(data,'figure5_incremental_capture')
    fig=plt.figure(figsize=(6.3,6.6))
    fig.text(.035,.97,'G1: deployment score + consensus     G2: G1 + range',fontsize=9.2,color=INK)
    forest_rows(fig,[.185,.585,.405,.28],data[data.target=='AllErr'],(-3,3),[-3,-1.5,0,1.5,3],.91,'A  All errors')
    forest_rows(fig,[.185,.135,.405,.28],data[data.target=='FN'],(-42,42),[-40,-20,0,20,40],.47,'B  False negatives')
    fig.text(.035,.035,'G1 room = min(review budget, target N) − G1 hits; an observed-sample ceiling.',fontsize=8.7,color=MUTED)
    save(fig,'incremental_capture_v1_10')


def selection(joined):
    choices=read(REV/'development_policy_choices.csv')
    choices=choices.query("variant == 'global' and learner == 'additive' and budget == .1 and target in ['AllErr','FN'] and policy_set in ['P_base','P_full']")
    rows=[]
    for b in BASES:
        for t in ['AllErr','FN']:
            g=choices[(choices.base==b)&(choices.target==t)].set_index('policy_set')
            rows.append(dict(base=b,target=t,base_policy=g.loc['P_base'].selected_policy,full_policy=g.loc['P_full'].selected_policy,
                             base_oof_hits=int(g.loc['P_base'].development_hits),full_oof_hits=int(g.loc['P_full'].development_hits),
                             target_n=int(g.loc['P_base'].development_target_n)))
    display=pd.DataFrame(rows);display['same_choice']=display.base_policy==display.full_policy
    assert len(display)==4 and display.same_choice.sum()==3
    transfer=joined[joined.contrast=='P_full-P_base'].copy(); assert len(transfer)==12
    same_cells=0
    for _,r in display.iterrows():
        g=transfer[(transfer.base==r.base)&(transfer.target==r.target)]
        if r.same_choice:
            assert len(g)==3 and (g.hits_difference==0).all()
            assert (g.family_adjusted_lower==0).all() and (g.family_adjusted_upper==0).all()
            same_cells+=len(g)
    assert same_cells==9
    export(display,'figure6_development_choices');export(transfer,'figure6_frozen_transfer')
    fig=plt.figure(figsize=(6.3,4.75))
    ax=fig.add_axes([.025,.535,.95,.41]);ax.set(xlim=(0,1),ylim=(0,5.8));ax.axis('off')
    ax.text(0,5.65,'A  Four development choices at 10% capacity',fontsize=10,fontweight='bold',color=INK)
    xs=[.0,.30,.56,.85];heads=['Institution / target','Base choice','Full choice','OOF hits']
    for x,h in zip(xs,heads):ax.text(x,4.72,h,fontsize=8.8,color=MUTED,ha='left' if x==0 else 'center')
    for y,(_,r) in zip([3.95,3.05,2.15,1.25],display.iterrows()):
        if not r.same_choice:ax.add_patch(plt.Rectangle((-.005,y-.35),1.01,.72,color='#EDF5FA',zorder=-1))
        vals=[('Freddie' if r.base.startswith('freddie') else 'Fannie')+' / '+('All errors' if r.target=='AllErr' else 'FN'),r.base_policy,r.full_policy,f'{r.base_oof_hits} → {r.full_oof_hits}']
        for x,v in zip(xs,vals):ax.text(x,y,v,ha='left' if x==0 else 'center',va='center',fontsize=9,color=INK)
    ax.text(0,.22,'Same choice in 3 of 4 decisions → identical queues in 9 of 12 later comparisons.',fontsize=9,color=INK)
    d=transfer[(transfer.base==BASES[0])&(transfer.target=='AllErr')]
    ax=forest_rows(fig,[.19,.135,.44,.245],d,(-3,4),[-3,-1.5,0,1.5,3],.435,'B  Freddie all-error transfer: full − base',columns=False)
    for y,(_,r) in enumerate(d.set_index('cohort').loc[COHORTS[:3]].iterrows()):
        ax.text(1.22,y,f'{int(r.hits_difference):+d}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=9)
        ax.text(1.58,y,f'{int(r.reference_hits)} → {int(r.augmented_hits)}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=9)
    ax.text(1.22,1.08,'Δ hits',transform=ax.transAxes,ha='center',fontsize=8.8,color=MUTED)
    ax.text(1.58,1.08,'Base → Full hits',transform=ax.transAxes,ha='center',fontsize=8.8,color=MUTED)
    save(fig,'frozen_selection_v1_10')


def stability():
    ranges=read(REBUILD/'range_stability.csv');comparisons=read(REBUILD/'comparisons.csv')
    ranges=ranges[ranges.cohort.isin(COHORTS)].copy()
    assert len(ranges)==18 and len(comparisons)==36
    assert set(ranges.seed)==set(SEEDS)
    assert np.allclose(comparisons.capture_difference,comparisons.hits_difference/comparisons.target_n)
    assert (comparisons.hits_difference==comparisons.entering_hits-comparisons.leaving_hits).all()
    assert ((ranges.range_original_jaccard>=0)&(ranges.range_original_jaccard<=1)).all()
    assert (ranges.M>=2).all()
    export(ranges,'figure7_range_stability');export(comparisons,'figure7_incremental_hits')
    fig,axes=plt.subplots(2,3,figsize=(6.3,4.35))
    fig.subplots_adjust(left=.155,right=.975,bottom=.17,top=.86,hspace=.72,wspace=.38)
    for i,b in enumerate(BASES):
        for j in range(3):
            ax=axes[i,j]
            for si,(seed,color,marker) in enumerate(zip(SEEDS,[BLUE,ORANGE,GREEN],['o','s','^'])):
                for yi,year in enumerate([2021,2022,2023]):
                    c=f'{b[:-4]}{year}';frame=ranges if j==0 else comparisons
                    g=frame[(frame.cohort==c)&(frame.seed==seed)]
                    if j>0:g=g[g.target==['AllErr','FN'][j-1]]
                    assert len(g)==1
                    x=g.iloc[0]['range_original_jaccard' if j==0 else 'hits_difference']
                    ax.scatter(x,yi+(si-1)*.18,s=27,color=color,marker=marker,zorder=3)
            ax.set(yticks=range(3),yticklabels=[str(y) for y in [2021,2022,2023]] if j==0 else [],ylim=(2.48,-.5))
            if j==0:ax.set(xlim=(0,1),xticks=[0,.5,1],xlabel='Range Jaccard')
            else:
                limit=6 if j==1 else 3
                ax.set(xlim=(-limit,limit),xticks=[-limit+1,0,limit-1],xlabel='G2 − G1 Δ hits')
                ax.axvline(0,color=MUTED,ls='--',lw=.9)
            ax.set_title(f'{chr(65+i*3+j)}  '+['Queue stability','All errors','False negatives'][j],loc='left',fontsize=9.3,fontweight='bold')
            style(ax)
        axes[i,0].set_ylabel('Freddie' if i==0 else 'Fannie',fontweight='bold',labelpad=15,fontsize=9.3)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,ls='',label=f'Seed {s}') for s,c,m in zip(['1','2','3'],[BLUE,ORANGE,GREEN],['o','s','^'])],loc='upper center',bbox_to_anchor=(.55,.985),ncol=3,fontsize=9,frameon=False)
    fig.text(.155,.04,'Three fixed reconstructions; points are individual results, not confidence intervals.',fontsize=8.6,color=MUTED)
    save(fig,'candidate_stability_v1_10')


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                         'xtick.labelsize':8.8,'ytick.labelsize':8.8,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none','axes.titlecolor':INK,
                         'text.color':INK,'axes.labelcolor':INK})
    protocol(); historical_data=historical(); geometry(historical_data)
    joined=main_comparisons(); incremental(joined); selection(joined); stability()
    for path,digest in SOURCES.items():
        assert sha(ROOT/path)==digest, f'Source changed during presentation generation: {path}'
    assert all(np.isclose(x['width_in'],6.3) for x in CHECKS.values())
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
              'scope':'presentation only; no new estimates, fitted models, queues or bootstrap draws',
              'source_sha256':SOURCES,'output_sha256':OUTPUTS,'figures':CHECKS,
              'checks':['20 historical captures equal hits / target_n; 1999 original draws',
                        'four signed decompositions: total = position + within',
                        'queue composition fractions sum to one',
                        'all 20 historical composition / target hit counts agree across source generations',
                        '24 primary contrasts equal augmented hits - reference hits and divide by target N',
                        'remaining ceiling = min(B,T)-reference_hits and bounds observed improvement',
                        'three of four same choices imply nine of twelve identical-policy comparisons',
                        '36 reconstruction capture and membership hit identities',
                        '18 later reconstruction Jaccards in [0,1], each seed present',
                        'all source hashes unchanged after rendering','all canvases exactly 6.3 inches wide'],
              'renderer_sha256':sha(__file__)}
    (OUT/'FIGURE_VALIDATION.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Six presentation figures and derived CSVs rendered; all identity/source checks passed.')


if __name__=='__main__':
    main()
