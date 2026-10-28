"""Render v1.9 displays from recorded aggregate evidence, without refitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter, NullFormatter, NullLocator
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/v1_9'
OUT = ROOT / 'results/tables/v1_9_presentation'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (PAPER/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'pdf.fonttype':42,'ps.fonttype':42})
    provenance = {}
    def read(path):
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    fig,ax=plt.subplots(figsize=(7.0,3.8));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    boxes=[(.1,3.25,3.0,1.2,'1  FIX THE QUESTION\nTarget, budget and decision\nCandidate and label contracts'),
           (3.55,3.25,2.9,1.2,'2  DEVELOP\nMatched G0 / G1 / G2\nChoose on development labels'),
           (6.9,3.25,3.0,1.2,'3  FREEZE\nModel, policy index, budget\nSources, hashes and analysis plan'),
           (1.0,.4,3.6,1.35,'HISTORICAL TRANSFER\nPreviously inspected outcomes\nExploratory fixed-system comparison'),
           (5.4,.4,3.6,1.35,'INDEPENDENT EVALUATION\nUnused mature outcomes required\nNot established in this revision')]
    for i,(x,y,w,h,txt) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=('#f3f6fa' if i<4 else '#fff1df'),edgecolor=('#375d7a' if i<4 else '#b36a18'),lw=1))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8.1,linespacing=1.6)
    for a,b in [((3.2,3.85),(3.45,3.85)),((6.55,3.85),(6.8,3.85)),((8.4,3.1),(7.2,1.9)),((8.2,3.1),(2.8,1.9))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.2,'color':'#516170'})
    ax.text(5,2.45,'Provenance and outcome-observability gate',ha='center',fontsize=9,fontweight='bold',bbox={'facecolor':'white','edgecolor':'none','pad':4})
    ax.text(5,.03,'Outputs: target count • hits • capture • yield • paired differences • unresolved conditions',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(PAPER/'figures/current_protocol.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/current_protocol.png',dpi=200,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7,5.6),sharex=True)
    rows=[];signed=[]
    for ci,(cohort,label) in enumerate([('freddie_mac_2018','Freddie Mac'),('fannie_mae_2018','Fannie Mae')]):
        base=ROOT/'results/tables/v1_9_revision/historical_2018_baselines'/cohort
        data=read(base/'performance.csv').set_index(['target','strategy'])
        dec=read(base/'decomposition.csv').set_index('target')
        for ti,(target,tlabel,ref) in enumerate([('AllErr','All errors','AboveTauLow'),('FN','False negatives','BelowTauTop')]):
            ax=axes[ci,ti];n=int(data.loc[(target,'Range')].target_n);policies=['ThresholdProb','Range',ref]
            for j,(policy,color) in enumerate(zip(policies,['#2878a8','#d87831','#47826a'])):
                r=data.loc[(target,policy)];value=float(r.capture);lo=float(r.lower);hi=float(r.upper);h=int(r.hits)
                ax.errorbar(value,j,xerr=[[value-lo],[hi-value]],fmt='o',color=color,capsize=3,ms=6)
                ax.text(.67,j,f'{h}/{n}',ha='right',va='center',fontsize=8,color=color)
                rows.append(dict(cohort=cohort,target=target,policy=policy,N=int(r.N),B=int(r.B),target_n=n,hits=h,capture=value,lower=lo,upper=hi,hits_per_1000_reviews=float(r.hits_per_1000_reviews),endpoint='historical recorded provider-age endpoint',candidate_set='family-conditioned',interval='paired row bootstrap with queues rebuilt; 1999 replicates; frozen credit models'))
            ax.set_yticks(range(3),['Threshold','Range',ref]);ax.invert_yaxis();ax.set_ylim(2.6,-.6);ax.set_xlim(0,.69);ax.set_xticks([0,.1,.2,.3,.4,.5]);ax.set_title(f'{label}: {tlabel}',loc='left',fontsize=10)
            ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.xaxis.set_major_formatter(PercentFormatter(1,0))
            a=dec.loc[target]
            signed.append(dict(cohort=cohort,target=target,comparison='Range - ThresholdProb',capture_difference=a.capture_difference,position_difference=a.position_difference,within_difference=a.within_difference,lower=a.lower,upper=a.upper,position_lower=a.position_lower,position_upper=a.position_upper,within_lower=a.within_lower,within_upper=a.within_upper,scope='historical decomposition; fixed development quantiles; signed arithmetic identity, not causal'))
        axes[ci,1].text(.02,-.29,'Labels: hits / target count; B = 1,000',transform=axes[ci,1].transAxes,fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Target capture')
    fig.tight_layout(h_pad=2.3,w_pad=2);fig.savefig(PAPER/'figures/historical_target_counts.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/historical_target_counts.png',dpi=200,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'historical_target_counts.csv',index=False)
    pd.DataFrame(signed).to_csv(OUT/'historical_signed_decomposition.csv',index=False)
    assert np.allclose([x['capture_difference'] for x in signed],[x['position_difference']+x['within_difference'] for x in signed])
    (OUT/'input_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrr}\toprule',r'Cohort / target & Total (pp) & Position (pp) & Within (pp)\\\midrule']
    for r in signed:
        label=('Freddie' if r['cohort'].startswith('freddie') else 'Fannie')+' / '+r['target']
        def entry(key,low,high):
            return f"{100*r[key]:+.2f} [{100*r[low]:+.2f}, {100*r[high]:+.2f}]"
        lines.append(f"{label} & {entry('capture_difference','lower','upper')} & {entry('position_difference','position_lower','position_upper')} & {entry('within_difference','within_lower','within_upper')}"+r'\\')
    lines += [r'\bottomrule\end{tabular}']
    (PAPER/'generated/current_decomposition.tex').write_text('\n'.join(lines)+'\n')
    rev=ROOT/'results/tables/v1_9_revision'
    main=read(rev/'main_historical_e2e3.csv')
    cohorts=[f'{base}_{year}' for base in ['freddie_mac','fannie_mae'] for year in [2021,2022,2023]]
    contrasts=main.set_index(['cohort','target','contrast'])
    perf={c:read(rev/'exploratory_retrospective'/c/'global__additive/performance.csv').query('budget == 0.1').set_index(['target','strategy']) for c in cohorts}
    def cohort_label(c):return ('Freddie ' if c.startswith('freddie') else 'Fannie ')+c[-4:]
    def ci(r,adjusted=True):
        lo,hi=('family_adjusted_lower','family_adjusted_upper') if adjusted else ('lower','upper')
        return f'{100*r.capture_difference:+.2f} [{100*r[lo]:+.2f}, {100*r[hi]:+.2f}]'
    lines=[r'\begin{tabular}{llrrrrl}\toprule',r'Cohort & Target & $B$ & $T_t$ & G1 hits & G2 hits & $\Delta$ capture (pp)\\\midrule']
    for t in ['AllErr','FN']:
        for c in cohorts:
            r=contrasts.loc[(c,t,'G2-G1')];p=perf[c]
            lines.append(f'{cohort_label(c)} & {t} & {int(r.B):,} & {int(r.target_n):,} & {int(p.loc[(t,"G1")].hits)} & {int(p.loc[(t,"G2")].hits)} & {ci(r)}'+r'\\')
        if t=='AllErr':lines.append(r'\midrule')
    lines.append(r'\bottomrule\end{tabular}')
    (PAPER/'generated/current_incremental.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrrl}\toprule',r'Cohort & $T_{\rm Err}$ & Base hits & Full hits & $\Delta$ hits & $\Delta$ capture (pp)\\\midrule']
    for c in cohorts:
        r=contrasts.loc[(c,'AllErr','P_full-P_base')];p=perf[c]
        lines.append(f'{cohort_label(c)} & {int(r.target_n):,} & {int(p.loc[("AllErr","P_base")].hits)} & {int(p.loc[("AllErr","P_full")].hits)} & {int(r.hits_difference):+d} & {ci(r)}'+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    (PAPER/'generated/current_selection.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrl}\toprule',r'Cohort & Target / contrast & $T_t$ & $\Delta$ hits & $\Delta$ capture (pp)\\\midrule']
    for c in ['freddie_mac_2018','fannie_mae_2018']:
        a=read(rev/'historical_2018_baselines'/c/'paired_contrasts.csv').set_index(['target','contrast'])
        for t,ct in [('AllErr','Range-ThresholdProb'),('AllErr','AboveTauLow-Range'),('FN','BelowTauTop-Range')]:
            r=a.loc[(t,ct)];nice=ct.replace('ThresholdProb','Threshold').replace('-', '--')
            lines.append(f'{cohort_label(c)} & {t}: {nice} & {int(r.target_n):,} & {int(r.hits_difference):+d} & {ci(r,False)}'+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    (PAPER/'generated/current_strong_baselines.tex').write_text('\n'.join(lines)+'\n')
    historical=ROOT/'results/tables/v1_4_mechanism'
    bins=read(historical/'mechanism_plot_bins.csv');spans=read(historical/'mechanism_plot_spans.csv');comp=read(historical/'mechanism_plot_composition.csv')
    colors={'ThresholdProb':'#2878a8','Range':'#d87831'}
    fig,axes=plt.subplots(2,2,figsize=(7,5.8),layout='constrained',gridspec_kw={'width_ratios':[1,1]})
    for row,c in enumerate(['freddie_mac_2018','fannie_mae_2018']):
        ax=axes[row,0];g=bins[bins.cohort==c];ss=spans[spans.cohort==c]
        ax.fill_between(g.p,g.q25,g.q75,color='#9CBBCD',alpha=.45);ax.plot(g.p,g['median'],color='#2878a8')
        ax.axvline(ss.tau.iloc[0],ls='--',color='black',lw=.8)
        for off,(_,r) in enumerate(ss.iterrows()):
            ax.plot([r.lo,r.hi],[.94-.09*off]*2,color=colors[r['queue']],lw=3,transform=ax.get_xaxis_transform(),clip_on=True)
        ax.set_xscale('logit');ax.set(xlabel='Deployment score',ylabel='Candidate probability range',title=cohort_label(c),xlim=(max(.0001,min(g.p.min(),ss.lo.min())*.8),min(.99,max(g.p.max(),ss.hi.max())*1.1)))
        ax.set_xticks([.003,.01,.04,.1,.3],['.003','.01','.04','.10','.30'])
        ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax=axes[row,1];qs=['ThresholdProb','Range','RiskTop','AboveTauLow','BelowTauTop'];cc=comp[comp.cohort==c].set_index('queue').loc[qs];left=np.zeros(len(cc))
        for col,color in [('FP','#E69F69'),('TP','#9C293C'),('FN','#2C659E'),('TN','#B9CFE2')]:
            ax.barh(range(len(cc)),cc[col],left=left,color=color,height=.65);left+=cc[col].to_numpy()
        ax.invert_yaxis();ax.set(yticks=range(len(qs)),yticklabels=['Threshold','Range','RiskTop','AboveTauLow','BelowTauTop'],xlabel='Fraction of queue',xlim=(0,1));ax.tick_params(axis='y',labelsize=8);ax.spines[['top','right']].set_visible(False)
    fig.legend(handles=[Line2D([],[],lw=3,color=colors[q],label=('Threshold' if q=='ThresholdProb' else q)+' 5–95% score span') for q in colors],loc='outside upper center',ncol=2,frameon=False)
    fig.legend(handles=[Line2D([],[],lw=6,color=c,label=l) for c,l in [('#E69F69','FP'),('#9C293C','TP'),('#2C659E','FN'),('#B9CFE2','TN')]],loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(PAPER/'figures/current_score_geometry.pdf');fig.savefig(PAPER/'figures/current_score_geometry.png',dpi=200);plt.close(fig)
    rebuilt=ROOT/'results/tables/v1_9_candidate_rebuild'
    if (rebuilt/'EVALUATION_COMPLETE.json').exists():
        seed_results=read(rebuilt/'across_seed_summary.csv').set_index(['cohort','target'])
        stability=read(rebuilt/'range_stability_summary.csv').set_index('cohort')
        lines=[r'\begin{tabular}{lrrrr}\toprule',r'Cohort & Retained $M$ & Range Jaccard & AllErr $\Delta$ hits & FN $\Delta$ hits\\\midrule']
        for c in cohorts:
            s=stability.loc[c];a=seed_results.loc[(c,'AllErr')];f=seed_results.loc[(c,'FN')]
            mr=f'{int(s.M_min)}--{int(s.M_max)}' if s.M_min!=s.M_max else str(int(s.M_min))
            jr=f'{s.range_original_jaccard_min:.3f}--{s.range_original_jaccard_max:.3f}'
            ar=f'[{int(a.hits_min):+d}, {int(a.hits_max):+d}]';fr=f'[{int(f.hits_min):+d}, {int(f.hits_max):+d}]'
            lines.append(f'{cohort_label(c)} & {mr} & {jr} & {ar} & {fr}'+r'\\')
        lines.append(r'\bottomrule\end{tabular}')
        (PAPER/'generated/current_candidate_rebuild.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'input_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Rendered protocol and historical evidence; source hashes retained.')

if __name__=='__main__':main()
