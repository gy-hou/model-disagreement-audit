"""Builds the full 7-dataset report from 14_* outputs and applies the frozen decision rules (no estimation)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; R = ROOT / 'idea-update/results'
sys.path.insert(0, str(HERE)); from mdtable import md
P = json.loads((ROOT / 'idea-update/protocol_audit_discovery_v1.json').read_text())
m = pd.read_csv(R / '14_methods.csv'); c = pd.read_csv(R / '14_contrasts.csv'); ll = pd.read_csv(R / '14_logloss.csv')
st = pd.read_csv(R / '14_status.csv') if (R / '14_status.csv').stat().st_size > 1 else pd.DataFrame(columns=['dataset', 'issue'])
roles = pd.read_csv(R / '13_discovery_roles.csv').set_index('dataset')
out = []
# 1. overview
ov = []
for d in P['datasets']:
    mm = m[m.dataset == d]
    T = lambda t: int(mm[mm.target == t]['T'].iloc[0]) if len(mm[mm.target == t]) else None
    ov.append(dict(数据集=d, N=int(mm.N.iloc[0]) if len(mm) else None, B=int(mm.B.iloc[0]) if len(mm) else None,
                   AllErr事件数=T('AllErr'), FN事件数=T('FN'), M_global=int(roles.loc[d, 'M_global']), 冗余=roles.loc[d, 'redundancy'],
                   问题='; '.join(st[st.dataset == d].issue.astype(str)) or '无'))
out += ['### 1. 数据集概览', md(pd.DataFrame(ov))]
# 2. methods
for t in ('AllErr', 'FN'):
    x = m[m.target == t].copy(); x['cell'] = x.apply(lambda r: f"{r.hits} / {100*r.capture:.1f}% / {100*r.precision:.1f}%", axis=1)
    pv = x.pivot(index='dataset', columns='method', values='cell').reindex(P['datasets'])
    pv = pv[[k for k in ('G1', 'G2', 'SYM', 'ID', 'GX', 'G1add') if k in pv.columns]].reset_index().fillna('未定义')
    out += [f'### 2{"a" if t=="AllErr" else "b"}. 各方法结果（{t}，10% 预算；格内依次为 命中 / capture / 队列 precision）', md(pv)]
# 3. primary
def fmt(r, adj):
    lo, hi = (r.lo_adjusted, r.hi_adjusted) if adj else (r.lo_pointwise, r.hi_pointwise)
    return f"[{lo:+.0f}, {hi:+.0f}]"
pr = c[(c.role == 'primary')].copy()
rows = []
for _, r in pr.iterrows():
    if not r.defined: rows.append({'数据集': r.dataset, '对比': r.contrast, 'ΔH': '未定义'}); continue
    rows.append({'数据集': r.dataset, '对比': r.contrast, '参照命中': int(r.hits_ref), 'ΔH': int(r.d_hits), 'Δ每万人': round(r.d_hits_per10k, 2),
                 'ΔH/B': round(r.d_hits_per_B, 4), '剩余空间': int(r.remaining_capacity_ref), '逐点95%': fmt(r, False), 'Bonferroni-14': fmt(r, True),
                 '实质增益': r.meaningful_gain, '实质损失': r.meaningful_loss, '可排除': r.excluded,
                 'Δlog loss [95%]': f"{r.d_logloss:+.5f} [{r.d_logloss_lo:+.5f}, {r.d_logloss_hi:+.5f}]",
                 '冗余标注': 'G1 已可恢复，不可能反映新信息' if roles.loc[r.dataset, 'M_global'] == 2 and roles.loc[r.dataset, 'reference_in_global'] else ''})
out += ['### 3. 主比较：AllErr，10%，ID − SYM 与 ID − G1（14 项，Bonferroni-14）', md(pd.DataFrame(rows))]
# 4. secondary
se = c[c.role == 'secondary'].copy(); rows = []
for _, r in se.iterrows():
    if not r.defined: rows.append({'数据集': r.dataset, '目标': r.target, '对比': r.contrast, 'ΔH': '未定义'}); continue
    rows.append({'数据集': r.dataset, '目标': r.target, '对比': r.contrast, '参照命中': int(r.hits_ref), 'ΔH': int(r.d_hits),
                 'Δ每万人': round(r.d_hits_per10k, 2), '剩余空间': int(r.remaining_capacity_ref), '逐点95%': fmt(r, False),
                 '实质增益(逐点)': r.meaningful_gain, '实质损失': r.meaningful_loss, '可排除': r.excluded, 'Δlog loss': round(r.d_logloss, 5),
                 'log loss 区间<0': bool(r.d_logloss_hi < 0)})
out += ['### 4. 次要比较（探索性，逐点区间）', md(pd.DataFrame(rows))]
# 5. replication and decision inputs
cd = c[c.defined == True]
rep = cd.groupby(['contrast', 'target', 'role']).agg(实质增益数据集数=('meaningful_gain', 'sum'), 实质损失数据集数=('meaningful_loss', 'sum'),
                                                      可排除数据集数=('excluded', 'sum'), 已定义数据集数=('defined', 'sum')).reset_index()
rep['可复现增益(≥3且无损失)'] = (rep.实质增益数据集数 >= 3) & (rep.实质损失数据集数 == 0)
out += ['### 5. 可复现性检查（按事先写定的规则）', md(rep)]
def gain(d, con, t='AllErr'):
    x = cd[(cd.dataset == d) & (cd.contrast == con) & (cd.target == t)]
    return bool(len(x) and x.meaningful_gain.iloc[0])
lead_id = [d for d in P['datasets'] if gain(d, 'ID-SYM') and gain(d, 'ID-G1')]
lead_sym = [d for d in P['datasets'] if (gain(d, 'SYM-G1') or gain(d, 'SYM-G2')) and not gain(d, 'ID-SYM')]
gx_only = [d for d in P['datasets'] if gain(d, 'GX-G1') and not any(gain(d, k) for k in ('ID-SYM', 'ID-G1', 'SYM-G1', 'SYM-G2', 'G2-G1'))]
loss_only = sorted({r.dataset + ':' + r.contrast + ':' + r.target for _, r in cd.iterrows() if r.d_logloss_hi < 0 and not r.meaningful_gain})
wide = sorted({r.dataset + ':' + r.contrast for _, r in cd[(cd.role == 'primary')].iterrows() if not r.excluded and not r.meaningful_gain})
dec = dict(身份线索_规则1=lead_id, 对称表示线索_规则2=lead_sym, 仅GX改善_规则3=gx_only, 只改善log_loss_规则4=loss_only, 主比较中未排除也未增益=wide)
out += ['### 6. 判定规则的机械输入', '```\n' + json.dumps(dec, ensure_ascii=False, indent=1) + '\n```']
(R / '14_report.md').write_text('\n\n'.join(out)); print('\n\n'.join(out))
