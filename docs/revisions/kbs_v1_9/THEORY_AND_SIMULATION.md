# E1: 机制、信息与固定预算边界

2026-10-19。依据用户提供的《KBS v1.8 第一性原理修改执行方案》第 4、7.1 节。作者使用本地执行前冻结协议；这不是公开预注册，不追溯改变历史分析的状态。只新增独立脚本、测试、聚合结果及图，保留历史 300-seed 结果。

## 1. 理论核验与适用范围

### 四组反例

组质量为 (.80,.05,.05,.10)，参考分数为 (.020,.039,.041,.200)，真实风险等于分数。阈值 .040，预算 .10。候选分别为 S 和 S ± (.001,.002,.002,.003)。各候选四组排序相同，因此总体 AUC 相同；不声称候选均校准或 Brier risk 相同。Threshold 选 B/C，Range 选 D；两者不重合。错误总质量 .1459，Threshold/Range 的错误质量 .0499/.0800。EM 只位于 B/C。完整算术由代码与测试复核，生成 `population_atoms.csv`、`candidate_population_metrics.csv` 和 `population_metrics.csv`。

因为所有候选及 D 都由 S 决定，给定 S 后没有新增 outcome 信息。这不等于边际独立，也不表示真实模型分歧普遍无用。表中 population capture 是概率质量的比率，不是有限样本随机 capture 的期望。有限样本收敛另用独立标签模拟显示。

### 信息无增益命题

对固定已拟合系统，若给定整批可用信息后 E[Z_i | W_1,D_1,...,W_N,D_N] = r(W_i)，则任何大小 B、仅依赖这些信息的名单的条件期望命中为所选 r(W_i) 的和，至多为最大的 B 项之和。单例条件等式 P(Z=1|W,D)=P(Z=1|W) 在独立案例及适当条件独立假设下推出这一批量条件；不能忽略案例间依赖后直接推广。

这是一项标准排序决策事实，不宣称原创理论。它针对指定 target；EM 或由 D 定义的 target 必须另外检验条件等式。它不保证两个有限拟合器相同，亦不使有限分箱成为精确条件独立检验。

若当前固定 N 个案例上 max_i |eta_hat_i-eta_i| <= epsilon，则 oracle top-B 相对拟合 top-B 的期望命中差不超过 2B epsilon：在两组名单之间插入 eta_hat，拟合排序差项非正，另两项各至多 B epsilon。该条件是理论前提，不由 cross-fitting 自动保证。仅对同一正的条件期望 target 总数，可以除以分母得到 ratio-of-expectations 界；没有把它用于 realized capture 的随机分母。

### 预算边界命题

令 r1=E[Z|W,D]、r0=E[r1|W]，pi0/pi1 为各自期望预算 b 的最优总体选择规则，允许并列随机化。对 r1 的预算截点 c1，

V(pi1)-V(pi0) = E[(pi1-pi0)(r1-c1)] >= 0。

等式利用相同预算；非负来自 r1 高于/低于截点时 pi1 分别为 1/0。无边界并列且两策略为 0/1 时，等于 E[|r1-c1| |pi1-pi0|]。有条件信息不保证改变给定 budget 的名单；拟合模型不享有此 oracle 不劣保证。总体并列使用统一随机化，不能按未知 U 的顺序人为给 W-only oracle 优惠。

### 位置分解

同一 target 分母和固定 score bins 下 C=C_pos+C_within；两队列相减得 Delta_C=Delta_pos+Delta_within。是描述性恒等式，不是因果分解。E1 使用预先给定的有限 score 支持点作为精确 bins；不冒称验证了历史 50/100-bin 方法的精确条件检验效力。空 target 的 capture 及分解为 undefined。

## 2. 执行前固定协议

- 正式重复：300，seed 从 261019000 到 261019299；6 情景共 1,800 次。机制均值是主要目标；在 p=.05 时拒绝率 MCSE 约 .01258，95% 正态半宽约 .02466，故不会声称精确校准了 5% 检验。
- 每情景每次：8,000 条开发标签、10,000 条独立评价标签。每个支持点人数固定等于预设质量，训练/评价 Bernoulli 标签使用独立随机流。评价行随机排列形成独立于标签的稳定 tie IDs。重复衡量固定设计下标签及估计变异，**不**包含随机 cohort 构成、候选训练搜索或真实环境变异。
- 预算 .05/.10/.15/.20。Pos、10% 是预算边界说明的主目标；FN、FP、AllErr 同时报，EM 只作诊断，且明确包含 candidate flip。所有队列人数为 ceil(bN)，targets 和分母保持共同。
- A_exact：方案四组反例。A_independent_jitter：D 在相同 S 下乘以 1±.10，Y 条件独立于此抖动；headline 反转仍可保持。
- B：S=(.04,.08,.11,.12)，质量 (.6,.2,.1,.1)，U=±1 等概率；q=S+alpha U，alpha=.010/.025。对称候选的 arithmetic consensus=S，D=.004+.002U。先由 oracle 检查额外信息和 10% Pos 边界交换，再做标签模拟。
- C：S=(.02,.04,.06,.20)，同样质量；q=S+.005U 仅作用于前三组，最高风险 .20 组不变。10% 时各 outcome target 的 oracle 入选集合相同；15% Pos 可以在入选边缘利用信息。此性质由构造与确定性测试检查，不按随机结果事后分类。
- D：S 与 consensus 构成 2×2 支持，q=logistic(-3+2 interaction)，range 由 interaction 决定。已有 W=(S,consensus) 完全决定 D，因此不存在额外 oracle 信息。相同 learner 的弱 additive W/WD 与正确表达 joint 支持的 saturated W/WD 对照，用来区分函数近似收益和额外信息。
- 学习器：有限支持的 saturated 条件频率，Beta(.5,.5) 平滑；弱加性 logistic 固定 ridge=1e-6、maxiter=500。同一 W/WD 对照仅改变输入，不调参。风险拟合后统一转换为 target probability 再 top-B。
- 另做总体层面 tau=.025/.040/.100 与阈值附近质量 .10/.20 的小型预设敏感性；不把每一格变成新的经验验证数据集。解析反例有限样本收敛 N=2,000/10,000/50,000，每规模同样 300 个独立标签重复。
- 对所有设计候选报告真实总体 AUC、Brier、相对最优 AUC 差和 .010 标记；**不**默认这些机制探针满足真实候选筛选，也不在看结果后删除不合格候选。不同情景不是“等质量候选”的普遍验证。
- 输出均值、MCSE、均值 ±1.96 MCSE，以及经验标签重复 2.5%/97.5% 分位数；前者是 Monte Carlo 均值精度，后者是设计下重复变异，均不能冒充真实未来人群置信区间。每次同时计算配对差、交换人数/命中、位置分量、log-loss。
- 不新增正式条件独立检验，也不对零显著结果宣称无信息。保留历史 low-prevalence Pos 的 50-bin 23/300=.0767 和 100-bin 双侧 22/300=.0733；历史文件 hash 在协议与运行后复核。

## 3. 可复用入口

```text
.venv/bin/python -m pytest tests/test_mechanism_v1_9.py -q
.venv/bin/python src/mechanism_v1_9.py --freeze
.venv/bin/python src/mechanism_v1_9.py --run
```

`--freeze` 拒绝覆盖已有协议；`--run` 要求当前协议及历史文件 hashes 完全匹配，且拒绝覆盖已有完成的 production manifest。源文件与冻结 hash 不同必须有显式 `source_amendments.json` 记录；未记录的统计实现改动将拒绝执行。每次运行写入 source hash、环境版本、产出 hash 和完成状态。脚本没有贷款输入、外部请求、paper 改写或旧结果覆盖路径。当前结果已完成，因此原样再调用 `--run` 会明确拒绝；复现须先在独立副本中保存本次协议及全部输出，避免覆盖已发表/已引用结果。

## 4. 执行状态与实际结果

正式运行已完成：6 情景各 300 次，共 1,800 次；288,000 行策略指标和 144,000 行配对比较。另有 3 个规模共 900 次解析反例收敛重复。九项理论/实现及防覆盖测试通过；历史四个 CSV 的 hash 与冻结协议一致。原始协议时间与代码 hash、生产运行 hash 和聚合产物 hash 均保存在 `protocol.json` 与 `run_manifest.json`。正式结果之后缩短图标签、增加聚合桥接表，并按独立代码复核加入“已完成运行禁止覆盖”和“源变更需显式 amendment”护栏；没有改变任何模拟参数或结果值。源变更原因及前后 hash 单独记入 `source_amendments.json`。

### 主要结果（Pos、10%预算；capture差以百分点计）

| 情景 | 精确总体 oracle capture 差 | 总体 oracle log-loss 改善 | 饱和 joint 拟合均值差 | 弱加性拟合均值差 |
|---|---:|---:|---:|---:|
| A 条件独立抖动 | 0 | 0 | 0 | 0 |
| B 弱信息 alpha=.010 | 0.79365 | 0.001024 | 0.5640 | 0.7406 |
| B 主信息 alpha=.025 | 3.17460 | 0.006731 | 3.1906 | 3.1576 |
| C 有信息、不动10%边界 | 0 | 0.000474 | 0 | 0 |
| D 控制近似不足 | 0 | 0 | 0 | 2.2358 |

B 主情景饱和 joint 拟合均值的 Monte Carlo 区间为 [2.9442,3.4371] 个百分点；单次标签重复的中央 95% 范围为 [-1.4103,7.2902] 个百分点，不能混用这两种精度。B oracle 每 10,000 个案例交换 500 人，期望增加 20 个 positive hits。B 弱信息 oracle 交换相同人数，期望只增加 5 个 hits，说明交换人数不直接等于收益。

C 在 10% 下所有 300 次 Pos oracle/拟合配对均相同；oracle log-loss 的样本均值仍改善 0.000466 nats。15% 下同一总体 oracle capture 增益为 0.54348 个百分点，而 5%、10%、20% 均为零。这里“有信息”来自已知生成模型及总体风险函数，不能用任意实证不显著的 capture 结果倒推无信息。

D 的 D 是 W=(S,consensus) 的确定函数，oracle 与饱和 joint 模型的增加 D 结果严格相同；弱加性拟合却平均增加 2.2358 个百分点（MC 均值区间 [1.5631,2.9084]）。总体弱模型的伪真极限在其完全并列时采用均匀随机化，得到 9.5144 个百分点差；有限拟合的随机系数会打破并列，故不把这个总体弱极限差当成有限拟合均值或其收敛保证。A 的弱模型也可因重新表达同一个 score 函数而改善 log-loss，但没有 oracle 信息增益。

四组反例的精确 AUC 为 .7292317708333333；参考 Brier=.0355199，两个扰动候选各高 .0000021。总体 all-error capture 为 34.2015079%（Threshold）与 54.8320768%（Range），EM 为 100% 与 0%。固定设计 N=50,000 的 300 次标签重复均值为 34.2031% 与 54.8310%；这仅显示有限样本近似总体构造，不是新的真实数据验证。

### 候选资格的实际限制

`candidate_population_metrics.csv` 显示：A 两种情景下所有三候选及参考部署都处于最佳总体 AUC 的 .010 范围。**B、C、D 只有 upper 候选通过该统一门槛**，参考部署/中心/下候选未通过。因此 B/C/D 是揭示信息与边界、模型近似差异的设计探针，不能宣称为同质量 Rashomon 候选集合中的一般验证；若把 .010 gate 施加到它们，只剩一个候选，range 退化。该限制在协议阶段已经承诺公开，结果没有通过换资格阈值或筛掉不利情景进行修补。

### 文件与未完成事项

- `mechanism_primary_summary.csv`：主文可直接引用的精确总体/有限拟合并排摘要。
- `population_atoms.csv`、`population_metrics.csv`、`population_comparisons.csv`：生成模型、所有 target/budget 的真值、分解和边界量。
- `simulation_queue_repetitions.csv`、`simulation_comparison_repetitions.csv`：全部 seed 级策略与配对指标；对应 `*_summary.csv` 为派生摘要。
- `population_threshold_mass_sensitivity.csv`：预设阈值/边界质量敏感性；改变阈值时目标随之改变。
- `counterexample_convergence_*.csv`：总体反例的有限标签近似。
- `historical_simulation_summary_unchanged.csv`：历史 300-seed/50–100 bins 结果带来源拷贝，未重跑、未优化拒绝率。
- `mechanism_information_boundary.pdf/png`：已目视检查的三面板机制图；图中误差线是 MC 均值精度，非个体研究的95%效应CI。

未做：新的真实贷款拟合/独立评价、随机 cohort 构成敏感性、正式条件独立检验效力校准、统一 .010 gate 后仍保留多个信息候选的 B/C/D 构造、人工审核/机构收益实验、理论原创性证明。E1 的 oracle 事实与固定设计标签模拟均不能替代这些事项。
