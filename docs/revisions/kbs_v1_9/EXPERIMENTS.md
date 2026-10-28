# v1.9 E2/E3：冻结系统的探索性评价

## 状态和证据范围

本轮执行新增分析，但没有获得真正独立的 D_eval。用户确认 Freddie 2019/2020 曾被使用；Fannie 的未抽中贷款也不能仅凭未入 test 就认定其源标签未被分析。所有新实证结果明确为 **exploratory retrospective**。程序提供独立评价入口，但默认拒绝缺失或不通过的数据隔离、研究者暴露、标签时间和冻结范围证据。合成软件测试不充当独立真实数据。

原 2018 test 现在明确作为 C_audit。历史信用模型、2018 validation 上的候选资格及阈值完全保留；这一重用不追溯构成原始 A/B/C 的完整嵌套划分。主历史评价使用两个机构各 2021、2022、2023 年的六个旧 cohort。它们已经被研究者查看，不能称为未用测试或独立时间外验证。

原目标保留为历史 **recorded loan-age-based endpoint**，不能再直接称为已验证的固定 calendar 24-month outcome。Freddie 另有严格日历窗口和结局可判定性审计，其敏感性同时改变终点和可纳入人群，不是单独的标签修正因果效应。详细规则以 DATA_AUDIT.md 与 outcome contract 为准。

## 冻结顺序

1. `analysis_plan.json` 冻结目标、预算、候选资格、学习器、选择规则、随机种子、统计家族、数据角色、失败和空目标处理；`PLAN.sha256.json` 记录身份。
2. 只在 C_audit 做统一五折 OOF 和政策选择，再在全 C_audit 拟合审核风险模型，保存 `FROZEN_DEVELOPMENT.json`。
3. 评价之前收到尾部 Monte Carlo 精度复核，将主分析从 1,999 提升到 9,999 次配对重抽样。`pre_evaluation_amendment_01/` 保存原协议、代码和开发元数据；`PRE_EVALUATION_AMENDMENT_01.json` 记录时间及原因。没有在看到本轮评价结果后改变学习器、目标或选择规则。
4. 冻结完成后才执行旧 2018 基线再分析、历史六 cohort 评价和严格终点敏感性。

历史数值生成后，独立代码复核要求两项未执行入口保护和一项说明修正：独立输入强制候选身份及预测来源；严格终点敏感性不得继承原主家族；EM 下的共同 flip mask 必须说明。`implementation_amendment_02/` 保存历史实际执行版本，`IMPLEMENTATION_AMENDMENT_02.json` 记录更正。没有修改历史主分析数值、学习器、政策或目标；严格敏感性在这次更正后首次执行。各阶段 receipt 保留实际执行 plan hash，不把版本沿革抹平。

这证明本轮文件的执行顺序，不是历史公开 preregistration。

## G0/G1/G2 与候选标准

- G0：`logit(clip(p_dep,1e-6,1-1e-6))`。
- G1：G0 加 `logit(clip(mean_f p_f,1e-6,1-1e-6))`。consensus 是平均**概率**后取 logit，不是平均 logits。
- G2：G1 加候选 probability range。

主学习器对每个输入分别做 5 个 quantile knots 的 cubic spline，再 StandardScaler 和 L2 logistic（C=1，max_iter=4000，tol=1e-7）。G1/G2 的区别只有输入；没有额外调参预算。交互敏感性统一使用 HGBT：100 iterations、7 leaves、minimum leaf 100、learning rate .05、L2=1、128 bins、无 early stopping。各模型使用完全相同的五折和随机种子 20261019；两种学习器都没有超参搜索。OOF 仅用于开发选择，最终风险模型在全 C_audit 重拟合。

主候选从原始 26-model grid 按 global validation AUC best-.010（数值容忍 1e-6）筛选。Freddie 主集合仅 **2 个 EBM 候选**；Fannie 为 **9 个候选**，须据 registry 报告 family，而不是仍称四族主集合。原 family-conditioned 集合 13/20 个候选作为敏感性。未新增 Brier gate，未声称经验 AUC 带意味着统计等价。本报告的主固定系统分析没有重新训练候选模型。其后另行登记并完成三固定 seed 的候选重建，属于看到主结果后追加的探索性分析，不改变原主比较家族；详见同目录 `CANDIDATE_REBUILD.md`。

## 政策选择与评价对象

P_score = ThresholdProb、RiskTop、AboveTauLow、BelowTauTop、G0。
P_base = P_score + G1。
P_full = P_base + Range + G2。

每个 target/budget 在 pooled OOF 人群上选择命中数最大者；并列按上述预定顺序偏向较早策略。目标为空时使用明确记录的 ThresholdProb fallback，不伪造目标收益。随后固定策略索引，不依据评价标签更换赢家。全部真实队列包含 `ceil(b*N)` 个不同 ID；单侧不足时按另一侧离阈值的次序补足。

主目标 AllErr/FN；预算 .10；.05/.20 为敏感性。FP/Pos 和 EM 为诊断。EM 本身依赖候选标签分歧；其目标对齐模型共同使用已知 candidate-flip mask，故 EM 下同名 G0/P_score 不代表仅部署分数的信息集合。纯 score / consensus / disagreement 增量解释限定在 AllErr/FN/FP/Pos。EM_fixed 保持原 family 集合定义，EM_variant 使用相应候选集合，分母分别报告。

主 24 项家族 = 6 个机构×年份 cohort × 2 targets × 2 contrasts（G2−G1、P_full−P_base），限 global + additive + 10% budget。它们不是 24 个独立总体。G1−G0、P_base−P_score、对固定 Threshold/Range 的比较和各敏感性单独作为描述。

## 配对区间与分解

每个 cohort 内，各目标、预算和策略共享同一批 multinomial 行重采样。按冻结分数次序、出现次数的权重重建 top-B；边界出现次数可部分计入以恰好达到 B。bootstrap 中多次抽到同一案例代表经验分布权重，真实审核名单仍不允许重复 ID。分母在每次重采样重算；无目标抽样记 undefined 并计数。

主 global/additive 使用 9,999 次；其余历史模型敏感性及 2018 基线用 1,999 次；严格终点子群敏感性用 999 次。先报告 pointwise 95% percentile intervals，主 24 项另报告 Bonferroni-adjusted percentile bounds（每项 alpha=.05/24）。这是近似 bootstrap simultaneous inference，不能称有限样本精确 FWER 保证。没有把 bootstrap tail proportion 当作有效 null p-value。

所有区间条件于已拟合信用模型、审核模型、候选集合、选择好的政策和阈值，不包括训练或选择变异。缺少跨贷款 borrower cluster 标识，故使用 row bootstrap；它不代表未来宏观环境的不确定性。

位置分解采用 C_audit 分数确定的最多 50 个 bins（阈值上下各最多 25），边界在评价前冻结。比较同目标同分母的 capture difference、position difference、within difference，并检查加和恒等式；不称因果分解，不计算总差接近零时的“解释百分比”。

## 程序入口与文件

在仓库根目录使用 `.venv/bin/python src/audit_revision_v1_9.py STAGE`。阶段为 `freeze`、`develop`、`evaluate-development-baselines`、`evaluate-historical`、`evaluate-strict`。已冻结阶段拒绝重复覆盖；只读复核可调用测试。

独立入口为 `evaluate-independent --dataset DATA.npz --gate GATE.json --base freddie_mac_2018`（或 Fannie 系统）。必须有两个事先固定的评价 cohort 及八项主对比家族，使用 9,999 次。未提供真实合格新数据时此入口不能产生独立实证结论。

输出根目录：`results/tables/v1_9_revision/`。`local_cache/` 内含逐行预测、审核模型和严格标签映射，只作本地受限材料；公开复现包应只包含代码、协议、聚合 CSV/JSON 和合法获取数据后的重建说明。

从已核验结果重建主文汇总：`.venv/bin/python results/tables/v1_9_revision/build_summary.py`。该步骤不拟合、不选择政策、不重新评价；输出 `main_results.json` 和 `main_historical_e2e3.csv` 等聚合表。

### 聚合 CSV 字段

- `performance.csv`：cohort/base/variant/learner/scope；target、budget、N、B、effective_budget、target_n；strategy、selected_policy；hits、capture、yield_、hits_per_1000_reviews；pointwise lower/upper、bootstrap 数量、有效/空目标重采样数和 status。
- `paired_contrasts.csv`：contrast、hits_difference、capture_difference、yield_difference；paired lower/upper；适用时 family_adjusted_lower/upper、family_size、comparison_role。正值代表前一个策略更高 capture。
- `decomposition.csv`：原 capture 差、position 差、within 差及各自区间；`identity_residual` 应在浮点误差内为零。
- `predictive.csv`：G2−G1 或 G1−G0 的 log-loss 差及配对区间，负值有利于新增输入模型。
- `budget_boundary.csv`：G2 相对 G1 的进入/离开人数、各自目标命中数及 queue Jaccard。
- `development_policy_choices.csv`：开发数据上冻结的政策及 OOF hits，不作为独立评价结果。
- `candidate_registry.csv`：候选、family、原 validation AUC/Brier、资格和 artifact hash。

## 未完成范围

没有独立 D_eval、没有候选重复训练、没有 reviewer effectiveness 或利润评价、没有真实分析者使用研究。所有这些缺口必须在主张中保留，不能由软件通过、旧数据新划分、更多 bootstrap 次数或新增表格替代。

## 已完成结果与如何解释

共完成 24 套历史配置（六 cohort × 两候选定义 × 两学习器）、12 套严格终点配置、两套 2018 基线。38 个执行 receipt 的文件哈希、预算、hits/capture/yield、目标分母和分解恒等式全部核验通过；实证模块 11 项测试通过，独立复核还验证了机制模块 9 项测试。最大分解残差约 2.1e-17。

主 24 项比较有 7 个正点估计，但没有一个 pointwise 或 Bonferroni-adjusted 区间下界高于零。它们不能被描述成无信息证明，也不是 24 次独立失败。完整政策集合在 12 个 target×cohort 配对中，有 9 个与基础集合选中完全相同的政策；剩下三个为 Freddie all-error，新增命中为 2、6、1，区间均包含零。开发时 Freddie P_full 选中 G2 仅因为 2018 OOF 比 AboveTauLow 多 1 个错误命中（963 vs 962），这一选择按预案保留。

Fannie 2023 提供具体的“预测损失与固定预算不同”例子：G2−G1 log-loss 为 −0.000552，pointwise 95% interval [−0.000754, −0.000343]；但 10% all-error 队列更换 24 人，进入者 23 个目标、离开者 24 个目标，净少 1 个。FN 队列更换 50 人，也净少 1 个。该例是历史样本上的条件性描述，不是额外信息的一般识别或外部效用结论。

候选/学习器依赖仍明显。例如 Freddie 2022 的 G2−G1 FN capture 在 global additive 下为 −8.64 个百分点，交互模型下为 +2.47 个百分点；后一项区间跨零。严格终点子群仍保留该 additive FN 减少，且主 E2/E3 没有变成稳定正收益。严格子群 N 为 8,799、8,762、5,640，改变了人群与结局；不能把它当成独立 replication。
