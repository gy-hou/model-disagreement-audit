# v1.9 数据隔离、结局与日历审计

审计日期：2026-10-19（Australia/Sydney）。依据用户提供的14节修改方案，检查本项目现有源文件、ETL、缓存、旧结果和官方数据字典。所有统计仅输出聚合量，不输出贷款标识。

## 1. 可用性结论

**本轮没有已证实合格的独立 D_eval。** 2018 和 2021–2023 的两家 GSE 样本已反复用于发现、解释、筛选和敏感性分析。用户已明确确认2019/2020 Freddie曾经使用过；随后“不确定，先按现有数据推进”回答的是是否另有完全未用的新数据，并未撤回前述已用确认。因此2019/2020也不能宣称未使用。重新划分或创建新时间戳不恢复独立性。本轮新增结果应标记为 retrospective / exploratory。

对 Freddie 2019、2020，本审计仅读取 origination 中贷款标识和首付款日期、压缩包目录，没有打开其 performance 内容。两年各50,000个不同贷款；八个本地 Freddie vintage 的原始贷款标识两两交集均为0。2019/2020 最晚首付款分别为2021-06/2022-04，日历 age24 最晚为2023-05/2024-03。**这些事实证明身份互斥及日历长度，不证明研究者从未使用，也不证明随访完整。**

Fannie 2018 原始全量1,834,047笔、2021/2022/2023全量4,780,993/2,012,171/969,574笔的标签，已在历史 ETL 中参与分层抽样和全量发生率统计。未被抽入50,000笔的余下贷款，不能直接称为来自从未接触标签的数据来源。旧 processed 文件只保留重新编号的 row_id。2018 的50,000笔来源标识可由私有 servicing cache 恢复；2021–2023 reservoir 未保留最终原始标识映射，需要按旧顺序与标签重放才能精确排除已用样本。该重放不会消除历史总体标签暴露。

本地 Freddie 2024/2025 快照没有任何贷款在2025-09截止前达到原首付款后 age24，不能作为完整窗口评价。按文件名检查 Downloads 与 Archive 未发现额外的2016/2017 Freddie或2019/2020/2024/2025 Fannie季度原文件；这是限定目录与文件名的检查，不是全设备不存在证明。按用户最新指示已停止新数据查找。

## 2. 原 A/B/C 角色不独立

| 原数据用途 | 实际行为 | 本轮允许的解释 |
|---|---|---|
| 2018 train，30,000笔/机构 | 拟合信用模型、拟合预处理 | 历史 A_base 的训练部分 |
| 2018 validation，10,000笔/机构 | AUC筛选候选、指定 tau、sigmoid calibration、outcome OOF、部分策略选择 | A_base 开发、B_cal、C_audit 被复用；不是四类隔离 |
| 2018 former test，10,000笔/机构 | 多版本结果和设计已看过 | 可作本轮 C_audit，但不得再次称未用测试 |
| 2021–2023 former test | 迁移、机制、阈值等多轮分析 | 历史探索评价，不能完成新独立验证 |

只对 outcome learner 做5折OOF，没有对候选资格、阈值及校准全链条重新训练。`src/control_comparison_v1_8.py` 与 `src/evidence_gate_v1_6.py` 的OOF是固定上游系统下的开发诊断；`src/gate_refit_v1_7.py`也明确不包含历史候选选择和信用模型重训。它们不能支持完整嵌套流程评价。

v1.9可在原2018 test上开发 G0/G1/G2及policy，继承已经固定的候选模型；这有助于避免同一批验证标签同时承担资格选择和当前outcome拟合，但仍只是历史开发分析。

## 3. 新发现：旧标签窗口不是始终固定的原始日历窗口

旧 Freddie/Fannie ETL 将 provider `loan_age` 的0–24（含两端）行聚合，数值delinquency>=2为事件，RA被映射为9；缺失或不可解析状态比较结果为false。有至少一行落入该窗口即进入标签合并，没有要求完整月份，也没有按退出原因区分未知阴性。

本地 Freddie 官方 user guide第11页规定，修改贷款的 `LOAN AGE` 使用修改后的 first-payment 重新计算。因而旧0–24算法可能计入原始首付款之后第24个月以后的事件。官方字典还把RA定义为REO acquisition；它是旧实现纳入的复合事件，不应写成一个数值60DPD代码。

**本轮实际完成了Freddie四个已用vintage的原始performance审计**，按 `monthly_reporting_period - original first_payment_date + 1` 建立日历age。原始文件行序、历史maturity gate与row_id对应，且所有train/val/test共183,704条旧标签全部精确重现；每年三分割ID互斥且覆盖全部原样本。因此下表可可靠地与旧test比较。

| Freddie test | N | 原 loan_age 阳性 | 原首付日历0–24事件阳性 | 标签改变 | 严格可判定 N | 排除未知 |
|---|---:|---:|---:|---:|---:|---:|
| 2018 | 10,000 | 391 | 383 | 8 | 5,707 | 4,293 |
| 2021 | 10,000 | 151 | 148 | 3 | 8,799 | 1,201 |
| 2022 | 10,000 | 286 | 284 | 2 | 8,762 | 1,238 |
| 2023 | 6,741 | 184 | 184 | 0 | 5,640 | 1,101 |

所有改变均为旧阳性在新原首付日历窗内没有事件；不存在新增日历阳性。严格子群内相对旧标签的改变数分别7/3/2/0；2018另1例原阳性因随访不足而是unknown，而非已证实阴性。

严格子群的定义单独命名为 `strict_calendar_age0_24_60plus_or_reo`：先要求日历成熟；观测到指定事件即可识别阳性；阴性需要0–24共25个可能age全部具有有效状态，无窗口内缺失、无法定位月份或冲突状态。任何提前退出都不自动作为阴性。这个保守定义是**可识别结局人群敏感性**，排除概率可能与结局和服务过程有关，不能称为恢复了全体贷款的无偏标签。窗口包含age0，故同时报告1–24这24个scheduled ages完整率；不要把25个可能age描述为恰好24个观测值。

四个旧test中窗口内缺失状态、无法定位年月、同月冲突有效状态均为0；未知主要来自完整月序列不足或退出。2018旧“valid age24或任意recorded exit覆盖100%”只证明原审计的宽松覆盖条件，不能证明每个缺失月都已判定。Fannie旧2018相同宽松审计为100%覆盖、0缺失状态，但**尚未按本轮原始日历严格规则重放Fannie**；不能把Freddie发现或修复自动推广到Fannie。

## 4. 2022成熟性和2023 flow澄清

Freddie四个已读performance快照的最大reporting period均为2025-09。旧代码仅对2023设置first_payment<=2023-10门槛。2022源样本有9笔晚于该first-payment门槛，因此旧文“2022全部成熟”在源池层面不准确；这9笔未进入旧2022 test（旧test 10,000笔日历均成熟）。不要把源池问题夸大成这10,000笔测试全部不成熟。

Freddie2023：50,000原生样本→33,704日历门槛保留→20,222/6,741/6,741三分割，池成熟比例0.67408。Fannie2023：969,574源贷款→排除14,616→954,958门槛保留（0.9849253383）→按标签分层抽样50,000→30,000/10,000/10,000。因此0.6741与0.9849及最终test N差异来自门槛与抽样顺序，并不构成数字矛盾。完整八行见 `cohort_flow.csv`。

## 5. 日历顺序与可用时间

下载记录：2018两家和全部本地Freddie样本为2026-06-26；Fannie2021–2023为2026-06-28。Freddie压缩包内文件时间为2026-01-06，**不是经过核实的官方发布日期**。旧源下载时使用的精确provider release ID/官方发布时点未被完整保留。

本轮拟合/重构发生于2026年10月；评价2018或2021–2023只能称回顾性。用2021完整24月标签选policy不能声称在2022实时运行。所谓“2018模型”是以2018源vintage拟合的研究参考模型，不是有证据的2018年银行部署。

未来若做历史时点可执行设计，需要分别证明观测窗口结束、数据发布/入库、策略冻结、队列形成的先后，且全部开发标签实际可获得时间<=freeze<=queue。现在仅可证明本地样本、模型及研究版本的可重建关系。

## 6. 可运行交付与限制

`src/build_eval_cohort_v1_9.py` 默认只读取已用的2018/2021/2022/2023 Freddie。入口：

```sh
.venv/bin/python src/build_eval_cohort_v1_9.py --write-private-test-labels
```

公开聚合输出：`outcome_audit.csv`、`outcome_audit_metadata.json`；定义：`outcome_contract.json`；私有衍生标签位于忽略目录 `results/tables/v1_9_revision/local_cache/`，含row_id不含原贷款ID。它们仍是贷款行级数据，不放公开包。源文件hash与脚本hash保留在metadata。

已执行四年份旧标签和三分割ID回放；另用完全合成的四笔贷款验证：完整阴性、早期阳性、修改后age重置导致旧阳性但日历阴性、提前清偿但缺随访保持unknown。测试通过。该程序是严格endpoint审计及旧test衍生表构建入口；**不是已经完成的新数据获取、独立验证或完整前瞻cohort构建器**。

Freddie2019/2020、Fannie未用余样本都不会由该入口自动解封。`independent_evaluation_gate.json`保留blocked，任何声称独立的评价入口必须拒绝通过。

## 7. 来源

主要本地证据：`src/preprocessing_freddie_mac.py`、`src/preprocessing_fannie_mae.py`、`src/preprocessing_gse_backtest_v0_7.py`、`src/servicing_audit_v1_5.py`；每个processed cohort的README；`results/tables/v1_4_mechanism/*/tableA0_cohort.csv`；两个2018 servicing_summary；原始数据目录RETRIEVAL及Freddie user guide。

2026-10-19读取的官方页面说明目前Freddie数据下载需注册登录Clarity，Fannie Data Dynamics也需账户。未尝试登录、下载新贷款或接受新条款。停止外搜后未继续寻找其他来源。

- Freddie数据与访问：[Single Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
- Freddie旧版数据字典：[General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf)
- Fannie访问：[Data Dynamics](https://capitalmarkets.fanniemae.com/tools-applications/data-dynamics)

上述结论针对当前checkout与本轮实查范围；没有将“未发现使用文件”当作“从未被研究者使用”的证明。
