# v1.9 主分析之后的候选重复构建敏感性

2026-10-19。此次追加由用户修改方案第7.3节的候选构建稳定性要求驱动。主E2/E3的历史结果已经看过，因此这里明确是 **post-primary exploratory sensitivity**，不是追溯预注册、追加确证试验或独立D_eval。主分析、原信用模型、原数据和原冻结记录不改写。

## 执行前固定的范围

- 两个2018机构系统，每个固定三个种子：2026101901、2026101902、2026101903；每次完整原26网格（M1=6、M2=7、M6=9、M7=4），总计156次候选拟合。
- 原30,000训练样本拟合，原10,000验证样本计算AUC和Brier；global AUC门槛为当次网格最好AUC减.010，数值容差1e-6，不另增Brier gate。所有通过和未通过候选均报告，不按结果停止或替换种子。
- 原deployment分数与tau完全固定。原2018 former test仍为C_audit；各次新候选的G1/G2采用与主分析相同的5折、同additive参数及完整C refit。不新增policy argmax。
- 六个已经使用的2021–2023 cohort，仅10%预算、AllErr和FN。每套固定候选和outcome learner使用999次配对行重抽样，报告点式区间，不纳入原24比较主族。
- 报告每seed候选成分、验证表现、原global Range与新Range的Jaccard、G2−G1的目标差与成员交换；案例入选次数保留私有，公开0/1/2/3次频率分布。跨seed范围是三个重建结果的描述，不是训练随机性的置信区间，也不是独立cohort重复。
- 任何候选拟合失败保留记录且该seed整套结果无效；M<2不得人为生成有意义的Range队列。无替代seed。开发预测、模型、行标识、贷款行级成员记录仅在`local_cache/`或`models/`。

## 工程与保护条件

训练复用原函数，但单独逐模板覆盖config内写死的random_state，并核验实际拟合模型的seed。直接调用原`--fit-grid`会写旧目录并清理候选，因此没有使用。两机构可各用一个进程，显式限制每个进程5个模型worker和1个BLAS线程。

冻结协议`results/tables/v1_9_candidate_rebuild/protocol.json`及独立SHA记录代码、网格、环境、train/val、原deployment、原候选模型、固定预测和主结果输入的hash。全部六次候选与G1/G2开发结束后，再冻结共同`FROZEN_DEVELOPMENT.json`，然后运行新增历史评价。运行前后复查输入未变。

可行性检查确认本地14 CPU、48 GiB内存及约145 GiB可用盘空间。每机构训练30,000行，one-hot矩阵约113/119 MB，原完整候选模型约0.52 GiB。没有找到逐模型原始计时日志；旧reference模型保存时间跨度提示本轮训练为数十分钟量级，此估计不当作实测耗时。新脚本逐候选记录真实秒数。

## 执行结果

156/156候选均成功完成，实际random_state逐一核验；六套G1/G2开发均成功，无替代种子、无省略候选。Freddie的global候选数为2/3/2；Fannie为8/10/12。候选构成为：

| 系统 | seed后四位 | M1 | M2 | M6 | M7 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| Freddie | 1901 | 0 | 0 | 0 | 2 | 2 |
| Freddie | 1902 | 0 | 1 | 0 | 2 | 3 |
| Freddie | 1903 | 0 | 0 | 0 | 2 | 2 |
| Fannie | 1901 | 6 | 2 | 0 | 0 | 8 |
| Fannie | 1902 | 6 | 3 | 0 | 1 | 10 |
| Fannie | 1903 | 6 | 4 | 0 | 2 | 12 |

每次完整26候选的验证AUC、Brier、是否入选、实际seed、模型hash、耗时及警告全部保留于六个`*_candidates.csv`。确定性配置可能在不同random_state下产生相同拟合；设置不同seed不意味着每个学习器一定改变预测。此次变化范围只包括固定train/val、固定特征和网格下的种子变化，不包含训练样本重抽样或网格选择不确定性。

36条新增G2−G1比较均有999个有效配对draw，全部点式95%区间包含零，没有正下界或负上界。AllErr命中差整体为−5至+5，FN为−2至+2。下面是三个预设重建结果的最小值与最大值，**不是跨训练随机性的置信区间**：

| 历史评价cohort | 目标 | G2−G1命中差范围 | capture差范围（百分点） |
|---|---|---:|---:|
| Freddie 2021 | AllErr | +1 至 +2 | +0.272 至 +0.543 |
| Freddie 2021 | FN | −1 至 +1 | −0.735 至 +0.735 |
| Freddie 2022 | AllErr | −5 至 +5 | −0.118 至 +0.118 |
| Freddie 2022 | FN | −2 至 −1 | −2.469 至 −1.235 |
| Freddie 2023 | AllErr | 0 至 +1 | 0 至 +0.023 |
| Freddie 2023 | FN | 0 | 0 |
| Fannie 2021 | AllErr | −1 至 +1 | −0.129 至 +0.129 |
| Fannie 2021 | FN | −2 至 +1 | −1.887 至 +0.943 |
| Fannie 2022 | AllErr | 0 至 +2 | 0 至 +0.075 |
| Fannie 2022 | FN | −2 至 +1 | −2.273 至 +1.136 |
| Fannie 2023 | AllErr | −2 至 0 | −0.043 至 0 |
| Fannie 2023 | FN | 0 至 +2 | 0 至 +5.882 |

Range与原global队列的Jaccard在Freddie为.224–.495、Fannie为.654–.795；每机构范围来自2018开发cohort与三个后来cohort各三个seed，共12个相关比较。它们表明Freddie的名单对所检查的候选重建较敏感。三个重建队列共同入选人数在Freddie四个cohort分别为305/197/339/186，在Fannie分别为717/768/774/707；每队列B为1,000，只有Freddie2023为675。公开的`selection_frequency.csv`按全cohort分母报告0/1/2/3次入选人数及比例，不把“全体中三次入选比例”误称为“每个队列的稳定比例”。

此次重建没有使原主分析成为未见数据评价，也不证明条件无信息。36个点式区间和三个seed不能当作独立重复或高精度训练分布推断。每套区间只条件于该套重训候选与对应G1/G2拟合。原主结果和原24比较族不变，新增结果只进入明确标记的探索性敏感性部分。

沿用历史provider loan_age 0–24的存储结局（numeric delinquency>=2或RA事件）；没有偷偷替换为严格calendar标签。原deployment分数、tau和目标分母固定。此次没有重复E2的policy-set选择，因此不能把G2−G1稳定性解释为完整策略选择流程的训练稳定性。

## 更正链、警告与验证

全部训练任务正常结束并通过原源码hash检查之后，独立实现复核指出三个CLI边界保护缺口：全部seed无效时空CSV汇总、冻结开发时未逐项核已有生成receipt、正的常数Range可能形成无意义tie队列。在任何新增capture评价之前归档原执行源码与测试，登记`SOURCE_AMENDMENT_01.json`及SHA；原`protocol.json`未修改。修正只加强失败处理、receipt链校验与数值退化检查，训练参数、候选资格和已经产生的模型/预测均不变。非退化检查要求各案例Range的最大值减最小值大于1e-12。

六份生成receipt与全部候选、C模型/预测hash通过`development_preflight.json`核验；随后共同`FROZEN_DEVELOPMENT.json`封存。`evaluation_preflight.json`逐一核验24组cohort×seed的行标识、旧标签、固定预测映射及新旧Range非退化，并记录未计算capture。完成这些检查和独立代码复核后才运行评价。

10项新增测试通过。独立复核核验了原protocol、源码更正链、156个候选模型hash、重新计算的资格门槛与开发预测。结果读回复核了36条目标比较、18条log-loss比较、24条Range对比、32个频率分布单元、ceil预算、进入/退出命中恒等式、私有成员索引与公开汇总、区间有效draw及所有结果hash。`VALIDATION.json`保存最终聚合CSV哈希。受保护的旧训练、验证、模型、固定预测及原主分析文件全部保持不变。

训练没有ConvergenceWarning。保留36条scikit-learn 1.8的penalty弃用提示以及12条显式L1与默认l1_ratio的提示；本地实现核查确认显式penalty仍传入liblinear，未为了消除警告改动冻结配置。候选流水线累计实测约1,440秒（包含验证评分、序列化和hash，不是纯CPU时间）；两个机构并行，从协议冻结到最后开发receipt约15分钟。完整候选表中的逐模型`seconds`是可核对的实际耗时记录。

## 完成产物

- `comparisons.csv`：36条逐seed目标命中、capture差、点式区间和成员交换。
- `predictive.csv`：18条逐seed G2−G1 log-loss差及点式区间。
- `range_stability.csv` / `range_stability_summary.csv`：原global队列与重建队列的逐seed及范围摘要。
- `selection_frequency.csv`：仅公开全cohort的入选频率分布；案例级文件不公开。
- `across_seed_summary.csv`：三个固定seed的描述性范围，明确不是跨训练区间。
- `EVALUATION_COMPLETE.json`、`VALIDATION.json`：完成及验证记录；没有失败或无效seed。
