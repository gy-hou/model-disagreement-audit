# Model disagreement and audit value

Research code for **From Model Disagreement to Audit Value: Conditional Information and Budget Boundaries**, by Guangyu Hou, Wei Zhang, Jie He, Zhiying Liu and Mengou Wang.

This repository accompanies the final-r5 manuscript prepared for submission. It contains the exact constructions, synthetic experiment programs, retrospective analysis code, preserved aggregate results and figure builders. The study distinguishes differences in review queues, information beyond existing scores, and additional target hits at a fixed review budget.

The versioned snapshot is **[kbs-final-r5](https://github.com/gy-hou/model-disagreement-audit/releases/tag/kbs-final-r5)**. This identifier records the code version; it is not a journal publication or DOI.

## Start here

1. Read [REPRODUCING.md](REPRODUCING.md) for the tested commands and data requirements.
2. Inspect the [final vector figures](paper/kbs_final_r5/figures/) and [recorded aggregate results](results/tables/).
3. Use [CITATION.cff](CITATION.cff) to cite this software snapshot. The repository URL is https://github.com/gy-hou/model-disagreement-audit.

The recorded environment is Python 3.14.3. From a fresh clone:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-render.txt
.venv/bin/python src/mechanism_v1_10.py design
.venv/bin/python -m pytest -q tests/test_mechanism_v1_9.py tests/test_audit_revision_v1_9.py tests/test_candidate_rebuild_v1_9.py tests/test_mechanism_v1_10.py
```

Figure reconstruction additionally requires Arial and Poppler's `pdffonts`; existing vector PDFs can be read without those tools. The verification record in [VALIDATION.json](VALIDATION.json) reports the commands actually checked. Verification used the recorded local Python environment; it does not establish a clean installation on every operating system.

## Repository map

| Location | Contents |
|---|---|
| `src/` | Core mechanisms, audit policies, preprocessing and empirical analysis programs. |
| `tests/` | Implementation checks using small fixtures and temporary outputs. |
| `configs/` | Recorded model-grid configurations. |
| `results/tables/` | Aggregate results, synthetic repetition records, frozen protocols and provenance. |
| `idea-update/` | Completed supplementary constructions and exploratory discovery analyses included in this paper. |
| `docs/revisions/` | Selected outcome contracts, figure builders and their numeric inputs. |
| `paper/kbs_final_r5/figures/` | Final artwork; this directory contains figures, not manuscript drafts. |
| `SOURCE_MANIFEST.json` | Hashes of the 491 unchanged files selected from the validated research archive. |

Historical version names are retained because scripts and provenance records use those paths. The current Figure 1 is built by the r3 program; Figures 2–7 use the r1 presentation program. See the exact mapping in the reproduction guide. The separate proposed human/agent review study is not part of this repository.

## Data and scope

The public files support exact calculations, synthetic reproduction and rebuilding displays from completed aggregate results. Loan-level records, individual predictions, fitted model objects, private caches and source datasets are excluded. Full empirical reruns require authorized provider data and the documented frozen inputs; a clone alone is not sufficient. Source-data access conditions are not changed by this code release.

The empirical findings are retrospective and exploratory. Previously examined cohorts are not independent prospective confirmation, and the study does not measure human-review corrections or costs. Code publication does not add new scientific results.

## License and assistance

Original project code is released under the [MIT License](LICENSE). Third-party Nature figure-QA utilities retain their [Apache-2.0 license](docs/revisions/kbs_final_r1/nature_qa/LICENSE.md); see [NOTICE.md](NOTICE.md). No provider data or proprietary fonts are licensed or redistributed here.

ChatGPT/Codex and Claude assisted with code development, language editing, manuscript organization and analytical-design suggestions. Figures are deterministic Python/Matplotlib renderings of recorded data or specified constructions. Authors remain responsible for the methods, code and reported results.

## 中文说明

这是论文《从模型分歧到审核价值：条件信息与预算边界》的公开代码仓库，采用 MIT 许可证。仓库包含代码、汇总结果、合成实验记录和绘图程序；不包含受限的贷款明细、逐行预测、模型文件或论文草稿。`kbs-final-r5` 固定对应本次投稿稿件的代码版本。复现范围和运行步骤见上方说明；真人与 agent 复核的后续研究未并入。
