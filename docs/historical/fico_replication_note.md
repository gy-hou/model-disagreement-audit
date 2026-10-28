# FICO HELOC Headline Replication - Phase 7

> Direction-only replication of the UCI H4 headline using a public Hugging Face mirror of the FICO HELOC xML Challenge dataset. Formal publication should confirm official FICO data-use permissions.

## Source

- Public mirror used for this run: https://huggingface.co/datasets/mstz/heloc/tree/main/risk
- FICO xML Challenge announcement: https://investors.fico.com/news-releases/news-release-details/fico-announces-xml-challenge

## Baseline Metrics

| model_id | auc | ks | brier | fn | cutoff_probability |
| --- | --- | --- | --- | --- | --- |
| M1 | 0.7826 | 0.4520 | 0.1901 | 290 | 0.5189 |
| M2 | 0.7798 | 0.4404 | 0.1957 | 295 | 0.5380 |
| M3 | 0.7768 | 0.4397 | 0.1997 | 295 | 0.5343 |

## Pairwise Disagreement

| pair | label_disagreement_rate | spearman_rank_correlation |
| --- | --- | --- |
| M1-M2 | 0.1568 | 0.8088 |
| M2-M3 | 0.1329 | 0.8662 |

## Headline Result

| dataset | target | budget | dgs_core_capture | thresholdprob_capture | gap | ci_low | ci_high | direction | fn_total | test_rows | test_positive_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FICO_HELOC_public_mirror | FN | 0.1000 | 0.1763 | 0.0881 | 0.0881 | 0.0357 | 0.1395 | replicates_positive_direction | 295 | 2092 | 0.5220 | M3 is an XGBoost seed/subsample challenger because HELOC features are anonymized and PAY_* monotonic constraints are unavailable. |

## Split Sensitivity

| n_splits | median_gap | min_gap | max_gap | positive_gap_share | median_dgs_capture | median_threshold_capture |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.0783 | 0.0403 | 0.1318 | 1.0000 | 0.1703 | 0.0845 |
