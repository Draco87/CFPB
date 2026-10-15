# Complaint Intelligence Platform (CFPB)

Triage, risk prediction and regulation-grounded response drafting for consumer financial complaints.

## Data
- `data/raw/narratives_archive/` - CFPB FOIA Reading Room archive exports (May 2021 - Feb 2026), **with** complaint narratives (3.03M complaints). The CFPB stopped publishing narratives in its live database on 14 Aug 2026; the archive preserves them.
- `data/raw/ccdb_current/complaints.csv` - current Consumer Complaint Database export (metadata only, no narratives). Used for tabular risk modelling and drift monitoring.

## Setup (uv)
```bash
uv sync          # creates .venv, installs dependencies + this project (editable)
```

## Module 1 - Product triage: which of 9 products is a complaint about?

### Pipeline
| Script | Where | What it does |
|---|---|---|
| `00_combine_archive.py` | local | 13 archive CSVs -> cleaned Parquet parts |
| `m1_00_make_splits.py` | local | time-based splits: train < 2025, val Jan-Jun 2025, test Jul 2025-Feb 2026 (frozen) |
| `m1_01_train_baseline.py` | local (CPU) | TF-IDF + logistic regression |
| `m1_02_finetune_distilbert.py` | Kaggle (T4) | fine-tuned DistilBERT |
| `m1_03_label_shift.py` | local | monthly product mix + label-shift correction (EM) |
| `m1_04_compare_models.py` | local | bootstrap CIs, McNemar test, per-month scores |
| `m1_05_ccdb_vs_archive.py` | local | checks the early-2026 narrative gap against the full database |
| `m1_06_llm_classify.py` | Kaggle (T4 x2) | Qwen2.5-7B-Instruct, zero- and few-shot |
| `m1_07_compare_llm.py` | local | LLM vs trained models on the same complaints |

### Results - full test set (50,000 complaints, real product mix)
| Model | Macro-F1 (95% CI) | Accuracy | Latency / complaint |
|---|---|---|---|
| TF-IDF + LogReg | 0.744 (0.736-0.752) | 0.876 | 1.6 ms (CPU) |
| TF-IDF + LogReg + EM label-shift correction | 0.751 (0.743-0.759) | 0.875 | 1.6 ms (CPU) |
| DistilBERT (fine-tuned) | **0.757** (0.749-0.764) | 0.876 | 1.3 ms (T4) |

DistilBERT vs TF-IDF: +0.013 macro-F1 (paired bootstrap CI 0.006-0.019); with EM correction on both, the gap (+0.005) is not significant.

### Results - LLM comparison (900 test complaints, 100 per product)
| Model | Macro-F1 (95% CI) | Latency / complaint | Labels needed |
|---|---|---|---|
| DistilBERT | **0.818** (0.791-0.840) | 1.3 ms | 174k |
| TF-IDF + LogReg | 0.783 (0.757-0.810) | 1.6 ms | 174k |
| Qwen2.5-7B few-shot (9 examples) | 0.668 (0.635-0.695) | 3,787 ms | 9 |
| Qwen2.5-7B zero-shot | 0.652 (0.620-0.678) | 354 ms | 0 |

### Findings
1. **Fine-tuned small models beat a 7B LLM by ~15 macro-F1 points at ~1/270th of the latency.** The LLM is weakest on debt collection vs credit reporting - a boundary set by CFPB convention, learnable from labels but not from common sense.
2. **Few-shot prompting: +1.6 points for 10x the cost** (3,355 vs 437 prompt tokens).
3. **Label drift is real.** A one-month spike (Jan 2025: money-transfer complaints 1% -> 24% of volume) distorted the validation mix. Unsupervised EM prior re-estimation recovered ~85% of the resulting macro-F1 loss without labels. It works for TF-IDF but not for DistilBERT, whose probabilities are less well calibrated.
4. **Label ambiguity caps performance.** Most remaining errors are debt collection <-> credit reporting; consumers choose the product themselves when filing.
5. **Data caveat:** Jan-Feb 2026 has very few published narratives and almost no credit-reporting ones, likely a publication change rather than consumer behaviour (`m1_05` checks this).

## Module 2 - Fine-tuning a small LLM for structured extraction

Can a small LLM, fine-tuned on our labels, close the gap to the trained classifiers - and extract more than one field?
The model reads a complaint and returns JSON: `{"product": ..., "sub_product": ..., "issue": ...}`
(56 possible sub-products, 90 possible issues).

### Pipeline
| Script | Where | What it does |
|---|---|---|
| `m2_01_finetune_slm.py` | Kaggle (1x T4) | Qwen2.5-1.5B-Instruct: untrained test, LoRA fine-tuning, JSON evaluation |
| `m2_02_compare.py` | local | all models on the same 900 complaints + field-level baselines |

**Training:** LoRA (r=16) on all attention and MLP layers - 18.5M trainable parameters (1.2% of the model);
18,000 complaints from 2023-2024 (2,000 per product, current CFPB category names); 1 epoch, 68 min on one T4;
loss 1.45 -> 0.07. Only the JSON answer is trained on (prompt tokens masked).

### Results - product (same 900 test complaints as Module 1, 100 per product)
| Model | Macro-F1 (95% CI) | Latency / complaint | Labels used |
|---|---|---|---|
| DistilBERT | **0.818** (0.791-0.840) | 1.3 ms | 174k |
| Qwen2.5-1.5B fine-tuned (LoRA) | 0.790 (0.766-0.815) | 232 ms | 18k |
| TF-IDF + LogReg | 0.783 (0.757-0.810) | 1.6 ms | 174k |
| Qwen2.5-7B few-shot | 0.668 (0.635-0.695) | 3,787 ms | 9 |
| Qwen2.5-7B zero-shot | 0.652 (0.620-0.678) | 354 ms | 0 |
| Qwen2.5-1.5B untrained | 0.346 (0.315-0.375) - 29% unparseable answers | 88 ms | 0 |

### Results - structured fields (fine-tuned 1.5B)
Valid JSON: **100%**. All three fields correct: 36.9%.
"Naive" = always guess the most common sub-product/issue for the product.

| Field | Fine-tuned | Naive (uses model's product) | Fine-tuned, product right | Naive, product right |
|---|---|---|---|---|
| Sub-product | **62.4%** | 55.2% | **79.0%** | 69.9% |
| Issue | **47.2%** | 34.4% | **58.0%** | 42.8% |

### Findings
1. **Fine-tuning took the 1.5B model from 0.35 to 0.79 macro-F1** and from 29% unparseable answers to 100% valid JSON - beating an untrained model 5x its size by 12+ points.
2. **It matches TF-IDF with 10x less labelled data** but is ~180x slower than DistilBERT; for product-only routing, DistilBERT remains the better choice.
3. **Its real value is multi-field extraction in one pass**, beating naive baselines by 7-15 points on sub-product and issue - a single model where a classifier approach would need one model per field.
4. **Errors cascade:** a wrong product usually means wrong sub-product and issue. Low-confidence cases should go to a human.
5. **Issue labels are ambiguous by design** (e.g. "debt not owed" vs "written notification about debt"); consumers choose them when filing.

## Module 3 - Risk prediction: will a complaint end with the company paying?

When a complaint arrives, how likely is it to end with **monetary relief**? Those are the complaints where the company most
likely made a mistake. The purpose is to **find them early** - help and repay the customer faster, fix the root cause,
and avoid regulatory escalation - not to reduce legitimate payouts.

### Pipeline
| Script | Where | What it does |
|---|---|---|
| `m3_01_build_features.py` | local | time windows, history-based rates, text model, keyword flags |
| `m3_02_train_explain.py` | local | 4 models, bootstrap CIs, SHAP, calibration + recalibration, group check, HTML report |
| `eda_report.py` | local | plain-English analysis of the whole dataset (trends and anomalies) |

**Time windows (each step uses only the past):** history May 2021-Dec 2022 (company/issue payout rates + text model) ->
train Jan 2023-Jun 2024 (300k) -> validation Jul-Dec 2024 (100k) -> test Jan 2025-Feb 2026 (200k).
**Leakage guard:** `company_response`, `company_public_response`, `timely` and `date_sent` are filled in after the
complaint arrives and are never used as inputs.

### Results - test set (200,000 complaints, 2025-26; payout rate 1.34%, about 1 in 75)
| Model | PR-AUC (95% CI) | ROC-AUC | Payouts found in riskiest 10% |
|---|---|---|---|
| Company track record only (no training) | 0.187 (0.176-0.197) | 0.950 | 86% |
| Logistic regression (details) | 0.284 (0.270-0.302) | 0.969 | 95% |
| XGBoost (details) | **0.287** (0.273-0.305) | 0.971 | 95% |
| XGBoost (details + text) | 0.285 (0.271-0.300) | 0.974 | **97%** |

Random ranking would score PR-AUC ~0.013. Adding the text: PR-AUC -0.003 (95% CI -0.011 to +0.005) - not significant.
Top SHAP drivers: company's past payout rate, the text score, company, issue, issue's past payout rate.

### Calibration
| | Average predicted | Actual |
|---|---|---|
| Original model | 1.86% | 1.34% |
| Recalibrated (Platt scaling on Jul-Dec 2024) | 1.82% | 1.34% |
Riskiest 2% of complaints: predicted ~46%, actual ~28%.

### Group check (first look; Module 6 goes deeper)
| Group | Complaints | Actual payout rate | Average predicted | PR-AUC |
|---|---|---|---|---|
| All | 200,000 | 1.3% | 1.9% | 0.285 |
| Older American | 3,731 | 8.0% | 9.2% | 0.376 |
| Servicemember | 9,599 | 3.2% | 5.0% | 0.217 |

### Findings
1. **Reviewing the riskiest 10% of complaints finds 97% of payout cases** (vs 10% in random order).
2. **Who the complaint is about matters more than what it says:** company history alone finds 86%; the text adds no significant PR-AUC.
3. **Simple beats fancy:** logistic regression ties with XGBoost - the more explainable model is the sensible choice.
4. **Ranking is stable, probabilities are not.** The payout rate fell from 3.0% (2021-22) to 2.3%, 1.5% and 1.3%. Predictions
   are ~40% too high on 2025-26 data, and recalibrating on H2 2024 barely helps - the same kinds of complaints were paid
   less often during the test period itself. Fix: recalibrate on a schedule with the latest resolved complaints (Module 7).
5. **Servicemember complaints are ranked less well** (PR-AUC 0.217 vs 0.285) - to be tested for significance in Module 6.
