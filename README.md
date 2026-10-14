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
