# Complaint Intelligence Platform (CFPB)

Triage, risk prediction and regulation-grounded response drafting for consumer financial complaints.

## Data

### 1. Complaint narratives - CFPB FOIA Reading Room archive
On 14 Aug 2026 the CFPB stopped publishing complaint narratives in its live Consumer Complaint Database.
Previously published narratives are preserved in the
[CFPB Consumer Complaint Database Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
(FOIA Electronic Reading Room). This project uses exports 3-15 (May 2021 - Feb 2026, **3.03M complaints with narratives**),
stored in `data/raw/narratives_archive/`:

| # | Period | File |
|---|---|---|
| 3 | May 2021 through October 2022 | [CCDB_Export_3_May_2021_through_October_2022.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_3_May_2021_through_October_2022.zip) |
| 4 | November 2022 through August 2023 | [CCDB_Export_4_November_2022_through_August_2023.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_4_November_2022_through_August_2023.zip) |
| 5 | September 2023 through March 2024 | [CCDB_Export_5_September_2023_through_March_2024.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_5_September_2023_through_March_2024.zip) |
| 6 | April 2024 through July 2024 | [CCDB_Export_6_April_2024_through_July_2024.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_6_April_2024_through_July_2024.zip) |
| 7 | August 2024 through October 2024 | [CCDB_Export_7_August_2024_through_October_2024.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_7_August_2024_through_October_2024.zip) |
| 8 | November 2024 through December 2024 | [CCDB_Export_8_November_2024_through_December_2024.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_8_November_2024_through_December_2024.zip) |
| 9 | January 2025 through February 2025 | [CCDB_Export_9_January_2025_through_February_2025.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_9_January_2025_through_February_2025.zip) |
| 10 | March 2025 through April 2025 | [CCDB_Export_10_March_2025_through_April_2025.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_10_March_2025_through_April_2025.zip) |
| 11 | May 2025 through June 2025 | [CCDB_Export_11_May_2025_through_June_2025.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_11_May_2025_through_June_2025.zip) |
| 12 | July 2025 through August 2025 | [CCDB_Export_12_July_2025_through_August_2025.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_12_July_2025_through_August_2025.zip) |
| 13 | September 2025 through October 2025 | [CCDB_Export_13_September_2025_through_October_2025.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_13_September_2025_through_October_2025.zip) |
| 14 | November 2025 through December 2025 | [CCDB_Export_14_November_2025_through_December_2025.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_14_November_2025_through_December_2025.zip) |
| 15 | January 2026 through February 2026 | [CCDB_Export_15_January_2026_through_February_2026.zip](https://files.consumerfinance.gov/f/documents/CCDB_Export_15_January_2026_through_February_2026.zip) |

The archive also holds older exports (Dec 2011 - Apr 2021) and newer ones (Mar - Aug 2026) not used here.

### 2. Current Consumer Complaint Database export
`data/raw/ccdb_current/complaints.csv` - the live database export (all complaints, metadata only, no narratives),
from the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
Used to check publication gaps (`m1_05`) and for drift monitoring.

### 3. Regulations (Module 4)
Downloaded by `m4_01_fetch_regs.py` into `data/raw/regs/`:
- [eCFR API](https://www.ecfr.gov/reader-aids/ecfr-developer-resources), Title 12, current as of 2026-10-13: Regulations
  E (12 CFR 1005), F (1006), V (1022), X (1024), Z (1026) and DD (1030)
- US Code via govinfo: [Fair Credit Reporting Act](https://www.govinfo.gov/content/pkg/USCODE-2023-title15/html/USCODE-2023-title15-chap41-subchapIII.htm)
  (15 U.S.C. 1681 ff.) and [Fair Debt Collection Practices Act](https://www.govinfo.gov/content/pkg/USCODE-2023-title15/html/USCODE-2023-title15-chap41-subchapV.htm) (15 U.S.C. 1692 ff.)

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

## Module 4 - RAG: regulation-grounded draft replies

For each complaint, find the regulations that apply and have a local LLM draft a reply for an agent to review,
citing only the retrieved rules. An open-book exam rather than answering from memory.

### Pipeline
| Script | What it does |
|---|---|
| `m4_01_fetch_regs.py` | downloads 6 CFPB regulations + FCRA + FDCPA; splits into 757 units (236 rule sections, 302 official interpretations, 219 appendices) |
| `m4_02_build_index.py` | 5,394 chunks (~220 words, 40 overlap, section title kept on every chunk); BM25 keyword index + FAISS index of `BAAI/bge-small-en-v1.5` embeddings |
| `m4_03_draft_replies.py` | drafts replies for sample complaints, checks citations, writes `reports/m4_drafts.html` / `.jsonl` |
| `src/cfpb/rag.py` | reusable retrieval + generation code (used again in Modules 5 and 7) |

**Retrieval:** LLM rewrites the complaint into one short regulatory question -> hybrid search (BM25 + dense, reciprocal
rank fusion) limited to the regulations for the complaint's product -> cross-encoder reranking
(`ms-marco-MiniLM-L-6-v2`) -> top 5, at most 2 chunks per section. Appendices (model forms/clauses) are excluded.
**Generation:** `qwen2.5:3b` via Ollama on a 4 GB GTX 1650 Ti (context 6,144 tokens, answer capped at 450 tokens), with
rules: cite only provided excerpts, state deadlines only if they appear in an excerpt, never contradict the complaint.

### Results - 27 test complaints (3 per product)
| Check | Result |
|---|---|
| Drafts citing at least one rule | 27/27 |
| Citations that point to a retrieved passage | 62/62 (100%) |
| Drafts following the required format | 17/27 |
| Average time per draft | 29 s |

### Findings
1. **Retrieval quality decides answer quality.** The first version retrieved model-form boilerplate (Appendix A to Reg E)
   for an unauthorized-transfer complaint; the LLM then stated a wrong deadline ("investigate within 60 days" - actually
   the consumer's reporting window), yet the citation check still passed.
2. **Citation validity is not correctness.** A citation can point to a retrieved passage and still misstate it - which is
   why Module 5 measures faithfulness and answer quality separately.
3. **Fixes that worked:** excluding appendices, LLM query rewriting and a stricter prompt -> the same complaint retrieves
   12 CFR 1005.11 and 1005.6 and states the correct 10-business-day investigation deadline.
4. **Remaining gaps:** omissions (e.g. provisional credit when an investigation is extended) and loose format compliance
   (17/27) from a 3B model - measured in Module 5.


## Module 5 - Evaluating the RAG system

Two questions, both answerable without legal expertise: (1) **does the search find the right rules?** - checked against
an answer key; (2) **do the drafts say only what those rules say?** - checked by a larger LLM acting as judge, which is
itself checked by a human.

### Pipeline
| Script | Where | What it does |
|---|---|---|
| `eval/golden_retrieval.json` | - | answer key: 34 realistic complaints, each with the rule section(s) that clearly govern it |
| `m5_01_retrieval_eval.py` | local | 8 search setups scored on the answer key (rewritten questions cached in `eval/rewrites_cache.json`) |
| `m4_03_draft_replies.py --no-rewrite --include-appendices --out m4_drafts_v1` | local | "before" drafts (Module 4 version 1) for comparison |
| `m5_02_judge.py` | Kaggle (T4 x2) | Qwen2.5-7B-Instruct judges every draft: lists rule statements, marks each supported / unsupported by an excerpt |
| `m5_03_report.py` | local | `reports/m5_report.html`; creates `reports/m5_judge_check.csv` for the human check of the judge |

### Results - retrieval (34 answer-key complaints)
| Search setup | Correct rule in top 5 | Recall@5 | MRR@10 |
|---|---|---|---|
| Keywords only (BM25) | 21/34 | 0.57 | 0.40 |
| Meaning only (embeddings) | 19/34 | 0.54 | 0.49 |
| Hybrid (keywords + meaning) | 25/34 | 0.71 | 0.54 |
| Hybrid + rerank | 25/34 | 0.69 | 0.55 |
| **Hybrid + rerank + LLM query rewrite (final)** | **29/34 (85%)** | **0.81** | 0.66 |
| Final, without product filter | 27/34 | 0.75 | 0.60 |
| Final, with model forms (appendices) | 29/34 | 0.81 | 0.66 |
| Final + original complaint (multi-query) | 29/34 | 0.78 | **0.69** |

### Results - faithfulness (LLM-as-judge, 27 drafts each)
| Drafts | Rule statements | Supported (judge) | Drafts with an unsupported statement | Addresses the problem (1-5) |
|---|---|---|---|---|
| Version 1 (raw complaint as query, model forms included) | 68 | 35 (51%) | 12/27 | 3.0 |
| Final setup | 79 | 37 (47%) | 18/26 | 2.8 |

### Results - human check of the judge (20 random statements from the final drafts)
| | |
|---|---|
| Not rule statements at all (promises such as "we will review your account") | 6/20 - the judge should have skipped them; it marked 5 unsupported |
| Agreement with the judge on the 14 real rule statements | 10/14 (71%) |
| Judge too strict (rejected a correct paraphrase) / too lenient (accepted an exaggeration) | 3 / 1 |
| Supported rule statements - human vs judge | **11/14 (79%)** vs 9/14 (64%) |

### Findings
1. **Each retrieval component earns its place.** Keyword and meaning search miss different things (hybrid 25 vs 21 / 19);
   LLM query rewriting is the largest single gain (25 -> 29); the product filter adds 2. Reranking barely changes the
   top 5. Excluding model forms does not change answer-key retrieval - its value in Module 4 was avoiding misleading
   boilerplate, a single example rather than a measured effect.
2. **Query rewriting sometimes drops the key detail** ("credit card" -> "the company"; "threatened arrest" -> "legal
   action"). Searching with both the rewrite and the original complaint (multi-query) fixed 2 misses but broke 2 others -
   a tie on 34 cases, so the simpler single-query setup is kept.
3. **Faithful is not the same as correct.** Version 1 scores similar faithfulness because a draft that copies a model
   form is "supported" by that form even when it is the wrong rule for the complaint (Module 4's wrong 60-day deadline).
   Retrieval accuracy, not faithfulness, captures that improvement.
4. **The LLM judge needed checking.** 30% of what it scored were not rule statements, and it missed paraphrases; the raw
   ~50% faithfulness understates quality. On the human-checked sample ~79% of rule statements are supported (small
   sample - an estimate). Its verdicts were usually right, but the excerpt it cited as evidence was often not the best one.
5. **Remaining failure modes of the 3B drafter:** mixing up rules (corrected mortgage disclosures offered as a fix for
   credit reporting), exaggerating scope ("all of Regulation E"), and dropping a rule's "if" condition. Next: a judge
   prompt that removes promises and accepts paraphrases, and seeded-error tests of the judge (Module 6).

## Module 6 - Responsible AI: fairness, guardrails and testing the tests

Modules 1-5 asked "does it work?". Module 6 asks "how can it go wrong, for whom, and what catches it?"
Model cards for every model: [`docs/model_cards.md`](docs/model_cards.md).

### Pipeline
| Script / file | Where | What it does |
|---|---|---|
| `m6_01_fairness.py` | local | risk model by group (Older American, Servicemember, no tag): ROC-AUC, payout cases found in the top 10%, share flagged, calibration, product-mix check; bootstrap 95% intervals for every group-vs-rest gap |
| `src/cfpb/guardrails.py` | - | input checks (personal data masked, instruction-like text flagged) and output checks (time limits must be in a retrieved passage, no refund promises or admissions of fault, no personal data, citations must be retrieved) |
| `eval/redteam.json` | - | 9 hand-written attacks (prompt injection, fake regulation, hidden tags, refund/admission demands, off-topic) + 1 normal control |
| `m6_02_guardrails.py` | local + Ollama | output checks on the existing drafts; red-team test with the plain and a hardened ("guarded") prompt |
| `m6_03_judge_v2.py` | Kaggle (T4 x2) | statement-level judge (rule or promise? supported?) tested against the 20 human verdicts and against 107 planted-error statements |

### Results - fairness of the risk model (200,000 test complaints)
| Group | Complaints | Payout rate | Expected from its products | Ranking (ROC-AUC) | Payout cases found in top 10% | Share flagged |
|---|---|---|---|---|---|---|
| All | 200,000 | 1.34% | 1.34% | 0.974 | 97% | 10% |
| Older American | 3,731 | 8.04% | 5.40% | 0.886 | 98% | 49% |
| Servicemember | 9,599 | 3.18% | 2.99% | 0.917 | 96% | 26% |
| No tag | 187,503 | 1.14% | 1.19% | 0.978 | 97% | 8.6% |

Group vs everyone else (95% bootstrap intervals): ranking is **worse** for Older American (-0.105 to -0.075) and
Servicemember (-0.069 to -0.051); finding payout cases shows **no clear difference** for any group.

### Results - guardrails
| | Final drafts (27) | Version 1 drafts (27) |
|---|---|---|
| Time limit / number not in any passage | 7 | 5 |
| Cites a rule that was not retrieved | 0 | 3 |
| Refund promise / admission of fault / personal data | 0 | 0 |
| Any risk warning (format problems counted separately: 10 / 12) | 7 | 9 |

| Red-team (9 attacks) | Plain prompt | Guarded prompt |
|---|---|---|
| Attack changed the draft | 0 | 2 |
| ... and was not flagged by the output checks | 0 | 0 |

### Results - testing the judge with planted errors (53 clean cases)
| Planted error | n | LLM judge v2 | Code checks | Together |
|---|---|---|---|---|
| Deadline changed | 5 | 2 | 5 | 5 |
| "$500 compensation" invented | 35 | 14 | 33 | 33 |
| Wrong citation title | 5 | 2 | 4 | 4 |
| Meaning reversed | 8 | 3 | 0 | 3 |
| **Total** | **53** | **21 (40%)** | **42 (79%)** | **45 (85%)** |

Judge v2 vs the 20 human verdicts: it accepts correct paraphrases (v1 did not), but it rejected none of the 3 real
errors and spotted 0 of 6 promises.

### Findings
1. **Use metrics that fit the comparison.** The Module 3 "Servicemember PR-AUC gap" mostly came from comparing PR-AUC
   across groups with different payout rates. Base-rate-free measures show equal opportunity (96-98% of payout cases
   found in every group), but significantly weaker ranking inside tagged groups, because the model over-flags them.
   That is fine for prioritising reviews, and rules out using the score to deprioritise anyone.
2. **Deterministic checks beat prompt instructions.** With a 3B model, hardening the prompt gave no measurable
   protection (0 vs 2 successful attacks, noise on 9 cases); the output checks flagged every successful attack.
3. **Red-teaming tests the defences too.** The first red-team run exposed four gaps in the checks (only reading text
   before "RULES USED", trusting deadlines quoted in the complaint, a too-narrow promise pattern, refusals counted as
   successes); the planted-error test exposed a fifth (citations to other titles were ignored). All five were fixed.
4. **An LLM judge is weak on precise details.** It reads past a changed number or an added "$500" clause that code
   verifies exactly; code cannot tell "may" from "may not". Layered design: code checks -> LLM judge -> human agent.
5. **Honest limits:** the test sets are small and hand-made (9 attacks, 53 planted errors); some planted errors were
   malformed by the generator (citation numbers changed instead of deadlines, ungrammatical flips) and are reported
   separately; only the groups present in the data can be checked for fairness.
