# Model cards - Complaint Intelligence Platform

One card per model: what it is for, what it is **not** for, how it was tested, and what is known to go wrong.
All numbers are from held-out data that is later in time than the training data (train < 2025, test 2025-26).

---

## 1. Product triage - which of 9 products is a complaint about?

| | |
|---|---|
| **Model** | DistilBERT fine-tuned on complaint narratives (baseline: TF-IDF + logistic regression) |
| **Purpose** | Route a new complaint to the right product team, so it reaches someone who can act on it faster |
| **Users** | Customer-care operations; a person can always change the route |
| **Not for** | Deciding whether a complaint is valid, or anything about the customer. It reads the product, not the person |
| **Training data** | 174k CFPB complaint narratives filed before 2025 (FOIA archive), 9 product groups |
| **Test result** | Macro-F1 **0.757** (95% CI 0.749-0.764) on 50,000 complaints, Jul 2025 - Feb 2026; accuracy 0.876 |
| **Speed** | ~1.3 ms per complaint on a T4 GPU (TF-IDF: 1.6 ms on CPU, macro-F1 0.744) |

**Known limitations**
- Most errors are *debt collection* vs *credit reporting*. Consumers choose the product themselves when filing, so the labels overlap.
- **Label drift:** the product mix changes month to month (e.g. money-transfer complaints jumped from 1% to 24% in Jan 2025). EM label-shift correction recovers about 85% of the lost macro-F1 for TF-IDF, but not for DistilBERT, whose probabilities are less well calibrated.
- Jan-Feb 2026 has very few published narratives, and almost none for credit reporting, which is likely a publication change. Recent performance is therefore measured on thin data.
- A 7B LLM with prompting is ~15 macro-F1 points worse and hundreds of times slower, so LLMs are not used for this step.

---

## 2. Structured extraction - product, sub-product and issue as JSON

| | |
|---|---|
| **Model** | Qwen2.5-1.5B-Instruct + LoRA (r=16, 1.2% of parameters trained), 18,000 complaints from 2023-24 |
| **Purpose** | Fill in the case-management fields (product, sub-product, issue) for an agent to confirm |
| **Not for** | Unreviewed automatic filing: all three fields are fully correct only 36.9% of the time |
| **Test result** | Product macro-F1 **0.790** on 900 test complaints (DistilBERT 0.818); valid JSON **100%**; ~232 ms per complaint |

**Known limitations**
- Sub-product (56 options) and issue (90 options) are much harder than product, and some categories are rare.
- The untrained model returns unusable answers 29% of the time. Fine-tuning is what makes it work.

---

## 3. Payout risk - will a complaint end with the company paying?

| | |
|---|---|
| **Model** | XGBoost on complaint details, company/issue track record and text signals; scores recalibrated (Platt) on the latest labelled period |
| **Purpose** | Help agents decide **which complaints to look at first** |
| **Users** | Customer-care team leads |
| **Not for** | Deprioritising, rejecting or delaying any complaint; deciding compensation; any decision about an individual customer |
| **Leakage guard** | Only information known when the complaint arrives is used (the company's response and timeliness are excluded) |
| **Test result** | 200,000 complaints from 2025-26 (payout rate 1.34%, about 1 in 75): PR-AUC **0.285**, ROC-AUC **0.974**; reviewing the riskiest 10% finds **97%** of payout cases |
| **Explanations** | SHAP values for every prediction (top drivers: the sub-product's and issue's past payout rate, how it was submitted, mentions of overdraft or double charges) |

**Fairness check (Module 6, Step 1)**: groups available in the data are the *Older American* and *Servicemember* tags. There is no gender, race or income, so other groups cannot be checked.

| Group | Complaints | Payout rate | Ranking (ROC-AUC) | Payout cases found in top 10% | Share of group flagged |
|---|---|---|---|---|---|
| All | 200,000 | 1.34% | 0.974 | 97% | 10% |
| Older American | 3,731 | 8.04% | 0.886 | 98% | 49% |
| Servicemember | 9,599 | 3.18% | 0.917 | 96% | 26% |
| No tag | 187,503 | 1.14% | 0.978 | 97% | 8.6% |

- **Equal opportunity holds:** every group's payout cases are found equally often (all gaps are within chance, bootstrap 95% intervals).
- **Ranking within the tagged groups is weaker, and that is not chance** (Older American -0.075 to -0.105 ROC-AUC vs everyone else). The model over-flags these groups (49% of Older American complaints land in the top 10%). That is acceptable for "who to review first", where being flagged means faster attention, and is the reason the score must **not** be used to deprioritise anyone.
- Older Americans are paid out more often than their product mix predicts (8.0% vs 5.4%); for Servicemembers the difference is mostly product mix.
- PR-AUC must not be compared across groups: it falls when a group's payout rate is lower, even with perfect ranking.

**Known limitations**
- **Calibration drift:** the payout rate keeps falling, so predicted percentages run high (1.82% predicted vs 1.34% actual on test, even after recalibration). Ranking is unaffected. To be fixed with rolling recalibration (Module 7).

---

## 4. RAG reply drafter, with guardrails and an LLM judge

| | |
|---|---|
| **Model** | Retrieval over 6 CFPB regulations + FCRA + FDCPA (BM25 + bge-small embeddings, reciprocal rank fusion, cross-encoder reranking, LLM query rewrite) -> `qwen2.5:3b` via Ollama writes a draft that cites the retrieved rules |
| **Purpose** | Give an agent a **first draft** that points to the regulation that applies |
| **Users** | Customer-care agents, who edit and approve every reply |
| **Not for** | Sending replies to customers without review; legal advice; deciding refunds or fault |

**Evaluation**
- **Retrieval:** a correct rule is in the top 5 for **29 of 34** answer-key complaints (Recall@5 0.81, MRR@10 0.66).
- **Faithfulness:** an LLM judge scored ~47% of rule statements as supported, but a human check of 20 statements showed 30% of them were not rule statements at all. By the human reading, ~79% (11/14) of real rule statements are supported (small sample).
- **Planted-error test (53 errors):** the LLM judge alone caught 40%, the code checks 79%, both together **85%**. Code checks catch wrong numbers, invented amounts and wrong citations. The judge catches some reversed meanings ("may" vs "may not"), which code cannot.

**Guardrails (Module 6, Step 2)**: rule-based checks that put a warning on a draft. They never block it, because a human reviews every draft anyway.
- Input: masks card/account numbers, SSNs, e-mails and phone numbers to XXXX; flags instruction-like text ("ignore previous instructions", fake `<system>` tags).
- Output: time limits must appear in a retrieved regulation passage; other numbers in a passage or the complaint; no refund/compensation promises; no admission of fault; no personal data; every citation must point to a retrieved passage.
- **Red-team test** (9 hand-written attacks + 1 normal complaint): 2 attacks changed the draft (a fake "3-day" deadline from an invented regulation; "we will waive all interest"). Both were flagged by the output checks, so **0 of 9 reached an agent unflagged**. Prompt hardening alone gave no measurable protection with a 3B model, so the plain prompt is kept and the checks do the work.
- On 27 real drafts, 7 got a risk warning, all for a time limit that no retrieved passage contains (e.g. "we will respond within 30 days").

**Known limitations**
- **Faithful is not the same as correct:** a draft can quote a passage accurately and still apply it to the wrong situation (e.g. a credit-card billing rule in a student-loan reply). Retrieval accuracy and human review address this; faithfulness scores do not.
- The 3B drafter sometimes mixes up rules, exaggerates their scope ("all of Regulation E"), drops a rule's "if" condition, or ignores the output format (10/27 drafts).
- **Reversed meanings are the blind spot:** of 8 planted ones, 5 passed both the judge and the code checks.
- The LLM judge is unreliable in both directions: a strict prompt rejects correct paraphrases; a lenient prompt accepted 60% of statements with an invented "$500 compensation". Its "supported" is weak evidence; its "unsupported" is more reliable.
- Rule-based checks only catch the failures they were written for; new kinds of attack need new tests.
- The regulations are a snapshot (eCFR as of 2026-10-13) and must be refreshed when they change.
