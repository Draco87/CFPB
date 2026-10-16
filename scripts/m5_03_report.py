"""
Module 5, Step 4: put the evaluation together in one readable report.

    uv run python scripts/m5_03_report.py

Needs reports/m5_retrieval*.csv (Step 1) and, from Kaggle (Step 3), models/m5/m5_judge_summary.json + judged_*.jsonl

Also creates reports/m5_judge_check.csv - 20 statements the judge ruled on, with the passages. Fill the column
`your_verdict` with yes (the passages say this) or no, save, and run this script again: the report then shows how
often you and the judge agree. This is how an LLM judge is validated - no legal knowledge needed, only careful reading.

Writes reports/m5_report.html
"""
import html
import json

import matplotlib.pyplot as plt
import pandas as pd

from cfpb import config
from cfpb.report import BLUE, INK, build_html, fig_html, pct, table_html

M5 = config.MODELS / "m5"
CHECK = config.REPORTS / "m5_judge_check.csv"
NAMES = {"m4_drafts": "Final setup (rewritten question, rules + interpretations only)",
         "m4_drafts_v1": "Version 1 (raw complaint as query, model forms included)"}


def search_section():
    res = pd.read_csv(config.REPORTS / "m5_retrieval.csv")
    det = pd.read_csv(config.REPORTS / "m5_retrieval_details.csv")
    s = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(s) + 0.8))
    ax.barh(s["setup"], s["found_rate"] * 100, color=BLUE, height=0.6)
    for y, (v, lab) in enumerate(zip(s["found_rate"], s["found_in_top5"])):
        ax.text(v * 100, y, f" {lab}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 115)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("Answer-key complaints where a correct rule was in the top 5", loc="left", color=INK, fontsize=11)
    tbl = pd.DataFrame({"Search setup": res["setup"], "Correct rule in top 5": res["found_in_top5"],
                        "Recall@5": [f"{v:.2f}" for v in res["recall_at_5"]],
                        "MRR@10": [f"{v:.2f}" for v in res["mrr_at_10"]]})
    best = res.loc[res["found_rate"].idxmax()]
    main = res[res["setup"].str.contains("final", case=False) & ~res["setup"].str.contains(",")]
    main = main.iloc[0] if len(main) else best
    sentences = [
        f"{len(det['id'].unique())} realistic complaints with a known correct rule (the answer key). For each search "
        "setup we check whether that rule is among the 5 passages the LLM would be given.",
        f"The setup used in Module 4 finds a correct rule for <b>{main['found_in_top5']}</b> complaints "
        f"(MRR {main['mrr_at_10']:.2f}: the correct rule is usually "
        f"{'1st' if main['mrr_at_10'] > 0.75 else '1st or 2nd' if main['mrr_at_10'] > 0.45 else 'further down'}).",
        f"Best setup: <b>{html.escape(best['setup'])}</b> ({best['found_in_top5']})."]
    final_name = main["setup"]
    missed = det[(det["config"] == final_name) & (~det["found_in_top5"])]
    bullets = [f"#{r.id} <b>{html.escape(r.product)}</b>: expected {html.escape(r.expected)} - got "
               f"{html.escape(str(r.top5))}<br><i>query: {html.escape(str(r.query))}</i>" for r in missed.itertuples()]
    if not bullets:
        bullets = ["The main setup found a correct rule for every answer-key complaint."]
    return ("1. Does the search find the right rules?", sentences,
            table_html(tbl, num_cols=("Correct rule in top 5", "Recall@5", "MRR@10")) + fig_html(fig),
            ["<b>Missed by the main setup:</b>"] + bullets)


def load_judged():
    out = {}
    for f in sorted(M5.glob("judged_*.jsonl")):
        out[f.stem.replace("judged_", "")] = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


def faithfulness_section(summary, judged):
    rows = []
    for stem, s in summary["files"].items():
        rows.append({"Drafts": NAMES.get(stem, stem),
                     "Rule statements": s["statements"],
                     "Supported by the passages": f"{s['supported']}/{s['statements']} ({pct(s['faithfulness'] or 0)})",
                     "Drafts with an unsupported statement": f"{s['drafts_with_unsupported_statement']}/{s['judged_ok']}",
                     "Addresses the problem (1-5)": f"{s['addresses_problem_mean']:.1f}" if s["addresses_problem_mean"] else "-",
                     "Contradicts the complaint": f"{s['contradicts_complaint']}/{s['judged_ok']}"})
    fin = summary["files"].get("m4_drafts") or next(iter(summary["files"].values()))
    sentences = [
        f"A larger model (<b>{summary['judge_model'].split('/')[-1]}</b>) acted as the judge. For every draft it listed "
        "each statement about what a rule requires, and marked it supported only if a retrieved passage clearly says it.",
        f"<b>Faithfulness</b> = supported statements / all statements. In the final setup, "
        f"<b>{pct(fin['faithfulness'] or 0)}</b> of rule statements are supported; "
        f"{fin['drafts_with_unsupported_statement']} of {fin['judged_ok']} drafts contain at least one unsupported statement."]
    if "m4_drafts_v1" in summary["files"] and "m4_drafts" in summary["files"]:
        v1, v2 = summary["files"]["m4_drafts_v1"], summary["files"]["m4_drafts"]
        if v1["faithfulness"] is not None and v2["faithfulness"] is not None:
            sentences.append(f"Compared with Version 1, faithfulness changed from {pct(v1['faithfulness'])} to "
                             f"{pct(v2['faithfulness'])} ({(v2['faithfulness'] - v1['faithfulness']) * 100:+.0f} points).")
    examples = []
    for j in judged.get("m4_drafts", next(iter(judged.values()), [])):
        for c in (j["judge"] or {}).get("claims", []):
            if not c.get("supported") and len(examples) < 6:
                examples.append(f"<b>{html.escape(str(j['product']))}</b>: \"{html.escape(str(c.get('claim')))}\"")
    return ("2. Do the drafts stick to the rules they were given? (faithfulness)", sentences,
            table_html(pd.DataFrame(rows)), ["<b>Examples of unsupported statements:</b>"] + (examples or ["none"]))


def judge_check_section(judged):
    recs = judged.get("m4_drafts") or next(iter(judged.values()), [])
    if not CHECK.exists():
        rows = []
        for j in recs:
            for c in (j["judge"] or {}).get("claims", []):
                ex = c.get("excerpt")
                passage = (j["retrieved"][ex - 1] if isinstance(ex, int) and 1 <= ex <= len(j["retrieved"]) else None)
                rows.append({"product": j["product"], "statement": c.get("claim"),
                             "judge_says_supported": bool(c.get("supported")),
                             "passage_judge_pointed_to": f"[{passage['citation']}] {passage['text']}" if passage else "",
                             "all_passages": "\n\n".join(f"[{p['citation']}] {p['text']}" for p in j["retrieved"]),
                             "your_verdict": ""})
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sample(min(20, len(df)), random_state=config.SEED).reset_index(drop=True)
        df.to_csv(CHECK, index=False, encoding="utf-8-sig")
        return ("3. Can we trust the judge?", [
            f"Created <b>{CHECK.name}</b> with {len(df)} statements the judge ruled on. Open it in Excel, read each "
            "statement and the passages, and write <b>yes</b> (a passage says this) or <b>no</b> in <i>your_verdict</i>. "
            "Save and run this script again to see how often you and the judge agree."], "", [])
    df = pd.read_csv(CHECK)
    done = df[df["your_verdict"].astype(str).str.strip().str.lower().isin(["yes", "no"])]
    if not len(done):
        return ("3. Can we trust the judge?", [f"<b>{CHECK.name}</b> exists but no verdicts are filled in yet."], "", [])
    you = done["your_verdict"].str.strip().str.lower() == "yes"
    judge = done["judge_says_supported"].astype(str).str.lower().isin(["true", "1"])
    agree = (you == judge).mean()
    return ("3. Can we trust the judge?", [
        f"You checked <b>{len(done)}</b> of the judge's rulings by reading the passages yourself. "
        f"You agreed with the judge on <b>{int((you == judge).sum())}/{len(done)}</b> ({pct(agree)}).",
        f"Where you disagreed: the judge said 'supported' but you said no in {int((judge & ~you).sum())} cases "
        f"(judge too lenient), and the reverse in {int((~judge & you).sum())} cases (judge too strict)."], "", [])


def main():
    sections = [("What Module 5 measures", [
        "Two questions, both answered without needing legal expertise: (1) does the search find the right rules - "
        "checked against an answer key; (2) do the drafts say only what those rules say - checked by a larger LLM "
        "acting as a judge, which is itself spot-checked by a human."], "", [])]
    if (config.REPORTS / "m5_retrieval.csv").exists():
        sections.append(search_section())
    else:
        print("Step 1 results not found - run scripts/m5_01_retrieval_eval.py first.")
    summ = M5 / "m5_judge_summary.json"
    if summ.exists():
        summary, judged = json.loads(summ.read_text()), load_judged()
        sections.append(faithfulness_section(summary, judged))
        sections.append(judge_check_section(judged))
    else:
        print(f"Judge results not found in {M5} - download them from Kaggle (Step 3).")
    out = config.REPORTS / "m5_report.html"
    out.write_text(build_html("RAG evaluation", "Answer-key retrieval tests and LLM-as-judge faithfulness.", sections),
                   encoding="utf-8")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
