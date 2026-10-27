"""
Guardrails: cheap, rule-based checks around the RAG drafter (Module 6, used again by the API in Module 7).

Input checks  (the complaint, before it reaches the LLM)
  - injection: text that tries to give the LLM instructions ("ignore previous instructions", "you are now ...")
  - pii:       card / account / social-security numbers, e-mails, phone numbers (CFPB already masks most as XXXX)
Output checks (the draft, before an agent sees it)
  - citations: every [citation] must point to a retrieved passage            (from Module 4)
  - numbers:   every number/deadline must appear in a retrieved passage or in the complaint itself
  - promises:  no refund / compensation promises, no admission of fault - an agent decides those, not the LLM
  - pii:       no card / social-security numbers etc. in the reply
  - format:    "DRAFT REPLY:" and "RULES USED:" present

Nothing here blocks a draft outright: drafts always go to a human agent. A failed check puts a visible warning on
the draft and marks it "needs careful review". These checks are deliberately simple and explainable - they catch
specific, common failures; they do not prove a draft is correct.
"""
import re

from cfpb.rag import check_citations

INJECTION = re.compile(
    r"ignore (all |any |the |your )?(previous|prior|above|earlier)? ?(instructions|rules|prompts?)"
    r"|disregard (all |any |the |your )?(previous |prior |above )?(instructions|rules)"
    r"|forget (all |your )?(previous |prior )?(instructions|rules)"
    r"|you are now\b|from now on,? you|new instructions?:|system prompt|developer mode|jailbreak"
    r"|pretend (to be|you are)|act as (a|an|my)\b|respond (only )?(as|like) a\b|<\s*/?\s*(system|complaint)\s*>",
    re.I)

PII = {
    "card or account number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "social security number": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "e-mail address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "phone number": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
}

PROMISE = re.compile(
    r"\b(we|i)( will|'ll| shall| are going to| can) (immediately |promptly |fully )?"
    r"(refund|reimburse|compensate|pay you|waive|reverse the charges?|forgive)"
    r"|\byou (will|shall) (receive|get|be (given|paid|issued)) (a |an |the )?(full |complete )?"
    r"(refund|reimbursement|compensation|credit of \$)"
    r"|\b(we|i) (guarantee|promise)\b",
    re.I)
ADMISSION = re.compile(
    r"\b(our|my) (fault|mistake|error|negligence)\b|\bwe (made|have made) (a|an) (mistake|error)\b"
    r"|\bwe (are|were) (at fault|responsible for (this|the) (error|mistake)|liable)\b|\bwe admit\b", re.I)

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-five": 45,
         "forty five": 45, "sixty": 60, "ninety": 90, "hundred": 100}
CITATION_LIKE = re.compile(r"\[[^\]]*\]|\b\d+ (?:CFR|U\.S\.C\.) [\d.\-–a-z()]+|§+ ?[\d.\-–a-z()]+"
                           r"|\b\d{3,4}\.\d+[a-z]?(?:\([a-z0-9]+\))*", re.I)
NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _numbers(text):
    """Numbers in the text, as plain digit strings; citations and section numbers removed; number words converted."""
    t = CITATION_LIKE.sub(" ", str(text))
    for w, v in WORDS.items():
        t = re.sub(rf"\b{w}\b", str(v), t, flags=re.I)
    out = set()
    for m in NUMBER.findall(t):
        v = m.strip("$%").replace(",", "")
        v = v[:-2] if v.endswith(".0") else v
        if v and v not in {"0", "1"}:          # "1" appears in too many harmless phrases ("1 of your payments")
            out.add(v)
    return out


def reply_part(draft):
    """Only the customer-facing part of the answer is checked for promises and numbers."""
    return str(draft).split("RULES USED:")[0].replace("DRAFT REPLY:", "")


def check_input(complaint):
    inj = sorted({m.group(0).lower() for m in INJECTION.finditer(str(complaint))})
    pii = sorted({k for k, rx in PII.items() if rx.search(str(complaint))})
    return {"injection": inj, "pii": pii}


def check_output(draft, passages, complaint=""):
    """passages: DataFrame with 'citation' and 'text' (as returned by Retriever.search) or a list of dicts."""
    if hasattr(passages, "to_dict"):
        plist = passages[["citation", "text"]].to_dict("records")
    else:
        plist = list(passages)
    import pandas as pd
    cite = check_citations(draft, pd.DataFrame(plist, columns=["citation", "text"]))
    reply = reply_part(draft)
    allowed = _numbers(" ".join(p["text"] for p in plist)) | _numbers(complaint)
    ungrounded = sorted(_numbers(reply) - allowed, key=lambda x: float(x))
    res = {
        "invalid_citations": cite["invalid"],
        "ungrounded_numbers": ungrounded,
        "promises": sorted({m.group(0) for m in PROMISE.finditer(reply)}),
        "admissions": sorted({m.group(0) for m in ADMISSION.finditer(reply)}),
        "pii": sorted({k for k, rx in PII.items() if rx.search(reply)}),
        "format_ok": cite["format_ok"],
        "has_citation": cite["has_citation"],
    }
    res["warnings"] = [k for k in ("invalid_citations", "ungrounded_numbers", "promises", "admissions", "pii")
                       if res[k]] + ([] if res["format_ok"] else ["format"]) + ([] if res["has_citation"] else ["no_citation"])
    res["needs_careful_review"] = bool(res["warnings"])
    return res
