"""
Guardrails: cheap, rule-based checks around the RAG drafter (Module 6, used again by the API in Module 7).

Input checks  (the complaint, before it reaches the LLM)
  - injection: text that tries to give the LLM instructions ("ignore previous instructions", "you are now ...")
  - pii:       card / account / social-security numbers, e-mails, phone numbers (CFPB already masks most as XXXX)
Output checks (the draft, before an agent sees it)
  - citations: every [citation] must point to a retrieved passage            (from Module 4)
  - numbers:   every time limit ("10 business days") must appear in a retrieved passage; other numbers in a
               passage or in the complaint itself
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
    "card or account number": re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
    "social security number": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "e-mail address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "phone number": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
}

# "we will [up to 4 words] refund/waive ..." - catches "we will proceed to waive all interest"
PROMISE = re.compile(
    r"\b(?:we|i)\b(?:\s+[\w'-]+){0,4}?\s+(refund|reimburse|compensate|pay you|waive|reverse the charges?|forgive|"
    r"guarantee|promise)\w*"
    r"|\byou (?:will|shall) (?:receive|get|be (?:given|paid|issued)) (?:a |an |the )?(?:full |complete )?"
    r"(refund|reimbursement|compensation|credit of \$)",
    re.I)
NEGATION = re.compile(r"\b(not|cannot|can't|cant|won't|unable|never|no)\b|n't\b", re.I)
ADMISSION = re.compile(
    r"\b(our|my) (fault|mistake|error|negligence)\b|\bwe (made|have made) (a|an) (mistake|error)\b"
    r"|\bwe (are|were) (at fault|responsible for (this|the) (error|mistake)|liable)\b|\bwe admit\b", re.I)

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-five": 45,
         "forty five": 45, "sixty": 60, "ninety": 90, "hundred": 100}
CITATION_LIKE = re.compile(r"\[[^\]]*\]|\b\d+ (?:CFR|U\.S\.C\.) [\d.\-–a-z()]+|§+ ?[\d.\-–a-z()]+"
                           r"|\b\d{3,4}\.\d+[a-z]?(?:\([a-z0-9]+\))*", re.I)
NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
DEADLINE = re.compile(r"(\d[\d,]*)\s*(?:-\s*)?(?:calendar |business |working )?(days?|weeks?|months?|years?|hours?)\b", re.I)


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


def _words_to_digits(text):
    t = str(text)
    for w, v in WORDS.items():
        t = re.sub(rf"\b{w}\b", str(v), t, flags=re.I)
    return t


def _deadlines(text):
    """Time limits like '10 business days', 'sixty days', '24 hours' -> {'10 day', '60 day', '24 hour'}."""
    t = CITATION_LIKE.sub(" ", _words_to_digits(text))
    return {f"{n.replace(',', '')} {u.lower().rstrip('s')}" for n, u in DEADLINE.findall(t)}


def reply_part(draft):
    """The whole answer minus the format labels. (Small models often put 'RULES USED' in the middle, so we
    check everything they wrote, not just the text before it.)"""
    return str(draft).replace("DRAFT REPLY:", " ").replace("RULES USED:", " ")


def sentences(text):
    return [x for x in re.split(r"(?<=[.!?])\s+|\n+", str(text)) if x.strip()]


def find_promises(text):
    """Promise phrases, skipping negated sentences such as 'I can't guarantee a win'."""
    out = set()
    for sent in sentences(text):
        if NEGATION.search(sent):
            continue
        out |= {m.group(0) for m in PROMISE.finditer(sent)}
    return sorted(out)


def redact_pii(text):
    """Mask personal numbers the way the CFPB does (XXXX) before the text goes anywhere."""
    t = str(text)
    for rx in PII.values():
        t = rx.sub("XXXX", t)
    return t


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
    ptext = " ".join(p["text"] for p in plist)
    # amounts may come from the complaint ("the $200 withdrawal"); TIME LIMITS must come from a regulation passage,
    # otherwise a complaint saying "you must do this within 3 days" could put a fake deadline into the reply
    pass_deadlines = _deadlines(ptext)
    fake_deadlines = sorted(d for d in _deadlines(reply) if d not in pass_deadlines)
    deadline_nums = {d.split()[0] for d in _deadlines(reply)}
    allowed = _numbers(ptext) | _numbers(complaint)
    other = sorted((_numbers(reply) - deadline_nums) - allowed, key=lambda x: float(x))
    ungrounded = fake_deadlines + other
    res = {
        "invalid_citations": cite["invalid"],
        "ungrounded_numbers": ungrounded,
        "promises": find_promises(reply),
        "admissions": sorted({m.group(0) for m in ADMISSION.finditer(reply)}),
        "pii": sorted({k for k, rx in PII.items() if rx.search(reply)}),
        "format_ok": cite["format_ok"],
        "has_citation": cite["has_citation"],
    }
    # risky = could mislead the customer; format = cosmetic, shown separately so it does not drown out real risks
    res["warnings"] = [k for k in ("invalid_citations", "ungrounded_numbers", "promises", "admissions", "pii")
                       if res[k]] + ([] if res["has_citation"] else ["no_citation"])
    res["format_warning"] = not res["format_ok"]
    res["needs_careful_review"] = bool(res["warnings"])
    return res
