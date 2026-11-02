"""
Complaint Intelligence API (Module 7).

    uv run uvicorn cfpb.api:app --reload          then open  http://127.0.0.1:8000/docs

Endpoints
  GET  /health    which parts are ready (models on disk, Ollama running)
  POST /triage    which of the 9 products is this complaint about?        (TF-IDF model, ~2 ms)
  POST /risk      how likely is a payout, and why? (top 3 reasons)        (XGBoost + SHAP, ~50 ms)
  POST /draft     regulation-cited draft reply + guardrail warnings        (RAG + local LLM, ~30 s)
  POST /analyze   all of the above in one call (draft optional)

Privacy: card/account numbers, SSNs, e-mails and phone numbers are masked before the text reaches any model,
and the request log (logs/requests.jsonl) stores numbers and labels only - never the complaint text.
"""
import json
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from cfpb import config
from cfpb.guardrails import check_input, check_output, redact_pii
from cfpb.text import clean_text

LOG = config.ROOT / "logs" / "requests.jsonl"
PRODUCTS = ["Bank account", "Credit / prepaid card", "Credit reporting", "Debt collection", "Money transfer / virtual currency",
            "Mortgage", "Payday / personal loan", "Student loan", "Vehicle loan"]


# ---------------------------------------------------------------- request / response shapes
class Complaint(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000, description="The complaint narrative")
    product: Optional[str] = Field(None, description=f"One of: {', '.join(PRODUCTS)}. If empty, /triage decides.")
    sub_product: Optional[str] = None
    issue: Optional[str] = None
    company: Optional[str] = None
    state: Optional[str] = Field(None, description="Two-letter US state, e.g. CA")
    submitted_via: Optional[str] = Field(None, description="e.g. Web, Phone, Referral")
    tags: Optional[str] = Field(None, description="e.g. 'Older American' or 'Servicemember'")
    date: Optional[str] = Field(None, description="YYYY-MM-DD; default today")

    model_config = {"json_schema_extra": {"examples": [{
        "text": "Someone made two ATM withdrawals of $200 from my checking account that I did not make. I reported "
                "it to the bank the same day but they have not investigated.",
        "company": "BANK OF AMERICA, NATIONAL ASSOCIATION", "state": "CA", "submitted_via": "Web"}]}}


class AnalyzeRequest(Complaint):
    include_draft: bool = Field(False, description="Also draft a reply (slow: ~30 s on a laptop GPU)")


# ---------------------------------------------------------------- the models, loaded once on first use
class Components:
    def __init__(self):
        self._triage = self._risk = self._retriever = None

    def triage(self):
        if self._triage is None:
            import joblib
            path = config.MODELS / "m1_tfidf_logreg.joblib"
            if not path.exists():
                raise HTTPException(503, f"Triage model missing: {path.name}. Run scripts/m1_01_train_baseline.py")
            self._triage = joblib.load(path)
        return self._triage

    def risk(self):
        if self._risk is None:
            from cfpb.risk import ARTIFACTS, RiskModel
            if not ARTIFACTS.exists():
                raise HTTPException(503, "Risk artifacts missing. Run scripts/m7_01_export_artifacts.py")
            self._risk = RiskModel()
        return self._risk

    def retriever(self):
        if self._retriever is None:
            from cfpb.rag import INDEX_DIR, Retriever
            if not (INDEX_DIR / "chunks.parquet").exists():
                raise HTTPException(503, "Regulation index missing. Run scripts/m4_02_build_index.py")
            self._retriever = Retriever(rerank=True)
        return self._retriever

    def answer(self, complaint, product, issue):
        from cfpb.rag import DEFAULT_LLM, answer_complaint, ollama_ok
        ok, msg = ollama_ok(DEFAULT_LLM)
        if not ok:
            raise HTTPException(503, msg)
        return answer_complaint(self.retriever(), complaint, product, issue)


_components = Components()


def get_components():
    return _components


# ---------------------------------------------------------------- helpers
def log_request(endpoint, started, **fields):
    """One JSON line per request - numbers and labels only, never the complaint text (privacy)."""
    LOG.parent.mkdir(exist_ok=True)
    rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "endpoint": endpoint,
           "latency_ms": round((time.perf_counter() - started) * 1000, 1), **fields}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def run_triage(c: Components, text):
    model = c.triage()
    probs = model.predict_proba([clean_text(text)])[0]
    order = probs.argsort()[::-1]
    classes = list(model.classes_)
    return {"product": classes[order[0]], "confidence": round(float(probs[order[0]]), 3),
            "alternatives": [{"product": classes[i], "probability": round(float(probs[i]), 3)} for i in order[1:3]]}


def run_risk(c: Components, req: Complaint, product):
    row = req.model_dump(include={"sub_product", "issue", "company", "state", "submitted_via", "tags", "date"})
    row.update(text=redact_pii(req.text), product=product)
    return c.risk().predict([row])[0]


def run_draft(c: Components, req: Complaint, product):
    flags = check_input(req.text)
    safe = redact_pii(req.text)                       # personal numbers never reach the search or the LLM
    res = c.answer(safe, product, req.issue)
    out = check_output(res["draft"], res["passages"], safe)
    return {"draft": res["draft"],
            "rules_retrieved": sorted(set(res["passages"]["citation"])),
            "input_flags": flags,
            "guardrail_warnings": out["warnings"],
            "format_warning": out["format_warning"],
            "needs_careful_review": out["needs_careful_review"] or bool(flags["injection"]),
            "details": {k: out[k] for k in ("ungrounded_numbers", "promises", "admissions", "invalid_citations")},
            "seconds": res["seconds"]}


# ---------------------------------------------------------------- the API
app = FastAPI(title="Complaint Intelligence API",
              description="Triage, payout risk and regulation-cited draft replies for US consumer-finance "
                          "complaints. Drafts are for a human agent to review - never sent automatically.",
              version="0.7.0")


@app.get("/health")
def health():
    from cfpb.rag import DEFAULT_LLM, INDEX_DIR, ollama_ok
    from cfpb.risk import ARTIFACTS
    ok, msg = ollama_ok(DEFAULT_LLM)
    return {"status": "ok",
            "triage_model": (config.MODELS / "m1_tfidf_logreg.joblib").exists(),
            "risk_model": ARTIFACTS.exists() and (config.MODELS / "m3_xgb_full.json").exists(),
            "regulation_index": (INDEX_DIR / "chunks.parquet").exists(),
            "llm": "ready" if ok else msg}


@app.post("/triage")
def triage(req: Complaint, c: Components = Depends(get_components)):
    t0 = time.perf_counter()
    out = run_triage(c, redact_pii(req.text))
    log_request("triage", t0, product=out["product"], confidence=out["confidence"], n_chars=len(req.text))
    return out


@app.post("/risk")
def risk(req: Complaint, c: Components = Depends(get_components)):
    t0 = time.perf_counter()
    product, source = req.product, "given"
    if product is None:
        product, source = run_triage(c, redact_pii(req.text))["product"], "triage"
    elif product not in PRODUCTS:
        raise HTTPException(422, f"Unknown product '{product}'. Use one of: {', '.join(PRODUCTS)}")
    out = {"product": product, "product_source": source, **run_risk(c, req, product)}
    log_request("risk", t0, product=product, payout_probability=out["payout_probability"],
                risk_band=out["risk_band"], n_chars=len(req.text))
    return out


@app.post("/draft")
def draft(req: Complaint, c: Components = Depends(get_components)):
    t0 = time.perf_counter()
    product = req.product or run_triage(c, redact_pii(req.text))["product"]
    out = {"product": product, **run_draft(c, req, product)}
    log_request("draft", t0, product=product, warnings=out["guardrail_warnings"],
                injection_flag=bool(out["input_flags"]["injection"]), n_chars=len(req.text))
    return out


@app.post("/analyze")
def analyze(req: AnalyzeRequest, c: Components = Depends(get_components)):
    t0 = time.perf_counter()
    tri = run_triage(c, redact_pii(req.text))
    product = req.product or tri["product"]
    out = {"triage": tri, "product_used": product, "risk": run_risk(c, req, product)}
    if req.include_draft:
        out["draft"] = run_draft(c, req, product)
    log_request("analyze", t0, product=product, confidence=tri["confidence"],
                payout_probability=out["risk"]["payout_probability"], n_chars=len(req.text),
                warnings=out.get("draft", {}).get("guardrail_warnings"))
    return out
