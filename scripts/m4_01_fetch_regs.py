"""
Module 4, Step 1: download the regulations that govern consumer complaints and split them into sections.

    uv add requests
    uv run python scripts/m4_01_fetch_regs.py

Sources (official, free):
  eCFR API (Code of Federal Regulations, current version) - CFPB regulations:
    12 CFR 1005  Regulation E  - electronic transfers, debit cards, prepaid, remittances
    12 CFR 1006  Regulation F  - debt collection
    12 CFR 1022  Regulation V  - credit reporting (furnishers)
    12 CFR 1024  Regulation X  - mortgage servicing
    12 CFR 1026  Regulation Z  - credit cards, loans, mortgages (truth in lending)
    12 CFR 1030  Regulation DD - deposit accounts (truth in savings)
  govinfo (US Code) - two laws whose key consumer rules live in the statute itself:
    15 U.S.C. 1681 ff.  Fair Credit Reporting Act      (credit report disputes, 15 U.S.C. 1681i)
    15 U.S.C. 1692 ff.  Fair Debt Collection Practices Act

Raw files -> data/raw/regs/ ; sections -> data/interim/reg_units.parquet
One row per section (or per commentary block): source, citation, title, text, products.
"""
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pandas as pd
import requests

from cfpb import config

RAW = config.DATA / "raw" / "regs"
RAW.mkdir(parents=True, exist_ok=True)
OUT = config.INTERIM / "reg_units.parquet"

CFR_PARTS = {  # part -> (short name, products it mainly applies to)
    "1005": ("Regulation E", ["Bank account", "Money transfer / virtual currency", "Credit / prepaid card"]),
    "1006": ("Regulation F", ["Debt collection"]),
    "1022": ("Regulation V", ["Credit reporting", "Debt collection"]),
    "1024": ("Regulation X", ["Mortgage"]),
    "1026": ("Regulation Z", ["Credit / prepaid card", "Mortgage", "Vehicle loan", "Payday / personal loan",
                              "Student loan"]),
    "1030": ("Regulation DD", ["Bank account"]),
}
USC = {  # file key -> (url, short name, products)
    "fcra": ("https://www.govinfo.gov/content/pkg/USCODE-2023-title15/html/USCODE-2023-title15-chap41-subchapIII.htm",
             "Fair Credit Reporting Act", ["Credit reporting", "Debt collection"]),
    "fdcpa": ("https://www.govinfo.gov/content/pkg/USCODE-2023-title15/html/USCODE-2023-title15-chap41-subchapV.htm",
              "Fair Debt Collection Practices Act", ["Debt collection"]),
}
HEADERS = {"User-Agent": "cfpb-portfolio-project (educational)", "Accept-Encoding": "gzip"}
NOTE_HEADINGS = ("Editorial Notes", "Statutory Notes and Related Subsidiaries", "Executive Documents")


def clean(s):
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n\s*\n+", "\n", s)).strip()


def get(url, path):
    if path.exists():
        return path.read_bytes()
    print(f"  downloading {url}")
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content


# ---------- eCFR (XML)
def text_of(el, skip=("HEAD", "CITA")):
    parts = []
    for child in el:
        if child.tag in skip:
            continue
        t = "".join(child.itertext()).strip()
        if t:
            parts.append(t)
    return clean("\n".join(parts))


def cite_block(part, head, block_title, kind):
    """'Section 1005.11-...' or '6(b)(3) Periodic Statement...' -> 12 CFR 1005.11 / 12 CFR 1005.6"""
    if kind == "interpretation":
        sec = re.search(r"Section\s+(\d+\.\d+[a-z]?)", block_title)
        if sec:
            return f"12 CFR {sec.group(1)} (Official Interpretation)"
        par = re.match(r"(?:Paragraph\s+|Comment\s+)?(\d+[a-z]?)\(", block_title)
        if par:
            return f"12 CFR {part}.{par.group(1)} (Official Interpretation)"
    return f"12 CFR {part}, {head.split('—')[0].strip()}"


def parse_cfr(xml_bytes, part, name, products):
    root = ET.fromstring(xml_bytes)
    units = []
    for el in root.iter():
        if not (el.tag.startswith("DIV") and el.get("TYPE")):
            continue
        typ, n = el.get("TYPE", "").upper(), el.get("N", "")
        head = clean("".join(el.find("HEAD").itertext())) if el.find("HEAD") is not None else n
        if typ == "SECTION":
            units.append({"citation": f"12 CFR {n}", "title": head, "text": text_of(el), "kind": "rule"})
        elif typ == "APPENDIX" or "SUPPLEMENT" in n.upper() or "APPENDIX" in n.upper():
            # Supplement I = official interpretations of the rules (kept for search).
            # Other appendices = model forms/clauses and formulas (wording templates, not rules).
            kind = "interpretation" if "SUPPLEMENT" in n.upper() else "appendix"
            block_title, buf = head, []

            def flush():
                body = clean("\n".join(buf))
                if len(body) > 40:
                    units.append({"citation": cite_block(part, head, block_title, kind),
                                  "title": block_title, "text": body, "kind": kind})

            for child in el:
                if child.tag == "HEAD":
                    continue
                if child.tag in ("HD1", "HD2") and buf:
                    flush()
                    buf = []
                if child.tag in ("HD1", "HD2"):
                    block_title = clean("".join(child.itertext()))
                else:
                    t = "".join(child.itertext()).strip()
                    if t:
                        buf.append(t)
            flush()
    for u in units:
        u.update(source=f"{name} (12 CFR {part})", products=products)
    return units


# ---------- govinfo US Code (HTML)
class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table"}

    def __init__(self):
        super().__init__()
        self.out = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data)


SECTION_RE = re.compile(r"^§\s*(\d+[0-9a-zA-Z\-–]*)\.\s*(.*)$")


def parse_usc(html_bytes, name, products):
    p = _HTMLText()
    p.feed(html_bytes.decode("utf-8", errors="replace"))
    lines = [ln.strip() for ln in "".join(p.out).replace("\xa0", " ").split("\n")]
    units, cur, buf, in_notes = [], None, [], False

    def flush():
        if cur and buf:
            units.append({"citation": f"15 U.S.C. {cur[0]}", "title": f"§{cur[0]}. {cur[1]}",
                          "text": clean("\n".join(buf)), "source": name, "products": products, "kind": "rule"})

    for ln in lines:
        if not ln:
            continue
        m = SECTION_RE.match(ln)
        if m:
            flush()
            cur, buf, in_notes = (m.group(1), m.group(2)), [], False
            continue
        if ln in NOTE_HEADINGS:          # amendment history etc. - not the rule itself
            in_notes = True
        if cur and not in_notes:
            buf.append(ln)
    flush()
    return [u for u in units if "Repealed" not in u["title"] and len(u["text"]) > 40]


def main():
    date = requests.get("https://www.ecfr.gov/api/versioner/v1/titles.json", headers=HEADERS,
                        timeout=60).json()
    date = next(t for t in date["titles"] if t["number"] == 12)["up_to_date_as_of"]
    print(f"eCFR title 12 is current as of {date}")

    units = []
    for part, (name, products) in CFR_PARTS.items():
        xml = get(f"https://www.ecfr.gov/api/versioner/v1/full/{date}/title-12.xml?part={part}",
                  RAW / f"12cfr{part}_{date}.xml")
        u = parse_cfr(xml, part, name, products)
        print(f"  {name:<14} 12 CFR {part}: {len(u):>4} units, {sum(len(x['text'].split()) for x in u):>8,} words")
        units += u
    for key, (url, name, products) in USC.items():
        u = parse_usc(get(url, RAW / f"usc_{key}.htm"), name, products)
        print(f"  {name:<34}: {len(u):>4} sections, {sum(len(x['text'].split()) for x in u):>8,} words")
        units += u

    df = pd.DataFrame(units)
    df["n_words"] = df["text"].str.split().str.len()
    print("\nUnits by kind:", df["kind"].value_counts().to_dict(), "(appendices = model forms, left out of search)")
    df.to_parquet(OUT, index=False)
    print(f"\n{len(df):,} units, {df['n_words'].sum():,} words -> {OUT}")
    print("\nExamples:")
    for cit in ["12 CFR 1005.11", "15 U.S.C. 1681i", "12 CFR 1026.13"]:
        row = df[df["citation"] == cit]
        if len(row):
            r = row.iloc[0]
            print(f"  [{r['citation']}] {r['title']}\n    {r['text'][:160]}...")
        else:
            print(f"  {cit}: NOT FOUND - paste this output to Claude")


if __name__ == "__main__":
    main()
