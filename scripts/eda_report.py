"""
Basic analysis of the CFPB complaints - trends and anomalies, explained in plain English.

    uv run python scripts/eda_report.py

Reads the cleaned archive (data/interim/narratives_parts; the complaint text is skipped, so it is quick)
Writes reports/eda_report.html  -> open it in your browser.

How anomalies are found (simple and explainable):
  every month is compared with the MEDIAN of the 6 months before it.
  spike = at least 1.5x that usual level; drop = half of it or less.
  For product shares a change must also be at least 3 percentage points to count.
"""
import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfpb import config
from cfpb.data import load_combined

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 110})
COLS = ["date", "product", "issue", "company", "state", "tags", "submitted_via",
        "company_response", "timely", "n_chars"]
WINDOW, SPIKE, DROP = 6, 1.5, 0.5


# ---------- plain-English helpers
def n(x):
    x = float(x)
    if x >= 1e6:
        return f"{x / 1e6:.1f} million"
    return f"{x:,.0f}"


def pct(p):
    return f"{p * 100:.0f}%" if p >= 0.01 else f"{p * 100:.1f}%"


def about(p):
    """0.74 -> 'about 3 in 4'"""
    for num, den in [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5),
                     (1, 10), (9, 10), (1, 20)]:
        if abs(p - num / den) < 0.025:
            return f"about {num} in {den}"
    return ""


def share_text(p):
    a = about(p)
    return f"{pct(p)} ({a})" if a else pct(p)


def mon(period):
    return period.strftime("%b %Y")


def fig_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def hbar(series, title, value_fmt=pct, color=BLUE):
    s = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(s) + 0.8))
    ax.barh([str(i)[:55] for i in s.index], s.values, color=color, height=0.6)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(s.values):
        ax.text(v, y, " " + value_fmt(v), va="center", color=INK, fontsize=9)
    ax.set_xlim(0, s.max() * 1.18)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    return fig_html(fig)


def dates(ax):
    loc = matplotlib.dates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(loc))


def pct_axis(ax, values):
    decimals = 1 if np.nanmax(values) - np.nanmin(values) < 5 else 0
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.{decimals}f}%"))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))


def flag(series, min_abs_change=0.0):
    """Compare each month with the median of the previous WINDOW months."""
    base = series.shift(1).rolling(WINDOW, min_periods=3).median()
    ratio = series / base
    out = []
    for m in series.index:
        r, b, v = ratio.get(m), base.get(m), series.get(m)
        if pd.isna(r) or abs(v - b) < min_abs_change:
            continue
        if r >= SPIKE:
            out.append((m, "spike", v, b, r))
        elif r <= DROP:
            out.append((m, "drop", v, b, r))
    return out


# ---------- report sections: each returns (title, sentences, chart_html, bullets)
def sec_overview(df):
    months = df["month"].nunique()
    s = [f"The archive holds <b>{n(len(df))}</b> complaints that have a written narrative, "
         f"from <b>{mon(df['month'].min())}</b> to <b>{mon(df['month'].max())}</b> ({months} months).",
         f"They name <b>{n(df['company'].nunique())}</b> companies, {df['product'].nunique()} products "
         f"and {df['issue'].nunique()} different issues."]
    return "At a glance", s, "", []


def sec_volume(df):
    m = df.groupby("month").size()
    anomalies = flag(m)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = m.index.to_timestamp()
    ax.plot(x, m.values, color=BLUE, lw=2)
    for a in anomalies:
        ax.plot(a[0].to_timestamp(), a[2], "o", color=ORANGE, ms=8, mec="white", mew=2)
    ax.set_ylim(0, m.max() * 1.1)
    dates(ax)
    ax.set_title("Complaints per month (orange = unusual month)", loc="left", color=INK, fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))

    per_month = df.groupby(df["date"].dt.year).size() / df.groupby(df["date"].dt.year)["month"].nunique()
    full = per_month.round()
    first, last = full.index.min(), full.index.max()
    s = ["A typical month by year: " + ", ".join(f"{y}: {n(v)}" for y, v in full.items()) + "."]
    if full.loc[first] > 0:
        s.append(f"From {first} to {last}, a typical month went from about {n(full.loc[first])} to "
                 f"{n(full.loc[last])} complaints ({full.loc[last] / full.loc[first]:.1f}x).")
    bullets = [f"<b>{mon(a[0])}</b>: {n(a[2])} complaints - a {a[1]}, {a[4]:.1f}x the usual "
               f"level of about {n(a[3])}." for a in anomalies]
    if not bullets:
        bullets = ["No month was unusually high or low compared with the 6 months before it."]
    return "How many complaints, and when", s, fig_html(fig), bullets


def sec_products(df):
    share = df["product"].value_counts(normalize=True)
    s = [f"<b>{share.index[0]}</b> dominates: {share_text(share.iloc[0])} of all complaints.",
         f"The next biggest are {share.index[1]} ({pct(share.iloc[1])}) and {share.index[2]} "
         f"({pct(share.iloc[2])}). The smallest is {share.index[-1]} ({pct(share.iloc[-1])})."]
    bullets = ["Because one product is so large, overall accuracy can look good even when rare "
               "products are handled badly - that is why the models are scored with macro-F1."]
    return "Which products people complain about", s, hbar(share, "Share of all complaints"), bullets


def sec_mix(df):
    share = pd.crosstab(df["month"], df["product"], normalize="index")
    order = df["product"].value_counts().index
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), sharex=True)
    bullets = []
    for ax, p in zip(axes.flat, order):
        sr = share[p]
        x = sr.index.to_timestamp()
        ax.plot(x, sr.values * 100, color=BLUE, lw=1.6)
        for a in flag(sr, min_abs_change=0.03):
            ax.plot(a[0].to_timestamp(), a[2] * 100, "o", color=ORANGE, ms=7, mec="white", mew=1.5)
            bullets.append(f"<b>{p}</b> in <b>{mon(a[0])}</b>: {pct(a[2])} of complaints, "
                           f"usually about {pct(a[3])} - a {a[1]}.")
        ax.set_title(p, loc="left", fontsize=9, color=INK)
        ax.tick_params(labelsize=7)
        pct_axis(ax, sr.values * 100)
        dates(ax)
    fig.suptitle("Each product's share of the month (each panel has its own scale; orange = unusual)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    s = ["Each small chart shows what share of each month's complaints went to one product. "
         "A flat line means a stable mix; a jump means something changed."]
    if not bullets:
        bullets = ["No product's share jumped or fell sharply in any month."]
    bullets.append("Sudden jumps can come from a real event (for example a problem at one big company) "
                   "or from a change in how the CFPB publishes or labels complaints.")
    return "Has the product mix changed over time?", s, fig_html(fig), bullets


def sec_companies(df):
    top = df["company"].value_counts(normalize=True).head(10)
    top3 = top.head(3)
    s = [f"The top 3 companies - {', '.join(top3.index)} - receive {share_text(top3.sum())} "
         f"of all complaints.",
         f"The top 10 together receive {pct(top.sum())}."]
    bullets = ["When a few companies account for most complaints, anything unusual at one of them "
               "shows up in the totals - a useful first check when a spike appears."]
    return "Which companies", s, hbar(top, "Share of all complaints - top 10 companies"), bullets


def sec_issues(df):
    top = df["issue"].value_counts(normalize=True).head(10)
    s = [f"The most common issue is <b>{top.index[0]}</b> ({pct(top.iloc[0])} of complaints)."]
    end = df["month"].max()
    recent = df[df["month"] > end - 12]
    before = df[(df["month"] <= end - 12) & (df["month"] > end - 24)]
    bullets = []
    if len(recent) and len(before):
        r = recent["issue"].value_counts(normalize=True)
        b = before["issue"].value_counts(normalize=True)
        both = pd.concat([r, b], axis=1, keys=["recent", "before"]).fillna(0)
        both = both[(both["recent"] >= 0.005) | (both["before"] >= 0.005)]
        both["change"] = both["recent"] - both["before"]
        big = both[both["change"].abs() >= 0.01]                # at least 1 percentage point
        for name, row in big.sort_values("change", ascending=False).head(3).iterrows():
            if row["change"] > 0:
                bullets.append(f"Rising: <b>{name}</b> - {pct(row['before'])} -> {pct(row['recent'])} "
                               f"of complaints (last 12 months vs the 12 before).")
        for name, row in big.sort_values("change").head(3).iterrows():
            if row["change"] < 0:
                bullets.append(f"Falling: <b>{name}</b> - {pct(row['before'])} -> {pct(row['recent'])}.")
        if len(big):
            bullets.append("Some rises and falls are the CFPB renaming issue categories, not real changes.")
        else:
            bullets.append("No issue's share changed by more than 1 percentage point between the "
                           "last 12 months and the 12 before.")
    return "What the complaints are about", s, hbar(top, "Share of all complaints - top 10 issues"), bullets


def sec_responses(df):
    resp = df["company_response"].value_counts(normalize=True)
    timely = (df["timely"] == "Yes").groupby(df["month"]).mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(timely.index.to_timestamp(), timely.values * 100, color=BLUE, lw=2)
    ax.set_ylim(max(0, timely.min() * 100 - 5), 100.5)
    pct_axis(ax, timely.values * 100)
    dates(ax)
    ax.set_title("Share of complaints answered on time, by month", loc="left", color=INK, fontsize=11)
    overall = (df["timely"] == "Yes").mean()
    low = timely[timely < timely.median() - 0.05]
    s = [f"Companies answered {share_text(overall)} of complaints on time (within 15 days).",
         f"The most common outcome is <b>{resp.index[0]}</b> ({pct(resp.iloc[0])})."]
    money = resp[[i for i in resp.index if "monetary relief" in str(i) and "non" not in str(i)]].sum()
    if money:
        s.append(f"Only {pct(money)} of complaints ended with money paid back to the consumer.")
    bullets = [f"<b>{mon(m)}</b>: only {pct(v)} answered on time." for m, v in low.items()]
    if not bullets:
        bullets = ["The on-time rate stayed steady - no month fell more than 5 points below normal."]
    return ("How companies respond", s,
            hbar(resp, "How complaints were closed") + fig_html(fig), bullets)


def sec_people(df):
    states = df["state"].value_counts(normalize=True).head(10)
    tags = df["tags"].fillna("None").value_counts(normalize=True)
    older = tags[[t for t in tags.index if "Older American" in t]].sum()
    service = tags[[t for t in tags.index if "Servicemember" in t]].sum()
    via = df["submitted_via"].value_counts(normalize=True)
    s = [f"The top 3 states - {', '.join(states.index[:3])} - account for {pct(states.iloc[:3].sum())} "
         f"of complaints (these are also the most populous states, so this is mostly about population).",
         f"{pct(older)} of complaints are tagged <i>Older American</i> and {pct(service)} <i>Servicemember</i>.",
         f"{share_text(via.iloc[0])} arrived via <b>{via.index[0]}</b>."]
    return "Where complaints come from", s, hbar(states, "Share of complaints - top 10 states"), []


def sec_length(df):
    med = df.groupby("product")["n_chars"].median().sort_values(ascending=False)
    short = (df["n_chars"] < 200).mean()
    s = [f"A typical complaint is about {n(df['n_chars'].median())} characters "
         f"(roughly {n(df['n_chars'].median() / 6)} words).",
         f"<b>{med.index[0]}</b> complaints are the longest (typical {n(med.iloc[0])} characters); "
         f"<b>{med.index[-1]}</b> the shortest ({n(med.iloc[-1])}).",
         f"{pct(short)} are very short (under 200 characters) - hard for any model to classify."]
    return "How long the complaints are", s, hbar(med, "Typical (median) length in characters", n), []


def sec_quality(df):
    miss = df[COLS].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    s = ["Share of complaints where each field is empty:"]
    bullets = [f"<b>{c}</b>: {pct(v)} empty" for c, v in miss.items()] or ["No field has missing values."]
    bullets.append("Empty <i>tags</i> is normal - most complaints have no tag.")
    return "Data quality", s, "", bullets


# ---------- HTML
CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;
color:#0b0b0b;background:#fcfcfb;line-height:1.5}h1{font-size:26px;margin-bottom:4px}
h2{font-size:19px;margin-top:40px;border-top:1px solid #e6e5e1;padding-top:20px}
.sub{color:#52514e}img{max-width:100%;display:block;margin:12px 0}
.notice{background:#fff4ec;border-left:4px solid #eb6834;padding:10px 16px;margin:12px 0}
.notice ul{margin:6px 0;padding-left:20px}"""


def build_html(sections):
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>CFPB complaints - basic analysis</title>"
             f"<style>{CSS}</style></head><body><h1>CFPB complaints - trends and anomalies</h1>"
             f"<p class='sub'>Complaints with a published narrative. Unusual months are compared with the "
             f"median of the 6 months before them (spike = 1.5x or more, drop = half or less).</p>"]
    for title, sentences, chart, bullets in sections:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts += [f"<p>{t}</p>" for t in sentences]
        parts.append(chart)
        if bullets:
            parts.append("<div class='notice'><b>What to notice</b><ul>"
                         + "".join(f"<li>{b}</li>" for b in bullets) + "</ul></div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    print("Loading ...")
    df = load_combined(COLS).dropna(subset=["product", "date"])
    df["month"] = df["date"].dt.to_period("M")
    sections = []
    for fn in [sec_overview, sec_volume, sec_products, sec_mix, sec_companies, sec_issues,
               sec_responses, sec_people, sec_length, sec_quality]:
        print(" ", fn.__name__[4:])
        sections.append(fn(df))

    out = config.REPORTS / "eda_report.html"
    out.write_text(build_html(sections), encoding="utf-8")

    print("\nKEY FINDINGS")
    for title, sentences, _, bullets in sections:
        print(f"\n## {title}")
        for t in sentences + bullets:
            print(" -", html.unescape(t.replace("<b>", "").replace("</b>", "")
                                      .replace("<i>", "").replace("</i>", "")))
    print(f"\nReport -> {out}  (open it in your browser)")


if __name__ == "__main__":
    main()
