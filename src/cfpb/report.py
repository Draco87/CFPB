"""Helpers for plain-English HTML reports with embedded charts."""
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 110})

CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;
color:#0b0b0b;background:#fcfcfb;line-height:1.5}h1{font-size:26px;margin-bottom:4px}
h2{font-size:19px;margin-top:40px;border-top:1px solid #e6e5e1;padding-top:20px}
.sub{color:#52514e}img{max-width:100%;display:block;margin:12px 0}
table{border-collapse:collapse;margin:12px 0;font-size:14px}td,th{padding:6px 12px;border-bottom:1px solid #e6e5e1;text-align:left}
th{color:#52514e;font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.notice{background:#fff4ec;border-left:4px solid #eb6834;padding:10px 16px;margin:12px 0}
.notice ul{margin:6px 0;padding-left:20px}"""


def n(x):
    x = float(x)
    return f"{x / 1e6:.1f} million" if x >= 1e6 else f"{x:,.0f}"


def pct(p, digits=None):
    if digits is not None:
        return f"{p * 100:.{digits}f}%"
    return f"{p * 100:.0f}%" if abs(p) >= 0.1 else f"{p * 100:.1f}%"


def about(p):
    """0.74 -> 'about 3 in 4'; small rates -> 'about 1 in 25'."""
    if 0 < p < 0.2:
        return f"about 1 in {round(1 / p)}"
    for num, den in [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5), (9, 10)]:
        if abs(p - num / den) < 0.025:
            return f"about {num} in {den}"
    return ""


def share_text(p):
    a = about(p)
    return f"{pct(p)} ({a})" if a else pct(p)


def fig_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def dates(ax):
    loc = matplotlib.dates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(loc))


def table_html(df, num_cols=()):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td class='num'>{v}</td>" if c in num_cols else f"<td>{v}</td>"
                                    for c, v in zip(df.columns, r)) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><tr>{head}</tr>{rows}</table>"


def build_html(title, subtitle, sections):
    """sections: list of (heading, [sentences], html_block, [bullets])"""
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style>"
             f"</head><body><h1>{title}</h1><p class='sub'>{subtitle}</p>"]
    for heading, sentences, block, bullets in sections:
        parts.append(f"<h2>{heading}</h2>")
        parts += [f"<p>{t}</p>" for t in sentences]
        parts.append(block)
        if bullets:
            parts.append("<div class='notice'><b>What to notice</b><ul>"
                         + "".join(f"<li>{b}</li>" for b in bullets) + "</ul></div>")
    parts.append("</body></html>")
    return "\n".join(parts)
