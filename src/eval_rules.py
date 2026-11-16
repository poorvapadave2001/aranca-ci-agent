"""Shared pieces for eval.py and eval_auto.py: the labeling rubric, the rule-based judge, and small stats helpers.
Edit RUBRIC to match how YOU want to judge items, then label consistently against it."""
import math, os, re
from urllib.parse import urlparse
import pandas as pd

CATS = ["M&A", "Deal", "Business Update", "Digital Presence"]

RUBRIC = """RELEVANT (y): the item tells a Practice Leader something new about the NAMED firm's own business.
  M&A              the firm (or its owner/portfolio) acquires, merges, is acquired, divests, invests or raises funding.
  Deal             the firm wins a client/mandate/project, signs a partnership or alliance, makes a senior appointment.
                   A case study about a named client engagement counts as Deal.
  Business Update  the firm launches a service, opens an office, enters a market, hires, wins an award, reports growth.
  Digital Presence the firm's OWN blog, report, webinar or website change that shows what it sells, its methods/tools/
                   capabilities, which clients/markets/industries it targets, or a change in positioning.
NOT RELEVANT (n):
  - about a different organisation with a similar name, or the firm is only mentioned in passing
  - generic market, industry or macro commentary that reveals nothing about the firm's own offering
    (market sizing, "X market guide", news analysis, commentary on someone else's deal or investment)
  - job ads, stock-holding/ticker news, SEO filler, pages with no substantive text
true_cat: the single best of M&A | Deal | Business Update | Digital Presence. Leave blank when true_relevant = n."""

# ---------------------------------------------------------------- rule-based judge (also used as a baseline)
PROMO = re.compile(r"market size|market share|forecast to 20|market report|industry outlook|prices?,? trends|marketbeat|\$[A-Z]{2,4}\b", re.I)
JOB = re.compile(r"\b(job|jobs|vacanc\w+|careers?|we are hiring|apply now|internship)\b", re.I)
SUFFIX = re.compile(r"\b(inc|ltd|llc|pvt|limited|corp|co|group)\b\.?", re.I)
RULES = [("M&A", r"acqui|merger|merge|stake|buyout|funding|raises|investment round|買収|合併|出資"),
         ("Deal", r"partnership|partners with|client|contract|mandate|wins|appointed|alliance|提携|受注"),
         ("Business Update", r"launch|expands?|opens?|new office|hires?|new service|awarded|開始|拡大")]

def dom(u):
    return urlparse(u if "//" in str(u) else "//" + str(u)).netloc.lower().removeprefix("www.")

_FIRM_DOM = None
def firm_domains():
    global _FIRM_DOM
    if _FIRM_DOM is None:
        try:
            c = pd.read_csv("data/competitors.csv", dtype=str).fillna("")
            _FIRM_DOM = {n: dom(u).split(".")[0] for n, u in zip(c["name"], c["url"])}
        except Exception:
            _FIRM_DOM = {}
    return _FIRM_DOM

def rule_judge(company, title, snippet, url):
    """-> ('y'|'n', category). Cheap deterministic judge: firm named (or own site), substance, no promo/job words."""
    text = f"{title} {snippet}".lower()
    name = SUFFIX.sub("", str(company).lower()).strip()
    entity = bool(name) and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text) is not None
    own = firm_domains().get(company, "@@") in dom(url)
    ok = (entity or own) and len(text.strip()) >= 40 and not PROMO.search(text) and not JOB.search(text)
    cat = next((c for c, p in RULES if re.search(p, text)), "Digital Presence") if ok else ""
    return ("y" if ok else "n"), cat

# ---------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)

def pct_ci(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n:.0%} ({lo:.0%}-{hi:.0%})"

def kappa(a, b):
    """Cohen's kappa for two boolean Series."""
    a, b = pd.Series(a).astype(bool), pd.Series(b).astype(bool)
    if len(a) == 0:
        return float("nan")
    po = (a.values == b.values).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0