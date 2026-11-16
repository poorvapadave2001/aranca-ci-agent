"""Pick approved=yes firms per region: relevance score + monitorability + recent activity.
  DRY_RUN=1 python src/shortlist.py | PER_REGION=8 | LOCK_NO=1 | RESET=1
"""
import os, re, shutil, sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
import pandas as pd

CSV = "data/competitors.csv"
DB = os.getenv("CI_DB", "data/ci.db")
PER_REGION = int(os.getenv("PER_REGION", "6"))
MIN_PER_REGION = 3
DRY_RUN = os.getenv("DRY_RUN") == "1"
RESET = os.getenv("RESET") == "1"
LOCK_NO = os.getenv("LOCK_NO") == "1"
OK_TYPES = {"strategy_consulting", "market_research", "ma_advisory"}
MON_BONUS = {"excellent": 10, "good": 6, "moderate": 2, "unknown": -3, "": -3}

BLOCK_NAMES = ["imarc", "gmi research", "techsci", "cerulli", "tokyo venture", "advisory corp",
               "saudi mergers", "precedence research", "peak ventures", "india market entry",
               "grand view", "spherical insights", "stratview", "360iresearch", "coherent market",
               "ken research", "business research company", "marketsandmarkets", "statista",
               "duckercarlisle"]
BLOCK_ROOTS = ("grandview", "sphericalinsights", "stratview", "360iresearch", "coherentmi",
               "kenresearch", "imarc", "thebusinessresearch", "marketsandmarkets", "statista",
               "precedenceresearch", "gmiresearch", "techsciresearch", "cerulli", "mordor",
               "bdo", "fticonsulting", "ipsos", "kantar", "protiviti", "bearingpoint", "icf",
               "duckercarlisle")
HEADLINE = re.compile(r"market research|strategy consulting|home|welcome|maximizing|consulting firm|"
                      r"company in|reports|homepage|about us", re.I)

df = pd.read_csv(CSV, dtype=str).fillna("")
for c in ("approved", "monitoring_status", "firm_type", "size_class", "score"):
    if c not in df.columns:
        df[c] = ""
df["score_n"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

def root(url):
    d = urlparse(url if "//" in url else "//" + url).netloc.lower().removeprefix("www.")
    return d.split(".")[0]

def blocked(r):
    n = r["name"].lower()
    return (any(b in n for b in BLOCK_NAMES) or any(b in root(r["url"]) for b in BLOCK_ROOTS)
            or len(r["name"]) > 40 or bool(HEADLINE.search(r["name"])))

def eligible(r):
    return (not blocked(r) and r["firm_type"] in OK_TYPES and r["size_class"] != "large"
            and r["monitoring_status"] != "poor")

active = set()
try:
    con = sqlite3.connect(DB)
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    active = {r[0] for r in con.execute("SELECT DISTINCT company FROM items WHERE published >= ?", (since,))}
    con.close()
except Exception:
    pass

if RESET:
    df["approved"] = ""
df["_rank"] = (df["score_n"] + df["monitoring_status"].map(MON_BONUS).fillna(-3)
               + df["name"].map(lambda n: 8 if n in active else 0))
df["_ok"] = df.apply(eligible, axis=1)
df["_dom"] = df["url"].map(root)

changes = []
for i, r in df[df["approved"].str.lower() == "yes"].iterrows():
    if blocked(r) or r["monitoring_status"] == "poor":
        df.at[i, "approved"] = "no"
        changes.append(f"demoted  {r['name']} [{r['region']}]")
for i, r in df.iterrows():
    if blocked(r) and r["approved"].lower() != "no":
        df.at[i, "approved"] = "no"
for region, g in df.groupby("region"):
    have = int((g["approved"].str.lower() == "yes").sum())
    taken = set(g[g["approved"].str.lower() == "yes"]["_dom"])
    pool = g[g["_ok"] & (g["approved"].str.lower() != "yes")]
    if LOCK_NO:
        pool = pool[pool["approved"].str.lower() != "no"]
    for i, r in pool.sort_values("_rank", ascending=False).iterrows():
        if have >= PER_REGION:
            break
        if r["_dom"] in taken:
            continue
        df.at[i, "approved"] = "yes"
        taken.add(r["_dom"])
        have += 1
        changes.append(f"approved {r['name']} [{region}] rank={r['_rank']:.0f} mon={r['monitoring_status'] or '?'}")

yes = df[df["approved"].str.lower() == "yes"].sort_values(["region", "_rank"], ascending=[True, False])
print("\n".join(changes) or "no changes")
print("\nAPPROVED FIRMS")
print(yes[["region", "name", "score", "monitoring_status", "firm_type", "url"]].to_string(index=False))
counts = yes.groupby("region").size()
print("\nPer region:", counts.to_dict())
for region in sorted(df["region"].unique()):
    if counts.get(region, 0) < MIN_PER_REGION:
        print(f"WARNING: {region} has only {counts.get(region, 0)} approved firms. "
              f"Add known firms to data/manual_competitors.csv and rerun discover.py.")

if DRY_RUN:
    print("\nDRY RUN: nothing written")
else:
    shutil.copy(CSV, "data/competitors_pre_shortlist.csv")
    df.drop(columns=["score_n", "_rank", "_ok", "_dom"]).to_csv(CSV, index=False)
    print(f"\nSaved {CSV} (backup: data/competitors_pre_shortlist.csv)")