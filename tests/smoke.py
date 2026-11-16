"""Offline smoke test (no API keys, no network):  python tests/smoke.py"""
import os, re, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, "src")
tmp = tempfile.mkdtemp()
shutil.copy("data/ci.db", f"{tmp}/ci.db")
env = {**os.environ, "DRY_RUN": "1", "CI_DB": f"{tmp}/ci.db", "REPORT_DIR": f"{tmp}/rep", "REPORT_DAYS": "365"}
for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY"):
    env.pop(k, None)

# 1. the report builds with no LLM key
r = subprocess.run([sys.executable, "src/report.py"], env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr
assert os.path.exists(f"{tmp}/rep/latest.html") and os.path.exists(f"{tmp}/rep/latest_digest.md")

# 2. every reported item has a source link and a publication date
import pandas as pd
df = pd.read_csv([f"{tmp}/rep/{f}" for f in os.listdir(f"{tmp}/rep") if f.startswith("items_")][0])
assert df["url"].str.startswith("http").all(), "item without a source link"
assert df["date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}").all(), "item without a date"

# 3. all four required categories + all five regions are present in the data
assert {"M&A", "Deal", "Business Update", "Digital Presence"} <= set(df["cat"]), set(df["cat"])
assert {"US", "India", "UAE", "KSA", "Japan"} <= set(df["region"]), set(df["region"])

# 4. LinkedIn post date decoding
os.environ["CI_DB"] = f"{tmp}/c.db"
import collect
ms = int(datetime(2024, 6, 1, 12, tzinfo=timezone.utc).timestamp() * 1000)
assert collect.linkedin_date(f"https://www.linkedin.com/posts/x-activity-{ms << 22}-ab").year == 2024
assert collect.linkedin_date("https://www.linkedin.com/company/x") is None
print(f"OK: {len(df)} items, regions={sorted(set(df['region']))}, categories={sorted(set(df['cat']))}")