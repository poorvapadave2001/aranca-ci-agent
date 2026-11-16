import sys, pandas as pd
sys.path.insert(0, "src")
from datetime import datetime, timedelta, timezone
import collect as C

cut = datetime.now(timezone.utc) - timedelta(days=120)
d = pd.read_csv("data/competitors.csv", dtype=str).fillna("")
d = d[(d.region == "KSA") & (d.approved.str.lower() != "no")
      & d.firm_type.isin(["strategy_consulting", "market_research", "ma_advisory"])
      & (d.size_class != "large")]
rows = []
for _, r in d.iterrows():
    ok = bool(C.get(r["url"]))
    urls = C.read_sitemap_urls(r["url"]) if ok else {}
    dated = [v for v in urls.values() if v]
    recent = sum(v >= cut for v in dated)
    rows.append((r["name"], r["approved"] or "-", ok, len(urls), len(dated), recent, r["url"]))
out = pd.DataFrame(rows, columns=["name", "appr", "home_ok", "urls", "dated", "recent120d", "url"])
print(out.sort_values(["home_ok", "recent120d", "urls"], ascending=False).to_string(index=False))