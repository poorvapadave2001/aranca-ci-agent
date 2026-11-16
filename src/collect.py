import hashlib, html, json, os, re, sqlite3, time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse
import feedparser, requests, trafilatura
import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))   # 30 for a seed run, 2 for daily runs
MAX_SITEMAP_ITEMS = 10                                   # per firm per run
DB = os.getenv("CI_DB", "data/ci.db")
SPEC_F = "data/monitor_specs.json"
ONLY = [x.strip().lower() for x in os.getenv("ONLY", "").split(",") if x.strip()]
UA = {"User-Agent": "Mozilla/5.0 (compatible; AranCI-Agent/1.0)"}
TAVILY_NEWS = os.getenv("TAVILY_NEWS") == "1"
TAVILY_MAX = int(os.getenv("TAVILY_MAX", "15"))
GOOGLE = {"streak": 0, "off": False}
NEWS_STATS = {"requests": 0, "http_fail": 0, "codes": set(), "raw": 0, "dated": 0, "matched": 0}

EDITIONS = {
    "US":    [("en-US", "US", "US:en", "en")],
    "India": [("en-IN", "IN", "IN:en", "en")],
    "UAE":   [("en", "AE", "AE:en", "en"), ("ar", "AE", "AE:ar", "ar")],
    "KSA":   [("en", "SA", "SA:en", "en"), ("ar", "SA", "SA:ar", "ar")],
    "Japan": [("en", "JP", "JP:en", "en"), ("ja", "JP", "JP:ja", "ja")],
}
TOPICS = {
    "ja": {"M&A": "買収 OR 合併 OR 出資 OR 資金調達",
           "Deal": "提携 OR 協業 OR 契約 OR 受注 OR 就任",
           "Business Update": "開始 OR 拡大 OR 新設 OR 発表 OR 設立"},
    "ar": {"M&A": "استحواذ OR اندماج OR استثمار OR تمويل",
           "Deal": "شراكة OR عقد OR اتفاقية OR تعيين",
           "Business Update": "إطلاق OR توسع OR افتتاح OR نمو"},
}
TOPIC_MAP = {"deals": "Deal", "m_and_a": "M&A", "business": "Business Update", "digital": "Digital Presence"}
FALLBACK_Q = {
    "Deal": "partnership OR client OR contract OR wins OR appointed OR mandate",
    "M&A": "acquisition OR acquires OR merger OR investment OR funding",
    "Business Update": "launches OR expands OR opens OR growth OR revenue OR hires OR new office"}
PAGES = ["", "/news", "/insights", "/blog", "/press", "/media", "/services", "/about", "/careers"]
CAT_RULES = [("M&A", r"acqui|merger|merge|invest|fund|raise|stake|buyout|買収|合併|出資|استحواذ|اندماج|استثمار"),
             ("Deal", r"partner|client|contract|win|mandate|alliance|appoint|提携|協業|受注|شراكة|عقد|اتفاقية"),
             ("Business Update", r"launch|expan|open|growth|revenue|hire|office|new service|開始|拡大|新設|إطلاق|توسع|افتتاح")]
CONTENT_PATH = re.compile(r"/(blog|news|insights?|articles?|press|posts?|publications?|resources?|"
                          r"case-stud\w*|reports?|media|perspectives?|thought\w*|events?|20\d\d)/", re.I)
STATIC_EXT = re.compile(r"\.(jpg|jpeg|png|gif|svg|pdf|css|js|xml|zip|mp4|webp)(\?|$)", re.I)
CTX = re.compile(r"consult|advis|research|strateg|partner|group|firm|analytics|intelligence|capital|"
                 r"コンサル|استشار", re.I)

# ---------------- db ----------------
os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
con = sqlite3.connect(DB)
con.executescript("""
CREATE TABLE IF NOT EXISTS items(
  id TEXT PRIMARY KEY, company TEXT, region TEXT, title TEXT, snippet TEXT, publisher TEXT,
  url TEXT, published TEXT, category_hint TEXT, source_type TEXT, fetched_at TEXT,
  status TEXT DEFAULT 'new');
CREATE TABLE IF NOT EXISTS pages(url TEXT PRIMARY KEY, hash TEXT, text TEXT, checked_at TEXT);
CREATE TABLE IF NOT EXISTS runs(ts TEXT, company TEXT, fetched INT, kept INT);
CREATE TABLE IF NOT EXISTS sitemap_seen(url TEXT PRIMARY KEY, site TEXT, first_seen TEXT);
""")
_cols = {r[1] for r in con.execute("PRAGMA table_info(items)")}
for _c in ("service_ids", "overlap_type", "lang"):
    if _c not in _cols:
        con.execute(f"ALTER TABLE items ADD COLUMN {_c} TEXT DEFAULT ''")
con.commit()

# ---------------- helpers ----------------
def norm(s): return re.sub(r"[\W_]+", "", (s or "").lower())
def hid(url): return hashlib.md5(url.split("#")[0].encode()).hexdigest()
def now_iso(): return datetime.now(timezone.utc).isoformat()

SLD = {"co", "com", "org", "net", "gov", "ac", "edu", "or", "ne"}
def canon(u):
    d = urlparse(u if "//" in (u or "") else "//" + (u or "")).netloc.lower().removeprefix("www.").split(":")[0]
    parts = d.split(".")
    if len(parts) <= 2:
        return d
    if len(parts[-1]) == 2 and parts[-2] in SLD:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def hint(text):
    for cat, pat in CAT_RULES:
        if re.search(pat, (text or "").lower()):
            return cat
    return "Digital Presence"

def dt_of(entry):
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime(*t[:6], tzinfo=timezone.utc) if t else None

def clean(s, n=500):
    return re.sub(r"\s+", " ", BeautifulSoup(s or "", "html.parser").get_text()).strip()[:n]

def get(url, html_only=True):
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code != 200:
            return None
        if html_only and "text/html" not in r.headers.get("content-type", ""):
            return None
        return r
    except Exception:
        return None

def _has_seq(tokens, sub):
    n = len(sub)
    return bool(n) and any(tokens[i:i + n] == sub for i in range(len(tokens) - n + 1))

def mentions(text, aliases):
    """The firm (any alias) must appear as whole words. Short one-word names also need business context.
    Non-ASCII aliases (Japanese/Arabic) use plain substring match."""
    t = (text or "").lower()
    toks = re.findall(r"\w+", t)
    for a in aliases:
        al = (a or "").lower().strip()
        if len(norm(al)) < 3:
            continue
        hit = _has_seq(toks, re.findall(r"\w+", al)) if al.isascii() else al in t
        if not hit:
            continue
        if " " not in al and len(al) <= 8 and not CTX.search(t):
            continue
        return True
    return False

recent_titles = {}
def load_recent():
    since = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
    for c, t in con.execute("SELECT company, title FROM items WHERE published >= ?", (since,)):
        recent_titles.setdefault(c, []).append(t)

def save(item):
    if not item["published"]:
        return False
    if con.execute("SELECT 1 FROM items WHERE id=?", (item["id"],)).fetchone():
        return False
    for t in recent_titles.get(item["company"], []):
        if fuzz.token_set_ratio(t, item["title"]) >= 88:
            return False
    recent_titles.setdefault(item["company"], []).append(item["title"])
    con.execute("INSERT INTO items(id,company,region,title,snippet,publisher,url,published,category_hint,"
                "source_type,fetched_at,status,service_ids,overlap_type,lang) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,'new',?,?,?)",
                (item["id"], item["company"], item["region"], item["title"], item["snippet"],
                 item["publisher"], item["url"], item["published"], item["category_hint"],
                 item["source_type"], now_iso(), item.get("service_ids", ""),
                 item.get("overlap_type", ""), item.get("lang", "")))
    return True

# ---------------- source 1: Google News RSS ----------------
def news_items(c, cutoff):
    name, region, names = c["name"], c["region"], c["aliases"]
    qname = " OR ".join(f'"{n}"' for n in names[:3])
    out = []
    if GOOGLE["off"]:
        return out
    for hl, gl, ceid, lang in EDITIONS.get(region, EDITIONS["US"]):
        if lang == "en":
            qs = ([(TOPIC_MAP.get(k, "Digital Presence"), q) for k, q in c["queries"].items()]
                  or [(t, f"({qname}) ({ors})") for t, ors in FALLBACK_Q.items()])
        else:
            qs = [(t, f"({qname}) ({ors})") for t, ors in TOPICS[lang].items()]
        for topic, base in qs:
            q = f"{base} when:{LOOKBACK_DAYS}d"
            url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={hl}&gl={gl}&ceid={ceid}"
            NEWS_STATS["requests"] += 1
            try:
                resp = requests.get(url, headers=UA, timeout=15)
                if resp.status_code != 200:
                    raise RuntimeError(resp.status_code)
                feed = feedparser.parse(resp.content)
                GOOGLE["streak"] = 0
            except Exception as ex:
                NEWS_STATS["http_fail"] += 1
                NEWS_STATS["codes"].add(ex.args[0] if isinstance(ex, RuntimeError) else "error")
                GOOGLE["streak"] += 1
                if GOOGLE["streak"] >= 6 and not GOOGLE["off"]:
                    GOOGLE["off"] = True
                    print("  Google News keeps failing (blocked?) -> disabled for this run; using free fallbacks")
                    return out
                continue
            NEWS_STATS["raw"] += len(feed.entries)
            for e in feed.entries:
                d = dt_of(e)
                title, snip = clean(e.get("title")), clean(e.get("summary"))
                if not d or d < cutoff:
                    continue
                NEWS_STATS["dated"] += 1
                if not mentions(title + " " + snip, names):
                    continue
                NEWS_STATS["matched"] += 1
                h = hint(title)
                out.append(dict(id=hid(e.link), company=name, region=region, title=title, snippet=snip,
                                publisher=e.get("source", {}).get("title", ""), url=e.link,
                                published=d.isoformat(), category_hint=h if h != "Digital Presence" else topic,
                                source_type="news", lang=lang))
            time.sleep(0.4)
    return out

# ---------------- source 1b: Tavily news (optional) ----------------
_tav, _tav_used = None, 0
def tavily_news(c, cutoff):
    global _tav, _tav_used
    if not (TAVILY_NEWS and os.getenv("TAVILY_API_KEY")) or _tav_used >= TAVILY_MAX:
        return []
    if _tav is None:
        from tavily import TavilyClient
        _tav = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    _tav_used += 1
    q = f'"{c["name"]}" (acquisition OR partnership OR client OR launches OR expands OR appoints OR funding)'
    try:
        res = _tav.search(q, topic="news", days=max(1, LOOKBACK_DAYS), max_results=6,
                          search_depth="basic").get("results", [])
    except Exception as e:
        print(f"  tavily failed for {c['name']}: {str(e)[:80]}")
        return []
    out = []
    for r in res:
        raw = r.get("published_date") or ""
        try:
            d = parsedate_to_datetime(raw)
        except Exception:
            try:
                d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except Exception:
                continue
        d = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        title, snip = clean(r.get("title")), clean(r.get("content"))
        if d < cutoff or not mentions(title + " " + snip, c["aliases"]):
            continue
        out.append(dict(id=hid(r["url"]), company=c["name"], region=c["region"], title=title, snippet=snip,
                        publisher=urlparse(r["url"]).netloc, url=r["url"], published=d.isoformat(),
                        category_hint=hint(title), source_type="search_news", lang="en"))
    return out

# ---------------- free news fallbacks: Bing News RSS -> DuckDuckGo -> Tavily ----------------
def _item(c, title, snip, url, d, publisher, src, lang="en"):
    h = hint(title)
    return dict(id=hid(url), company=c["name"], region=c["region"], title=title, snippet=snip,
                publisher=publisher, url=url, published=d.isoformat(),
                category_hint=h if h != "Digital Presence" else "Business Update",
                source_type=src, lang=lang)

def _real_url(link):
    q = parse_qs(urlparse(link).query)
    return q["url"][0] if q.get("url") else link

def bing_news(c, cutoff):
    qname = " OR ".join(f'"{n}"' for n in c["aliases"][:2])
    url = f"https://www.bing.com/news/search?q={quote_plus('(' + qname + ')')}&format=rss"
    try:
        resp = requests.get(url, headers=UA, timeout=15)
        if resp.status_code != 200:
            return []
        feed = feedparser.parse(resp.content)
    except Exception:
        return []
    out = []
    for e in feed.entries:
        d = dt_of(e)
        title, snip = clean(e.get("title")), clean(e.get("summary"))
        if not d or d < cutoff or not mentions(title + " " + snip, c["aliases"]):
            continue
        real = _real_url(e.link)
        out.append(_item(c, title, snip, real, d, urlparse(real).netloc, "news_bing"))
    time.sleep(0.5)
    return out

def ddg_news(c, cutoff):
    try:
        from ddgs import DDGS
        tl = "d" if LOOKBACK_DAYS <= 1 else "w" if LOOKBACK_DAYS <= 7 else "m" if LOOKBACK_DAYS <= 31 else None
        res = DDGS().news(f'"{c["aliases"][0]}"', max_results=10, timelimit=tl)
    except Exception:
        return []
    out = []
    for r in res or []:
        try:
            d = datetime.fromisoformat(str(r.get("date", "")).replace("Z", "+00:00"))
        except Exception:
            continue
        d = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        title, snip = clean(r.get("title")), clean(r.get("body"))
        if d < cutoff or not mentions(title + " " + snip, c["aliases"]):
            continue
        out.append(_item(c, title, snip, r["url"], d, r.get("source") or urlparse(r["url"]).netloc, "news_ddg"))
    time.sleep(1)
    return out

def fallback_news(c, cutoff):
    for fn in (bing_news, ddg_news, tavily_news):
        try:
            got = fn(c, cutoff)
        except Exception as e:
            print(f"  {fn.__name__} failed for {c['name']}: {str(e)[:60]}")
            got = []
        if got:
            return got
    return []

# ---------------- source 2: RSS feed ----------------
def site_feed_items(c, cutoff):
    name, region, site = c["name"], c["region"], c["url"]
    feeds = {c["rss"]} if c.get("rss") else set()
    if not feeds:
        r = get(site)
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        feeds = {urljoin(site, l["href"]) for l in soup.find_all("link", href=True)
                 if "rss" in (l.get("type") or "") or "atom" in (l.get("type") or "")}
    out = []
    for f in list(feeds)[:3]:
        for e in feedparser.parse(f).entries[:20]:
            d = dt_of(e)
            if not d or d < cutoff:
                continue
            title = clean(e.get("title"))
            out.append(dict(id=hid(e.link), company=name, region=region, title=title,
                            snippet=clean(e.get("summary")), publisher=urlparse(site).netloc,
                            url=e.link, published=d.isoformat(), category_hint=hint(title),
                            source_type="website_feed"))
    return out

# ---------------- source 3: sitemap diffing ----------------
def _locs(text, tag):
    return [html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", m)).strip()
            for m in re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)]

def _lastmod(s):
    try:
        d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def read_sitemap_urls(site):
    base = site.rstrip("/")
    queue, seen, urls = [], set(), {}
    robots = get(base + "/robots.txt", html_only=False)
    if robots:
        queue += re.findall(r"(?im)^sitemap:\s*(\S+)", robots.text)
    queue += [base + p for p in ("/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml")]
    fetched = 0
    while queue and fetched < 8 and len(urls) < 3000:
        sm = queue.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        r = get(sm, html_only=False)
        if not r or "<" not in r.text[:200]:
            continue
        fetched += 1
        text = r.text
        if "<sitemapindex" in text:
            subs = _locs(text, "loc")
            subs.sort(key=lambda u: 0 if re.search(r"post|blog|news|insight|article|publication", u, re.I) else 1)
            queue = subs[:6] + queue
        else:
            for blk in re.findall(r"<url>(.*?)</url>", text, re.S):
                loc = _locs(blk, "loc")
                lm = _locs(blk, "lastmod")
                if loc:
                    urls[loc[0]] = _lastmod(lm[0]) if lm else None
        if urls:
            break
    return urls

def page_meta(url):
    r = get(url)
    if not r:
        return None, ""
    soup = BeautifulSoup(r.text, "html.parser")
    og = soup.find("meta", property="og:title")
    title = clean(og["content"] if og and og.get("content") else (soup.title.string if soup.title else ""), 200)
    desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    snip = clean(desc["content"] if desc and desc.get("content") else "", 400)
    if not snip:
        snip = clean(trafilatura.extract(r.text) or "", 400)
    return title, snip

def looks_like_article(path):
    segs = [p for p in path.split("/") if p]
    if not segs:
        return False
    if len(segs) >= 2 and CONTENT_PATH.search(path):
        return True
    slug = re.sub(r"\.(php|html?|aspx?)$", "", segs[-1], flags=re.I)
    return len(segs) == 1 and slug.count("-") >= 3     # root-level "long-hyphenated-title" pages

def sitemap_items(c, cutoff):
    name, region, site = c["name"], c["region"], c["url"]
    urls = read_sitemap_urls(site)
    if not urls:
        return []
    dom = urlparse(site).netloc
    known = {r[0] for r in con.execute("SELECT url FROM sitemap_seen WHERE site=?", (dom,))}
    first_run = len(known) == 0
    now = datetime.now(timezone.utc)
    new = [u for u in urls if u not in known]
    con.executemany("INSERT OR IGNORE INTO sitemap_seen VALUES(?,?,?)",
                    [(u, dom, now.isoformat()) for u in new])
    cands = []
    for u in new:
        path = urlparse(u).path
        if STATIC_EXT.search(u) or not looks_like_article(path):
            continue
        if len([p for p in path.split("/") if p]) < 2:
            continue
        lm = urls[u]
        if lm:
            if lm < cutoff:
                continue
            cands.append((lm, u))
        elif not first_run:
            cands.append((now, u))
    cands.sort(reverse=True)
    out = []
    for pub, u in cands[:MAX_SITEMAP_ITEMS]:
        title, snip = page_meta(u)
        if not title:
            title = urlparse(u).path.strip("/").split("/")[-1].replace("-", " ").title()
        out.append(dict(id=hid(u), company=name, region=region, title=title, snippet=snip,
                        publisher=dom, url=u, published=pub.isoformat(),
                        category_hint=hint(title + " " + urlparse(u).path), source_type="sitemap"))
        time.sleep(0.3)
    return out

# ---------------- source 4: page-change detection ----------------
def page_changes(c, cutoff=None):
    name, region, site = c["name"], c["region"], c["url"]
    urls = [site] + [c[k] for k in ("blog", "press") if c.get(k)]
    if len(urls) == 1:
        urls = [site.rstrip("/") + p for p in PAGES]
    out = []
    for url in dict.fromkeys(urls):
        r = get(url)
        if not r:
            continue
        text = trafilatura.extract(r.text) or ""
        if len(text) < 200:
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        row = con.execute("SELECT hash, text FROM pages WHERE url=?", (url,)).fetchone()
        now = now_iso()
        if row and row[0] != h:
            old = set(row[1].splitlines())
            added = [l.strip() for l in text.splitlines() if len(l.strip()) >= 30 and l not in old]
            if sum(map(len, added)) >= 120:
                out.append(dict(id=hid(url + h), company=name, region=region,
                                title=f"Website updated: {urlparse(url).path or '/'}",
                                snippet=" ".join(added)[:500], publisher=urlparse(site).netloc,
                                url=url, published=now, category_hint="Digital Presence",
                                source_type="website_change"))
        con.execute("INSERT OR REPLACE INTO pages VALUES(?,?,?,?)", (url, h, text[:20000], now))
        time.sleep(0.3)
    return out

# ---------------- firm context ----------------
def load_specs():
    try:
        return json.load(open(SPEC_F, encoding="utf-8"))
    except Exception:
        print(f"WARNING: {SPEC_F} not found -> falling back to firm names only.")
        return {}

def build_firm(row, specs):
    sp = specs.get(canon(row["url"]), {})
    src = sp.get("sources", {})
    url = row["url"] if row["url"].startswith("http") else "https://" + row["url"]
    ids = ",".join(t["id"] for t in sp.get("threatens", []) if t.get("id")) or row.get("overlap_service_ids", "")
    return {"name": row["name"], "region": row["region"], "url": url,
            "aliases": list(dict.fromkeys([row["name"]] + list(sp.get("aliases", [])))),
            "queries": sp.get("news_queries", {}),
            "blog": src.get("blog", ""), "press": src.get("press", ""), "rss": src.get("rss", ""),
            "service_ids": ids, "overlap_type": sp.get("overlap_type") or row.get("overlap_type", "")}

# ---------------- main ----------------
if __name__ == "__main__":
    df = pd.read_csv("data/competitors.csv", dtype=str).fillna("")
    if (df["approved"].str.lower() == "yes").any():
        df = df[df["approved"].str.lower() == "yes"]
    else:
        print("No rows marked approved=yes -> using non-rejected competitors")
        df = df[df["approved"].str.lower() != "no"]
    if ONLY:
        df = df[df["name"].str.lower().map(lambda n: any(o in n for o in ONLY))]
    specs = load_specs()
    print(f"Collecting for {len(df)} firms, lookback {LOOKBACK_DAYS} days, specs for {len(specs)} firms")

    load_recent()
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    SOURCES = [("news", news_items), ("feed", site_feed_items),
               ("sitemap", sitemap_items), ("pagechg", page_changes)]
    tot = {s: 0 for s, _ in SOURCES}
    total_f = total_k = 0

    for _, row in df.iterrows():
        c = build_firm(row.to_dict(), specs)
        items, per = [], {}
        for label, fn in SOURCES:
            try:
                got = fn(c, cutoff)
            except Exception as e:
                print(f"  {label} failed for {c['name']}: {str(e)[:80]}")
                got = []
            per[label] = len(got)
            tot[label] += len(got)
            items += got
            if label == "news" and not got:
                fb = fallback_news(c, cutoff)
                per["fallback"] = len(fb)
                tot["fallback"] = tot.get("fallback", 0) + len(fb)
                items += fb
        for i in items:
            i["service_ids"], i["overlap_type"] = c["service_ids"], c["overlap_type"]
        kept = sum(save(i) for i in items)
        con.execute("INSERT INTO runs VALUES(?,?,?,?)", (now_iso(), c["name"], len(items), kept))
        con.commit()
        total_f += len(items)
        total_k += kept
        detail = " ".join(f"{k}={v}" for k, v in per.items())
        print(f"{c['name'][:30]:30} [{c['region']:5}] new={kept:2}  ({detail})")

    print(f"\nTOTAL fetched={total_f}  new after dedup={total_k}  by source: {tot}")
    ns = NEWS_STATS
    print(f"NEWS DIAGNOSTICS: requests={ns['requests']} http_fail={ns['http_fail']} codes={sorted(map(str, ns['codes']))} "
          f"| feed entries={ns['raw']} -> within lookback={ns['dated']} -> mention firm={ns['matched']}")
    if ns["requests"] and ns["http_fail"] >= ns["requests"] * 0.5:
        print("  -> Google News is blocking this machine: Bing/DuckDuckGo fallbacks were used instead.")
    elif ns["raw"] == 0:
        print("  -> Google News answered but returned no entries (low press coverage).")
    elif ns["matched"] == 0:
        print("  -> Entries exist but none mention the firm by name.")
    print(pd.read_sql("SELECT region, source_type, COUNT(*) n FROM items GROUP BY 1,2 ORDER BY 1,3 DESC", con))