import hashlib, json, os, re, shutil, sqlite3, sys, time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urljoin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
import llm
from llm import Exhausted
from tavily import TavilyClient

# ============================ CONFIG ============================
BUDGET = int(os.getenv("LLM_BUDGET", "20"))        # max UNCACHED LLM *calls* per run (0 = no LLM)
BATCH = int(os.getenv("BATCH", "6"))               # firms per LLM call  (6 calls = up to 36 firms)
TOP_K = int(os.getenv("TOP_K", "18"))              # max firms per region sent to the LLM
TARGET = int(os.getenv("TARGET_PER_REGION", "20")) # stop a region at this many firms
MIN_SCORE = int(os.getenv("MIN_SCORE", "45"))
ROLE = os.getenv("SCORE_ROLE", "fast")             # "fast" = cheap model, "best" = better
EV_CHARS = 600
EV_LLM_CHARS = int(os.getenv("EV_LLM_CHARS", "800"))
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")

DRY_RUN = os.getenv("DRY_RUN") == "1"              # nothing is written to disk, no LLM calls
if DRY_RUN:
    BUDGET = 0
ENRICH = os.getenv("ENRICH", "1") == "1"
ENRICH_TOP = int(os.getenv("ENRICH_TOP", "20"))
ENRICH_APPROVED_MAX = int(os.getenv("ENRICH_APPROVED_MAX", "40"))
ENRICH_REFRESH = os.getenv("ENRICH_REFRESH") == "1"
HARVEST = os.getenv("HARVEST", "1") == "1"
MAX_HARVEST = int(os.getenv("MAX_NEW", "25"))
SNOWBALL = os.getenv("SNOWBALL") == "1"
SNOWBALL_SEEDS = int(os.getenv("SNOWBALL_SEEDS", "3"))
SERVICE_EXPAND = os.getenv("SERVICE_EXPAND") == "1"
REFRESH_QUERIES = os.getenv("REFRESH_QUERIES") == "1"
REVISIT_GATED = os.getenv("REVISIT_GATED", "1") == "1"
FEEDBACK_WEIGHT = float(os.getenv("FEEDBACK_WEIGHT", "0.10"))
FEEDBACK_NEG = os.getenv("FEEDBACK_NEG") == "1"
W = {"service": 0.35, "model": 0.20, "region": 0.15, "evidence": 0.15, "publishing": 0.15}

ALL_REGIONS = {"US": "United States", "KSA": "Saudi Arabia", "UAE": "UAE Dubai Abu Dhabi",
               "India": "India", "Japan": "Japan"}
ONLY = [r.strip() for r in os.getenv("REGIONS_ONLY", "").split(",") if r.strip()]
REGIONS = {k: v for k, v in ALL_REGIONS.items() if not ONLY or k in ONLY}

GENERIC = ["boutique strategy consulting market entry growth advisory firm {r}",
           "market intelligence competitive benchmarking consultancy {r}",
           "commercial due diligence M&A advisory boutique {r}"]
LOCAL = {
    "US": ["independent growth strategy consulting firm Boston Chicago New York"],
    "KSA": ["strategy consulting firm Riyadh Vision 2030 market entry",
            "business advisory market research firm Jeddah Riyadh"],
    "UAE": ["strategy consulting firm Dubai market entry advisory",
            "business advisory market research firm Abu Dhabi DIFC"],
    "India": ["boutique strategy consulting firm Mumbai Delhi Bangalore"],
    "Japan": ["Japan market entry consulting firm Tokyo", "市場参入 経営コンサルティング 会社 東京"],
}
SERVICE_QUERIES = {
    "Market Research": "market research and market intelligence consultancy {r}",
    "Business Research": "business research corporate intelligence firm {r}",
    "Financial Research": "financial analytics investment research advisory firm {r}",
    "Consulting": "strategy and management consulting boutique {r}",
    "Data Analytics": "business analytics data intelligence consulting company {r}",
    "Investment Research": "equity research and financial advisory boutique {r}",
}
EXCL_ROOTS = {"mckinsey", "bcg", "bain", "deloitte", "pwc", "kpmg", "ey", "accenture", "kearney",
              "gartner", "forrester", "frost", "capgemini", "ibm", "idc", "nielsen", "aranca",
              "rolandberger", "oliverwyman", "strategyand", "lek", "oliverwyman"}
BAD_DOMAINS = ["wikipedia", "linkedin", "crunchbase", "glassdoor", "clutch", "youtube", "facebook",
               "twitter", "x.com", "medium", "reddit", "quora", "indeed", "yelp", "zoominfo",
               "tracxn", "ambitionbox", "goodfirms", "sortlist", "designrush", "forbes",
               "bloomberg", "reuters", "dnb", "owler", "wsj", "ft.com", "instagram", "pinterest",
               "scribd", "slideshare", "gov", "edu", "marketbeat", "yahoo", "google",
               "apple.com", "microsoft", "amazon", "wordpress", "wixsite", "bit.ly"]
# large firms and report publishers: dropped before any LLM call
BIG_FIRM_ROOTS = {"bdo", "fticonsulting", "fti", "grantthornton", "rsm", "crowe", "marsh", "mercer",
                  "oliverwyman", "alvarezandmarsal", "protiviti", "bearingpoint", "atkinsrealis",
                  "dxc", "broadcom", "slb", "hatch", "nielseniq", "niq", "ipsos", "kantar",
                  "comscore", "wooodmac", "woodmac", "jdpower", "icf"}
PUBLISHER_ROOTS = ("grandview", "sphericalinsights", "stratview", "360iresearch", "coherentmi",
                   "kenresearch", "imarc", "thebusinessresearch", "marketsandmarkets", "statista",
                   "mordorintelligence", "fortunebusinessinsights", "alliedmarket",
                   "precedenceresearch", "cbinsights", "greenbook")
HEADLINE = re.compile(r"market research|strategy consulting|home|welcome|maximizing|consulting firm|"
                      r"company in|reports|homepage|about us|^\W*$", re.I)
LIST = re.compile(r"\b(top \d+|top|best|leading|list of|directory|rankings?|companies in|firms in)\b", re.I)
POS = re.compile(r"consult|advis|strategy|market research|market intelligence|due diligence|"
                 r"market entry|benchmark|コンサル|戦略|市場調査|アドバイザリー|M&A", re.I)
NEG = re.compile(r"software|saas|staffing|recruit|venture capital|hedge fund|university|"
                 r"training institute|crypto", re.I)
CORE_TYPES = {"strategy_consulting", "market_research", "ma_advisory"}
REGION_PAT = {
    "US": r"united states|\bUSA\b|boston|chicago|new york|san francisco|washington|los angeles",
    "KSA": r"saudi|riyadh|jeddah|dammam|\bKSA\b|vision 2030",
    "UAE": r"\bUAE\b|dubai|abu dhabi|emirates|difc|\bGCC\b",
    "India": r"india|mumbai|delhi|bangalore|bengaluru|hyderabad|chennai|pune",
    "Japan": r"japan|tokyo|osaka|日本|東京|大阪",
}
TLD_HINT = {"US": (".us",), "KSA": (".sa",), "UAE": (".ae",), "India": (".in",), "Japan": (".jp",)}
INDUSTRY_WORDS = ["healthcare", "pharma", "life sciences", "technology", "energy", "chemicals",
                  "financial services", "banking", "consumer", "retail", "manufacturing", "automotive",
                  "telecom", "real estate", "logistics", "education", "agriculture", "mining",
                  "infrastructure", "public sector", "private equity", "fmcg"]
LINK_PAT = {
    "about": r"about|who-we-are|our-story|company|firm|team|会社概要|私たちについて|من نحن",
    "services": r"service|solution|what-we-do|capabilit|expertise|offering|practice|サービス|事業内容|خدمات",
    "blog": r"blog|insight|article|thought|perspective|publication|research|resource|knowledge|"
            r"コラム|インサイト|ブログ|記事|مدونة|مقالات",
    "press": r"press|news|media|announce|ニュース|お知らせ|プレス|أخبار",
}
UA = {"User-Agent": "Mozilla/5.0 (compatible; AranCI-Agent/1.0)"}

profile = json.load(open("data/aranca_profile.json"))
SL = profile["service_lines"]
LINES_SHORT = "; ".join(f"{s['id']} {s['name']}" for s in SL)
LINE_TEXTS = [f"{s['name']}. {s.get('description', '')[:300]} " + "; ".join(s.get("solutions", [])[:10])
              + " " + ", ".join(s.get("capability_keywords", [])[:10]) for s in SL]

os.makedirs("data", exist_ok=True)
CSV = "data/competitors.csv"
REGISTRY_F = "data/candidates_registry.csv"
COLS = ["name", "region", "url", "hq", "firm_type", "size_class", "score", "tier", "llm_score",
        "emb_score", "overlap_service_ids", "evidence_quote", "rationale", "approved",
        "monitoring_status", "local_score", "discovery_method", "source_url",
        "first_discovered_at", "last_verified_at"]

# ============================ STATE ============================
def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}

STATE_F, SEARCH_F = "data/discover_state_v2.json", "data/search_cache.json"
state = load_json(STATE_F)
state.setdefault("cands", {})
state.setdefault("queries", {})
state.setdefault("monitor", {})
state.setdefault("harvested_pages", [])
search_cache = load_json(SEARCH_F)

def save_state():
    if DRY_RUN:
        return
    json.dump(state, open(STATE_F, "w"))

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ============================ HELPERS ============================
def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())
def domain(u): return urlparse(u if "//" in u else "//" + u).netloc.lower().replace("www.", "")
def bad(d): return (not d) or any(b in d for b in BAD_DOMAINS)

SLD = {"co", "com", "org", "net", "gov", "ac", "edu", "or", "ne"}
def canon(u):
    """Canonical registered domain: strips www, ports and subdomains."""
    d = domain(u or "").split(":")[0]
    parts = d.split(".")
    if len(parts) <= 2:
        return d
    if len(parts[-1]) == 2 and parts[-2] in SLD:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def num(v, hi):
    try:
        return max(0, min(int(float(v)), hi))
    except Exception:
        return 0

def clean_name(name, d):
    """Use the domain as the brand when the page title looks like an SEO headline."""
    if len(name or "") > 35 or HEADLINE.search(name or ""):
        root = d.split(".")[0]
        root = re.sub(r"(consulting|consultants|advisory|advisors|research|group|insights|partners)$",
                      lambda m: " " + m.group(1).title(), root)
        return root.replace("-", " ").title().strip()
    return name

def brand(title, d):
    root = d.split(".")[0]
    for p in [x.strip() for x in re.split(r"\s[\|\-–—:]\s", title or "") if x.strip()]:
        if norm(root)[:6] in norm(p):
            return clean_name(p[:60], d)
    return root.replace("-", " ").title()

def region_of(text):
    for code, pat in REGION_PAT.items():
        if re.search(pat, text or "", re.I):
            return code
    return None

def http_get(url, tries=2, timeout=6, html_only=True):
    """GET with retries and backoff. Returns the response or None."""
    for a in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                if html_only and "html" not in r.headers.get("content-type", ""):
                    return None
                return r
            if r.status_code in (401, 403, 404, 410):
                return None
        except Exception:
            pass
        time.sleep(1.5 * (a + 1))
    return None

def text_of(html):
    try:
        import trafilatura
        t = trafilatura.extract(html)
        if t:
            return t
    except Exception:
        pass
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

class BudgetReached(Exception):
    pass

calls = 0
def llm_call(prompt):
    """Counts only UNCACHED calls against the budget."""
    global calls
    k = hashlib.md5((ROLE + prompt).encode()).hexdigest()
    if k not in llm.cache:
        if calls >= BUDGET:
            raise BudgetReached(f"LLM budget of {BUDGET} calls reached")
        calls += 1
    return llm.ask(ROLE, prompt)

def csv_df():
    if not os.path.exists(CSV):
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(CSV, dtype=str).fillna("").reindex(columns=COLS, fill_value="")

def have(code):
    df = csv_df()
    sub = df[(df["region"] == code) & (df["approved"].str.lower() != "no")]
    doms = {canon(u) for u in sub["url"] if u}
    doms |= {d for d, c in state["cands"].items() if c["region"] == code and c["status"] == "scored"}
    return len(doms)

# ---------- provenance ----------
def ensure_meta(d, c, method=None, src=None, frm=None):
    c.setdefault("regions", [c.get("region")])
    c.setdefault("domain", d)
    c.setdefault("discovery_method",
                 method or ("manual" if c.get("manual") else "harvest" if c.get("harvested") else "legacy"))
    c.setdefault("source_url", src or (c["urls"][0] if c.get("urls") else ""))
    c.setdefault("discovered_from", frm or "")
    c.setdefault("first_discovered_at", now_iso())
    c.setdefault("last_verified_at", "")
    c.setdefault("monitoring_status", "unknown")

RANK = {"scored": 5, "filtered": 4, "ready": 3, "new": 2, "gated": 1, "thin": 0}

def normalize_state():
    """Global dedup + one-time repairs of earlier bugs."""
    cands, changed = state["cands"], 0
    for d in list(cands):
        k = canon(d)
        if k == d:
            continue
        src = cands.pop(d)
        tgt = cands.get(k)
        regs = set(src.get("regions", [src.get("region")]))
        if tgt is None:
            cands[k] = src
        else:
            regs |= set(tgt.get("regions", [tgt.get("region")]))
            if RANK.get(src.get("status"), 0) > RANK.get(tgt.get("status"), 0):
                cands[k] = src
        cands[k]["regions"] = sorted(r for r in regs if r)
        if "row" in cands[k]:
            cands[k]["row"]["url"] = f"https://{k}"
        changed += 1
    requeued = cleaned = 0
    for d, c in cands.items():
        ensure_meta(d, c)
        # repair 1: boutiques wrongly filtered because the model returned "a|b|c" as the type
        if c.get("status") == "filtered" and "|" in str(c.get("competitor_type", "")):
            c["status"] = "ready"
            c.pop("competitor_type", None)
            requeued += 1
        # repair 2: headline-style names
        nm = clean_name(c.get("name", ""), d)
        if nm != c.get("name"):
            c["name"] = nm
            cleaned += 1
    if changed or requeued or cleaned:
        print(f"Normalized {changed} duplicates | re-queued {requeued} wrongly filtered | "
              f"cleaned {cleaned} names")
        save_state()

def backup():
    if DRY_RUN:
        return
    os.makedirs("data/backups", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for f in (STATE_F, CSV):
        if os.path.exists(f):
            shutil.copy(f, f"data/backups/{ts}_{os.path.basename(f)}")
    for suffix in ("_discover_state_v2.json", "_competitors.csv"):
        files = sorted(f for f in os.listdir("data/backups") if f.endswith(suffix))
        for old in files[:-10]:
            os.remove(os.path.join("data/backups", old))

# ============================ SEARCH (cached) ============================
tav = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]) if os.getenv("TAVILY_API_KEY") else None
BACKEND = "tavily" if tav else "ddg"
SEARCH_OK = True
fail_streak = 0

def _ddg(q, dom, n):
    try:
        from ddgs import DDGS
        hits = DDGS().text(f"site:{dom[0]} {q}" if dom else q, max_results=n)
        time.sleep(1)
        return [{"url": h["href"], "title": h["title"], "content": h["body"]} for h in hits]
    except Exception as e:
        print("  ddg fail:", str(e)[:80])
        return None

def search_health():
    """Validate fresh-search connectivity before depending on it. Cached results always work."""
    global SEARCH_OK
    if BACKEND == "ddg" and not _ddg("strategy consulting firm", None, 3):
        SEARCH_OK = False
        print("WARNING: DuckDuckGo returned no results. Fresh searches are disabled this run; "
              "cached results, harvesting and manual seeds still work. Set TAVILY_API_KEY to fix.")

def tsearch(q, include_domains=None, max_results=10):
    global BACKEND, SEARCH_OK, fail_streak
    key = f"{q}|{include_domains}|{max_results}"
    for k in (key, f"{q}|{include_domains}|8"):
        if k in search_cache:
            return search_cache[k]
    if not SEARCH_OK:
        return []
    res = None
    if BACKEND == "tavily":
        for a in range(3):
            try:
                kw = {"include_domains": include_domains} if include_domains else {}
                res = tav.search(q, search_depth="basic", max_results=max_results, **kw)["results"]
                time.sleep(0.2)
                break
            except Exception as e:
                m = str(e).lower()
                if any(x in m for x in ("limit", "usage", "432", "433", "credit")):
                    print("  Tavily credits exhausted -> DuckDuckGo")
                    BACKEND = "ddg"
                    break
                time.sleep(2 * (a + 1))
    if res is None and BACKEND == "ddg":
        res = _ddg(q, include_domains, max_results)
    if res is None:
        fail_streak += 1
        if fail_streak >= 5:
            SEARCH_OK = False
            print("  5 consecutive search failures -> fresh search disabled for this run")
        return []
    fail_streak = 0
    search_cache[key] = res
    if not DRY_RUN:
        json.dump(search_cache, open(SEARCH_F, "w"))
    return res

# ============================ EMBEDDINGS (local, free) ============================
_emb = None
_line_vecs = None
def embed_norm(texts):
    global _emb
    if _emb is None:
        from fastembed import TextEmbedding
        _emb = TextEmbedding(EMB_MODEL)
    F = np.array(list(_emb.embed(texts)))
    return F / np.linalg.norm(F, axis=1, keepdims=True)

def line_vecs():
    global _line_vecs
    if _line_vecs is None:
        _line_vecs = embed_norm(LINE_TEXTS)
    return _line_vecs

# ============================ ENRICHMENT + MONITORING (free) ============================
def monitoring_from(info):
    if not info.get("reachable"):
        return "unknown"
    pub = bool(info.get("blog_url") or info.get("press_url"))
    feed = bool(info.get("rss_url") or info.get("sitemap"))
    n = sum(map(bool, [info.get("blog_url"), info.get("press_url"), info.get("rss_url"), info.get("sitemap")]))
    if pub and feed:
        return "excellent"
    if n >= 2 or pub:
        return "good"
    if n == 1:
        return "moderate"
    return "poor"

def enrich(d, c=None):
    """Fetch homepage + about/services pages, detect blog/press/RSS/sitemap. No LLM, no search credits."""
    base = f"https://{d}"
    info = {"at": now_iso(), "reachable": False}
    r = http_get(base)
    if r:
        info["reachable"] = True
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
        md = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
        desc = md["content"].strip() if md and md.get("content") else ""
        found = {}
        for a in soup.find_all("a", href=True):
            href = urljoin(base, a["href"])
            if not href.startswith("http") or canon(href) != d:
                continue
            blob = urlparse(href).path.lower() + " " + a.get_text(" ", strip=True).lower()
            for k, pat in LINK_PAT.items():
                if k not in found and re.search(pat, blob):
                    found[k] = href
        feeds = [urljoin(base, l["href"]) for l in soup.find_all("link", href=True)
                 if any(x in (l.get("type") or "") for x in ("rss", "atom"))]
        rb = http_get(base + "/robots.txt", html_only=False, tries=1)
        sm = bool(rb and re.search(r"(?im)^sitemap:\s*\S+", rb.text))
        if not sm:
            sx = http_get(base + "/sitemap.xml", html_only=False, tries=1)
            sm = bool(sx and "<" in sx.text[:300])
        extra = {}
        for k in ("about", "services"):
            u = found.get(k)
            if u and u.rstrip("/") != base.rstrip("/"):
                rr = http_get(u, tries=2)
                extra[k] = text_of(rr.text)[:500] if rr else ""
            else:
                extra[k] = ""
        home_txt = text_of(r.text)[:1500]
        alltext = " ".join([title, desc, extra["about"], extra["services"], home_txt])
        info.update({
            "title": title[:150], "desc": desc[:300], "about": extra["about"][:300],
            "services": extra["services"][:300],
            "industries": [w for w in INDUSTRY_WORDS if w in alltext.lower()][:8],
            "geo": [code for code, pat in REGION_PAT.items() if re.search(pat, alltext, re.I)],
            "blog_url": found.get("blog", ""), "press_url": found.get("press", ""),
            "rss_url": feeds[0] if feeds else "", "sitemap": sm})
    info["monitoring_status"] = monitoring_from(info)
    state["monitor"][d] = info
    if c is not None:
        c["enrich"] = info
        c["monitoring_status"] = info["monitoring_status"]
        c["last_verified_at"] = now_iso()
        if info["reachable"]:
            c["ev_llm"] = (f"{info.get('title', '')}. {info.get('desc', '')} "
                           f"Services: {info.get('services', '')[:250]} About: {info.get('about', '')[:250]} "
                           f"Industries: {', '.join(info.get('industries', []))}. "
                           f"Markets mentioned: {', '.join(info.get('geo', []))}.")[:1500]
    return info

def phase_enrich():
    if not ENRICH:
        return
    cands = state["cands"]
    known = {canon(u) for u in csv_df()["url"] if u}
    n = 0
    for code in REGIONS:
        todo = sorted([(d, c) for d, c in cands.items()
                       if c["region"] == code and c["status"] == "ready" and d not in known
                       and (ENRICH_REFRESH or "enrich" not in c)],
                      key=lambda x: (not x[1].get("manual"), -x[1].get("emb", 0)))[:ENRICH_TOP]
        for d, c in todo:
            enrich(d, c)
            n += 1
        save_state()
    df = csv_df()
    df = df[df["approved"].str.lower() == "yes"]
    m = 0
    for _, r in df.iterrows():
        d = canon(r["url"])
        if (d in state["monitor"] and not ENRICH_REFRESH) or m >= ENRICH_APPROVED_MAX:
            continue
        enrich(d, cands.get(d))
        m += 1
    save_state()
    print(f"Enriched {n} candidates and {m} approved firms (0 tokens, 0 search credits)")

# ============================ LOCAL PRE-SCREENING SCORE (free) ============================
def region_score(c, d):
    code = c["region"]
    text = (c.get("ev_llm") or "") + " " + c["ev"]
    s = 0.0
    if re.search(REGION_PAT.get(code, "$^"), text, re.I):
        s += 0.6
    if d.endswith(TLD_HINT.get(code, ())):
        s += 0.2
    if code in (c.get("enrich") or {}).get("geo", []):
        s += 0.2
    return min(1.0, s)

def evidence_score(c):
    s = min(1.0, len(c["ev"]) / 1500) * 0.5
    e = c.get("enrich")
    if e and e.get("reachable"):
        s += 0.5 if len((e.get("about") or "") + (e.get("services") or "")) > 80 else 0.25
    return min(1.0, s)

def publishing_score(c):
    return {"excellent": 1.0, "good": 0.75, "moderate": 0.4, "poor": 0.0}.get(c.get("monitoring_status"), 0.3)

def feedback_scores(items):
    """Similarity to approved firms (positives) and optionally rejected ones (negatives)."""
    df = csv_df()
    def texts(flag):
        out = []
        for _, r in df[df["approved"].str.lower() == flag].iterrows():
            c = state["cands"].get(canon(r["url"]))
            t = ((c.get("ev_llm") or c.get("ev")) if c else "") or f"{r['name']}. {r['rationale']} {r['evidence_quote']}"
            out.append(t[:800])
        return out
    pos = texts("yes")
    neg = texts("no") if FEEDBACK_NEG else []
    if not pos and not neg:
        return np.zeros(len(items))
    V = embed_norm([(c.get("ev_llm") or c["ev"])[:800] for _, c in items])
    fb = np.zeros(len(items))
    if pos:
        P = embed_norm(pos)
        fb += (V @ P.T).max(axis=1)
    if neg:
        N = embed_norm(neg)
        fb -= (V @ N.T).max(axis=1)
    return fb - fb.mean() if len(items) > 1 else np.zeros(len(items))

def compute_local_scores():
    items = [(d, c) for d, c in state["cands"].items() if c["status"] in ("ready", "scored")]
    if not items:
        return
    raw = np.array([c.get("emb", 0.5) for _, c in items])
    s1 = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9) if len(raw) > 1 else np.array([0.5])
    fb = feedback_scores(items)
    for i, (d, c) in enumerate(items):
        t = (c.get("ev_llm") or "") + " " + c["ev"]
        pos = len({m.lower() for m in POS.findall(t)})
        neg = len(NEG.findall(t))
        s2 = max(0.0, min(1.0, pos / 5) - 0.2 * neg)
        s3, s4, s5 = region_score(c, d), evidence_score(c), publishing_score(c)
        local = 100 * (W["service"] * s1[i] + W["model"] * s2 + W["region"] * s3
                       + W["evidence"] * s4 + W["publishing"] * s5)
        local += 100 * FEEDBACK_WEIGHT * float(np.clip(fb[i], -1, 1))
        c["local_score"] = round(max(0.0, min(100.0, local)), 1)
        c["keyword_score"] = round(100 * s2)
        c["local_dims"] = {"service": round(float(s1[i]), 2), "model": round(s2, 2),
                           "region": round(s3, 2), "evidence": round(s4, 2), "publishing": round(s5, 2)}

# ============================ PHASE 0: manual seeds (optional) ============================
def phase_manual():
    p = "data/manual_competitors.csv"        # columns: name,region,website
    if not os.path.exists(p):
        return
    for _, r in pd.read_csv(p, dtype=str).fillna("").iterrows():
        d = canon(r["website"]) if r["website"] else ""
        if not d or d in state["cands"] or r["region"] not in REGIONS:
            continue
        res = tsearch(f"{r['name']} services clients about", include_domains=[d], max_results=4)
        ev = " ".join((x.get("content") or "")[:500] for x in res)[:1200]
        if len(ev) < 100:
            print(f"  manual seed skipped (no content found): {r['name']}")
            continue
        c = state["cands"][d] = {"name": r["name"], "region": r["region"], "ev": ev, "urls": [],
                                 "status": "ready", "emb": 0.6, "manual": True}
        ensure_meta(d, c, "manual", f"https://{d}", "data/manual_competitors.csv")
    save_state()

# ============================ PHASE 1: search -> candidates (NO LLM) ============================
def gap_regions():
    known = {canon(u) for u in csv_df()["url"] if u}
    out = set()
    for code in REGIONS:
        pool = sum(1 for d, c in state["cands"].items()
                   if c["region"] == code and c["status"] in ("new", "ready") and d not in known)
        if have(code) + pool < TARGET:
            out.add(code)
    return out

def add_result(code, r, q, method="search"):
    d = canon(r["url"])
    title = r.get("title") or ""
    if bad(d) or d.split(".")[0] in EXCL_ROOTS or LIST.search(title):
        return
    c = state["cands"].get(d)
    if c is None:
        c = state["cands"][d] = {"name": brand(title, d), "region": code, "ev": "", "urls": [], "status": "new"}
        ensure_meta(d, c, method, r["url"], q)
    elif code not in c.setdefault("regions", [c["region"]]):
        c["regions"].append(code)
    if c["status"] in ("new", "gated") and r["url"] not in c["urls"] and len(c["ev"]) < 1200:
        c["urls"].append(r["url"])
        c["ev"] += " " + (r.get("content") or "")[:500]

def phase_search():
    gaps = gap_regions() if SERVICE_EXPAND else set()
    for code, rname in REGIONS.items():
        before = len(state["cands"])
        qs = [t.format(r=rname) for t in GENERIC] + LOCAL[code]
        if code in gaps:
            qs += [t.format(r=rname) for t in SERVICE_QUERIES.values()]
        skipped = 0
        for q in qs:
            if q in state["queries"] and not REFRESH_QUERIES:
                skipped += 1
                continue
            res = tsearch(q)
            if res:
                state["queries"][q] = {"at": now_iso(), "n": len(res)}
            for r in res:
                add_result(code, r, q)
        save_state()
        tag = " +service-level queries" if code in gaps else ""
        print(f"{code}: +{len(state['cands']) - before} candidate sites, {skipped} queries already done{tag}")

# ============================ PHASE 1b: free harvesting from listicle pages ============================
def harvest_links(page_url):
    """Outbound links that look like firm homepages: {canonical_domain: anchor_text}."""
    r = http_get(page_url)
    if not r:
        return {}
    src, out = canon(page_url), {}
    for a in BeautifulSoup(r.text, "html.parser").find_all("a", href=True):
        u = urljoin(page_url, a["href"])
        if not u.startswith("http"):
            continue
        d = canon(u)
        root = d.split(".")[0]
        if d == src or bad(d) or root in EXCL_ROOTS or root in BIG_FIRM_ROOTS \
                or any(p in root for p in PUBLISHER_ROOTS):
            continue
        if len([p for p in urlparse(u).path.split("/") if p]) > 1:
            continue
        out.setdefault(d, a.get_text(" ", strip=True)[:60])
    return out

def homepage_evidence(d):
    r = http_get(f"https://{d}")
    if not r:
        return "", d.split(".")[0].title()
    soup = BeautifulSoup(r.text, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    md = soup.find("meta", attrs={"name": "description"})
    desc = md["content"] if md and md.get("content") else ""
    ev = f"{title}. {desc} {text_of(r.text)[:700]}".strip()
    nm = re.split(r"\s[\|\-–—:]\s", title)[0][:60] or d.split(".")[0].title()
    return ev, clean_name(nm, d)

def snowball_pages():
    """'alternatives to X' searches, gap regions only (costs search credits, results are cached)."""
    out = {}
    if not (SNOWBALL and SEARCH_OK):
        return out
    df = csv_df()
    df = df[df["approved"].str.lower() != "no"].copy()
    df["s"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    for code in gap_regions() & set(REGIONS):
        for _, r in df[df.region == code].sort_values("s", ascending=False).head(SNOWBALL_SEEDS).iterrows():
            q = f"{r['name']} competitors alternatives similar firms"
            res = tsearch(q)
            if res:
                state["queries"][q] = {"at": now_iso(), "n": len(res)}
            for x in res:
                if LIST.search(x.get("title") or ""):
                    out.setdefault(x["url"], (code, "snowball", q))
                else:
                    add_result(code, x, q, "snowball")
    return out

def phase_harvest():
    if not HARVEST:
        return
    MAX_PAGES = int(os.getenv("MAX_PAGES", "40"))
    pages = {}
    for key, results in search_cache.items():
        q = key.split("|")[0]
        code = region_of(q)
        if code not in REGIONS:
            continue
        for r in results or []:
            if LIST.search(r.get("title") or ""):
                pages.setdefault(r["url"], (code, "harvest", q))
    pages.update(snowball_pages())
    done = set(state["harvested_pages"])
    todo = {u: v for u, v in pages.items() if u not in done}
    batch = dict(list(todo.items())[:MAX_PAGES])
    print(f"Harvesting {len(batch)} of {len(todo)} pending pages this run "
          f"({len(done)} done before). Rerun to continue.", flush=True)
    added = {}
    for n, (url, (code, method, frm)) in enumerate(batch.items(), 1):
        print(f"  [{n}/{len(batch)}] {code} {domain(url)}", flush=True)
        capped = False
        for d in list(harvest_links(url))[:40]:
            if d in state["cands"]:
                c = state["cands"][d]
                if code not in c.setdefault("regions", [c["region"]]):
                    c["regions"].append(code)
                continue
            if added.get(code, 0) >= MAX_HARVEST:
                capped = True
                continue
            ev, nm = homepage_evidence(d)
            if len(ev) < 120:
                continue
            c = state["cands"][d] = {"name": nm, "region": code, "ev": ev[:1200], "urls": [url],
                                     "status": "new", "harvested": True}
            ensure_meta(d, c, method, url, frm)
            added[code] = added.get(code, 0) + 1
            print(f"      + {nm[:40]}", flush=True)
        if not capped:
            state["harvested_pages"].append(url)
        save_state()
    print("New harvested sites by region:", added or "none", "(0 LLM tokens)")

# ============================ PHASE 2: free gate + local ranking ============================
def phase_rank():
    cands = state["cands"]
    if REVISIT_GATED:
        n = 0
        for c in cands.values():
            if c["status"] == "gated" and len(c["ev"]) >= c.get("gate_len", 0) + 150:
                c["status"] = "new"
                n += 1
        if n:
            print(f"Revisiting {n} gated candidates with new evidence")
    pend = [d for d, c in cands.items() if c["status"] == "new"]
    ok = []
    for d in pend:
        t = cands[d]["ev"]
        pos, neg = len({m.lower() for m in POS.findall(t)}), len(NEG.findall(t))
        if pos < 2 or neg > pos:
            cands[d]["status"] = "gated"
            cands[d]["gate_len"] = len(t)
        else:
            ok.append(d)
    if ok:
        F = embed_norm([cands[d]["ev"][:1000] for d in ok])
        for d, r in zip(ok, (F @ line_vecs().T).max(axis=1)):
            cands[d]["emb"] = float(r)
            cands[d]["status"] = "ready"
    save_state()
    print(f"Free pre-filter: {len(ok)} ready, {len(pend) - len(ok)} gated out (0 API calls)")
    phase_enrich()
    compute_local_scores()
    save_state()

# ============================ PHASE 3: batched LLM scoring (top-K only) ============================
def score_batch(code, items):
    # free pre-exclusions: large firms and report publishers never reach the LLM
    keep = []
    for d, c in items:
        root = d.split(".")[0]
        if root in BIG_FIRM_ROOTS:
            c["status"] = "filtered"
            c["competitor_type"] = "big_firm"
            print(f"  FILTERED (big firm, no LLM): {c['name']}")
        elif any(p in root for p in PUBLISHER_ROOTS):
            c["status"] = "filtered"
            c["competitor_type"] = "publisher"
            print(f"  FILTERED (report publisher, no LLM): {c['name']}")
        else:
            c["name"] = clean_name(c["name"], d)
            keep.append((d, c))
    items = keep
    if not items:
        save_state()
        return

    txt = "\n".join(f"[{i}] {c['name']} ({d})\n{(c.get('ev_llm') or c['ev'])[:EV_LLM_CHARS]}"
                    for i, (d, c) in enumerate(items))
    out = llm_call(f"""Aranca Growth Advisory lines: {LINES_SHORT}

Judge each candidate: is it a real competitor (boutique or mid-sized strategy consulting, market research / intelligence consultancy, or M&A advisory firm)? Target region: {ALL_REGIONS[code]}. Use ONLY the text given.
A firm that mainly sells syndicated market-report databases is type "other". A directory, vendor or headline page is "other".
Return JSON: {{"r":[{{"i":0,"type":"strategy_consulting","size":"boutique|mid-sized|large|unknown","fit":0-100,"in_region":true,"hq":"city, country or null","svc":["SVC01"],"why":"max 15 words"}}]}}
Pick exactly ONE value for "type": strategy_consulting, market_research, ma_advisory, or other. Never combine values with "|".
fit = overlap with Aranca's lines + size fit + presence in the target region. in_region = the text shows an office, clients or market presence in the target region. Be strict.

CANDIDATES:
{txt}""")
    res = {}
    for x in out.get("r", []):
        try:
            res[int(x.get("i"))] = x
        except Exception:
            pass
    for i, (d, c) in enumerate(items):
        x = res.get(i)
        if not x:
            continue                                   # stays 'ready': retried next run
        raw_type = str(x.get("type") or "")
        types = {t.strip() for t in re.split(r"[|,/]", raw_type) if t.strip()}
        core = next((t for t in ("strategy_consulting", "ma_advisory", "market_research")
                     if t in types), None)
        c["competitor_type"] = core or raw_type
        c["last_verified_at"] = now_iso()
        if not core or x.get("size") == "large":
            c["status"] = "filtered"
            print(f"  FILTERED ({raw_type}/{x.get('size')}): {c['name']}")
            continue
        svc = x.get("svc") or []
        fit = num(x.get("fit"), 100)
        if x.get("in_region") is False:
            fit = max(0, fit - 15)
        c["status"] = "scored"
        c["llm_score"] = fit
        c["row"] = {"name": c["name"], "region": code, "url": f"https://{d}", "hq": x.get("hq"),
                    "firm_type": core, "size_class": x.get("size"),
                    "llm_score": fit,
                    "overlap_service_ids": ",".join(svc) if isinstance(svc, list) else str(svc),
                    "evidence_quote": "", "rationale": x.get("why"),
                    "monitoring_status": c.get("monitoring_status", "unknown"),
                    "local_score": c.get("local_score", ""),
                    "discovery_method": c.get("discovery_method", ""),
                    "source_url": c.get("source_url", ""),
                    "first_discovered_at": c.get("first_discovered_at", ""),
                    "last_verified_at": c["last_verified_at"]}
        print(f"  + {c['name']} [{code}] fit={fit} local={c.get('local_score')} "
              f"monitor={c.get('monitoring_status')}")
    save_state()

def phase_score():
    known = {canon(u) for u in csv_df()["url"] if u}
    queues = {}
    for code in REGIONS:
        need = TARGET - have(code)
        if need <= 0:
            print(f"{code}: already has {have(code)} firms (target {TARGET}), skipping LLM")
            continue
        ready = sorted([(d, c) for d, c in state["cands"].items()
                        if c["region"] == code and c["status"] == "ready" and d not in known],
                       key=lambda x: (not x[1].get("manual"),
                                      -x[1].get("local_score", x[1].get("emb", 0) * 100)))
        queues[code] = ready[:min(TOP_K, need * 2)]
    print("To send to LLM:", {c: len(q) for c, q in queues.items()},
          f"| budget {BUDGET} calls, batch {BATCH}")
    while any(queues.values()):
        for code in list(queues):
            q = queues[code]
            if q:
                queues[code] = q[BATCH:]
                score_batch(code, q[:BATCH])

# ============================ REGISTRY, COVERAGE, YIELD ============================
def registry_df():
    old = csv_df()
    appr = {canon(u): a.lower() for u, a in zip(old["url"], old["approved"]) if u}
    status_of = {"yes": "approved", "no": "rejected"}
    stage_of = {"new": "discovered", "gated": "discovered", "thin": "discovered",
                "ready": "qualified", "scored": "llm_scored", "filtered": "llm_scored"}
    rows, seen = [], set()
    for d, c in state["cands"].items():
        seen.add(d)
        rows.append({
            "domain": d, "company_name": c["name"], "region": c["region"],
            "discovered_from": c.get("discovered_from", ""), "source_url": c.get("source_url", ""),
            "discovery_method": c.get("discovery_method", "legacy"),
            "first_discovered_at": c.get("first_discovered_at", ""),
            "last_verified_at": c.get("last_verified_at", ""),
            "keyword_score": c.get("keyword_score", ""),
            "embedding_score": round(c["emb"], 3) if "emb" in c else "",
            "local_score": c.get("local_score", ""), "llm_score": c.get("llm_score", ""),
            "competitor_type": c.get("competitor_type", ""),
            "monitoring_status": (state["monitor"].get(d, {}).get("monitoring_status")
                                  or c.get("monitoring_status", "unknown")),
            "approval_status": status_of.get(appr.get(d, ""), "pending"),
            "stage": stage_of.get(c["status"], "discovered")})
    for _, r in old.iterrows():
        d = canon(r["url"])
        if not d or d in seen:
            continue
        seen.add(d)
        rows.append({
            "domain": d, "company_name": r["name"], "region": r["region"],
            "discovered_from": "", "source_url": r.get("source_url", ""),
            "discovery_method": r.get("discovery_method") or "legacy",
            "first_discovered_at": r.get("first_discovered_at", ""),
            "last_verified_at": r.get("last_verified_at", ""),
            "keyword_score": "", "embedding_score": r.get("emb_score", ""),
            "local_score": r.get("local_score", ""), "llm_score": r.get("llm_score", ""),
            "competitor_type": r.get("firm_type", ""),
            "monitoring_status": (state["monitor"].get(d, {}).get("monitoring_status")
                                  or r.get("monitoring_status") or "unknown"),
            "approval_status": status_of.get(appr.get(d, ""), "pending"),
            "stage": "llm_scored"})
    return pd.DataFrame(rows)

def active_firms(days=30):
    try:
        con = sqlite3.connect("data/ci.db")
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        out = {r[0] for r in con.execute("SELECT DISTINCT company FROM items WHERE published >= ?", (since,))}
        con.close()
        return out
    except Exception:
        return set()

def coverage():
    reg = registry_df()
    if reg.empty:
        print("\nNo candidates yet.")
        return
    active = active_firms()
    print("\n=== REGIONAL COVERAGE TRACKER ===")
    print(f"{'Region':7}{'Discov':>8}{'Qualif':>8}{'LLM':>6}{'Approv':>8}{'Collect':>9}{'Active':>8}{'Target':>8}")
    short = []
    for code in ALL_REGIONS:
        s = reg[reg.region == code]
        appr = s[s.approval_status == "approved"]
        coll = appr[appr.monitoring_status.isin(["excellent", "good", "moderate"])]
        act = appr[appr.company_name.isin(active)]
        qual = s[s.stage.isin(["qualified", "llm_scored"])]
        llm_n = s[s.stage == "llm_scored"]
        print(f"{code:7}{len(s):8}{len(qual):8}{len(llm_n):6}{len(appr):8}{len(coll):9}{len(act):8}{TARGET:8}")
        if len(appr) < 3 or len(coll) < 3:
            short.append(code)
    print("Discov = found | Qualif = passed local gate | LLM = scored by LLM | Approv = approved=yes | "
          "Collect = approved and a monitorable channel exists | Active = published in last 30 days")
    if short:
        print(f"Regions with fewer than 3 collectable approved firms: {', '.join(short)}")
        print("  -> add known firms to data/manual_competitors.csv, or run with SNOWBALL=1 SERVICE_EXPAND=1")
    silent = reg[(reg.approval_status == "approved") & (reg.monitoring_status == "poor")]
    if len(silent):
        print("\nApproved but hard to monitor (no blog/press/RSS/sitemap found), consider replacing:")
        for _, r in silent.iterrows():
            print(f"  {r.company_name} [{r.region}] {r.domain}")
    print("\n=== DISCOVERY YIELD BY METHOD ===")
    y = reg.groupby("discovery_method").agg(
        discovered=("domain", "count"),
        approved=("approval_status", lambda s: int((s == "approved").sum())))
    y["yield_%"] = (100 * y["approved"] / y["discovered"]).round(1)
    print(y.to_string())
    if not DRY_RUN:
        reg.to_csv(REGISTRY_F, index=False)
        print(f"\nProvenance registry -> {REGISTRY_F}")

# ============================ OUTPUT ============================
def finalize():
    cands = state["cands"]
    prev = [{"name": c["name"], "region": c["region"], "url": f"https://{d}", "status": c["status"],
             "local_score": c.get("local_score", ""), "keyword_score": c.get("keyword_score", ""),
             "emb": round(c.get("emb", 0), 3), "monitoring": c.get("monitoring_status", "unknown"),
             "method": c.get("discovery_method", ""), "source_url": c.get("source_url", "")}
            for d, c in cands.items() if c["status"] in ("ready", "scored")]
    if prev and not DRY_RUN:
        pd.DataFrame(prev).sort_values(["region", "local_score"], ascending=[True, False]
                                       ).to_csv("data/candidates_preview.csv", index=False)

    scored = [c for c in cands.values() if c["status"] == "scored"]
    old = csv_df()
    new = pd.DataFrame()
    if scored:
        raw = np.array([c.get("emb", 0.5) for c in scored])
        emb_score = 100 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-9) if len(raw) > 1 else np.array([50.0])
        new = pd.DataFrame([c["row"] for c in scored])
        new["emb_score"] = np.round(emb_score)
        new["score"] = (0.7 * new["llm_score"] + 0.3 * new["emb_score"]).round(0)
        new["tier"] = pd.cut(new["score"], [-1, 59, 74, 101], labels=["low", "medium", "high"]).astype(str)
        new = new[new["score"] >= MIN_SCORE].copy()
        appr = {canon(u): a for u, a in zip(old["url"], old["approved"])}
        new["approved"] = new["url"].map(lambda u: appr.get(canon(u), ""))
    keep = old[~old["url"].map(canon).isin(new["url"].map(canon))] if not new.empty else old
    out = pd.concat([new, keep], ignore_index=True).reindex(columns=COLS, fill_value="")
    if out.empty:
        print("Nothing to write yet.")
        return
    out["monitoring_status"] = [
        (m if m and m != "unknown" else state["monitor"].get(canon(u), {}).get("monitoring_status", m or ""))
        for u, m in zip(out["url"], out["monitoring_status"])]
    out["_s"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)
    out["_d"] = out["url"].map(canon)
    out = (out.drop_duplicates("_d").sort_values(["region", "_s"], ascending=[True, False])
              .drop(columns=["_s", "_d"]))
    if DRY_RUN:
        print(f"\nDRY RUN: would save {len(out)} competitors to {CSV} (nothing written)")
        return
    if os.path.exists(CSV):
        shutil.copy(CSV, "data/competitors_backup.csv")
    out.to_csv(CSV, index=False)
    print(f"\nSaved {len(out)} competitors -> {CSV} (backup: data/competitors_backup.csv)")

# ============================ MAIN ============================
print(f"Resuming: {len(state['cands'])} candidate sites known, "
      f"{sum(c['status'] == 'scored' for c in state['cands'].values())} scored, "
      f"{len(state['queries'])} queries done" + ("  [DRY RUN]" if DRY_RUN else ""))
try:
    backup()
    normalize_state()
    search_health()
    phase_manual()
    phase_search()
    phase_harvest()
    phase_rank()
    phase_score()
except BudgetReached as e:
    print("\n" + str(e) + ". Progress saved; rerun to continue.")
except Exhausted as e:
    print("\n" + str(e))
except KeyboardInterrupt:
    print("\nInterrupted. Progress saved.")
finally:
    save_state()
    print(f"LLM calls used this run: {calls}/{BUDGET}")
    finalize()
    coverage()