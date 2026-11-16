"""RAG layer. Local embeddings (fastembed) + numpy: no API tokens spent on retrieval.
  python src/rag.py build | match "text" | ask "question"
"""
import json, os, re, sqlite3, sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")
ARANCA_DIR = "data/aranca"
INDEX_F, META_F, ITEM_F = "data/aranca_index.npz", "data/aranca_index.json", "data/item_vecs.npz"
DB = os.getenv("CI_DB", "data/ci.db")
GENERIC_TITLES = {"our approach", "impact", "approach", "overview", "benefits", "outcome", "outcomes"}

_emb = None
def embed(texts):
    global _emb
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    if _emb is None:
        from fastembed import TextEmbedding
        _emb = TextEmbedding(EMB_MODEL)
    M = np.asarray(list(_emb.embed([t[:1000] for t in texts])), dtype="float32")
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

def _rd(name):
    p = f"{ARANCA_DIR}/aranca_{name}.csv"
    return pd.read_csv(p, dtype=str).fillna("") if os.path.exists(p) else pd.DataFrame()

def build_index():
    svc, sol = _rd("services"), _rd("solutions")
    if svc.empty:
        sys.exit(f"{ARANCA_DIR}/aranca_services.csv not found")
    svc_name = dict(zip(svc["service_id"], svc["service_name"]))
    sol_name = dict(zip(sol["solution_id"], sol["solution_name"])) if not sol.empty else {}
    sol_svc = dict(zip(sol["solution_id"], sol["service_id"])) if not sol.empty else {}
    chunks = []
    def add(kind, sid, title, text):
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 15:
            chunks.append({"kind": kind, "sid": sid, "title": title[:120], "text": text[:700]})
    for _, r in svc.iterrows():
        add("service", r["service_id"], r["service_name"],
            f"{r['service_name']}. {r['description']} {r['key_offerings']}")
    for _, r in sol.iterrows():
        add("solution", r["service_id"], r["solution_name"],
            f"{r['solution_name']} (Aranca service: {svc_name.get(r['service_id'], '')})")
    for _, r in _rd("capabilities").iterrows():
        if r["capability"].strip().lower() in GENERIC_TITLES:
            continue
        sid = sol_svc.get(r["solution_id"], "")
        add("capability", sid, r["capability"],
            f"{r['capability']}. {r['keywords']}. Solution: {sol_name.get(r['solution_id'], '')}")
    for _, r in _rd("differentiators").iterrows():
        add("differentiator", r["service_id"], r["title"],
            f"{r['title']}. {r['description']} {r['client_impact']}")
    for _, r in _rd("experience").iterrows():
        add("experience", r["service_id"], r["assignment_example"][:100],
            f"{r['assignment_example']} [{r['geography']}; {r['industry']}]")
    for _, r in _rd("case_studies").iterrows():
        add("case_study", "", r["project_title"],
            f"{r['project_title']} ({r['industry']}). {r['challenge']} {r['impact']}")
    V = embed([f"{c['title']}. {c['text']}" for c in chunks])
    np.savez_compressed(INDEX_F, vecs=V.astype("float16"))
    json.dump({"chunks": chunks, "svc": svc_name, "model": EMB_MODEL},
              open(META_F, "w", encoding="utf-8"))
    print(f"Aranca index: {len(chunks)} chunks -> {INDEX_F}")
    return len(chunks)

_idx = None
def _index():
    global _idx
    if _idx is None:
        if not (os.path.exists(INDEX_F) and os.path.exists(META_F)):
            if not os.path.exists(f"{ARANCA_DIR}/aranca_services.csv"):
                _idx = (np.zeros((0, 384), dtype="float32"), {"chunks": [], "svc": {}})
                return _idx
            build_index()
        _idx = (np.load(INDEX_F)["vecs"].astype("float32"), json.load(open(META_F, encoding="utf-8")))
    return _idx

def match_many(texts, k=4):
    A, meta = _index()
    if not len(A) or not texts:
        return [[] for _ in texts]
    sims = embed(texts) @ A.T
    out = []
    for row in sims:
        res, seen = [], set()
        for j in np.argsort(-row)[:k * 3]:
            c = meta["chunks"][j]
            key = (c["sid"], c["title"].lower())
            if key in seen:
                continue
            seen.add(key)
            res.append({**c, "score": float(row[j])})
            if len(res) == k:
                break
        out.append(res)
    return out

def top_services(matches, n=2, min_score=0.45):
    agg = {}
    for m in matches:
        if m["sid"] and m["score"] >= min_score:
            agg[m["sid"]] = agg.get(m["sid"], 0) + m["score"]
    return [s for s, _ in sorted(agg.items(), key=lambda x: -x[1])[:n]]

def brief(matches, n=2):
    _, meta = _index()
    parts = [f"{meta['svc'].get(m['sid'], m['kind'])}: {m['title'][:70]}" for m in matches[:n] if m["score"] >= 0.4]
    return "; ".join(parts)

REGION_WORDS = {"KSA": ["saudi", "ksa", "riyadh", "jeddah", "gulf", "gcc"],
                "UAE": ["uae", "dubai", "abu dhabi", "emirates", "gulf", "gcc"],
                "Japan": ["japan", "japanese", "tokyo"], "India": ["india", "indian", "mumbai", "delhi"],
                "US": ["usa", "united states", "america", "american"]}
STOP = set("what have has the and for are with from that this recent latest announced any new about did does who".split())

def load_items():
    if not os.path.exists(DB):
        return pd.DataFrame()
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql("""SELECT i.id, i.company, i.region, i.title, i.url, i.published, i.category_hint,
                   i.snippet, n.cat, n.rel, n.summary, n.why, n.state
            FROM items i LEFT JOIN ci_insights n ON n.id = i.id
            WHERE COALESCE(n.state, '') != 'skipped'""", con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()
    done = (df["state"] == "done") & df["cat"].notna()
    df["category"] = df["cat"].where(done, df["category_hint"]).fillna("Digital Presence")
    df["rel"] = pd.to_numeric(df["rel"], errors="coerce").where(done, 1).fillna(1)
    df["summary"] = df["summary"].where(done, df["snippet"].fillna("").str[:200]).fillna("")
    df["why"] = df["why"].fillna("")
    df = df[(df["category"] != "Noise") & (df["rel"] > 0) & df["published"].notna()].copy()
    df["doc"] = (df["company"] + " (" + df["region"] + ") " + df["category"] + ": "
                 + df["summary"].where(df["summary"] != "", df["title"]) + " " + df["why"])
    return df.reset_index(drop=True)

def _item_vecs(df):
    cache = {}
    if os.path.exists(ITEM_F):
        z = np.load(ITEM_F, allow_pickle=False)
        cache = dict(zip(z["ids"].tolist(), z["vecs"].astype("float32")))
    miss = [i for i, x in enumerate(df["id"]) if x not in cache]
    if miss:
        V = embed([df["doc"].iloc[i] for i in miss])
        for i, v in zip(miss, V):
            cache[df["id"].iloc[i]] = v
        ids = np.array(list(cache.keys()))
        np.savez_compressed(ITEM_F, ids=ids, vecs=np.stack([cache[x] for x in ids]).astype("float16"))
    return np.stack([cache[x] for x in df["id"]]) if len(df) else np.zeros((0, 384), dtype="float32")

def _parse_filters(question, df):
    q = question.lower()
    regions = [r for r, ws in REGION_WORDS.items() if any(re.search(rf"\b{re.escape(w)}\b", q) for w in ws)]
    if re.search(r"\bUS\b", question):
        regions.append("US")
    comps = [c for c in df["company"].unique() if len(c) > 3 and c.lower() in q]
    days = None
    m = re.search(r"last (\d+) days", q)
    if m:
        days = int(m.group(1))
    elif "today" in q or "yesterday" in q:
        days = 2
    elif "this week" in q or "past week" in q:
        days = 7
    elif "this month" in q or "recent" in q or "latest" in q:
        days = 30
    return regions, comps, days

def retrieve(question, k=8):
    df = load_items()
    if df.empty:
        return df
    regions, comps, days = _parse_filters(question, df)
    mask = pd.Series(True, index=df.index)
    if regions:
        mask &= df["region"].isin(regions)
    if comps:
        mask &= df["company"].isin(comps)
    if days:
        mask &= df["published"] >= (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if mask.sum() < 3 and regions:           # keep the region, drop the time window
        mask = df["region"].isin(regions)
        if comps:
            mask &= df["company"].isin(comps)
    elif mask.sum() < 3:
        mask = pd.Series(True, index=df.index)
    sub = df[mask].copy()
    if sub.empty:
        return sub
    sub = sub.drop_duplicates(subset=["company", "summary"])
    V = _item_vecs(sub)
    sims = V @ embed([question])[0]
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 3 and w not in STOP}
    kw = sub["doc"].str.lower().map(lambda t: sum(w in t for w in words) / (len(words) or 1))
    sub["score"] = sims + 0.15 * kw.values + 0.02 * sub["rel"].values
    ranked = sub.sort_values("score", ascending=False)
    return ranked.groupby("company", sort=False).head(3).head(k)

def answer(question, k=8):
    df = load_items()
    regions, _, _ = _parse_filters(question, df) if not df.empty else ([], [], None)
    covered = set(df["region"]) if not df.empty else set()
    missing = [r for r in regions if r not in covered]
    note = f" (No stored items for: {', '.join(missing)}.)" if missing else ""
    hits = retrieve(question, k)
    if hits.empty:
        return {"answer": text + note, "sources": src}
    ar = match_many([d for d in hits["doc"]], k=2)
    src = [{"company": r.company, "region": r.region, "date": r.published[:10], "summary": r.summary or r.title,
            "url": r.url, "aranca": brief(m)} for r, m in zip(hits.itertuples(), ar)]
    ctx = "\n".join(f"[{i}] {s['company']} ({s['region']}, {s['date']}): {s['summary']}"
                    + (f" | Aranca offering affected: {s['aranca']}" if s["aranca"] else "")
                    for i, s in enumerate(src, 1))
    try:
        import llm
        out = llm.ask("fast", f"""Answer the question using ONLY the numbered developments below. Describe blog posts and reports as publications, and only call deals, hires, launches and M&A announcements. Cite sources as [n]. Where useful, say which Aranca offering is affected (given in the text). If they do not answer the question, say so.
Return JSON {{"answer": "..."}}
QUESTION: {question}
DEVELOPMENTS:
{ctx}""")
        text = out.get("answer") or "No answer returned."
    except BaseException as e:
        text = f"LLM unavailable ({str(e)[:60]}). Matching developments are listed below."
    return {"answer": text, "sources": src}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = " ".join(sys.argv[2:])
    if cmd == "build":
        build_index()
    elif cmd == "match":
        for m in match_many([arg])[0]:
            print(f"{m['score']:.2f}  {m['sid']:6} {m['kind']:14} {m['title']}")
    elif cmd == "ask":
        res = answer(arg)
        print(res["answer"], "\n")
        for i, s in enumerate(res["sources"], 1):
            print(f"[{i}] {s['company']} ({s['region']}, {s['date']}): {s['summary'][:90]}\n    {s['url']}")
    else:
        print(__doc__)