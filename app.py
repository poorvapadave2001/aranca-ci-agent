import json, os, sqlite3, sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
from rapidfuzz import fuzz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
st.set_page_config(page_title="Aranca CI Agent", page_icon="🎯", layout="wide")

DB = os.getenv("CI_DB", "data/ci.db")
URG_W = {"high": 3, "med": 2, "low": 1}
URG_BADGE = {"high": "🔴 HIGH", "med": "🟠 MEDIUM", "low": "🟢 LOW"}
CAT_ICON = {"M&A": "🤝", "Deal": "📝", "Business Update": "📈", "Digital Presence": "🌐", "Noise": "🔇"}
CAT_W = {"M&A": 4, "Deal": 3, "Business Update": 2, "Digital Presence": 1, "Noise": 0}
HINT_REL = {"M&A": 3, "Deal": 3, "Business Update": 2, "Digital Presence": 1}
SYN = {"ksa": ["saudi", "riyadh", "jeddah"], "uae": ["dubai", "abu dhabi", "emirates", "gulf", "gcc"],
       "japan": ["japanese", "tokyo"], "india": ["indian", "mumbai", "delhi"],
       "us": ["america", "united states"]}


def load_json(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}


SVC_NAME = {s["id"]: s["name"] for s in load_json("data/aranca_profile.json").get("service_lines", [])}
PL = load_json("data/practice_leaders.json")


def svc_label(ids):
    out = []
    for i in str(ids or "").split(","):
        i = i.strip()
        if i:
            pl = PL.get(i, "")
            out.append(f"{SVC_NAME.get(i, i)}" + (f" ({pl})" if pl else ""))
    return ", ".join(out)


# ------------------------- data -------------------------
@st.cache_data(ttl=60)
def load():
    if not os.path.exists(DB):
        return pd.DataFrame(), {}
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql("""SELECT i.id, i.company, i.region, i.title, i.url, i.published, i.source_type,
                   i.publisher, i.category_hint, i.service_ids, i.snippet,
                   n.cat, n.rel, n.summary, n.why, n.svc, n.urg, n.state, n.analyzed_at
            FROM items i LEFT JOIN ci_insights n ON n.id = i.id
            WHERE COALESCE(n.state, '') != 'skipped'""", con)
    except Exception:       # ci_insights not created yet: show collection hints only
        try:
            df = pd.read_sql("SELECT id, company, region, title, url, published, source_type, publisher, "
                             "category_hint, service_ids, snippet FROM items", con)
        except Exception:
            con.close()
            return pd.DataFrame(), {}
        for c in ("cat", "rel", "summary", "why", "svc", "urg", "state", "analyzed_at"):
            df[c] = None
    con.close()
    if df.empty:
        return df, {}
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["published"])
    df["analyzed"] = (df["state"] == "done") & df["cat"].notna()
    df["category"] = df["cat"].where(df["analyzed"], df["category_hint"]).fillna("Digital Presence")
    df.loc[~df["category"].isin(CAT_ICON), "category"] = "Digital Presence"
    df["rel"] = pd.to_numeric(df["rel"], errors="coerce")
    df["rel"] = df["rel"].where(df["analyzed"], df["category"].map(HINT_REL)).fillna(1).astype(int)
    df["summary"] = df["summary"].where(df["analyzed"], df["snippet"].fillna("").str[:180])
    df["why"] = df["why"].where(df["analyzed"], "").fillna("")
    df["urg"] = df["urg"].where(df["urg"].isin(URG_W), "low")
    df["svc_ids"] = df["svc"].where(df["analyzed"] & df["svc"].fillna("").ne(""), df["service_ids"]).fillna("")
    df["priority"] = df["rel"] * 10 + df["urg"].map(URG_W).fillna(1) * 3 + df["category"].map(CAT_W).fillna(0)
    # near-duplicate removal per company (same story from several sources), best copy first
    df = df.sort_values(["analyzed", "rel", "published"], ascending=[False, False, False])
    seen, keep = {}, []
    for idx, r in df.iterrows():
        s = r["summary"] or r["title"] or ""
        if any(fuzz.token_set_ratio(s, t) >= 85 for t in seen.get(r["company"], [])):
            continue
        seen.setdefault(r["company"], []).append(s)
        keep.append(idx)
    df = df.loc[keep].sort_values("published", ascending=False).reset_index(drop=True)
    return df, {"analyzed": int(df["analyzed"].sum()), "total": len(df)}


ins_all, stats = load()


def digest_md(df, days):
    lines = ["# Aranca Competitive Intelligence Digest",
             f"_Generated {datetime.now().strftime('%d %b %Y %H:%M')} | last {days} days_\n",
             f"**{len(df)} developments** across **{df.company.nunique()} firms** in **{df.region.nunique()} regions**.\n"]
    for _, r in df.sort_values("priority", ascending=False).head(10).iterrows():
        lines += [f"### {r.company} ({r.region}) - {r.category} [relevance {r.rel}/5]",
                  f"{r.summary}",
                  f"- **Why it matters:** {r.why or 'not yet analysed'}",
                  f"- **Aranca service lines / PL:** {svc_label(r.svc_ids) or 'unmapped'}",
                  f"- Source: {r.url} | Published: {r.published:%Y-%m-%d}\n"]
    return "\n".join(lines)


# ------------------------- sidebar -------------------------
st.sidebar.title("🎯 Aranca CI Agent")
st.sidebar.caption("Daily competitive intelligence for Growth Advisory Practice Leaders")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()
show_noise = st.sidebar.checkbox("Show items classed as Noise", value=False)
min_rel = st.sidebar.slider("Minimum relevance", 0, 5, 1)

if ins_all.empty:
    st.title("Aranca Competitive Intelligence")
    st.warning("No items yet. Run `python src/collect.py` then `python src/report.py`.")
    st.stop()

ins = ins_all[(ins_all.rel >= min_rel) & (show_noise | (ins_all.category != "Noise"))].copy()

last = pd.to_datetime(ins_all["analyzed_at"], utc=True, errors="coerce").max()
if pd.notna(last):
    st.sidebar.write(f"**Last analysed:** {last:%d %b %Y %H:%M} UTC")
st.sidebar.markdown("---")
st.sidebar.write("**Pipeline stats**")
st.sidebar.write(f"Items after dedup: {stats['total']}")
st.sidebar.write(f"Analysed by LLM: {stats['analyzed']}")
st.sidebar.write(f"Shown with current filters: {len(ins)}")
if os.path.exists("data/competitors.csv"):
    comp = pd.read_csv("data/competitors.csv", dtype=str).fillna("")
    appr = comp[comp["approved"].str.lower() == "yes"]
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Monitored firms:** {len(appr)}")
    st.sidebar.write(appr.groupby("region").size().to_dict())

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📰 Today's Brief", "🔎 Explorer", "🏢 Competitor Profiles",
                                         "🧭 By Practice Leader", "💬 Ask the Agent"])

# ------------------------- tab 1: brief -------------------------
with tab1:
    st.header("Today's Brief")
    days = st.select_slider("Time window (days)", [1, 2, 7, 14, 30, 60, 90], value=30)
    win = ins[ins["published"] >= datetime.now(timezone.utc) - timedelta(days=days)]
    if win.empty:
        st.info("Nothing in this window. Widen it, or run the collector again.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Developments", len(win))
        c2.metric("High relevance (4+)", int((win.rel >= 4).sum()))
        c3.metric("Firms active", win.company.nunique())
        c4.metric("Regions covered", f"{win.region.nunique()} / 5")
        top = win.sort_values(["priority", "published"], ascending=False)
        lead = top.iloc[0]
        by_reg = ", ".join(f"{k}: {v}" for k, v in win.region.value_counts().items())
        st.info(f"**Executive summary:** {len(win)} developments from {win.company.nunique()} competitors "
                f"({by_reg}). Top signal: **{lead.company}** ({lead.region}) - {lead.summary}")
        st.download_button("⬇️ Download digest (Markdown)", digest_md(win, days),
                           file_name=f"aranca_digest_{datetime.now():%Y%m%d}.md")
        st.subheader("Top developments")
        for _, r in top.head(8).iterrows():
            with st.container(border=True):
                a, b = st.columns([5, 1])
                a.markdown(f"**{CAT_ICON.get(r.category, '•')} {r.company}** · {r.region} · "
                           f"{r.category} · {r.published:%d %b %Y}")
                b.markdown(f"{URG_BADGE.get(r.urg, '')} · rel {r.rel}/5")
                st.markdown(f"**[{r.title}]({r.url})**" if str(r.url).startswith("http") else f"**{r.title}**")
                st.write(r.summary)
                if r.why:
                    st.markdown(f"**Why it matters:** {r.why}")
                st.caption(" | ".join(x for x in [
                    f"Service lines: {svc_label(r.svc_ids)}" if r.svc_ids else "",
                    f"Source: {r.source_type}" + (f" - {r.publisher}" if r.publisher else ""),
                    "" if r.analyzed else "not yet analysed"] if x))

# ------------------------- tab 2: explorer -------------------------
with tab2:
    st.header("Explorer")
    f1, f2, f3, f4 = st.columns(4)
    reg = f1.multiselect("Region", sorted(ins.region.dropna().unique()))
    comp_sel = f2.multiselect("Company", sorted(ins.company.dropna().unique()))
    cat = f3.multiselect("Category", sorted(ins.category.dropna().unique()))
    src = f4.multiselect("Source type", sorted(ins.source_type.dropna().unique()))
    q = st.text_input("Search text")
    d = ins.copy()
    if reg: d = d[d.region.isin(reg)]
    if comp_sel: d = d[d.company.isin(comp_sel)]
    if cat: d = d[d.category.isin(cat)]
    if src: d = d[d.source_type.isin(src)]
    if q:
        blob = d.company.fillna("") + " " + d.title.fillna("") + " " + d.summary.fillna("") + " " + d.why.fillna("")
        d = d[blob.str.contains(q, case=False, regex=False)]
    st.caption(f"{len(d)} results")
    show = d[["published", "company", "region", "category", "rel", "urg", "summary", "why",
              "svc_ids", "source_type", "url"]].copy()
    show["published"] = show["published"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("Source", display_text="open")})
    st.download_button("⬇️ Export CSV", show.to_csv(index=False), file_name="aranca_ci_export.csv")

# ------------------------- tab 3: profiles -------------------------
with tab3:
    st.header("Competitor Profiles")
    if ins.empty:
        st.info("No items under the current filters.")
    else:
        k1, k2 = st.columns(2)
        k1.plotly_chart(px.bar(ins.groupby(["region", "category"]).size().reset_index(name="n"),
                               x="region", y="n", color="category", title="Activity by region and category"),
                        use_container_width=True)
        t = ins.copy()
        t["week"] = t["published"].dt.to_period("W").dt.start_time
        k2.plotly_chart(px.bar(t.groupby(["week", "category"]).size().reset_index(name="n"),
                               x="week", y="n", color="category", title="Activity over time"),
                        use_container_width=True)
        st.markdown("---")
        firm = st.selectbox("Select a competitor", sorted(ins.company.dropna().unique()))
        fd = ins[ins.company == firm].sort_values("published", ascending=False)
        m1, m2, m3 = st.columns(3)
        m1.metric("Region", fd.region.iloc[0])
        m2.metric("Developments tracked", len(fd))
        m3.metric("Highest relevance", f"{fd.rel.max()}/5")
        st.plotly_chart(px.pie(fd, names="category", title=f"{firm}: activity mix", hole=0.4),
                        use_container_width=True)
        st.subheader("Timeline")
        for _, r in fd.iterrows():
            link = f" [source]({r.url})" if str(r.url).startswith("http") else ""
            st.markdown(f"**{r.published:%d %b %Y}** · {CAT_ICON.get(r.category, '•')} {r.category} · "
                        f"rel {r.rel}/5  \n{r.summary}{link}")

# ------------------------- tab 4: by practice leader -------------------------
with tab4:
    st.header("By service line / Practice Leader")
    rows = []
    for _, r in ins.iterrows():
        ids = [i.strip() for i in str(r.svc_ids).split(",") if i.strip()] or ["unmapped"]
        for i in ids:
            rows.append({**r.to_dict(), "svc_one": i})
    sv = pd.DataFrame(rows)
    if sv.empty:
        st.info("No items under the current filters.")
    else:
        counts = sv.groupby("svc_one").size().reset_index(name="n")
        counts["service line"] = counts["svc_one"].map(lambda i: SVC_NAME.get(i, i))
        st.plotly_chart(px.bar(counts, x="service line", y="n", title="Developments per Aranca service line"),
                        use_container_width=True)
        for sid in sorted(sv.svc_one.unique()):
            g = sv[sv.svc_one == sid].sort_values(["rel", "published"], ascending=False)
            pl = PL.get(sid, "")
            with st.expander(f"{SVC_NAME.get(sid, sid)}" + (f" - {pl}" if pl else "") + f"  ({len(g)})"):
                for _, r in g.head(10).iterrows():
                    st.markdown(f"**{r.company}** ({r.region}) · {r.category} · {r.published:%d %b} · rel {r.rel}/5  \n"
                                f"{r.summary} [source]({r.url})")

# ------------------------- tab 5: ask the agent -------------------------
with tab5:
    st.header("Ask the Agent")
    st.caption("Answers come only from the stored, sourced items. Retrieval is keyword-based "
               "(an embedding upgrade is on the roadmap).")
    question = st.text_input("Example: What have Japan competitors announced recently?")

    def retrieve(question, k=8):
        ql = question.lower()
        tokens = {w for w in ql.replace("?", " ").replace(",", " ").split() if len(w) > 2}
        for key, vs in SYN.items():
            if key in tokens or any(v in ql for v in vs):
                tokens |= {key, *vs}
        blob = (ins.company.fillna("") + " " + ins.region.fillna("") + " " + ins.category.fillna("") + " "
                + ins.title.fillna("") + " " + ins.summary.fillna("") + " " + ins.why.fillna("")).str.lower()
        score = blob.apply(lambda t: sum(1 for w in tokens if w in t))
        out = ins.assign(_s=score + ins.priority / 100)
        return out[score > 0].sort_values("_s", ascending=False).head(k)

    if st.button("Ask") and question:
        hits = retrieve(question)
        if hits.empty:
            st.warning("No matching items found.")
        else:
            ctx = "\n".join(f"[{i + 1}] {r.company} ({r.region}, {r.published:%Y-%m-%d}, {r.category}): "
                            f"{r.summary} | Why it matters: {r.why}" for i, (_, r) in enumerate(hits.iterrows()))
            try:
                from llm import ask
                out = ask("fast", f"""Answer the question using ONLY the numbered items below.
Cite sources as [n]. If the items don't answer it, say so.
Return JSON: {{"answer": "..."}}

QUESTION: {question}

ITEMS:
{ctx}""")
                st.markdown(out.get("answer", "No answer returned."))
            except BaseException as e:   # SystemExit if no provider, Exhausted if limits hit
                st.warning(f"LLM unavailable ({str(e)[:80]}). Showing the matching items instead.")
            with st.expander("Sources"):
                for i, (_, r) in enumerate(hits.iterrows()):
                    st.markdown(f"[{i + 1}] **{r.company}** · {r.published:%d %b %Y} · {r.summary} [link]({r.url})")