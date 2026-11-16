"""Evaluate the CI agent against labels.

  python src/eval.py make              # blind labeling sheet (~45 items, NO agent output shown) + eval/RUBRIC.md
  python src/eval.py extend [N]        # add N fresh unlabeled items as the held-out TEST split (default 40)
  python src/eval.py score [tag]       # metrics with 95% CIs, baselines, threshold sweep, confusion matrix, dev vs test
  python src/eval.py reset yes         # delete agent output for labeled items so report.py re-analyses them

Workflow: make -> label by hand -> score baseline_v1 -> change prompt -> reset yes -> report.py -> score v2
          -> extend -> label the new rows -> score (the 'test' split is the number to quote).
Agent output is read live from the database, never stored in the label sheet (so labeling stays blind).
"""
import os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from eval_rules import CATS, RUBRIC, kappa, pct_ci, rule_judge, wilson

DB = os.getenv("CI_DB", "data/ci.db")
F, HIST, RES, JUDGE_F = "eval/labeled_sample.csv", "eval/history.csv", "eval/results.md", "eval/judge_labels.csv"
N = int(os.getenv("N", "45"))
MIN_REL = int(os.getenv("MIN_REL", "1"))      # must match what the dashboard shows by default (report.py MIN_REL)
COLS = ["id", "split", "company", "region", "date", "title", "snippet", "url",
        "true_relevant", "true_cat", "label_source", "notes"]

# ------------------------------------------------------------------ data access
def load_agent():
    con = sqlite3.connect(DB)
    df = pd.read_sql("""SELECT i.id, i.company, i.region, i.title, i.snippet, i.url, i.published,
              i.category_hint, n.cat AS agent_cat, n.rel AS agent_rel, n.summary
        FROM items i JOIN ci_insights n ON n.id = i.id WHERE n.state = 'done'""", con)
    con.close()
    return df

def read_labels():
    if not os.path.exists(F):
        sys.exit("No label sheet yet. Run: python src/eval.py make")
    d = pd.read_csv(F, dtype=str).fillna("")
    for c in COLS:
        if c not in d.columns:
            d[c] = ""
    d["split"] = d["split"].replace("", "dev")
    return d

def stratified(df, n, seed):
    per = max(1, n // max(1, df["region"].nunique()))
    s = pd.concat([g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby("region")])
    if len(s) < n:
        rest = df[~df["id"].isin(s["id"])]
        s = pd.concat([s, rest.sample(min(len(rest), n - len(s)), random_state=seed)])
    return s.sample(frac=1, random_state=seed + 1)        # shuffled: do not label region by region

def sheet(s, split):
    return pd.DataFrame({"id": s["id"], "split": split, "company": s["company"], "region": s["region"],
                         "date": s["published"].str[:10], "title": s["title"],
                         "snippet": s["snippet"].fillna("").str[:300], "url": s["url"],
                         "true_relevant": "", "true_cat": "", "label_source": "", "notes": ""})[COLS]

# ------------------------------------------------------------------ make / extend / reset
def make():
    if os.path.exists(F):
        sys.exit(f"{F} already exists (re-running would erase your labels). Use 'extend' to add items, or delete the file to restart.")
    df = load_agent()
    if df.empty:
        sys.exit("No analysed items yet. Run report.py first.")
    os.makedirs("eval", exist_ok=True)
    out = sheet(stratified(df, N, 7), "dev")
    out.to_csv(F, index=False)
    open("eval/RUBRIC.md", "w", encoding="utf-8").write("# Labeling rubric (edit, then label consistently)\n\n" + RUBRIC + "\n")
    print(f"Wrote {len(out)} rows -> {F}   (agent output is hidden on purpose)\nRubric -> eval/RUBRIC.md - read it and edit it first.")
    print("Fill true_relevant (y/n) and true_cat for each row from title/snippet/url. Put 'hand' in label_source.")

def extend():
    d = read_labels()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    pool = load_agent()
    pool = pool[~pool["id"].isin(d["id"])]
    if pool.empty:
        sys.exit("No unlabeled analysed items left. Run report.py on newer data first.")
    new = sheet(stratified(pool, min(n, len(pool)), 11), "test")
    pd.concat([d[COLS], new]).to_csv(F, index=False)
    print(f"Added {len(new)} TEST rows ({len(pool) - len(new)} analysed items still unused). Label them, then run score.")
    print("Do NOT tune the prompt on the test split - it is your held-out check.")

def reset():
    if len(sys.argv) < 3 or sys.argv[2] != "yes":
        sys.exit("This deletes agent output (ci_insights) for the labeled items so report.py re-analyses them with the current prompt.\nRe-run as: python src/eval.py reset yes")
    ids = read_labels()["id"].tolist()
    con = sqlite3.connect(DB)
    con.executemany("DELETE FROM ci_insights WHERE id = ?", [(i,) for i in ids])
    con.commit(); con.close()
    print(f"Cleared agent output for {len(ids)} items. Now run:  REPORT_DAYS=90 LLM_BUDGET=20 python src/report.py\n"
          "(REPORT_DAYS must cover the oldest labeled item; the LLM cache is keyed on the prompt, so a changed prompt re-queries.)")

# ------------------------------------------------------------------ scoring
def confusion(shown, truth):
    tp, fp = int((shown & truth).sum()), int((shown & ~truth).sum())
    fn, tn = int((~shown & truth).sum()), int((~shown & ~truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "p": p, "r": r, "spec": tn / (tn + fp) if tn + fp else 0.0,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}

def row(name, c):
    return [name, str(c["tp"] + c["fp"]), pct_ci(c["tp"], c["tp"] + c["fp"]), pct_ci(c["tp"], c["tp"] + c["fn"]),
            pct_ci(c["tn"], c["tn"] + c["fp"]), f"{c['f1']:.2f}"]

def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return lines

def score():
    tag = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    lab, ag = read_labels(), load_agent()[["id", "agent_cat", "agent_rel", "category_hint"]]
    lab["true_relevant"] = lab["true_relevant"].str.lower().str.strip().str[:1]
    n_lab = lab["true_relevant"].isin(["y", "n"]).sum()
    d = lab[lab["true_relevant"].isin(["y", "n"])].merge(ag, on="id", how="left")
    missing = int(d["agent_cat"].isna().sum())
    d = d.dropna(subset=["agent_cat"]).copy()
    if len(d) < 10:
        sys.exit(f"Only {len(d)} labeled + analysed rows ({n_lab} labeled). Label at least 30 rows in {F}.")
    d["agent_rel"] = pd.to_numeric(d["agent_rel"], errors="coerce").fillna(0)
    d["truth"] = d["true_relevant"] == "y"
    d["rule_base"] = [rule_judge(r.company, r.title, r.snippet, r.url)[0] == "y" for r in d.itertuples()]
    shown = lambda t: (d["agent_cat"] != "Noise") & (d["agent_rel"] >= t)
    L = ["# Evaluation results", "",
         f"Labeled sample: **{len(d)} items** (dev {int((d.split != 'test').sum())}, test {int((d.split == 'test').sum())}). "
         f"Dashboard threshold: relevance >= {MIN_REL}. Intervals are 95% Wilson CIs.", ""]
    src = d["label_source"].replace("", "unspecified").value_counts().to_dict()
    L += [f"Label sources: {src}", ""]
    if any(k.startswith("auto") for k in src):
        L += ["> WARNING: some labels were produced by an automatic judge, not by hand. Say so when quoting these numbers.", ""]
    if missing:
        L += [f"> {missing} labeled items have no agent output (cleared or not analysed yet) and are excluded.", ""]

    # 1) agent vs baselines
    a = confusion(shown(MIN_REL), d["truth"])
    systems = [("**Agent (LLM triage)**", a), ("Rule judge (no LLM)", confusion(d["rule_base"], d["truth"])),
               ("Show every keyword hint", confusion(d["category_hint"].isin(CATS), d["truth"]))]
    L += ["## Agent vs baselines", ""] + table(["System", "Shown", "Precision", "Recall", "Noise rejected", "F1"],
                                               [row(n, c) for n, c in systems])
    L += ["", f"Agent confusion: TP={a['tp']} FP={a['fp']} FN={a['fn']} TN={a['tn']}", ""]

    # 2) threshold sweep
    L += ["## Relevance threshold sweep (what the dashboard filter does)", ""]
    L += table(["Show rel >=", "Shown", "Precision", "Recall", "Noise rejected", "F1"],
               [row(str(t), confusion(shown(t), d["truth"])) for t in (1, 2, 3, 4)]) + [""]

    # 3) dev vs test
    L += ["## Dev vs held-out test", ""]
    rows = []
    for nm, sub in (("dev (tuned on)", d[d.split != "test"]), ("test (held out)", d[d.split == "test"]), ("all", d)):
        if len(sub) < 10:
            rows.append([nm, f"n={len(sub)} (<10, not scored)", "", "", "", ""])
            continue
        s = (sub["agent_cat"] != "Noise") & (sub["agent_rel"] >= MIN_REL)
        rows.append([f"{nm} n={len(sub)}"] + row("", confusion(s, sub["truth"]))[1:])
    L += table(["Split", "Shown", "Precision", "Recall", "Noise rejected", "F1"], rows) + [""]

    # 4) categories
    c = d[d.truth & d.true_cat.isin(CATS)].copy()
    if len(c):
        c["agent_cat2"] = c["agent_cat"].where(c["agent_cat"].isin(CATS + ["Noise"]), "Noise")
        L += ["## Category accuracy (true positives only)", "",
              f"Agent: {pct_ci(int((c.agent_cat2 == c.true_cat).sum()), len(c))}   |   keyword hint: "
              f"{pct_ci(int((c.category_hint == c.true_cat).sum()), len(c))}   (n={len(c)})", ""]
        per = []
        for k, g in c.groupby("true_cat"):
            per.append([k, len(g), pct_ci(int((g.agent_cat2 == k).sum()), len(g)), "indicative only (n<5)" if len(g) < 5 else ""])
        L += table(["True category", "n", "Agent accuracy", "Note"], per) + ["", "Confusion matrix (rows = your label, columns = agent):", "", "```",
              pd.crosstab(c["true_cat"], c["agent_cat2"], margins=True).to_string(), "```", ""]

    # 5) agreement of automatic judges with hand labels
    if os.path.exists(JUDGE_F):
        j = pd.read_csv(JUDGE_F, dtype=str).fillna("")
        h = d[d["label_source"].isin(["hand", "unspecified", ""])].merge(j, on="id", how="inner")
        if len(h) >= 10:
            L += ["## Do the automatic judges agree with your hand labels?", ""]
            for nm, col in (("Rule judge", "rule_rel"), ("LLM judge", "llm_rel")):
                m = h[h[col].isin(["y", "n"])]
                if len(m) >= 10:
                    L.append(f"- {nm}: {(m[col].eq('y') == m.truth).mean():.0%} raw agreement, Cohen's kappa "
                             f"{kappa(m[col].eq('y'), m.truth):.2f} (n={len(m)})")
            L += ["", "Kappa above ~0.6 means the judge can reasonably extend your labels; below that, keep labeling by hand.", ""]

    # 6) other checks
    has_src = ((d["url"].str.startswith("http")) & (d["date"].str.len() == 10)).mean()
    L += [f"Items with source link and publication date: {has_src:.0%}", ""]

    # 7) error analysis
    shown_now = shown(MIN_REL)
    fp, fn = d[shown_now & ~d.truth], d[~shown_now & d.truth]
    wrong = c[c.agent_cat2 != c.true_cat] if len(c) else c
    L += ["## Error analysis", "", "False positives (shown but irrelevant):", ""]
    L += [f"- {r.company}: {r.title[:90]} (agent: {r.agent_cat}, rel {int(r.agent_rel)})" + (f" - note: {r.notes}" if r.notes else "")
          for r in fp.head(10).itertuples()] or ["- none"]
    L += ["", "False negatives (relevant but hidden):", ""]
    L += [f"- {r.company}: {r.title[:90]} (agent: {r.agent_cat}, rel {int(r.agent_rel)})" for r in fn.head(8).itertuples()] or ["- none"]
    L += ["", "Wrong category on true positives:", ""]
    L += [f"- {r.company}: {r.title[:80]} (you: {r.true_cat} | agent: {r.agent_cat2})" for r in wrong.head(8).itertuples()] or ["- none"]

    # 8) history (before/after)
    cat_acc = (c.agent_cat2 == c.true_cat).mean() if len(c) else float("nan")
    h_row = {"tag": tag, "n": len(d), "min_rel": MIN_REL, "precision": round(a["p"], 3), "recall": round(a["r"], 3),
             "noise_rejected": round(a["spec"], 3), "f1": round(a["f1"], 3), "cat_acc": round(cat_acc, 3)}
    hist = pd.read_csv(HIST) if os.path.exists(HIST) else pd.DataFrame()
    hist = pd.concat([hist[hist.get("tag", pd.Series(dtype=str)) != tag] if len(hist) else hist, pd.DataFrame([h_row])])
    os.makedirs("eval", exist_ok=True)
    hist.to_csv(HIST, index=False)
    L += ["", "## History (compare versions)", ""] + table(list(hist.columns), hist.tail(8).values.tolist())
    open(RES, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\n-> {RES}   (history -> {HIST})")

if __name__ == "__main__":
    {"make": make, "extend": extend, "score": score, "reset": reset}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()