"""Second-opinion judges for the label sheet: a rule judge and an LLM judge. They do NOT overwrite your hand labels.

  python src/eval_auto.py                    # rule judge + LLM judge -> eval/judge_labels.csv, prints agreement
  JUDGE=rules python src/eval_auto.py        # rules only, 0 tokens
  JUDGE_ROLE=best python src/eval_auto.py    # which llm.py role judges (default best = a different model than the agent's 'fast')
  FILL=1 python src/eval_auto.py             # ALSO fill rows whose true_relevant is blank (tagged label_source=auto_*)

Use it to (1) check how well a judge agrees with your hand labels (Cohen's kappa) and (2) only then, if kappa is high,
extend labels with FILL=1. Auto-filled labels are flagged in the results so they are never passed off as hand labels.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from eval_rules import CATS, RUBRIC, kappa, rule_judge

F, OUT = "eval/labeled_sample.csv", "eval/judge_labels.csv"
MODE = os.getenv("JUDGE", "both")
ROLE = os.getenv("JUDGE_ROLE", "best")
FILL = os.getenv("FILL") == "1"
BATCH = 9

if not os.path.exists(F):
    sys.exit("Run: python src/eval.py make")
d = pd.read_csv(F, dtype=str).fillna("")
for c in ("true_relevant", "true_cat", "label_source"):
    if c not in d.columns:
        d[c] = ""

j = pd.DataFrame({"id": d["id"], "rule_rel": "", "rule_cat": "", "llm_rel": "", "llm_cat": ""})
for k, r in enumerate(d.itertuples()):
    j.loc[k, ["rule_rel", "rule_cat"]] = rule_judge(r.company, r.title, r.snippet, r.url)

if MODE != "rules":
    try:
        import llm
        for s in range(0, len(d), BATCH):
            ch = d.iloc[s:s + BATCH]
            items = "\n".join(f'[{i}] firm="{r.company}" title="{r.title[:160]}" text="{r.snippet[:260]}" url={r.url}'
                              for i, r in enumerate(ch.itertuples()))
            out = llm.ask(ROLE, f"""You are an independent evaluator of competitive-intelligence items. Judge each item ABOUT the named firm using this rubric.

{RUBRIC}

Return JSON {{"r":[{{"i":0,"rel":"y","cat":"Digital Presence"}}]}} with cat only when rel is y.
{items}""")
            for x in out.get("r", []):
                try:
                    k = d.index.get_loc(ch.index[int(x["i"])])
                    j.loc[k, "llm_rel"] = str(x.get("rel", "")).lower()[:1]
                    j.loc[k, "llm_cat"] = x.get("cat", "") if x.get("cat") in CATS else ""
                except Exception:
                    pass
    except (Exception, SystemExit) as e:          # not BaseException: Ctrl+C must still stop the script
        print(f"LLM judge unavailable or stopped early ({str(e)[:90]}). Using what was judged so far + the rule judge.")

j["disagree"] = ((j.rule_rel != j.llm_rel) & j.llm_rel.isin(["y", "n"])).map({True: "yes", False: ""})
os.makedirs("eval", exist_ok=True)
j.to_csv(OUT, index=False)

# ---------------- agreement reports
has_llm = j.llm_rel.isin(["y", "n"])
if has_llm.sum() >= 10:
    a, b = j.loc[has_llm, "rule_rel"] == "y", j.loc[has_llm, "llm_rel"] == "y"
    print(f"Rule judge vs LLM judge: {(a == b).mean():.0%} raw agreement, kappa {kappa(a, b):.2f} (n={int(has_llm.sum())})")
hand = d["true_relevant"].str.lower().str[:1].isin(["y", "n"]) & ~d["label_source"].str.startswith("auto")
if hand.sum() >= 10:
    t = d.loc[hand, "true_relevant"].str.lower().str[:1] == "y"
    for nm, col in (("Rule judge", "rule_rel"), ("LLM judge", "llm_rel")):
        m = j.loc[hand, col].isin(["y", "n"])
        if m.sum() >= 10:
            p = j.loc[hand, col][m] == "y"
            print(f"{nm} vs YOUR hand labels: {(p == t[m]).mean():.0%} raw agreement, kappa {kappa(p, t[m]):.2f} (n={int(m.sum())})")
else:
    print(f"Only {int(hand.sum())} hand-labeled rows - label at least 30 by hand to measure judge agreement.")

# ---------------- optional fill of blank rows only
if FILL:
    blank = d["true_relevant"].str.strip() == ""
    n = 0
    for k in d.index[blank]:
        pos = d.index.get_loc(k)
        use_llm = j.loc[pos, "llm_rel"] in ("y", "n")
        rel = j.loc[pos, "llm_rel" if use_llm else "rule_rel"]
        cat = j.loc[pos, "llm_cat" if use_llm else "rule_cat"] if rel == "y" else ""
        d.loc[k, ["true_relevant", "true_cat", "label_source"]] = [rel, cat, "auto_llm" if use_llm else "auto_rules"]
        n += 1
    d.to_csv(F, index=False)
    print(f"Filled {n} blank rows (hand labels untouched). They are tagged label_source=auto_* and flagged in eval.py score.")
print(f"Wrote {OUT}. Next: python src/eval.py score")