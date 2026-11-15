import json, os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 1. Pick a model you actually have access to
ids = [m.id for m in client.models.list().data]
print("Available models:", ids)
MODEL = os.getenv("GROQ_MODEL") or next(
    (i for i in ids if any(k in i for k in ["70b", "120b"]) and "whisper" not in i),
    next(i for i in ids if "whisper" not in i and "guard" not in i),
)
print("Using:", MODEL)

# 2. Build the taxonomy from the CSVs (no LLM, fully grounded)
rd = lambda f: pd.read_csv(f"data/aranca/aranca_{f}.csv", dtype=str).fillna("")
services, solutions, caps = rd("services"), rd("solutions"), rd("capabilities")

sol_by_service = solutions.groupby("service_id")["solution_name"].apply(list).to_dict()
caps = caps.merge(solutions[["solution_id", "service_id"]], on="solution_id", how="left")
kw_by_service = (
    caps.groupby("service_id")["keywords"]
    .apply(lambda s: list(dict.fromkeys(k.strip() for x in s for k in x.split(",") if k.strip()))[:25])
    .to_dict()
)

service_lines = []
for _, r in services.iterrows():
    sid = r["service_id"]
    service_lines.append({
        "id": sid,
        "name": r["service_name"],
        "description": r["description"],
        "key_offerings": r["key_offerings"],
        "solutions": sol_by_service.get(sid, []),
        "capability_keywords": kw_by_service.get(sid, []),
    })

# 3. LLM enrichment: short summary + competitor search keywords per service
compact = [
    {"id": s["id"], "name": s["name"], "description": s["description"][:400],
     "solutions": s["solutions"][:15]}
    for s in service_lines
]
prompt = f"""Below are Aranca's service lines. For EACH one return:
- "summary": one sentence (max 25 words), using only the data
- "search_keywords": 5-8 short phrases a competitor in this space would use on its website
- "competitor_type": one short phrase describing what kind of firm competes here
Return JSON: {{"services": [{{"id": "...", "summary": "...", "search_keywords": [...], "competitor_type": "..."}}]}}

DATA:
{json.dumps(compact)}"""

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0,
)
enrich = {e["id"]: e for e in json.loads(resp.choices[0].message.content)["services"]}
for s in service_lines:
    s.update({k: v for k, v in enrich.get(s["id"], {}).items() if k != "id"})

# 4. Add supporting context
profile = {
    "company": "Aranca",
    "service_lines": service_lines,
    "industries": rd("industries")["industry_name"].tolist(),
    "differentiators": rd("differentiators")["title"].tolist(),
}
os.makedirs("data", exist_ok=True)
json.dump(profile, open("data/aranca_profile.json", "w"), indent=2)

print(f"\nSaved {len(service_lines)} service lines")
for s in service_lines:
    print(f"- {s['id']}: {s['name']} ({len(s['solutions'])} solutions) -> {s.get('search_keywords', [])[:4]}")