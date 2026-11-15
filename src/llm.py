import atexit, hashlib, json, os, re, time
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

# (base_url, key_env, fast_model, best_model). Override with GEMINI_EXTRACT / GEMINI_SCORE etc.
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-20b", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
               "gemini-2.5-flash", "gemini-2.5-flash"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "gpt-oss-120b", "gpt-oss-120b"),
}
CHAIN = []
for p in [x.strip() for x in os.getenv("LLM_PROVIDERS", "groq").split(",") if x.strip()]:
    if p in PROVIDERS and os.getenv(PROVIDERS[p][1]):
        base, key, m_fast, m_best = PROVIDERS[p]
        CHAIN.append({"name": p, "client": OpenAI(base_url=base, api_key=os.environ[key]),
                      "fast": os.getenv(f"{p.upper()}_EXTRACT", m_fast),
                      "best": os.getenv(f"{p.upper()}_SCORE", m_best), "dead": False})
if not CHAIN:
    raise SystemExit("No LLM provider configured. Set LLM_PROVIDERS and API keys in .env")
print("LLM chain:", " -> ".join(c["name"] for c in CHAIN))

CACHE_F = os.getenv("LLM_CACHE", "data/llm_cache2.json")
os.makedirs(os.path.dirname(CACHE_F) or ".", exist_ok=True)
try:
    cache = json.load(open(CACHE_F, encoding="utf-8"))
except Exception:
    cache = {}

USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

class Exhausted(Exception):
    pass

DAILY = ("per day", "tpd", "rpd", "daily", "exceeded your current")
LIMIT_RE = re.compile(r"rate.?limit|too many requests|quota|resource_exhausted|tokens per", re.I)
MAX_SLEEP = 90          # seconds: wait this long at most for a short rate limit
LONG_WAIT = 120         # seconds: a longer wait means a daily limit, so switch provider

def _save_cache():
    tmp = CACHE_F + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_F)             # atomic: Ctrl+C can no longer corrupt the cache

def _json(text):
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("no JSON")
    out = json.loads(text[a:b + 1])
    if not isinstance(out, dict):
        raise ValueError("JSON is not an object")
    return out

def _wait_seconds(msg):
    """Parse 'try again in 7m12.5s' / '2h3m' / '850ms' from a provider message. None if absent."""
    m = re.search(r"try again in\s+((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s)?(?:[\d.]+ms)?)", msg, re.I)
    if not m or not m.group(1):
        return None
    t, s = m.group(1), 0.0
    for val, unit in re.findall(r"([\d.]+)(ms|h|m|s)", t):
        v = float(val)
        s += v * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return s

def _models_hint(p):
    try:
        ids = [m.id for m in p["client"].models.list().data][:15]
        print(f"  [{p['name']}] available models: {ids}  (set {p['name'].upper()}_EXTRACT / _SCORE)")
    except Exception:
        pass

def ask(role, prompt):
    """role: 'fast' (cheap) or 'best' (higher quality). Cached on disk; failures are never cached."""
    k = hashlib.md5((role + prompt).encode()).hexdigest()
    if k in cache:
        return cache[k]
    for p in CHAIN:
        if p["dead"]:
            continue
        model = p[role]
        for a in range(3):
            try:
                kw = {"reasoning_effort": "low"} if p["name"] == "groq" and "gpt-oss" in model else {}
                r = p["client"].chat.completions.create(
                    model=model, temperature=0, max_completion_tokens=6000, **kw,
                    messages=[{"role": "system", "content": "Reply with a single valid JSON object only. No markdown."},
                              {"role": "user", "content": prompt}])
                out = _json(r.choices[0].message.content)
                USAGE["calls"] += 1
                u = getattr(r, "usage", None)
                if u:
                    USAGE["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                    USAGE["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
                if out:                                   # never cache an empty answer
                    cache[k] = out
                    _save_cache()
                return out
            except Exception as e:
                m = str(e)
                low = m.lower()
                if "429" in m or LIMIT_RE.search(low):
                    wait = _wait_seconds(m)
                    if any(x in low for x in DAILY) or (wait is not None and wait > LONG_WAIT):
                        print(f"  [{p['name']}] daily limit (resets in ~{int(wait // 60) if wait else '?'} min) -> next provider")
                        p["dead"] = True
                        break
                    w = min((wait or 20) + 1, MAX_SLEEP)
                    print(f"  [{p['name']}] rate limited, waiting {w:.0f}s")
                    time.sleep(w)
                elif "404" in m or "not_found" in low or "does not exist" in low:
                    print(f"  [{p['name']}] model '{model}' unavailable -> next provider")
                    _models_hint(p)
                    p["dead"] = True
                    break
                elif "401" in m or "invalid api key" in low or "permission" in low:
                    print(f"  [{p['name']}] authentication problem -> next provider: {m[:100]}")
                    p["dead"] = True
                    break
                else:                                     # bad JSON, 5xx, network: short retry
                    print(f"  [{p['name']}] retry {a + 1}/3:", m[:120])
                    time.sleep(3 * (a + 1))
    if all(p["dead"] for p in CHAIN):
        raise Exhausted("All providers exhausted. Progress is cached; rerun later.")
    return {}

def usage():
    return dict(USAGE)

@atexit.register
def _summary():
    if USAGE["calls"]:
        print(f"[llm] {USAGE['calls']} calls, {USAGE['prompt_tokens']} prompt + "
              f"{USAGE['completion_tokens']} completion tokens")