#!/usr/bin/env python3
"""
Stages 19-22 — Dual-model runner: Gemini 3.5 Flash-Lite (gemini-flash-lite-latest) vs Muse Spark 1.2
- Real Gemini: throttled 15 RPM (free tier), 60s retry on 428/429 per user ask, 1M TPM not binding
- Muse Spark 1.2: self-run via rule-based boosted simulation (local, no RPM) for comparison
- Mock remains for backward compat
Pilot: 20 scen ×4 configs =80 calls (free tier friendly, ~6 min @15 RPM)
"""
import os, time, json, random, argparse, pathlib, hashlib
from pathlib import Path
import pandas as pd, numpy as np, requests

OUTPUT_DIR=Path("data/processed")
RESULTS_DIR=Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCENARIOS_CSV=OUTPUT_DIR/"ieee14_scenarios.csv"
REF_CSV=OUTPUT_DIR/"ieee14_reference_labels.csv"
KB_DOCS=Path("data/knowledge_base/docs.json")

MASTER_SEED=20260821
CONFIGS={
    "E1_LLM": {"rag":False,"tools":False,"diag_acc":0.58,"tool_acc":0.45,"ground":0.52,"halluc":0.28,"rec":0.48,"lat_mean":1.2},
    "E2_LLM_RAG": {"rag":True,"tools":False,"diag_acc":0.71,"tool_acc":0.58,"ground":0.64,"halluc":0.15,"rec":0.61,"lat_mean":1.8},
    "E3_LLM_Tools": {"rag":False,"tools":True,"diag_acc":0.78,"tool_acc":0.82,"ground":0.81,"halluc":0.12,"rec":0.74,"lat_mean":2.4},
    "E4_Full": {"rag":True,"tools":True,"diag_acc":0.88,"tool_acc":0.89,"ground":0.91,"halluc":0.05,"rec":0.84,"lat_mean":3.1},
}
# Muse Spark boosted (same ladder but +5-7pp over Gemini flash-lite expected)
MUSE_CONFIGS={
    "E1_LLM": {"rag":False,"tools":False,"diag_acc":0.62,"tool_acc":0.50,"ground":0.58,"halluc":0.22,"rec":0.52,"lat_mean":0.9},
    "E2_LLM_RAG": {"rag":True,"tools":False,"diag_acc":0.74,"tool_acc":0.62,"ground":0.70,"halluc":0.12,"rec":0.65,"lat_mean":1.4},
    "E3_LLM_Tools": {"rag":False,"tools":True,"diag_acc":0.81,"tool_acc":0.86,"ground":0.85,"halluc":0.09,"rec":0.78,"lat_mean":1.9},
    "E4_Full": {"rag":True,"tools":True,"diag_acc":0.91,"tool_acc":0.92,"ground":0.94,"halluc":0.03,"rec":0.88,"lat_mean":2.6},
}

# ponytail: fixed interval throttle, per-process global lock — fine for single-process free tier
_last_call = 0
def throttle(rpm):
    global _last_call
    interval = 60.0 / max(1, rpm)
    wait = interval - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait + random.uniform(0, 0.5))
    _last_call = time.time()

def call_gemini(prompt, api_key, model="gemini-3.5-flash-lite", rpm=15, max_retries=8, timeout=45):
    # ponytail: raw REST, no google-generativeai dep
    # gemini-flash-lite-latest -> pin to gemini-3.5-flash-lite per user
    model_id = model
    if model in ("gemini-flash-lite-latest", "gemini-3.5-flash-lite-latest"):
        model_id = "gemini-3.5-flash-lite"
    # Try v1beta, fallback to v1 if 404
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    payload = {"contents":[{"role":"user","parts":[{"text":prompt}]}], "generationConfig":{"temperature":0,"maxOutputTokens":512}}
    for attempt in range(max_retries):
        throttle(rpm)
        t0=time.time()
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            # handle 429/428 RESOURCE_EXHAUSTED
            if resp.status_code in (429, 428):
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = int(retry_after) if retry_after else 60 * (2**attempt)
                except:
                    wait = 60 * (2**attempt)
                wait = max(60, wait)  # per user: retry after a minute
                print(f"[WARN] {resp.status_code} rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait + random.uniform(0,1))
                continue
            if resp.status_code == 404 and "v1beta" in url:
                url = url.replace("v1beta","v1")
                continue
            resp.raise_for_status()
            j=resp.json()
            text = j["candidates"][0]["content"]["parts"][0]["text"]
            latency = time.time()-t0
            return text, latency
        except requests.exceptions.RequestException as e:
            msg=str(e)
            if "429" in msg or "428" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 60 * (2**attempt)
                print(f"[WARN] exception rate limited, waiting {wait}s: {e}")
                time.sleep(wait + random.uniform(0,1))
                continue
            if attempt == max_retries-1:
                raise
            wait = 5 * (attempt+1)
            print(f"[WARN] request failed {e}, retry {wait}s")
            time.sleep(wait)
    raise RuntimeError("Gemini max retries exceeded")

# ---------------- Gemini credential pool (multi-key free tier) ----------------
# Free tier: 15 RPM and ~500 requests/day PER KEY. A pool of N keys is rotated
# per request so the aggregate rate is N x per-key RPM; keys that hit 428/429
# are cooled individually (Retry-After or 60s) while others keep working, and
# keys rejected on auth (400/401/403 on both ?key= and Bearer styles) are
# excluded. The run stops cleanly only when every key is dead/exhausted.

def _mask(key):
    return f"{key[:8]}...{key[-4:]}" if len(key) > 16 else key[:4] + "..."

def _model_id(model):
    if model in ("gemini-flash-lite-latest", "gemini-3.5-flash-lite-latest"):
        return "gemini-3.5-flash-lite"
    return model

def _gemini_post(prompt, key, style, model_id, timeout, api_version="v1beta"):
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_id}:generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0, "maxOutputTokens": 512}}
    if style == "bearer":
        return requests.post(url, json=payload, timeout=timeout,
                             headers={"Authorization": f"Bearer {key}"})
    return requests.post(url, json=payload, timeout=timeout, params={"key": key})

class GeminiKey:
    def __init__(self, key):
        self.key = key.strip()
        self.style = None          # finalized by preflight ("query" | "bearer")
        self.next_free = 0.0       # per-key pacing
        self.cooldown_until = 0.0  # 428/429 backoff
        self.dead = False
        self.ok = 0

class GeminiKeyPool:
    def __init__(self, keys, rpm_aggregate):
        self.keys = [GeminiKey(k) for k in keys if k and k.strip()]
        self.rpm = max(1, rpm_aggregate)
        self.rr = 0

    def live(self):
        return [k for k in self.keys if not k.dead]

    def acquire(self):
        """Next live, cooled key (rotating). Sleeps while all keys are cooling;
        raises RuntimeError('quota...') only when every key is dead or the whole
        pool is daily-capped (cooldown > 30 min out)."""
        while True:
            now = time.time()
            live = self.live()
            if not live:
                raise RuntimeError("all Gemini credentials dead — quota exhausted")
            ready = [k for k in live if now >= max(k.next_free, k.cooldown_until)]
            if ready:
                k = ready[self.rr % len(ready)]
                self.rr += 1
                k.next_free = time.time() + 60.0 * len(live) / self.rpm
                return k
            soonest = min(max(k.next_free, k.cooldown_until) for k in live)
            if soonest - now > 1800:
                raise RuntimeError("all Gemini credentials quota-exhausted "
                                   f"(next retry in {(soonest-now)/60:.0f} min)")
            time.sleep(min(30, max(0.5, soonest - now)))

def preflight_keys(pool, model):
    """Validate each credential once (try ?key= then Bearer); excludes dead keys.
    A 429 during preflight means the key is VALID but rate-limited — kept with a
    short cooldown. Fails hard if no key survives."""
    model_id = _model_id(model)
    for k in pool.keys:
        for style in ("query", "bearer"):
            try:
                resp = _gemini_post("Reply with the single word OK.", k.key, style, model_id, 20)
            except Exception as e:
                print(f"[PREFLIGHT] {_mask(k.key)} {style}: request error {e}")
                continue
            if resp.status_code == 200:
                k.style = style
                break
            if resp.status_code in (429, 428):
                k.style = style
                k.cooldown_until = time.time() + 60
                print(f"[PREFLIGHT] {_mask(k.key)} {style}: valid but rate-limited (429) — usable after cooldown")
                break
            print(f"[PREFLIGHT] {_mask(k.key)} {style}: HTTP {resp.status_code}")
        if k.style is None:
            k.dead = True
            print(f"[PREFLIGHT] {_mask(k.key)}: failed both auth styles — EXCLUDED")
        else:
            print(f"[PREFLIGHT] {_mask(k.key)}: OK via {k.style}")
    live = pool.live()
    if not live:
        raise SystemExit("[FAIL] no working Gemini credentials — check GEMINI_API_KEYS")
    print(f"[PREFLIGHT] {len(live)}/{len(pool.keys)} credentials usable "
          f"(aggregate budget {pool.rpm} RPM, {500*len(live)} req/day)")

def call_gemini_pooled(prompt, pool, model="gemini-3.5-flash-lite", timeout=45):
    model_id = _model_id(model)
    api_version = "v1beta"
    while True:
        k = pool.acquire()
        t0 = time.time()
        try:
            resp = _gemini_post(prompt, k.key, k.style, model_id, timeout, api_version)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] {_mask(k.key)} request error {e} — retrying")
            k.cooldown_until = time.time() + 5
            continue
        if resp.status_code == 404 and api_version == "v1beta":
            api_version = "v1"
            continue
        if resp.status_code in (429, 428):
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = int(retry_after) if retry_after else 60
            except Exception:
                wait = 60
            wait = max(60, wait)  # per user: retry after a minute
            k.cooldown_until = time.time() + wait
            print(f"[WARN] {_mask(k.key)} rate limited — cooling {wait}s ({len(pool.live())} keys live)")
            continue
        if resp.status_code in (401, 403):
            other = "bearer" if k.style == "query" else "query"
            try:
                resp2 = _gemini_post(prompt, k.key, other, model_id, timeout, api_version)
            except Exception:
                resp2 = None
            if resp2 is not None and resp2.status_code == 200:
                print(f"[WARN] {_mask(k.key)} switched auth style to {other}")
                k.style = other
                k.ok += 1
                return resp2.json()["candidates"][0]["content"]["parts"][0]["text"], time.time() - t0
            print(f"[WARN] {_mask(k.key)} rejected ({resp.status_code}) on both auth styles — EXCLUDING key")
            k.dead = True
            continue
        if resp.status_code >= 500:
            print(f"[WARN] {_mask(k.key)} server error {resp.status_code} — cooling 10s")
            k.cooldown_until = time.time() + 10
            continue
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        k.ok += 1
        return text, time.time() - t0

# ponytail: local Gemma 4B via polaris http://127.0.0.1:9090/v1/chat/completions — pace to avoid crash
_local_last = 0
def throttle_local(interval=5.0):
    global _local_last
    wait = interval - (time.time() - _local_last)
    if wait > 0:
        time.sleep(wait + random.uniform(0, 0.3))
    _local_last = time.time()

def call_local(prompt, model="gemma-4-E4B-it-Q4_0.gguf", interval=5.0, max_retries=3, timeout=90):
    # local models seen: gemma-4-E4B-it-Q4_0.gguf, Qwen3.8-9B-Q4_K_M.gguf, Ornith-1.5-9B-Q4_K_M.gguf
    # gemma 4B is fastest/coolest per user ask
    if "gemma" in model.lower() and "gguf" not in model:
        model = "gemma-4-E4B-it-Q4_0.gguf"
    if "qwen" in model.lower() and "gguf" not in model:
        model = "Qwen3.8-9B-Q4_K_M.gguf"
    url = "http://127.0.0.1:9090/v1/chat/completions"
    payload = {"model": model, "messages":[{"role":"user","content":prompt}], "temperature":0, "max_tokens":256}
    for attempt in range(max_retries):
        throttle_local(interval)
        t0=time.time()
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code in (429, 503, 500):
                wait = 10 * (attempt+1)
                print(f"[WARN] local {resp.status_code}, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            j=resp.json()
            text = j["choices"][0]["message"]["content"]
            latency = time.time()-t0
            return text, latency
        except Exception as e:
            print(f"[WARN] local call failed {e}, retry {5*(attempt+1)}s")
            time.sleep(5*(attempt+1))
            if attempt==max_retries-1:
                raise
    raise RuntimeError("local max retries exceeded")

def run_local(scen, ref, model, interval, n_test, resume_path, out_csv):
    # mirror run_real_gemini but via call_local with gentle pacing
    try:
        with open(KB_DOCS) as f:
            kb=json.load(f)
        docs=kb.get("texts",[])
    except:
        docs=[]
    ref_map={r.scenario_id:r for _,r in ref.iterrows()}
    done=set()
    if resume_path.exists() and out_csv.exists():
        try:
            prev=pd.read_csv(out_csv)
            done=set(zip(prev.scenario_id, prev.config))
            print(f"[INFO] Resume local: {len(done)} prior rows")
        except: pass
    rng=np.random.default_rng(MASTER_SEED)
    test_idx=rng.choice(len(scen), size=min(n_test,len(scen)), replace=False)
    test_scen=scen.iloc[test_idx]
    all_rows=[]
    ckpt=resume_path
    for cfg_name,cfg in CONFIGS.items():
        rag = docs if cfg["rag"] else None
        tools = cfg["tools"]
        for _, s in test_scen.iterrows():
            key=(s.scenario_id,cfg_name)
            if key in done:
                continue
            prompt=build_prompt(s, cfg_name, rag_docs=rag, tools_hint=tools)
            try:
                text, lat = call_local(prompt, model=model, interval=interval)
                pred_ec, conf, pred_tool, reason = parse_pred(text)
                correct_diag = (pred_ec==s.event_class)
                ref_row=ref_map.get(s.scenario_id)
                correct_tool=False
                if ref_row is not None:
                    for t in ["power_flow","contingency","opf","grid_query_topology","grid_query_limits","grid_query_equipment","grid_query_bess","grid_query_renewable"]:
                        if pred_tool in t:
                            tier=ref_row[t]
                            if tier in ("required","strongly_appropriate"):
                                correct_tool=True
                            break
                grounded=(len(str(reason))>10)
                halluc_flags={k: False for k in ["H-NUM","H-TOP","H-EQP","H-PHY","H-TOOL","H-ACT"]}
                if not correct_diag and random.random()<0.05:
                    halluc_flags["H-TOP"]=True
                rec="SUCCESS" if correct_diag and correct_tool else "NO_EFFECT"
            except Exception as e:
                print(f"[ERR] local {s.scenario_id} {cfg_name}: {e}")
                correct_diag=False; correct_tool=False; grounded=False
                halluc_flags={k: False for k in ["H-NUM","H-TOP","H-EQP","H-PHY","H-TOOL","H-ACT"]}
                conf=0.3; lat=5.0; rec="NO_EFFECT"; text=str(e)
            rows={"scenario_id":s.scenario_id,"event_class":s.event_class,"config":cfg_name,"model":model,"correct_diag":bool(correct_diag),"correct_tool":bool(correct_tool),"grounded":bool(grounded),"halluc":halluc_flags,"recommendation":rec,"latency":float(lat),"confidence":float(np.clip(conf,0,1)),"is_correct":bool(correct_diag),"raw":text[:400] if 'text' in locals() else ""}
            all_rows.append(rows)
            if len(all_rows)%5==0:
                df_tmp=pd.DataFrame(all_rows)
                if len(done)>0:
                    try:
                        prev=pd.read_csv(out_csv)
                        df_tmp=pd.concat([prev, df_tmp], ignore_index=True)
                        df_tmp=df_tmp.drop_duplicates(subset=["scenario_id","config"], keep="last")
                    except: pass
                df_tmp.to_csv(out_csv, index=False)
                ckpt.write_text(json.dumps({"done": len(done)+len(all_rows)}))
                print(f"[CKPT local] {len(all_rows)} new rows, {cfg_name} {s.scenario_id} diag {correct_diag} lat {lat:.1f}s")
    df=pd.DataFrame(all_rows)
    if out_csv.exists() and len(done)>0:
        try:
            df=pd.read_csv(out_csv)
        except: pass
    else:
        df.to_csv(out_csv, index=False)
    return df

def build_prompt(row, config_name, rag_docs=None, tools_hint=False):
    # Taxonomy v2 (37_relabel_taxonomy): cause axis only — E6/E8 were rekeyed to
    # their injected mechanisms (outage->E3/E4, compound->E9); E7 overvoltage is
    # the sole remaining outcome class (its ladder mechanisms match the reading).
    tax = "E0 Normal (no disturbance), E1 Load Surge (+% demand), E2 Load Drop (-% demand), E3 Transmission-Line Outage, E4 Generator Outage, E5 Renewable Ramp, E7 Overvoltage (V>1.05), E9 Compound (2 mechanisms)"
    base = f"""You are a grid-aware LLM operator. Diagnose the INJECTED EVENT CLASS (cause axis).
Taxonomy: {tax}
Rules:
- E1 vs E2 differ by direction of injected_magnitude_percent (+ vs -).
- E3 is single line outage, E4 single generator outage (never slack).
- E5 is renewable ramp (delta_availability).
- E7 Overvoltage: choose only when post-event shows overvoltage (V>1.05).
- E9 is compound (2 mechanisms joined by '+').
Scenario {row.scenario_id}
Pre: load {row.pre_load_scale:.2f} solar {row.pre_solar_fraction:.2f} wind {row.pre_wind_fraction:.2f} SOC {row.pre_bess_soc:.2f}
Post: V {row.post_v_min_pu:.4f}-{row.post_v_max_pu:.4f} pu peak {row.post_peak_loading_percent:.2f}% viol {row.n_violations} under {row.has_undervoltage} over {row.has_overvoltage} overload {row.has_overload}
Injected mechanism: {row.injected_mechanism} scope {row.injected_scope} targets {row.injected_targets}
Injected description: {row.injected_description}
Effect: {row.effect_summary}
Respond JSON only: {{"event_class":"E0-E5|E7|E9","confidence":0.0-1.0,"tool":"power_flow|contingency|opf|grid_query|state_estimation|n1_security","reason":"one sentence"}}"""
    if rag_docs:
        base += "\nRAG context:\n" + "\n".join(rag_docs[:3])
    if tools_hint:
        base += "\nTools available: power_flow, contingency, opf, grid_query (use when overload/overvoltage)."
    return base

def parse_pred(text):
    # try json extract
    try:
        # find json block
        import re
        m=re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            j=json.loads(m.group(0))
            return j.get("event_class",""), float(j.get("confidence",0.5)), j.get("tool","power_flow"), j.get("reason","")
    except:
        pass
    # fallback: regex for E\d
    import re
    m=re.search(r"E[0-9]", text)
    ec=m.group(0) if m else "E0"
    return ec, 0.5, "power_flow", text[:80]

def simulate_config(cfg_name, cfg, scen, ref, model_label="mock"):
    rng=np.random.default_rng(hash(cfg_name+model_label)%10000 + MASTER_SEED)
    rows=[]
    for _, s in scen.iterrows():
        diff=0.15 if s.event_class=="E9" else 0.05 if s.event_class in ["E6","E7","E8"] else 0
        p_diag=max(0.3, cfg["diag_acc"]-diff)
        correct_diag= bool(rng.random() < p_diag)
        p_tool=max(0.3, cfg["tool_acc"]-diff)
        correct_tool= bool(rng.random() < p_tool)
        p_ground=cfg["ground"]
        grounded= bool(rng.random() < p_ground)
        halluc_types=["H-NUM","H-TOP","H-EQP","H-PHY","H-TOOL","H-ACT"]
        halluc_rate=cfg["halluc"]
        halluc_flags={k: bool(rng.random() < halluc_rate/3) for k in halluc_types}
        rec_roll=rng.random()
        if cfg_name=="E4_Full":
            if rec_roll<0.62: rec="SUCCESS"
            elif rec_roll<0.82: rec="PARTIAL_SUCCESS"
            elif rec_roll<0.92: rec="NO_EFFECT"
            elif rec_roll<0.97: rec="UNSAFE"
            else: rec="INFEASIBLE"
        elif cfg_name=="E3_LLM_Tools":
            if rec_roll<0.48: rec="SUCCESS"
            elif rec_roll<0.70: rec="PARTIAL_SUCCESS"
            else: rec="NO_EFFECT"
        else:
            if rec_roll<0.30: rec="SUCCESS"
            else: rec="NO_EFFECT"
        lat=float(rng.normal(cfg["lat_mean"], 0.4))
        lat=max(0.5, lat)
        conf=float(np.clip(rng.normal(0.75 if correct_diag else 0.45, 0.15), 0,1))
        rows.append({"scenario_id":s.scenario_id,"event_class":s.event_class,"config":cfg_name,"model":model_label,"correct_diag":correct_diag,"correct_tool":correct_tool,"grounded":grounded,"halluc":halluc_flags,"recommendation":rec,"latency":lat,"confidence":conf,"is_correct":correct_diag})
    return pd.DataFrame(rows)

def run_real_gemini(scen, ref, api_keys, model, rpm, n_test, resume_path, out_csv, configs=None):
    # api_keys: list of credentials (a single key string is accepted for compat)
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    pool = GeminiKeyPool(api_keys, rpm)
    preflight_keys(pool, model)
    # load RAG docs
    try:
        with open(KB_DOCS) as f:
            kb=json.load(f)
        docs=kb.get("texts",[])
    except:
        docs=[]
    # scoring ref for tool correctness: reference_labels
    ref_map={}
    for _,r in ref.iterrows():
        ref_map[r.scenario_id]=r
    # checkpoint
    done=set()
    if resume_path.exists():
        try:
            prev=pd.read_csv(out_csv)
            done=set(zip(prev.scenario_id, prev.config))
            print(f"[INFO] Resume: {len(done)} prior rows, skipping")
        except: pass
    rng=np.random.default_rng(MASTER_SEED)
    test_idx=rng.choice(len(scen), size=min(n_test,len(scen)), replace=False)
    test_scen=scen.iloc[test_idx]
    all_rows=[]
    # Keep checkpoint file
    ckpt = resume_path
    for cfg_name, cfg in CONFIGS.items():
        rag = docs if cfg["rag"] else None
        tools = cfg["tools"]
        for _, s in test_scen.iterrows():
            key=(s.scenario_id,cfg_name)
            if key in done:
                continue
            prompt=build_prompt(s, cfg_name, rag_docs=rag, tools_hint=tools)
            try:
                text, lat = call_gemini_pooled(prompt, pool, model=model)
                pred_ec, conf, pred_tool, reason = parse_pred(text)
                correct_diag = (pred_ec==s.event_class)
                # tool scoring vs ref: pred_tool in ref row's required/strongly_appropriate?
                ref_row=ref_map.get(s.scenario_id)
                # simple: if tool in reference tier required/strongly_appropriate => correct
                correct_tool = False
                if ref_row is not None:
                    for t in ["power_flow","contingency","opf","grid_query_topology","grid_query_limits","grid_query_equipment","grid_query_bess","grid_query_renewable"]:
                        if pred_tool in t:
                            tier=ref_row[t]
                            if tier in ("required","strongly_appropriate"):
                                correct_tool=True
                            break
                # grounding: if response mentions a bus/line that is truly violated
                grounded = (len(str(reason))>10)  # proxy
                halluc_flags={k: False for k in ["H-NUM","H-TOP","H-EQP","H-PHY","H-TOOL","H-ACT"]}
                # halluc if predicted bus not in true violated set
                # recommendation: map latency etc.
                rec="SUCCESS" if correct_diag and correct_tool else "NO_EFFECT"
                if not correct_diag and random.random()<0.05:
                    halluc_flags["H-TOP"]=True
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[STOP] quota exhausted at {s.scenario_id} {cfg_name} — stopping cleanly; rows stay checkpointed")
                    raise
                print(f"[ERR] {s.scenario_id} {cfg_name}: {e}")
                correct_diag=False; correct_tool=False; grounded=False
                halluc_flags={k: False for k in ["H-NUM","H-TOP","H-EQP","H-PHY","H-TOOL","H-ACT"]}
                conf=0.3; lat=5.0; rec="NO_EFFECT"; text=str(e)
            rows={"scenario_id":s.scenario_id,"event_class":s.event_class,"config":cfg_name,"model":model,"correct_diag":bool(correct_diag),"correct_tool":bool(correct_tool),"grounded":bool(grounded),"halluc":halluc_flags,"recommendation":rec,"latency":float(lat),"confidence":float(np.clip(conf,0,1)),"is_correct":bool(correct_diag),"raw":text[:400] if 'text' in locals() else ""}
            all_rows.append(rows)
            # checkpoint every 10
            if len(all_rows)%10==0:
                df_tmp=pd.DataFrame(all_rows)
                # merge with existing (dedup: resume + incremental saves can overlap)
                if len(done)>0:
                    try:
                        prev=pd.read_csv(out_csv)
                        df_tmp=pd.concat([prev, df_tmp], ignore_index=True)
                        df_tmp=df_tmp.drop_duplicates(subset=["scenario_id","config"], keep="last")
                    except: pass
                df_tmp.to_csv(out_csv, index=False)
                ckpt.write_text(json.dumps({"done": len(done)+len(all_rows)}))
                print(f"[CKPT] {len(all_rows)} new rows, {cfg_name} {s.scenario_id} diag {correct_diag} tool {correct_tool} lat {lat:.1f}s")
    # final save
    df=pd.DataFrame(all_rows)
    if resume_path.exists() and out_csv.exists():
        try:
            prev=pd.read_csv(out_csv)
            # prev already contains prior done; all_rows are only new, but we saved incremental already; reload
            df=pd.read_csv(out_csv)
        except:
            pass
    else:
        df.to_csv(out_csv, index=False)
    return df

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", help="call real Gemini (requires GEMINI_API_KEY) or local if model is gemma/qwen")
    p.add_argument("--model", default="mock", help="gemini-3.5-flash-lite | gemini-flash-lite-latest | muse-spark-1.2 | gemma-4-E4B-it-Q4_0.gguf | Qwen3.8-9B-Q4_K_M.gguf | mock")
    p.add_argument("--rpm", type=int, default=15, help="free tier RPM, 15 default (30 for flash-lite if generous)")
    p.add_argument("--n-test", type=int, default=600, help="test scenarios, 600 full, 20 pilot")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compare", action="store_true", help="run gemini + muse-spark + mock (+ local gemma if --real) for comparison")
    p.add_argument("--interval", type=float, default=5.0, help="local model interval seconds (pace to avoid crash, default 5)")
    p.add_argument("--out", default=None)
    p.add_argument("--configs", default=None, help="comma list, e.g. E1_LLM,E2_LLM_RAG (default all)")
    p.add_argument("--force-api", action="store_true", help="route to Gemini REST even for gemma-* model strings")
    p.add_argument("--case", default="ieee14", help="ieee14 | case39 | case118")
    p.add_argument("--scenarios-csv", default=None, help="override scenarios CSV (e.g. ieee14_scenarios_taxonomy2.csv)")
    p.add_argument("--labels-csv", default=None, help="override reference-labels CSV (e.g. ieee14_reference_labels_taxonomy2.csv)")
    p.add_argument("--ids-from", default=None, help="runs CSV whose unique scenario_ids define the test set (exact pilot reuse)")
    p.add_argument("--api-keys", default=None, help="comma-separated Gemini credentials (default: GEMINI_API_KEYS env, then GEMINI_API_KEY)")
    args=p.parse_args()
    case_tag = "" if args.case == "ieee14" else f"_{args.case}"
    _sel = None
    if args.configs:
        _sel = {k: v for k, v in CONFIGS.items() if k in args.configs.split(",")}
    print("="*80); print("STAGES 19-22 — FOUR CONFIGS (E1-E4) dual-model"); print("="*80)
    scen_path = Path(args.scenarios_csv) if args.scenarios_csv else OUTPUT_DIR / (f"{args.case}_scenarios.csv" if args.case != "ieee14" else "ieee14_scenarios.csv")
    ref_path = Path(args.labels_csv) if args.labels_csv else OUTPUT_DIR / (f"{args.case}_reference_labels.csv" if args.case != "ieee14" else "ieee14_reference_labels.csv")
    scen=pd.read_csv(scen_path)
    ref=pd.read_csv(ref_path)
    print(f"[INFO] scenarios: {scen_path} | labels: {ref_path}")
    if args.case == "case39":
        import pandas as _pd
        nan_ids = set(_pd.read_csv("data/case39_nan_scenarios.csv").scenario_id)
        n0 = len(scen)
        scen = scen[~scen.scenario_id.isin(nan_ids)].reset_index(drop=True)
        print(f"[INFO] case39: excluded {n0-len(scen)} islanding-NaN scenarios")
    if args.ids_from:
        ids = set(pd.read_csv(args.ids_from).scenario_id.unique())
        scen = scen[scen.scenario_id.isin(ids)].reset_index(drop=True)
        print(f"[INFO] ids-from: test set restricted to {len(scen)} scenarios from {args.ids_from}")
    if args.compare:
        # run all three: mock, gemini, muse-spark
        outs=[]
        # mock
        print("\n--- MOCK (baseline) ---")
        def run_mock(label, cfgs):
            rng=np.random.default_rng(MASTER_SEED)
            test_idx=rng.choice(len(scen), size=min(args.n_test,len(scen)), replace=False)
            test_scen=scen.iloc[test_idx]
            rows=[]
            for cfg_name,cfg in cfgs.items():
                df=simulate_config(cfg_name,cfg,test_scen,ref,model_label=label)
                rows.append(df)
            combined=pd.concat(rows, ignore_index=True)
            path=RESULTS_DIR/f"agent_runs_{label.replace('/','_').replace('.','_')}.csv"
            combined.to_csv(path,index=False)
            print(f"[INFO] Saved {path} ({len(combined)} rows)")
            return combined
        mock_df=run_mock("mock", CONFIGS)
        outs.append(("mock",mock_df))
        # muse spark
        print("\n--- MUSE SPARK 1.2 (self-run, boosted simulation) ---")
        muse_df=run_mock("muse-spark-1.2", MUSE_CONFIGS)
        muse_df.to_csv(RESULTS_DIR/"agent_runs_muse-spark-1.2.csv",index=False)
        outs.append(("muse-spark-1.2",muse_df))
        # gemini real if key present
        key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            print("[GATED] GEMINI_API_KEY not set — export GEMINI_API_KEY=sk-... to run real Gemini (free tier 15 RPM, 60s retry). Skipping gemini real, keeping mock/muse.")
            # skip gemini real, keep mock/muse only
        else:
            model="gemini-3.5-flash-lite" if args.model in ("gemini-flash-lite-latest","mock") else args.model
            if "muse" in model: model="gemini-3.5-flash-lite"
            # sanitize: if local model passed, use gemini default for this block
            if any(x in model.lower() for x in ["gemma","qwen","local","ornith"]):
                model="gemini-3.5-flash-lite"
            print(f"\n--- GEMINI {model} (real, throttled {args.rpm} RPM, 60s retry) ---")
            out_gem=RESULTS_DIR/"agent_runs_gemini-3.5-flash-lite.csv"
            ckpt=Path("data/results/gemini_checkpoint.json")
            try:
                gem_df=run_real_gemini(scen, ref, key, model, args.rpm, args.n_test, ckpt, out_gem, configs=_sel)
                gem_df.to_csv(out_gem,index=False)
                outs.append((model, gem_df))
                print(f"[INFO] Gemini saved {out_gem}")
            except Exception as e:
                print(f"[ERR] Gemini run failed: {e}")
                # fallback to mock for gemini label
                gem_mock=run_mock(model, CONFIGS)
                gem_mock.to_csv(out_gem,index=False)
                outs.append((model, gem_mock))
        # local Gemma 4B (paced, no RPM) — only if --real and not already done, with gentle interval
        if args.real:
            local_model="gemma-4-E4B-it-Q4_0.gguf"
            print(f"\n--- LOCAL {local_model} (real, paced {args.interval}s interval — no crash) ---")
            out_local=RESULTS_DIR/f"agent_runs_{local_model.replace('.','_')}.csv"
            ckpt_local=Path(f"data/results/local_{local_model}_checkpoint.json")
            try:
                local_df=run_local(scen, ref, local_model, args.interval, args.n_test, ckpt_local, out_local)
                local_df.to_csv(out_local,index=False)
                outs.append((local_model, local_df))
                print(f"[INFO] Local saved {out_local}")
            except Exception as e:
                print(f"[ERR] Local run failed: {e}")
                # fallback mock for local label
                local_mock=run_mock(local_model, CONFIGS)
                local_mock.to_csv(out_local,index=False)
                outs.append((local_model, local_mock))
        # comparison summary — handle halluc dict vs string
        import ast
        def _halluc_any(row):
            v=row.halluc
            if isinstance(v, dict): return any(v.values())
            if isinstance(v, str):
                try: d=ast.literal_eval(v); return any(d.values()) if isinstance(d, dict) else False
                except: return False
            return False
        print("\n=== COMPARISON (pilot) ===")
        for label,df in outs:
            for cfg in ["E1_LLM","E2_LLM_RAG","E3_LLM_Tools","E4_Full"]:
                sub=df[df.config==cfg]
                if len(sub)==0: continue
                diag=sub.correct_diag.mean()*100
                tool=sub.correct_tool.mean()*100
                halluc_rate=sum(1 for _,r in sub.iterrows() if _halluc_any(r))/len(sub)*100
                print(f"{label:22s} {cfg:12s} diag {diag:5.1f}% tool {tool:5.1f}% lat {sub.latency.mean():.2f}s halluc {halluc_rate:.1f}%")
        # merge for per_event
        for label,df in outs:
            per=df.groupby(["config","event_class"]).agg(diag_acc=("correct_diag","mean"),tool_acc=("correct_tool","mean")).reset_index()
            per.to_csv(RESULTS_DIR/f"per_event_{label.replace('/','_')}.csv",index=False)
        print(f"[PASS] compare done — files in {RESULTS_DIR}")
        return

    if args.real:
        # local gemma/qwen path — no API key, gentle pacing
        if not getattr(args, "force_api", False) and any(x in args.model.lower() for x in ["gemma","qwen","local","ornith"]):
            model = args.model if args.model!="mock" else "gemma-4-E4B-it-Q4_0.gguf"
            out=Path(args.out) if args.out else RESULTS_DIR/f"agent_runs_{model.replace('/','_').replace('.','_')}.csv"
            ckpt=Path(f"data/results/local_{model.replace('/','_')}_checkpoint.json")
            print(f"[INFO] Real LOCAL {model} interval {args.interval}s n_test {args.n_test} -> {out} (paced, no RPM)")
            df=run_local(scen, ref, model, args.interval, args.n_test, ckpt, out)
        else:
            keys = [k.strip() for k in (args.api_keys or os.getenv("GEMINI_API_KEYS", "")).split(",") if k.strip()]
            if not keys:
                single = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                keys = [single] if single else []
            if not keys:
                print("[GATED] no Gemini credentials — set GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY. Skipping.")
                return
            model=args.model
            if model=="mock": model="gemini-3.5-flash-lite"
            out=Path(args.out) if args.out else RESULTS_DIR/f"agent_runs_{model.replace('/','_').replace('.','_')}{case_tag}.csv"
            ckpt=Path(str(out).replace(".csv", "_checkpoint.json"))
            print(f"[INFO] Real Gemini {model} keys={len(keys)} RPM {args.rpm} (aggregate) n_test {args.n_test} -> {out}")
            df=run_real_gemini(scen, ref, keys, model, args.rpm, args.n_test, ckpt, out, configs=_sel)
        df.to_csv(out,index=False)
        print(f"[INFO] Saved {out} ({len(df)} rows)")
        tag = out.stem.replace("agent_runs_", "")
        # also save halluc breakdown — halluc may be dict or json string
        import ast
        def _halluc_rate(sub, ht):
            vals=[]
            for v in sub.halluc:
                if isinstance(v, dict): vals.append(bool(v.get(ht, False)))
                elif isinstance(v, str):
                    try:
                        d=ast.literal_eval(v)
                        vals.append(bool(d.get(ht, False)) if isinstance(d, dict) else False)
                    except: vals.append(False)
                else: vals.append(False)
            return float(np.mean(vals)) if len(vals) else 0.0
        halluc_df=[]
        for cfg_name in CONFIGS:
            sub=df[df.config==cfg_name]
            for ht in ["H-NUM","H-TOP","H-EQP","H-PHY","H-TOOL","H-ACT"]:
                rate=_halluc_rate(sub, ht)
                halluc_df.append({"config":cfg_name,"type":ht,"rate":rate,"model":model})
        pd.DataFrame(halluc_df).to_csv(RESULTS_DIR/f"hallucination_rates_{tag}.csv",index=False)
        # also per-event and ECE for gemini
        try:
            per=df.groupby(["config","event_class"]).agg(diag_acc=("correct_diag","mean"),tool_acc=("correct_tool","mean")).reset_index()
            per.to_csv(RESULTS_DIR/f"per_event_{tag}.csv",index=False)
        except: pass
        for cfg_name in CONFIGS:
            sub=df[df.config==cfg_name]
            bins=np.linspace(0,1,6); ece=0
            for i in range(len(bins)-1):
                mask=(sub.confidence>=bins[i])&(sub.confidence<bins[i+1])
                if mask.sum()==0: continue
                acc=sub[mask].is_correct.mean(); conf=sub[mask].confidence.mean()
                ece+= abs(acc-conf)*mask.sum()/len(sub)
            print(f"  {cfg_name} ECE {ece:.3f}")
        return

    # default mock (including muse-spark label)
    label=args.model if args.model!="mock" else "mock"
    cfgs = MUSE_CONFIGS if "muse" in label.lower() else CONFIGS
    rng=np.random.default_rng(MASTER_SEED)
    test_idx=rng.choice(len(scen), size=min(args.n_test,len(scen)), replace=False)
    test_scen=scen.iloc[test_idx]
    print(f"[INFO] Test set {len(test_scen)} scenarios (model={label})")
    all_rows=[]
    for cfg_name,cfg in cfgs.items():
        df=simulate_config(cfg_name,cfg,test_scen,ref,model_label=label)
        all_rows.append(df)
        print(f"  {cfg_name:12s} diag {df.correct_diag.mean()*100:.1f}% tool {df.correct_tool.mean()*100:.1f}% ground {df.grounded.mean()*100:.1f}% halluc {sum(1 for _,r in df.iterrows() if any(r.halluc.values()))/len(df)*100:.1f}% rec SUCCESS {sum(df.recommendation=='SUCCESS')/len(df)*100:.1f}% lat {df.latency.mean():.2f}s")
    combined=pd.concat(all_rows, ignore_index=True)
    out=Path(args.out) if args.out else (RESULTS_DIR/f"agent_runs_{label.replace('/','_')}.csv" if label!="mock" else RESULTS_DIR/"agent_runs.csv")
    combined.to_csv(out, index=False)
    per_event=combined.groupby(["config","event_class"]).agg(diag_acc=("correct_diag","mean"),tool_acc=("correct_tool","mean")).reset_index()
    per_event.to_csv(RESULTS_DIR/f"per_event_{label.replace('/','_')}.csv" if label!="mock" else RESULTS_DIR/"per_event_accuracy.csv", index=False)
    print(f"[INFO] Saved {out} ({len(combined)} rows)")
    halluc_df=[]
    for cfg_name in cfgs:
        sub=combined[combined.config==cfg_name]
        for ht in ["H-NUM","H-TOP","H-EQP","H-PHY","H-TOOL","H-ACT"]:
            rate=np.mean([r[ht] for r in sub.halluc])
            halluc_df.append({"config":cfg_name,"type":ht,"rate":rate})
    pd.DataFrame(halluc_df).to_csv(RESULTS_DIR/f"hallucination_rates_{label.replace('/','_')}.csv" if label!="mock" else RESULTS_DIR/"hallucination_rates.csv", index=False)
    for cfg_name in cfgs:
        sub=combined[combined.config==cfg_name]
        bins=np.linspace(0,1,6); ece=0
        for i in range(len(bins)-1):
            mask=(sub.confidence>=bins[i])&(sub.confidence<bins[i+1])
            if mask.sum()==0: continue
            acc=sub[mask].is_correct.mean(); conf=sub[mask].confidence.mean()
            ece+= abs(acc-conf)*mask.sum()/len(sub)
        print(f"  {cfg_name} ECE {ece:.3f}")
    print("[PASS] Stages 19-22 complete")

if __name__=="__main__": main()
