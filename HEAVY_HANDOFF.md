# Heavy Run Handoff — Fresh Agent Context

> **One-command resume:** `bash heavy_resume.sh`  (or `nohup bash heavy_resume.sh > heavy.log 2>&1 &`)
> All stages are idempotent — skips completed outputs, picks up where interrupted. LLM stages remain **gated** (require API key / Ollama).

## What Has Been Done (2026-08-30)

### Rebuilt from zero (no prior `data/` existed)
- **03_renewables_bess.py** — IEEE14-RE: PV12MW@14, Wind15MW@6, BESS 20MW/40MWh@9, gen vm 1.02-1.03, limits **line 3.0% / trafo 4.0%** (tuned — base 1.45% < limit, compound 4.1% → overload). Hash `2580e77e...05cfb64`. IEEE-14 stiffness + E7 reactive mechanisms documented as limitation.
- **03_build_39_118.py** — IEEE39 (39b 35l 21 loads 9 gens peak 75.5% @1.0) and IEEE118 (118b 173l 99 loads 53 gens peak 4.49%). Limits **100%** for large nets (realistic), vm max 1.10 (vs 1.05 for 14). Hashes `1ea90b0a`, `6ed03378`. Files: `data/processed/case39_net_re.json`, `case118_net_re.json`.
- **04_operating_point_generator.py** — IEEE14 4,000 normals (0.70-1.10 ±5%, 195s, 100% keep). **Heavy:** `04_heavy_op.py` parameterized — case39 test `1000 → 77/1000 in 261s (1.5% keep, too low, needs range tuning)`. Current best ranges: 14:0.70-1.10, 39:0.65-0.90 (still low), 118:0.80-1.10. Needs further tuning (39 load_scale 0.55-0.70).
- **05_event_generator.py** — IEEE14 **3,000 scenarios 300/class in 1087s**, 21/21 checks, replay 40/40 dV0, E6 mean 5.83 attempts, E7 1.00, E8 9.95. Outputs: `ieee14_scenarios.*`, post voltages/loadings.
- **06_measurement_generator.py** — 361,764 readings (up to 122 minus outages) σ_V 0.003, σ_P max(0.0075·|S_true|,0.05) from true, max|z| 4.06/4.27/4.67, 182s.
- **07_state_estimator.py** — Synthetic WLS: 3k/3k iters 4-6 med5, RMSE_V 0.000689, RMSE_θ 0.0196, J/ν 1.0028, 5.65% exceedance (SE 0.40%), per-topology table, Jacobian 4.36e-10 over 24×3.
- **08_violation_detector.py** — Severity `S=0.6*…+0.4*…`, boundaries 0.0263/0.0526/0.1053 =1/38 fractions, 5-set CV, reconciled **2977/3000 =99.23%** (TN2021 FP14 FN9 TP956).
- **09_reference_labels.py** — 3k×10 tiers, 3 constant trivial, 7 varying, leakage audit, terminology “rule-based reference policy labels”. Saved `ieee14_reference_labels.csv`.
- **10_power_flow_tool.py** — `--scenario-id`/`--list` verified (example 000001 losses 11.84MW)
- **11_grid_query_tool.py**, **12_contingency_tool.py** (`--batch` 15/15 converged 0 islanded), **13_n1_security_tool.py**, **14_opf_tool.py** (minimal), **15_tool_validation.py** 21/21 PASS
- **16-18** KB 8 docs → `data/knowledge_base/faiss.index` (all-MiniLM-L6-v2 384-dim) Recall@1 100%
- **19-22** Mocked agents E1 55% → E4 87% on 600 tests, latency 1.2→3.1s, ECE 0.17, halluc 9.7→2.0% → `data/results/agent_runs.csv` (2400 rows). **NOT heavy** — rule-based, no real LLM calls.
- **23-25** Validation/recommendation/hallucination, **26** generalization 87→82→76%, **27** McNemar chi2 134 p4.5e-31 g0.72, bootstrap 84-90%, **28** 9 figs (Fig1,2,3,5,6,7,8,11,12) + tables, **29** `app.py` (streamlit, needs `streamlit` pkg)

### Current Data Inventory
```
data/processed/
  ieee14_* (complete 3k corpus), case39_net_re.json, case118_net_re.json,
  case39_operating_points.csv  (77 rows, incomplete — needs 5k),
  case39_*_factors/voltages/loadings (77 rows, incomplete)
data/results/agent_runs.csv (mocked), figs/ (9 pngs)
```

### What's Pending for Heavy Headless (Non-LLM, Gated LLM Skipped)
- [ ] **04 heavy OP:** Regenerate case39 5k + case118 7k with tuned load_scale (39:0.55-0.70, 118:0.75-1.05) — ~15-20min each, idempotent
- [ ] **05 heavy scenarios:** 5k (39) + 7k (118) with ladders (E6/E8 6-10 PF each) — ~12-15min each (≈30k PF solves), outputs `case39_scenarios.csv`, `case118_scenarios.csv`
- [ ] **06-09 heavy:** Measurements/SE/violations/labels for 39/118 (real, not synthetic WLS) — ~10min each case if using `pp.estimation.estimate`
- [ ] **13,15 heavy validation** on 39/118
- [ ] **27-28 re-run** on true heavy outputs (replace mocked `agent_runs.csv`)

### LLM-Gated (Do NOT Run Without Key/Ollama)
- Stages 19-22 with real LLM: needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` **or** `ollama pull llama3.1` + `OLLAMA_MODEL` env. Current 19-22 are mocked. To run heavy LLM: `python 19_22_run_agents.py --real --model gpt-4o` (to be implemented) — 2400 calls ≈80min + $20-40. Leave gated until user provides key.

### How Heavy Resume Works
`heavy_resume.sh` checks each output existence + row count:
- If `case39_operating_points.csv` has <5000 rows → re-run `04_heavy_op.py case39 5000`
- If `case39_scenarios.csv` missing or <5000 → run `05` variant for case39 (to be built as `05_heavy.py --case case39 --n 500`)
- Same for 118
- All `pp.runpp(numba=True)` with snapshot/restore, no stale tables

### Resume Command
```bash
cd /home/touhid/Documents/llmpaper
bash heavy_resume.sh
# or headless:
nohup bash heavy_resume.sh > heavy.log 2>&1 &
tail -f heavy.log
```

### Environment
- Python 3.14.6, pandapower 3.5.4, pandas 2.3.3, numpy 2.4.6, numba on
- Installed: sentence-transformers 6.0, faiss-cpu 1.15, statsmodels 0.15, streamlit 1.62
- No Ollama, no API keys set (checked 2026-08-30)
- Working dir: `/home/touhid/Documents/llmpaper`

### For Fresh Agent — Start Here
1. `cat HEAVY_HANDOFF.md`
2. `bash heavy_resume.sh --dry-run` to see what would run
3. `python3 -c "import pandapower; print(power)"` sanity
4. If heavy OP still low keep-rate, edit `04_heavy_op.py` load_scale ranges (see above) and re-run.

