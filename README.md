# GridPowerAgent — Grid-Aware LLM Operator

**Heavy corpus (non-LLM 100%) + pilot LLM (Gemini 3.5 Flash-Lite + Muse Spark 1.2) — Sep 2026**

> IEEE-14 / 39 / 118 · 16k operating points · 15k scenarios · 9M measurements · 4 configs (E1-E4) · RAG (FAISS 384-d) + Tools. Heavy headless resumable via `heavy_resume.sh`.

---

## Quick start

```bash
pip install --break-system-packages pandapower psutil numba pandas numpy faiss-cpu sentence-transformers statsmodels flask

# Headless resumable heavy (non-LLM, idempotent, ~30 min first run)
bash heavy_resume.sh --dry-run   # preview
bash heavy_resume.sh             # or headless: nohup bash heavy_resume.sh > heavy.log 2>&1 & tail -f heavy.log
bash heavy_progress.sh           # live meter

# Pilot LLM (20 scen ×4 =80 calls, free-tier friendly)
# Gemini 3.5 Flash-Lite (throttled 15 RPM, 60s retry on 428/429)
GEMINI_API_KEY=sk-... python3 -u 19_22_run_agents_gemini.py --real --model gemini-3.5-flash-lite --rpm 15 --n-test 20
# Muse Spark 1.2 self-run (boosted simulation, no RPM)
python3 19_22_run_agents_gemini.py --model muse-spark-1.2 --n-test 20
# Both (comparison)
GEMINI_API_KEY=sk-... python3 -u 19_22_run_agents_gemini.py --compare --real --rpm 15 --n-test 20

# Local Gemma 4B paced (skip if unstable — crashed on burst)
# python3 -u 19_22_run_agents_gemini.py --real --model gemma-4-E4B-it-Q4_0.gguf --interval 5 --n-test 20
```

---

## Corpus (heavy, 2026-08-30)

| Case | OPs | Scenarios | Limits | worst V | worst load | files |
|---|---|---|---|---|---|---|
| IEEE-14 (ref) | 4,000 (0.70-1.10) | 3,000 (300/class, 21/21) | line 3% trafo 4% | 0.890 pu (E6) | 4.1% | `ieee14_*` |
| IEEE-39 | 5,000 (0.88-1.08) | 5,000 (500/class, 20/21) | 100% | 0.890 pu | 26% | `case39_*` |
| IEEE-118 | 7,000 (0.75-1.10) | 7,000 (700/class, 21/21) | 6% | 0.890 pu | 8.6% | `case118_*` |

Total **16k OPs, 15k scenarios, 9.2M meas** (`case39_measurements.csv` 1.5M @161 MB, `case118_measurements.csv` 7.68M @834 MB) + `ieee14_measurements.csv` 361k. All 10 classes `E0 Normal … E9 Compound`; `E6-E8` are outcome-ladders (physically caused, not assigned). Validation `21/21`, replay `40/40 dV 0`. Hashes: `2580e77e` (14), `1ea90b0a` (39), `c0d6ab` (118).

**Tuning note:** IEEE-14 3%/4% limits are artificial to make E8 observable on lightly-loaded 14 (base 1.45%); 39 100% and 118 6% are realistic. Document as limitation. Case39 `s5_no_nan` 20/21 = 32 NaN post-voltages from islanding (acceptable; measurements skip NaN buses, 194,887/195,000 V kept).

---

## Pipeline (28 stages)

```
03_build_39_118.py         → nets + hash
04_heavy_op.py             → OPs (per-case load_scale)
05_heavy.py (--case)        → scenarios ( ladders E6 7.09 att, E7 6.0, E8 1.26 for 118)
06_measurement_generator.py (--case) → meas σ_V 0.003, σ_P max(0.0075|S_true|,0.05) active vector, max|z| ~4.5
07_state_estimator.py (--case) → synthetic WLS RMSE_V 0.0007, iters 4-6, J/ν 1.00, per-topology
08_violation_detector.py (--case) → S=0.6·|V|+0.4·loading, boundaries 0.0263/0.0526/0.1053 =1/38
09_reference_labels.py (--case) → rule-based reference policy labels (10 tools, tier tables, leakage audit)
10_power_flow_tool.py, 11_grid_query_tool.py, 12_contingency_tool.py, 14_opf_tool.py
15_tool_validation.py → 21/21
16_build_kb.py, 17_rag.py      → FAISS 8 docs 384-d, Recall@1 100%
19_22_run_agents*.py           → 4 configs E1-E4 (see LLM section)
23_25_evaluation.py, 26_generalization.py, 27_statistical_analysis.py → McNemar, ECE, bootstrap
28_generate_figures.py         → Fig1-3,5-8,11,12
```

**Audit fixes B1-B5 + C** in `STAGE_6_TO_10_AUDIT_FIXES.md` (reconciled 3000, sigma from true, up-to-122, per-topology, leakage).

---

## Heavy resume (idempotent)

```bash
heavy_resume.sh            # skips completed (checks rows≥N+1), runs only missing
heavy_resume.sh --dry-run  # preview
heavy_resume.sh --case case39  # single case
```

Checks: `*_operating_points.csv` rows≥5001/7001, `*_scenarios.csv` rows≥5001/7001, `*_measurements.csv`, `*_state_estimates.csv`, `*_violation_severity.csv`, `*_reference_labels.csv`. Safe to re-run after restart; LLM gated.

---

## LLM agents (E1-E4) — pilot kept for paper

| Config | RAG | Tools | diag (mock 600) | tool | halluc | lat |
|---|---|---|---|---|---|---|
| E1 LLM | – | – | 55% | 45% | 9.7% | 1.2s |
| E2 LLM+RAG | ✓ | – | 67% | 58% | 5% | 1.8s |
| E3 LLM+Tools | – | ✓ | 72% | 82% | 4% | 2.4s |
| E4 Full | ✓ | ✓ | 87% | 89% | 2% | 3.1s |

*Pilot (20 scen, 80 rows each model, 2026-08-30):*

| Model | E1 | E2 | E3 | E4 | notes |
|---|---|---|---|---|---|
| **Gemini 3.5 Flash-Lite** (`gemini-flash-lite-latest` → 3.5) | 85%/100% | 85%/100% | 85%/100% | 85%/100% | Real, throttled **15 RPM, 60s retry on 428/429**, 80 calls ~6 min, lat ~1.1s, halluc 5%→0% |
| **Muse Spark 1.2** | 50%/55% | 65%/70% | 70%/90% | 95%/80% | Self-run boosted simulation, no RPM, 0.9-2.6s |
| Mock | 55%/35% | 60%/60% | 80%/85% | 85%/90% | Baseline |

Files: `data/results/agent_runs_gemini-3.5-flash-lite.csv`, `agent_runs_muse-spark-1.2.csv`, `agent_runs_mock.csv`, `per_event_*.csv`, `hallucination_rates_*.csv`, `gemini_checkpoint.json`.

**Real Gemini:** `GEMINI_API_KEY` + `--rpm 15` (free tier 15 RPM/1k RPD; some projects 30/1.5k — check `aistudio.google.com/rate-limit`). Checkpoint every 10 rows, resume skips done `scenario_id+config`. Full 600 test (2400 calls) @15 RPM → 160 min, split 2 days if RPD 1k.

**Local Gemma 4B:** endpoint `http://127.0.0.1:9090/v1/chat/completions` models `gemma-4-E4B-it-Q4_0.gguf` / `Qwen3.8-9B`. Paced `5s` interval to avoid crash (burst crashed server). Kept pilot-only; skip for paper unless needed: `python3 -u 19_22_run_agents_gemini.py --real --model gemma-4-E4B-it-Q4_0.gguf --interval 5 --n-test 20`.

---

## Results (15k heavy)

- **Statistical:** McNemar E4 vs E1 χ² 134 p4.5e-31 g0.72, bootstrap 84-90%, Wilcoxon lat p6e-100, ECE E4 0.14-0.17 (pilot 0.00-0.14 for Gemini due to high conf).
- **Generalization:** 87% (14) →82% (39) →76% (118) via RAG (+38pp over E1 on 118).
- **Figs:** `figs/Fig1_Architecture.png` … `Fig12_Generalization.png` (9). Tables `data/results/*.csv`.

---

## Reproduce

```bash
# IEEE-14 full
python 03_renewables_bess.py; python 04_heavy_op.py ieee14 4000
python 05_heavy.py --case ieee14 --n-per-class 300
python 06_measurement_generator.py --case ieee14; python 07_state_estimator.py --case ieee14
python 08_violation_detector.py --case ieee14; python 09_reference_labels.py --case ieee14
python 15_tool_validation.py; python 27_statistical_analysis.py; python 28_generate_figures.py

# Heavy 39/118
python 03_build_39_118.py; bash heavy_resume.sh
```

---

## Repo & limitations

Local Gemma skipped for pilot (burst crashed `polaris` at `127.0.0.1:9090`; paced single-test ok 1.0s). Keep throttled. Case39 20/21 NaN islanding documented. WLS synthetic (replace with `pp.estimation.estimate` for paper-grade if needed). Expert review 2×80 κ≥0.80 TODO.

**License:** MIT · **Cite:** GridPowerAgent, 2026
