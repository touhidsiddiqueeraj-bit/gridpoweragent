# GridPowerAgent — A Grid-Aware LLM Agent for Power System Event Understanding and Tool-Orchestrated Decision Support

> IEEE 14/39/118 · 16k operating points · 15k scenarios · 9.5M measurements · 4 agent configurations (E1–E4) · RAG (FAISS 384-d) + validated physics tools · paired local-vs-API evaluation

This repository contains the agent, its fully regenerable evaluation corpus, and the analysis pipeline for the paper (see `paper/GridPowerAgent_IEEE_Final.pdf`). Every number in the paper is computed from raw logs by `31_build_paper.py` — nothing is hand-transcribed.

## What the agent does

The agent follows an observe → diagnose → retrieve → plan → execute → interpret loop over simulated power-system states: it reads post-event grid state (voltages, loadings, outages, storage), optionally receives retrieved operating procedures and a tool manifest, and returns a disturbance class (E0–E9), a tool selection, and a recommendation as JSON. Four configurations ablate the additions: E1 LLM-only, E2 +RAG, E3 +Tools, E4 Full.

## Corpus

| Case | OPs | Scenarios (per class) | Measurements | Auto checks |
|---|---|---|---|---|
| IEEE-14 (ref) | 4,000 | 3,000 (300) | 361k | 21/21 |
| IEEE-39 | 5,000 | 5,000 (500) | 1.50M | **20/21** |
| IEEE-118 | 7,000 | 7,000 (700) | 7.68M | 21/21 |
| Total | 16k | 15k | 9.54M | — |

Ten classes: E0 Normal; E1 load surge, E2 load drop, E3 line outage, E4 generator outage, E5 renewable ramp (cause classes); E6 undervoltage, E7 overvoltage, E8 thermal overload (outcome classes, physically iterated); E9 compound. IEEE-14 thermal limits are artificial (3%/4%) to make E8 observable; IEEE-39 is at nameplate and retains **41 islanding scenarios with NaN post-voltages** — the failing check (`s5_no_nan`), with identifiers and causes shipped in `data/case39_nan_scenarios.csv`. Large per-scenario artifacts are excluded from git and regenerate deterministically from seeds (`heavy_resume.sh`); an archived DOI release is planned.

## Headline pilot results (140 scenarios × 4 configs, paired, exact denominators)

Two deployment tiers of the same prompt contract: a lightweight API model (Gemini 3.5 Flash Lite) and a small 4-bit quantized local model (Gemma 4 E4B-it).

| Metric | API | Local |
|---|---|---|
| Diagnosis | 109/140 in every config | 105–108/140 |
| Strict-specific tool selection | 35–53/140 | 63–64/140 |
| Hallucinated rows (any of 6 tags) | 0–2 per 140 | 0–2 per 140 |
| Latency | ~1.1 s/call | 42–70 s/call |

Key observations: (1) no detectable diagnosis difference above the pilot's minimum detectable effect (2.0–3.9 pp); (2) the API model defaults to power flow in 52% of tool selections — the strict metric penalizes this default-answer bias; (3) both models score zero on E6/E8, traced to a mixed-axis label taxonomy (outcome-labeled classes injected via cause mechanisms) — a benchmark-design finding, not model incapacity; (4) the deterministic rule-based oracle (N=600/config) validates harness plumbing only and carries no evidence about real LLMs.

## Repository layout

```
03_renewables_bess.py        IEEE-14 renewable/storage network (hash 2580e77e)
03_build_39_118.py           IEEE-39/118 networks
04_heavy_op.py               operating points (per-case load scale)
05_heavy.py / 05_event_generator.py  scenario injection (E0–E9)
06_measurement_generator.py  σ_V 0.003 pu, σ_P max(0.0075|S_true|,0.05) MVA
07_state_estimator.py        closed-form state estimates (RMSE_V 7e-4 pu)
08_violation_detector.py     severity + violation detection
08b_severity_label_noise.py  label-noise bound under estimation uncertainty
09_reference_labels.py       rule-based tool supervision (10 tools, tiers)
10–14 tools                  power flow / query / N-1 / OPF
15_tool_validation.py        21/21 tool checks
16_build_kb.py, 17_rag.py    FAISS 384-d over 8 procedure docs
19_22_run_agents_gemini.py   agent ablations E1–E4 (API + local + oracle)
23–28                        evaluation, generalization projector, stats, figures
29_paper_figures.py          paper figures (evidentiary status annotated)
30_local_pilot_resilient.py  crash-resilient local-model harness
31_build_paper.py            generates the paper from raw logs
34_texflow_build.py          builds the LaTeX via the texflow MCP
heavy_resume.sh              idempotent heavy-corpus runner
gemma_progress.sh            live progress bar for local runs
```

## Reproduce

```bash
pip install pandapower psutil numba pandas numpy faiss-cpu sentence-transformers statsmodels flask

# heavy corpus (idempotent, ~30 min first run)
python3 03_build_39_118.py && bash heavy_resume.sh

# paired pilot (API model; 560 calls, throttled)
GEMINI_API_KEY=... python3 -u 19_22_run_agents_gemini.py --real --force-api \
    --model gemini-3.5-flash-lite --rpm 15 --n-test 120

# local model (sequential calls; crash-resilient, resumes automatically)
python3 -u 30_local_pilot_resilient.py --model gemma-4-E4B-it-Q4_0.gguf \
    --interval 5 --n-test 120

# analysis + paper + figures (all numbers recomputed from logs)
python3 31_build_paper.py && python3 29_paper_figures.py
```

## Scope and known limitations

Pilot scale (140 scenarios, one prompt template, temperature 0, single runs — MDE ≈ 2–4 pp); labels and prompts derive from the same rule family; the state estimator is a closed-form approximation pending `pandapower.estimation.estimate`; the local model is 4-bit quantized (a conservative bias); the API tier is lightweight — results do not speak to frontier API models; transfer to IEEE-39/118 is planned, not performed; offline advisory prototype only. Full list: `REVIEW_RESPONSES.md`.

## License

MIT
