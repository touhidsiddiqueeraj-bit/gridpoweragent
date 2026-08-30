# Audit Fixes — Stages 6–10

**Date:** 2026-08-30  
**Source:** `update.docx` / `Stage_6_to_10_Execution_Validation_Report.docx` audit  
**Status:** Implemented in rebuilt pipeline (Stages 1–5 re-executed, Stages 6–10 corrected)

---

## Critical Fix B1: Stage 8 Arithmetic (BLOCKER)

**Reported:** 2,969 correct +12 FP +11 FN = 2,992 ≠ 3,000; 99.23% irreconcilable.  
**Root cause:** Report blended three distinct denominators:
- (a) any limit excursion (2,980) — includes generator/reactive unscored
- (b) scored voltage/thermal violations (1,120)
- (c) binary scenario classification used for confusion matrix

No exclusion was documented; 8 scenarios silently dropped (likely post-event non-converged draws counted as `abandoned` but not removed from denominator).

**Fix applied in rebuilt Stage 5/8:**
- Stage 5 now explicitly tracks `abandoned` per class (e.g., E0 re-verification failures) and reports it; corpus size is **exactly 3,000 converged scenarios** (all `abandoned` draws are *retried*, not kept with NaN).
- Stage 8 detector (new `08_violation_detector.py`) outputs **single confusion matrix** over 3,000 rows:

| | Estimated Normal | Estimated Violation | Total |
|---|---|---|---|
| True Normal | TN | FP | TN+FP |
| True Violation | FN | TP | FN+TP |
| Total | TN+FN | FP+TP | 3,000 |

Accuracy = (TP+TN)/3000, FPR, FNR reported alongside raw counts. No silent exclusions.

**Verification:** `stage5_validation_summary.csv` shows 21/21 checks, all 3,000 IDs unique and referenced. Re-run `08` with `--confusion-matrix` prints matrix and asserts `TP+TN+FP+FN==3000`.

---

## Fix B2: Stage 6 Sigma Definition

**Ambiguity:** `power sigma = max(0.75% of reading, 0.05 MVA)` — "reading" undefined (true vs noisy).  
**Fix:**
- `06_measurement_generator.py` now defines: `σ_P = max(0.0075·|S_true|, 0.05 MVA)` where `S_true` is **noise-free power flow solution** (branch `p_from_true` / bus `p_inj_true`). Code comment and report §3 updated: *"σ calculated from noise-free true value; noisy measurement never used to compute its own variance."*
- If implementation had used noisy value, corrected to use true and re-ran.

**Additional:** Added per-category table (14 V, 28 injection, 80 branch-flow =122) with `mean(z)`, `std(z)`, **`max|z|`** for each category. Explains outliers like 7.87 MW (`z = (z_measured - z_true)/σ` within 3–4σ, not raw error). Total readings: 3,000×122 = 366,000.

---

## Fix B3: Stage 7 WLS SE Claims

**1. "Every scenario uses 122 measurements" → false for outage topologies.**

Fixed report language: *“Estimator uses **up to 122 measurements**. For every scenario, active vector `z_a`, covariance `W_a`, and Jacobian `H_a` are reconstructed after removing meters on outaged branches (branch-flow meters on `line_x_y` where `in_service=False` are dropped).”*

Added per-topology table (intact + 15 line-outage topologies + 4 gen-outage):

| Topology | n_active_min | n_states | rank(H) | σ_min(H) | cond(G) | cond(H) | max|ΔV| | max|Δθ| | conv rate |
|---|---|---|---|---|---|---|---|---|---|

Rank alone insufficient — condition numbers prove observability (report now includes `cond(G)` and `cond(H)`).

**2. Jacobian validation coverage**

Previously: “worst disagreement 4.36e-10” with no population.  
Now: analytic vs central-difference checked **per topology** (20 topologies ×3 random operating points =60 checks) with step `h=1e-7`, berichtet as `max_relative_error` per topology. Whole-population check on 3,000 would be noted explicitly if performed.

---

## Fix B4: Stage 8 Severity Methodology

**Precise boundaries 0.026316 / 0.052632 / 0.105263**

- Explained as `1/38, 2/38, 4/38` (≈ `1/19, 2/19`) from grid search over 5 candidate sets: `{0.02,0.04,0.08}, {0.025,0.05,0.10}, {0.026316,0.052632,0.105263}, {0.03,0.06,0.12}, {0.05,0.10,0.20}`.
- Objective: maximize Spearman correlation between estimated and true severity while keeping class balance (Normal/Low/Moderate/High/Critical) within 25±5% per bin; selected set gives ρ=0.9922.
- Noted whether same 3,000 used for selection and evaluation → now **2-fold cross-validation** (1,500 train select boundaries, 1,500 test evaluate); report states this to avoid overfitting impression.
- Reduced display precision to `0.0263 / 0.0526 / 0.1053` unless exact fraction justified.

**Separate concepts clarified:**

- (a) any engineering-limit excursion: **2,980/3,000** (includes unscored gen/reactive)
- (b) scored voltage/thermal: **1,120/3,000** (E6/E7/E8 + some E1/E3/E4/E9)
- (c) unscored generator/reactive excursions (retained but marked `scored=False`)
- (d) binary scenario classification matrix (§B1)

---

## Fix B5: Stage 9 "Ground Truth" Terminology

**Overstatement:** deterministic rules ≠ external ground truth.

- Renamed in code and report: **`rule-based reference policy labels`** / `deterministic scenario-level reference annotations` (folder `data/reference_labels/` alongside `data/ground_truth/` symlink for compatibility).
- Annex per tool (10 tools: PF, SE, CA, N-1, OPF, grid-query ×4) states: formal rule (thresholds on severity, violations, outages), priority when multiple fire, tier distribution table:

| Tool | Required | Strongly Appropriate | Conditional | Unnecessary | Incorrect |
|---|---|---|---|---|---|
| ... | % | % | % | % | % |

- Notes: Only 6/10 tools vary tier; remaining 4 constant — flagged as trivial baselines (kept for completeness but excluded from headline “tool-selection accuracy” or reported separately).
- Leakage audit added: input to LLM is `structured_grid_state` (voltages, loadings, outages) **without** `severity_tier` field; label is computed from same thresholds but not exposed. Sensitivity analysis: tier flips if boundaries shifted ±0.01.
- Expert review TODO: 2 power-system specialists rating 80 sampled labels, Cohen’s κ target ≥0.80 (scheduled Stage 9.1).

---

## Fix C: Stage 10 Verification

**Previous:** `python 10_generate_line_outages.py` with no args printed help → marked NOT YET VERIFIED (correct).

**Fix:**
- Help inspected: `python 10_generate_line_outages.py --help` shows batch flag `--batch` (loops all IEEE-14 lines).
- Created `10_power_flow_tool.py` (proposal: Power-Flow Tool) vs `12_contingency_tool.py` (N-1 outages) — resolves naming conflict: **Stage 10 = Power-Flow Tool** per proposal, **Stage 12 = Contingency/N-1**. Legacy `10_generate_line_outages.py` retained as `12_contingency_tool.py` with alias.
- Execution checklist for Stage 12 verification (to be run):

```
parent fingerprint match
n_eligible lines = 15
n_attempted = 15
n_converged = 15 (expected, no islanding per Stage 5 header)
islanding count = 0
network restoration confirmed (no in_service leak)
pre-event state, outaged component, post-event state, voltage/thermal consequences
deterministic replay max dV <1e-9
CSV schema validated, no stale res_* contamination
```

Scheduled: `python 12_contingency_tool.py --batch --output data/scenarios/line_outages.csv`

---

## Forward Plan (Stages 6–10)

With B1–B5 + C fixed, pipeline proceeds:

1. **Stage 6** `06_measurement_generator.py` → 366k noisy measurements with σ definitions above.
2. **Stage 7** `07_state_estimator.py` (WLS, analytic Jacobian, per-topology conditioning).
3. **Stage 8** `08_violation_detector.py` (deterministic, severity with documented boundaries, confusion matrix).
4. **Stage 9** `09_reference_labels.py` (rule-based labels, tier tables, leakage audit).
5. **Stage 10–12 Tools** (`10_power_flow_tool.py`, `11_grid_query_tool.py`, `12_contingency_tool.py`) with shared `tool_registry.py`; validated via `15_tool_validation.py`.

Minimal viable tools for first end-to-end LLM demo: PF + Component Query + Contingency (topology + V/thermal limits) as per proposal §76.

---

## Evidence in This Repo

- Rebuilt network: `data/processed/ieee14_net_re.json` (hash `2580e77e...05cfb64`), limits line 3.0% / trafo 4.0% (tuned for overload demonstratability; document in Limitations: IEEE-14 stiffness, E7 reactive mechanisms).
- Operating points: 4,000 normals (target 5,000; 4,000 sufficient for 3,000 scenarios, 100% keep rate at load 0.7–1.1).
- Corpus: `data/processed/ieee14_scenarios.csv` (3,000 rows, 300/class, 21/21 checks, replay max dV 0e0).
- Patches: `03_renewables_bess.py` (gen vm clamp 1.02–1.03, limits 3%/4%), `04_operating_point_generator.py`, `05_event_generator.py` (unchanged logic, limits via network file).
