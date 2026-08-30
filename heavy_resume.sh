#!/usr/bin/env bash
set -u
# heavy_resume.sh — Resumable headless heavy run (non-LLM)
# Skips completed outputs; safe to re-run after restart. LLM stages remain gated.
# Usage: bash heavy_resume.sh [--dry-run] [--case case39|case118|all]
# Headless: nohup bash heavy_resume.sh > heavy.log 2>&1 &  tail -f heavy.log

DRY_RUN=""
CASE="all"
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN="1"; echo "[DRY-RUN]"; shift
fi
if [[ "${1:-}" == "--case" ]]; then CASE="$2"; shift 2; fi

exists_and_valid() { [[ -f "$1" ]] && [[ $(wc -l < "$1") -gt 1 ]]; }

run_or_skip() {
  local desc="$1" check_file="$2" min_rows="$3" cmd="$4"
  if [[ -f "$check_file" ]] && [[ $(wc -l < "$check_file") -ge $min_rows ]]; then
    echo "[SKIP] $desc — $check_file has $(wc -l < "$check_file") rows ≥ $min_rows"
  else
    echo "[RUN] $desc"
    if [[ -n "$DRY_RUN" ]]; then echo "  would run: $cmd"; else eval "$cmd"; fi
  fi
}

echo "=== Heavy Resume $(date -Iseconds) CASE=$CASE ==="
echo "PWD=$(pwd)  Python=$(python3 -c 'import pandapower; print(pandapower.__version__)')"

# Stage 03 — nets (if missing)
if [[ ! -f data/processed/case39_net_re.json ]]; then
  echo "[RUN] Building case39/118 nets"
  [[ -n "$DRY_RUN" ]] || python3 03_build_39_118.py
else echo "[SKIP] case39/118 nets exist"; fi

# Stage 04 heavy OP — idempotent per case
for c in case39 case118; do
  if [[ "$CASE" != "all" && "$CASE" != "$c" ]]; then continue; fi
  N=5000; [[ "$c" == "case118" ]] && N=7000
  NEED=$((N+1)) # header + N rows
  run_or_skip "04 OP $c $N" "data/processed/${c}_operating_points.csv" "$NEED" "python3 04_heavy_op.py $c $N"
done

# Stage 05 heavy scenarios — generic heavy generator (to be implemented as 05_heavy.py)
# Currently 05_event_generator.py is hardcoded to ieee14. For heavy 39/118 we need 05_heavy.py.
# If missing, run per-case with patched 05 (temporary: reuse 05 logic via python snippet)
for c in case39 case118; do
  if [[ "$CASE" != "all" && "$CASE" != "$c" ]]; then continue; fi
  N=500; [[ "$c" == "case118" ]] && N=700 # 500/class for 39 (10 classes) -> 5000, 700/class for 118 ->7000
  OUT="data/processed/${c}_scenarios.csv"
  NEED_SCEN=$((N*10+1))
  if exists_and_valid "$OUT" && [[ $(wc -l < "$OUT") -ge $NEED_SCEN ]]; then
    echo "[SKIP] 05 scenarios $c — $OUT exists"
  else
    echo "[RUN] 05 scenarios $c (needs 05_heavy.py --case $c --n $N)"
    if [[ -n "$DRY_RUN" ]]; then echo "  would run heavy scenario gen for $c"; 
    else 
      if [[ -f "05_heavy.py" ]]; then python3 -u 05_heavy.py --case "$c" --n-per-class "$N"
      else echo "[WARN] 05_heavy.py not yet implemented — skipping $c scenarios (create it next)"; fi
    fi
  fi
done

# Stage 06-09 heavy — measurements / SE / violations / labels (now case-aware, synthetic WLS ~0.0007)
for c in case39 case118; do
  if [[ "$CASE" != "all" && "$CASE" != "$c" ]]; then continue; fi
  # 06 measurements
  if [[ -f "data/processed/${c}_scenarios.csv" ]]; then
    run_or_skip "06 measurements $c" "data/processed/${c}_measurements.csv" 100 "python3 -u 06_measurement_generator.py --case $c"
    run_or_skip "07 SE $c" "data/processed/${c}_state_estimates.csv" 100 "python3 -u 07_state_estimator.py --case $c"
    run_or_skip "08 violation $c" "data/processed/${c}_violation_severity.csv" 100 "python3 -u 08_violation_detector.py --case $c"
    run_or_skip "09 labels $c" "data/processed/${c}_reference_labels.csv" 100 "python3 -u 09_reference_labels.py --case $c"
  else
    echo "[SKIP] 06-09 $c — scenarios not yet existent"
  fi
done

# Stage 15 / 27-28 heavy re-run (lightweight, always after 06-09)
if [[ -z "$DRY_RUN" ]]; then
  echo "[RUN] Stage 15 heavy validation (best-effort)"
  python3 15_tool_validation.py || echo "[WARN] Stage15 failed (non-blocking)"
  echo "[RUN] Stage 27 stats + 28 figs (best-effort)"
  python3 27_statistical_analysis.py || echo "[WARN] Stage27 failed"
  python3 28_generate_figures.py || echo "[WARN] Stage28 failed"
else
  echo "[DRY-RUN] would run: python3 15_tool_validation.py; python3 27_...; python3 28_..."
fi

# LLM gated warning
echo "[GATED] Stages 19-22 real LLM remain gated — need OPENAI_API_KEY or Ollama. Mocked results in data/results/agent_runs.csv retained."

echo "=== Resume done $(date -Iseconds) ==="
echo "Next: bash heavy_resume.sh --dry-run to verify, then bash heavy_resume.sh for real"
