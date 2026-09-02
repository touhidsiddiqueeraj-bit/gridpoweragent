#!/usr/bin/env bash
# local_stack_restart.sh — bring the local Gemma evaluation stack back up after
# a reboot / crash, and relaunch the taxonomy-v2 pilot run if it is not running.
#
# Manual use (after a machine restart):
#   bash /home/touhid/Documents/llmpaper/local_stack_restart.sh
#
# Optional fully-automatic restart after reboot — add a crontab entry:
#   crontab -e
#   @reboot sleep 90 && bash /home/touhid/Documents/llmpaper/local_stack_restart.sh >> /home/touhid/Documents/llmpaper/local_restart.log 2>&1
#
# Stack: PolarisStudio (AppImage, GUI proxy on :9090) -> spawns llama-server on
# :8080 with gemma-4-E4B-it-Q4_0.gguf. The run harness (30_local_pilot_resilient.py)
# is itself crash-resilient: it health-polls the engine forever and resumes from
# its output CSV — this script only needs to (re)start the processes.
set -u
REPO=/home/touhid/Documents/llmpaper
POLARIS=/home/touhid/.local/bin/polarisstudio
LLAMA_SERVER=$HOME/llama.cpp/build/bin/llama-server
GGUF=/mnt/backup/llm-models/gemma-4-E4B-it-Q4_0.gguf

# --- pilot run configuration (taxonomy v2, IEEE-14 pilot) ---------------------
RUN_ARGS="--scenarios-csv data/processed/ieee14_scenarios_taxonomy2.csv --labels-csv data/processed/ieee14_reference_labels_taxonomy2.csv --ids-from data/results/agent_runs_gemini-3.5-flash-lite.csv --out data/results/agent_runs_gemma-4-E4B-it-Q4_0_gguf_tax2.csv --interval 5 --n-test 140"

engine() { curl -s -m 3 http://127.0.0.1:9090/v1/models >/dev/null 2>&1; }

echo "== local_stack_restart $(date '+%F %T') =="

# 1) engine (polaris 9090, which owns llama-server 8080)
if engine; then
    echo "[OK] engine already up on :9090"
else
    echo "[..] engine down — launching polarisstudio (spawns llama-server :8080)"
    if [[ -x "$POLARIS" ]]; then
        setsid nohup "$POLARIS" >/dev/null 2>&1 < /dev/null &
    fi
    for i in $(seq 1 24); do   # wait up to 120 s
        sleep 5
        engine && break
        # fallback: start llama-server directly if polaris did not come up
        if [[ $i -eq 8 ]] && ! curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
            echo "[..] polaris slow/absent — starting llama-server directly on :8080"
            setsid nohup "$LLAMA_SERVER" -m "$GGUF" --port 8080 --host 0.0.0.0 \
                -c 8192 -ngl 99 -t 4 -tb 4 -b 2048 -ub 512 -fa on -np 1 \
                --cont-batching --no-webui > "$REPO/llama_server.log" 2>&1 < /dev/null &
        fi
    done
    engine && echo "[OK] engine up on :9090" || { echo "[FAIL] engine still down — aborting (run harness will keep polling)"; }
fi

# 2) pilot harness (relaunches only if not already running; resumes from CSV)
cd "$REPO" || exit 1
if pgrep -f 30_local_pilot >/dev/null 2>&1; then
    echo "[OK] pilot harness already running ($(pgrep -f 30_local_pilot | tr '\n' ' '))"
else
    if engine; then
        echo "[..] launching local pilot harness"
        setsid nohup python3 -u 30_local_pilot_resilient.py \
            --model gemma-4-E4B-it-Q4_0.gguf $RUN_ARGS \
            > local_tax2.log 2>&1 < /dev/null &
        echo "[OK] harness started (pid $!) -> local_tax2.log"
    else
        echo "[SKIP] harness not started (engine down) — re-run this script once the engine is up"
    fi
fi

# 3) status
ROWS=$(wc -l < data/results/agent_runs_gemma-4-E4B-it-Q4_0_gguf_tax2.csv 2>/dev/null || echo 0)
echo "[STATUS] rows in tax2 CSV (incl. header): $ROWS / 561 target"
echo "[STATUS] monitor: tail -f $REPO/local_tax2.log  |  bash $REPO/gemma_progress.sh"
