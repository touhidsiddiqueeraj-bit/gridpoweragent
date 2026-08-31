#!/usr/bin/env bash
# Live progress for the Gemma extended run: bash gemma_progress.sh
CSV="data/results/agent_runs_gemma-4-E4B-it-Q4_0_gguf.csv"
TOTAL=560
read E1 E2 E3 E4 <<< $(python3 - <<PYEOF
import pandas as pd
try:
    df=pd.read_csv("$CSV")
    s=df.groupby("config").size()
    print(*(s.get(c,0) for c in ["E1_LLM","E2_LLM_RAG","E3_LLM_Tools","E4_Full"]))
except Exception:
    print(0, 0, 0, 0)
PYEOF
)
DONE=$((E1+E2+E3+E4))
PCT=$(python3 -c "print(f'{$DONE/$TOTAL*100:.1f}')")
BARLEN=30
FILL=$(python3 -c "print(int($DONE/$TOTAL*$BARLEN))")
BAR=$(python3 -c "print('█'*$FILL + '░'*($BARLEN-$FILL))")
ALIVE=$(pgrep -f 30_local_pilot >/dev/null && echo "RUNNING" || echo "STOPPED")
ENGINE=$(curl -s -m 3 http://127.0.0.1:9090/v1/models >/dev/null && echo "up" || echo "DOWN")
ETA=$(python3 -c "
rem=$TOTAL-$DONE
print(f'~{rem*41/3600:.1f}h' if rem>0 else 'done')" 2>/dev/null)
echo "Gemma-4-E4B extended run  [$BAR] $DONE/$TOTAL ($PCT%)  ETA $ETA"
echo "  E1 $(printf %3d $E1)/140 | E2 $(printf %3d $E2)/140 | E3 $(printf %3d $E3)/140 | E4 $(printf %3d $E4)/140   harness:$ALIVE  engine:$ENGINE"
tail -1 gemma_pilot.log 2>/dev/null | cut -c1-100
