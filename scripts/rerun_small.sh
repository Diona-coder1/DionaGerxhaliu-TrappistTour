#!/bin/bash
# re-run greedy and B=10 after the selection fix (waits for the main batch)
cd "$(dirname "$0")/.."
R=results/runs
while pgrep -f run_experiments.sh > /dev/null; do sleep 20; done
python3 -W ignore scripts/run_beam.py --width 1  --root-width 1  --out $R/greedy.pkl   > $R/greedy.log 2>&1
python3 -W ignore scripts/run_beam.py --width 10 --root-width 20 --out $R/beam_w10.pkl > $R/beam_w10.log 2>&1
echo SMALL DONE
