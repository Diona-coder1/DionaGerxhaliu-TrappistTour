#!/bin/bash
# extra large run for the final front (waits for the other batches)
cd "$(dirname "$0")/.."
R=results/runs
while pgrep -f "run_experiments.sh|rerun_small.sh" > /dev/null; do sleep 20; done
python3 -W ignore scripts/run_beam.py --width 300 --root-width 400 --seed 3 --de-G 90 --offsets 0 -20 --out $R/beam_w300_s3.pkl > $R/beam_w300_s3.log 2>&1
echo EXTRA DONE
# widest run (B = 1200, 8 cores, ~12 min); added for the improved final front
python3 -W ignore scripts/run_beam.py --width 1200 --root-width 1200 --seed 3 --de-G 90 --offsets 0 -20 --workers 8 --out $R/beam_w1200.pkl > $R/beam_w1200.log 2>&1
