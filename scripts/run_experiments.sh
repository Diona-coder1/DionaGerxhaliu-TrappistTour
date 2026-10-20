#!/bin/bash
# All experiments of the report (about 2 hours on 2 cores).
set -x
cd "$(dirname "$0")/.."
R=results/runs
python3 -W ignore scripts/run_beam.py --width 300 --root-width 300 --seed 1 --out $R/beam_w300.pkl > $R/beam_w300.log 2>&1
python3 -W ignore scripts/run_beam.py --width 75  --root-width 120 --out $R/beam_w75.pkl  > $R/beam_w75.log 2>&1
python3 -W ignore scripts/run_beam.py --width 30  --root-width 60  --out $R/beam_w30.pkl  > $R/beam_w30.log 2>&1
python3 -W ignore scripts/run_beam.py --width 10  --root-width 20  --out $R/beam_w10.pkl  > $R/beam_w10.log 2>&1
python3 -W ignore scripts/run_beam.py --width 1   --root-width 1   --out $R/greedy.pkl    > $R/greedy.log 2>&1
python3 -W ignore scripts/run_beam.py --width 50  --root-width 80 --no-de     --out $R/abl_screen_only.pkl > $R/abl_screen_only.log 2>&1
python3 -W ignore scripts/run_beam.py --width 50  --root-width 80 --no-screen --out $R/abl_de_only.pkl     > $R/abl_de_only.log 2>&1
python3 -W ignore scripts/run_beam.py --width 50  --root-width 80             --out $R/beam_w50.pkl        > $R/beam_w50.log 2>&1
python3 -W ignore scripts/run_beam.py --width 150 --root-width 300 --offsets -40 -80 --seed 2 --out $R/beam_offsets.pkl > $R/beam_offsets.log 2>&1
for s in 0 1 2; do
  python3 -W ignore scripts/run_baselines.py --algo random --evals 100000 --seed $s > $R/random_s$s.log 2>&1 &
  python3 -W ignore scripts/run_baselines.py --algo nsga2  --evals 100000 --seed $s > $R/nsga2_s$s.log 2>&1
  wait
done
echo ALL DONE
