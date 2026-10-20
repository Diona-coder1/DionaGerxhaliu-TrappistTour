# Trappist Tour – Dizajni dhe Analiza e Algoritmeve

**Autore:** Diona Gerxhaliu · Universiteti i Prishtinës · lënda *Dizajni dhe Analiza e Algoritmeve*

Projekt për problemin **[SpOC: Trappist Tour](https://optimize.esa.int/challenge/spoc-trappist-tour/About)**
të Agjencisë Hapësinore Evropiane (ESA, GECCO 2022): një sondë hyn në sistemin TRAPPIST-1 dhe duhet
të vizitojë të 7 planetët, duke minimizuar njëkohësisht **karburantin (ΔV)** dhe **kohën e misionit**.
Rezultati vlerësohet me **hipervëllimin** e frontit Pareto (pika referuese [2500 m/s, 4000 ditë]).

📄 **Raporti i plotë:** [`docs/RAPORTI.pdf`](docs/RAPORTI.pdf) (burimi: [`docs/RAPORTI.md`](docs/RAPORTI.md))

## Rezultati

{{README_RESULT}}

![Fronti Pareto](figures/fronti_pareto.png)

## Algoritmet

| # | Algoritmi | Ideja | Kompleksiteti |
|---|---|---|---|
| A1 | Kërkim i rastësishëm | bazë, mostra uniforme | O(N) vlerësime |
| A2 | NSGA-II | evolucionar shumë-objektiv gjenerik (SBX + OX1) | O(N) vlerësime + O(MN²)/gjeneratë |
| A3 | Greedy | turi ndërtohet hap pas hapi, zgjedhja më e mirë lokale | O(n²) zgjerime |
| A4 | **Beam search shumë-objektiv** | dekompozim sipas hapave; Lambert-screening + Differential Evolution; seleksion Pareto me *lookahead* | O(n²·B·(m + W·NP·G)) |
| A5 | CMA-ES | lustrim lokal i tureve të plota (luginat e ngushta) | O(E·d²) |
| A6 | Programim dinamik | nënbashkësia optimale me ≤ 100 pika (hipervëllim maksimal) | O(k·n²) |

## Struktura

```
official/trappist_tour_official.py   kodi zyrtar i ESA-s (i paprekur, kërkon pykep)
trappist/astro.py                    Lambert (Izzo), propagim Kepler, fly-by  – port i pykep
trappist/problem.py                  funksioni i fitnesit, rresht për rresht si ai zyrtar
trappist/vec.py, lambert_vec.py      versione të vektorizuara (NumPy) për kërkimin
trappist/mga.py                      modeli i shpejtë Lambert + fly-by analitik
trappist/legopt.py                   Differential Evolution për një hap (i vektorizuar)
trappist/beam.py                     A3/A4 – beam search shumë-objektiv
trappist/baselines.py                A1/A2 – kërkim i rastësishëm, NSGA-II
trappist/local.py                    A5 – CMA-ES
trappist/hv.py                       hipervëllimi + A6 (programim dinamik)
trappist/plotting.py                 figurat (pa pykep)
scripts/                             ekzekutimi i eksperimenteve, figurat, verifikimi
tests/                               verifikimi i modelit fizik me integrim numerik
results/final/submission.json        zgjidhja për platformën Optimize (≤ 100 vektorë)
```

## Si ekzekutohet

```bash
pip install -r requirements.txt
python -m pytest -q tests                                  # verifikimi (~10 s)
python scripts/verify_submission.py results/final/submission.json
python scripts/run_beam.py --width 150                     # A4, ~13 min në 2 bërthama
python scripts/run_baselines.py --algo nsga2 --seed 0      # A2
python scripts/make_final.py results/runs/beam_*.pkl --out results/final
python scripts/make_figures.py
```

Verifikimi me kodin origjinal të ESA-s (opsional):

```bash
conda install -c conda-forge pykep pygmo
python scripts/verify_submission.py results/final/submission.json --official
```

---

### English summary

Multi-objective optimisation of the ESA SpOC 2022 *Trappist Tour* problem for a university course on
algorithm design and analysis. The pykep routines used by the official evaluator are re-implemented
in NumPy and validated against numerical integration. The main method is a multi-objective beam
search that decomposes the tour leg by leg (Lambert screening + vectorised Differential Evolution per
leg, Pareto-aware selection with an A*-like lookahead), followed by CMA-ES polishing and an exact
dynamic-programming selection of the 100 solutions with maximal hypervolume. Generic baselines
(random search, NSGA-II, greedy) and ablations are included for comparison.
