"""Fills the {{...}} placeholders of docs/templates/*.template.md with the
measured numbers (results/summary.md, results/final/final.pkl) and writes
docs/RAPORTI.md and README.md. The PDF is then built with scripts/build_pdf.sh."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trappist.hv import hypervolume

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results", "runs")


def load(name):
    with open(os.path.join(R, name), "rb") as fh:
        return pickle.load(fh)


def hv(name):
    d = load(name)
    return hypervolume([f[:2] for _, f in d["solutions"]]), d["seconds"]


with open(os.path.join(ROOT, "results", "final", "final.pkl"), "rb") as fh:
    FIN = pickle.load(fh)
P = np.array(sorted([f[:2] for _, f in FIN["final"]]))
HV = FIN["hv"]
n_sel = len(P)
dv_min, t_at_dvmin = P[0]
t_min = P[P[:, 0] < 2500, 1].min()
dv_at_tmin = P[P[:, 1] == t_min, 0][0]
LB = [("Space Coders", 5870436.818), ("fcmaes", 5731151.887), ("Team HRI", 5601395.897),
      ("haiven", 5503790.681), ("ai_research", 5502957.03), ("yuricst", 5421736.524)]
rank = 1 + sum(1 for _, s in LB if s > HV)

hv150, s150 = hv("beam_main.pkl")
hv300, s300 = hv("beam_w300.pkl")
hv10, s10 = hv("beam_w10.pkl")
hv50, s50 = hv("beam_w50.pkl")
hv_de, s_de = hv("abl_de_only.pkl")
widths = [(10, "beam_w10.pkl"), (30, "beam_w30.pkl"), (50, "beam_w50.pkl"), (75, "beam_w75.pkl"),
          (150, "beam_main.pkl"), (300, "beam_w300.pkl")]
W = np.array([w for w, _ in widths], dtype=float)
S = np.array([load(n)["seconds"] for _, n in widths])
k, c = np.polyfit(W, S, 1)
r2 = 1 - np.sum((S - (k * W + c)) ** 2) / np.sum((S - S.mean()) ** 2)
nsga_best = min(min(f[0] for _, f in load(f"nsga2_s{s}.pkl")["solutions"]) for s in range(3))
nsga_feas = np.mean([load(f"nsga2_s{s}.pkl").get("n_feasible") for s in range(3)])
rnd_feas = np.mean([load(f"random_s{s}.pkl").get("n_feasible") for s in range(3)])
rnd_best = min(min(f[0] for _, f in load(f"random_s{s}.pkl")["solutions"]) for s in range(3))

T = {}
T["REZULTATI_KRYESOR"] = (
    f"Pipeline-i i propozuar arrin hipervëllim **{HV:,.0f}** (rezultati në platformë: "
    f"$-{HV:,.0f}$) me {n_sel} zgjidhje, që do ta vendoste në **vendin e {rank}-të** të renditjes zyrtare "
    f"të SpOC 2022. Algoritmet gjenerike, me buxhet të krahasueshëm llogaritës, nuk gjejnë asnjë zgjidhje "
    f"brenda pikës referuese (hipervëllim 0).")

with open(os.path.join(ROOT, "results", "summary.md")) as fh:
    table = fh.read()
T["TABELA_REZULTATEVE"] = (
    "Tabela 2 përmbledh të gjitha ekzekutimet. Algoritmet e rastësishme janë ekzekutuar me 3 farë (*seeds*), "
    "beam search-i është determinist për një farë të dhënë.\n\n" + table.split("\n*")[0].rstrip() +
    "\n\n: Krahasimi i algoritmeve (HV = hipervëllimi, sa më i madh aq më mirë; "
    "T min = koha më e shkurtër ndër zgjidhjet me ΔV < 2500 m/s).\n")

T["DISKUTIMI_KRAHASIMIT"] = (
    f"* **Kërkimi i rastësishëm** gjen mesatarisht vetëm {rnd_feas:.0f} zgjidhje të lejuara në 100 000 "
    f"vlerësime, dhe më e mira ka $\\Delta V = {rnd_best/1000:,.1f}$ km/s, 13 herë mbi kufirin prej 2.5 km/s. "
    "Kjo tregon se zgjidhjet e mira zënë një pjesë jashtëzakonisht të vogël të hapësirës.\n"
    f"* **NSGA-II** gjen shumë zgjidhje të lejuara (mesatarisht {nsga_feas:,.0f}), por më e mira ka "
    f"$\\Delta V = {nsga_best/1000:,.1f}$ km/s, pra hipervëllimi mbetet 0. Operatorët gjenerikë (SBX, OX1) "
    "nuk mund t'i ndjekin luginat e ngushta të zgjidhjeve balistike: ndryshimi i një kohe me pak sekonda "
    "pa ndryshuar fly-by-n njëkohësisht prish gjithë turin (shih seksionin 4e). Edhe renditja e planetëve "
    "ndryshon kuptimin e të gjitha gjeneve të tjera, prandaj kryqëzimi i dy prindërve të mirë rrallë jep fëmijë të mirë.\n"
    "* **Greedy** (B = 1) dështon plotësisht: zgjedh hyrjen me shpejtësinë relative më të vogël (në dukje "
    "më e mira), por nga ajo gjendje asnjë hap i mëtejshëm nuk përshtatet brenda buxhetit prej 2500 m/s. "
    "Mungesa e kthimit prapa (*backtracking*) e çon në rrugë qorre.\n"
    f"* **Beam search** me vetëm B = 10 jep tashmë {hv10:,.0f} brenda {s10:.0f} s. Kjo konfirmon se "
    "dekompozimi sipas hapave, jo fuqia llogaritëse, është çelësi.\n")

T["DISKUTIMI_SHKALLEZIMIT"] = (
    f"Koha e ekzekutimit rritet **linearisht** me gjerësinë: përshtatja $t(B) \\approx {k:.2f}\\,B + {c:.0f}$ s "
    f"ka $R^2 = {r2:.3f}$, siç parashikon analiza teorike $O(n^2 B (m + W N_P G))$. Cilësia rritet në mënyrë "
    f"monotone por me kthim zvogëlues (*diminishing returns*): nga B = 150 në B = 300 koha dyfishohet "
    f"({s150/60:.0f} → {s300/60:.0f} min) ndërsa HV rritet me {100*(hv300-hv150)/hv150:.1f}%. "
    "Kjo është sjellja tipike e beam search-it: gjerësia kontrollon kompromisin mes cilësisë dhe kohës.")

T["DISKUTIMI_ABLACIONIT"] = (
    "Për të kuptuar rolin e secilit komponent të gjenerimit të fëmijëve, me B = 50:\n\n"
    "* **Vetëm Lambert-screening** (pa DE): **asnjë tur i plotë.** Modeli me DSM te planeti është shumë "
    "pesimist kur fly-by nuk mund ta kthejë shpejtësinë mjaftueshëm (planete të lehta ose shpejtësi relative "
    "e madhe): atëherë DSM duhet bërë larg, pranë apoapsit, ku kushton shumë më pak. Në këto raste kostoja "
    "e vlerësuar ishte dhjetëra km/s, kurse në modelin e saktë 60–130 m/s.\n"
    f"* **Vetëm DE** (pa screening): HV = {hv_de:,.0f} në {s_de:.0f} s, që është {100*(hv50-hv_de)/hv50:.1f}% "
    f"më pak se hibridi ({hv50:,.0f}). DE vetë gjen hapa të mirë, por zgjidhjet balistike janë minimume "
    "shumë të ngushta që DE i gjen vetëm rastësisht.\n"
    "* **Hibridi** kombinon të dyja: screening jep pika nisjeje balistike (0 m/s) për DE, dhe DE trajton "
    "rastet ku duhet DSM në hapësirë të thellë.\n\n"
    "Përfundimi i ablacionit: komponentët plotësojnë njëri-tjetrin, dhe DE në modelin e saktë është "
    "komponenti i domosdoshëm.")

T["REZULTATI_FINAL"] = (
    f"Pipeline-i final bashkon turet e plota nga të gjitha ekzekutimet e beam search-it, i lustron me CMA-ES "
    f"dhe zgjedh me programim dinamik nënbashkësinë optimale. Rezultati:\n\n"
    f"* **Hipervëllimi: {HV:,.0f}** (rezultati në Optimize: $-{HV:,.3f}$), me {n_sel} zgjidhje "
    f"(fronti ka më pak se 100 pika jo-të-dominuara, prandaj përdoren të gjitha).\n"
    f"* Zgjidhja më ekonomike: $\\Delta V = {dv_min:.1f}$ m/s për {t_at_dvmin:.0f} ditë.\n"
    f"* Zgjidhja më e shpejtë: {t_min:.0f} ditë për $\\Delta V = {dv_at_tmin:.0f}$ m/s.\n"
    f"* Krahasuar me renditjen zyrtare të SpOC 2022 (1. Space Coders 5 870 437; 2. fcmaes 5 731 152; "
    f"3. Team HRI 5 601 396; 4. haiven 5 503 791), rezultati do të ishte në **vendin e {rank}-të** nga 20 ekipe.\n"
    "* Çdo vektor në `results/final/submission.json` është rivlerësuar me funksionin e fitnesit dhe i "
    "plotëson të dy kufizimet.\n\n"
    "Turet e mira kanë të njëjtën strukturë: hyrje me $T_0 \\approx 1500$ ditë dhe $\\Delta V_{init} \\approx 0$; "
    "fly-by i parë kap sondën në një orbitë me periodë 120–200 ditë; fly-by-t e mëpasshëm e zvogëlojnë "
    "periodën hap pas hapi (p.sh. 158 → 79 → 53 → 45 → 34 ditë), dhe hapat e fundit zgjasin vetëm 20–35 ditë. "
    "Planeti c (i dyti nga ylli) shfaqet shpesh si planet i parë: ka masë të madhe dhe është afër yllit, "
    "kështu që fly-by i tij e ndryshon shumë orbitën.")

T["PERFUNDIME"] = (
    "1. **Analiza e strukturës ia vlen më shumë se fuqia llogaritëse.** Hyrja pothuajse e ndarë, "
    "kostot në formë 'V' dhe luginat e ngushta shpjegojnë pse algoritmet gjenerike dështojnë dhe "
    "drejtuan dizajnin e algoritmit kryesor.\n"
    "2. **Dekompozimi + beam search** e kthen një problem 34-dimensional të përzier në një varg "
    "nënproblemesh 4-dimensionale, me kosto lineare në gjerësinë $B$ dhe polinomiale në numrin e planetëve, "
    "në vend të $O(n!)$.\n"
    "3. **Greedy dështon** sepse vendimet e hershme (hyrja, fly-by i parë) përcaktojnë se çfarë është e "
    "mundur më vonë; beam search me *lookahead* zgjidh këtë me kosto të kontrolluar.\n"
    "4. **Algoritmet ndihmëse** kanë rol të qartë: Lambert-screening (shpejtësi), DE (saktësi në modelin "
    "real), CMA-ES (kondicionimi i keq), programimi dinamik (zgjedhja optimale, $O(kn^2)$, e provuar me "
    "kërkim shterues).\n"
    "5. **Vërtetimi i modelit** me integrim numerik të pavarur na lejoi të punojmë pa bibliotekat zyrtare "
    "dhe zbuloi veçori të kodit zyrtar (p.sh. periapsi negativ për hiperbola) që ndikojnë drejtpërdrejt në hapësirën e zgjidhjeve.\n\n"
    "**Punë e ardhshme.** (i) Beam më i gjerë ose disa ekzekutime paralele (cilësia rritet me $B$); "
    "(ii) *lookahead* më i saktë, p.sh. kosto minimale e vlerësuar e hapave të mbetur; "
    "(iii) optimizim lokal që ruan strukturën (ndryshim i njëkohshëm i kohës dhe i fly-by-t) për të "
    "mbushur boshllëqet e frontit; (iv) implementim në C/Numba për 10–50× shpejtësi.")

README_RESULT = (
    f"| | Hipervëllimi | Vendi në renditjen e SpOC 2022 |\n|---|---:|---:|\n"
    f"| **Ky projekt** (beam search + CMA-ES + DP) | **{HV:,.0f}** | **{rank}** / 20 |\n"
    f"| NSGA-II, 100k vlerësime | 0 | – |\n| Kërkim i rastësishëm, 100k vlerësime | 0 | – |\n\n"
    f"Zgjidhja më ekonomike: ΔV = {dv_min:.1f} m/s për {t_at_dvmin:.0f} ditë; më e shpejta: {t_min:.0f} ditë "
    f"për ΔV = {dv_at_tmin:.0f} m/s. Dorëzimi: [`results/final/submission.json`](results/final/submission.json).")


def fill(src, dst, mapping):
    with open(src) as fh:
        s = fh.read()
    for key, val in mapping.items():
        s = s.replace("{{" + key + "}}", val)
    assert "{{" not in s, [l for l in s.splitlines() if "{{" in l]
    with open(dst, "w") as fh:
        fh.write(s)


fill(os.path.join(ROOT, "docs", "templates", "RAPORTI.template.md"), os.path.join(ROOT, "docs", "RAPORTI.md"), T)
fill(os.path.join(ROOT, "docs", "templates", "README.template.md"), os.path.join(ROOT, "README.md"),
     {"README_RESULT": README_RESULT})
print("HV", HV, "rank", rank, "fit", k, c, r2)
