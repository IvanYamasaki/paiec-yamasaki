"""Writes formativo/RESULTADOS.md from subs.csv and metrics/*.json (aggregates only)."""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = ("B0", "B1", "B3", "B7", "B15", "B31")
DESC = {
    "v6": "linha A v6 (nota LLM ligada, mas sem cobertura no teste oculto)",
    "l0": "v7 λ=0 (= v6 sem nota)",
    "l3": "v7 λ=0,3 (mistura de níveis 10-95% no prior)",
    "l6": "v7 λ=0,6",
    "mean": "baseline oficial de média empírica",
    "J1": "linha J: v7 λ=0,3, nível centrado em p=0,5, variância 8x",
    "J2": "linha J mínimo: Beta do par em torno do nível dos outros sujeitos",
    "J3": "linha J híbrido: α·J2 + (1−α)·J1, α = 1/(1 + k + n_outros/20)",
    "J4": "linha J: J1 sem metadados do sujeito, com efeitos de item do Rasch",
}


def valid(m):
    """Finished, or Failed after scoring (infra error on Codabench's side) with the full
    scoring output: every budget present and the parsed mean equal to the leaderboard score."""
    if "brier_alc" not in m or not all("brier_" + b in m for b in B):
        return False
    if m.get("status") == "Finished":
        return True
    return "score_brier" in m and abs(m["score_brier"] - m["brier_alc"]) < 1e-5


def load():
    rows = list(csv.DictReader(open(os.path.join(HERE, "subs.csv"))))
    out = {}
    for r in rows:
        p = os.path.join(HERE, "metrics", f"{r['id']}.json")
        if os.path.exists(p):
            m = json.load(open(p))
            if valid(m):
                out.setdefault(r["arm"], []).append(m)
    return rows, out


def main():
    rows, data = load()
    L = ["# Formativo do Codabench (PAIEC 17828, fase 29785)", "",
         "Mantido por a33. Cada submissão sorteia pares novos do teste oculto; a métrica é a média simples",
         "do Brier ALC dos pares (igual à coluna `brier` do leaderboard). B0..B31 = média do Brier dos pares",
         "em cada orçamento. Submissões com status Failed ou presas em Running por erro de infraestrutura depois do scoring",
         "contam se o scoring output veio completo e bate com o score do leaderboard (marcadas com *). IC 95% = média ± 1,96·DP/√n. Só números agregados; nenhum ID do teste é guardado.", "",
         "| braço | o que é | n | brier_alc média | DP | IC 95% | ECE | " + " | ".join(B) + " | pares/sub |",
         "|---|---|---|---|---|---|---|" + "---|" * len(B) + "---|"]
    for arm in ("v6", "l0", "l3", "l6", "J1", "J2", "J3", "J4", "mean"):
        ms = data.get(arm, [])
        if not ms:
            continue
        x = np.array([m["brier_alc"] for m in ms])
        sd = x.std(ddof=1) if len(x) > 1 else float("nan")
        h = 1.96 * sd / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        bs = " | ".join(f"{np.mean([m['brier_' + b] for m in ms]):.4f}" for b in B)
        L.append(f"| {arm} | {DESC.get(arm, '')} | {len(x)} | {x.mean():.4f} | {sd:.4f} | "
                 f"[{x.mean() - h:.4f}, {x.mean() + h:.4f}] | {np.mean([m['ece_alc'] for m in ms]):.4f} | {bs} | "
                 f"{np.mean([m['n_pairs'] for m in ms]):.1f} |")
    L += ["", "## Submissões", "", "| braço | ID | enviada (UTC) | brier_alc | B0 | B31 | pares |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        p = os.path.join(HERE, "metrics", f"{r['id']}.json")
        m = json.load(open(p)) if os.path.exists(p) else {}
        f = lambda k: f"{m[k]:.4f}" if k in m else "-"
        star = "*" if m.get("status") != "Finished" and valid(m) else ""
        L.append(f"| {r['arm']} | {r['id']}{star} | {r['submitted_utc']} | {f('brier_alc')} | {f('brier_B0')} | "
                 f"{f('brier_B31')} | {m.get('n_pairs', m.get('status', 'pendente'))} |")
    open(os.path.join(HERE, "RESULTADOS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
