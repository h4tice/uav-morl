"""Statistical analysis for the paper: significance tests and paired
weight-effect analysis on evaluation CSVs.

Usage:
    # Yontem karsilastirmasi (Welch t-testi + Mann-Whitney U):
    python -m src.analyze --compare results/metrics/ppo_hard_v3_episodes.csv results/metrics/apf_hard_episodes.csv results/metrics/astar_hard_episodes.csv

    # Pareto agirlik etkisi (ayni dunyalar uzerinde ESLESTIRILMIS analiz):
    python -m src.analyze --pareto results/metrics/ppo_hard_v3_pareto.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def compare_methods(csv_paths: list[str]) -> None:
    frames = {Path(p).stem.replace("_episodes", ""): pd.read_csv(p) for p in csv_paths}
    print("\n=== YONTEM OZETI ===")
    for name, df in frames.items():
        succ = df[df["success"]]
        pool = succ if len(succ) else df
        print(f"{name:20s} SR={df['success'].mean():.3f} CR={df['collision'].mean():.3f} "
              f"E={pool['energy_j'].mean():.0f}±{pool['energy_j'].std():.0f} J "
              f"L={pool['path_length_m'].mean():.1f} m "
              f"clr={pool['min_clearance_m'].mean():.2f} m (n={len(df)})")

    names = list(frames)
    print("\n=== IKILI TESTLER (basarili bolumler, enerji) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = frames[names[i]]; b = frames[names[j]]
            ea = a[a["success"]]["energy_j"].values
            eb = b[b["success"]]["energy_j"].values
            if len(ea) < 5 or len(eb) < 5:
                print(f"{names[i]} vs {names[j]}: yetersiz basarili bolum"); continue
            t, p_t = stats.ttest_ind(ea, eb, equal_var=False)
            u, p_u = stats.mannwhitneyu(ea, eb)
            print(f"{names[i]} vs {names[j]}: dE={ea.mean()-eb.mean():+.0f} J | "
                  f"Welch p={p_t:.4f} | Mann-Whitney p={p_u:.4f}")

    print("\n=== IKILI TESTLER (basari orani, iki-oranli z / Fisher) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = frames[names[i]]["success"]; b = frames[names[j]]["success"]
            table = [[int(a.sum()), int(len(a) - a.sum())],
                     [int(b.sum()), int(len(b) - b.sum())]]
            _, p = stats.fisher_exact(table)
            print(f"{names[i]} (SR={a.mean():.2f}) vs {names[j]} (SR={b.mean():.2f}): "
                  f"Fisher p={p:.4f}")


def pareto_paired(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    keys = ["w_goal", "w_energy", "w_safety"]
    weights = df[keys].drop_duplicates().values.tolist()
    print(f"\n=== PARETO ESLESTIRILMIS ANALIZ ({Path(csv_path).name}) ===")
    print("Ayni 'episode' indeksi = ayni dunya (tohum), agirliklar arasi dogrudan kiyas.\n")

    # Ozet tablo
    for w in weights:
        sub = df[(df.w_goal == w[0]) & (df.w_energy == w[1]) & (df.w_safety == w[2])]
        print(f"w={w}: SR={sub['success'].mean():.2f} "
              f"E={sub['energy_j'].mean():.0f} J L={sub['path_length_m'].mean():.1f} m "
              f"clr={sub['min_clearance_m'].mean():.2f} m")

    # En uc iki agirlik: en hedef-agirlikli vs en enerji-agirlikli
    w_goal_heavy = max(weights, key=lambda w: w[0])
    w_energy_heavy = max(weights, key=lambda w: w[1])
    a = df[(df.w_goal == w_goal_heavy[0]) & (df.w_energy == w_goal_heavy[1]) &
           (df.w_safety == w_goal_heavy[2])].set_index("episode")
    b = df[(df.w_goal == w_energy_heavy[0]) & (df.w_energy == w_energy_heavy[1]) &
           (df.w_safety == w_energy_heavy[2])].set_index("episode")
    common = a.index.intersection(b.index)
    # Her iki agirlikta da BASARILI olan ayni dunyalar
    both = [ep for ep in common if a.loc[ep, "success"] and b.loc[ep, "success"]]
    print(f"\nUc karsilastirma: goal-agirlikli {w_goal_heavy} vs enerji-agirlikli {w_energy_heavy}")
    print(f"Ortak dunya sayisi: {len(common)}, her ikisinde basarili: {len(both)}")
    if len(both) >= 10:
        de = a.loc[both, "energy_j"].values - b.loc[both, "energy_j"].values
        dl = a.loc[both, "path_length_m"].values - b.loc[both, "path_length_m"].values
        w_stat, p = stats.wilcoxon(de)
        t, p_t = stats.ttest_rel(a.loc[both, "energy_j"], b.loc[both, "energy_j"])
        print(f"Eslestirilmis enerji farki (goal - enerji agirlikli): "
              f"{de.mean():+.0f} ± {de.std():.0f} J")
        print(f"  Wilcoxon p={p:.4f} | paired t p={p_t:.4f}")
        print(f"Eslestirilmis rota farki: {dl.mean():+.2f} ± {dl.std():.2f} m")
        med = np.median(de)
        print(f"  Medyan fark: {med:+.0f} J "
              f"({'enerji-agirlikli daha az tuketiyor' if med > 0 else 'fark yok / ters yonde'})")
    else:
        print("Eslestirilmis test icin yeterli ortak basarili bolum yok; "
              "--episodes arttirarak tekrar degerlendirin.")

    # Guvenlik agirliginin min. mesafeye etkisi (ayni mantik)
    w_safety_heavy = max(weights, key=lambda w: w[2])
    c = df[(df.w_goal == w_safety_heavy[0]) & (df.w_energy == w_safety_heavy[1]) &
           (df.w_safety == w_safety_heavy[2])].set_index("episode")
    both2 = [ep for ep in a.index.intersection(c.index)
             if a.loc[ep, "success"] and c.loc[ep, "success"]]
    if len(both2) >= 10:
        dc = c.loc[both2, "min_clearance_m"].values - a.loc[both2, "min_clearance_m"].values
        _, p = stats.wilcoxon(dc)
        print(f"\nGuvenlik etkisi: w={w_safety_heavy} vs w={w_goal_heavy} "
              f"min. mesafe farki {dc.mean():+.2f} m (Wilcoxon p={p:.4f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare", nargs="+", default=None,
                   help="*_episodes.csv dosyalari")
    p.add_argument("--pareto", default=None, help="*_pareto.csv dosyasi")
    args = p.parse_args()
    if args.compare:
        compare_methods(args.compare)
    if args.pareto:
        pareto_paired(args.pareto)
    if not args.compare and not args.pareto:
        print("Kullanim icin --compare veya --pareto verin (bkz. docstring).")


if __name__ == "__main__":
    main()
