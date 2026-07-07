"""Full experiment pipeline: train -> evaluate -> baselines -> Pareto -> figures.

    python -m src.run_experiment                                  # default.yaml
    python -m src.run_experiment --config configs/hard.yaml       # zor senaryo
    python -m src.run_experiment --quick                          # duman testi
    python -m src.run_experiment --skip-train --model results/models/<run>/best_model.zip

DUZELTMELER (v2):
- --config artik TUM adimlara (evaluate, pareto, plots) iletiliyor.
  Onceki surumde yalnizca egitime iletiliyordu; baseline'lar ve
  degerlendirme her zaman default.yaml ile kosuyordu.
- Run adi ve metrik etiketleri config adini iceriyor (ppo_hard_seed42 gibi),
  boylece farkli senaryolarin modelleri/metrikleri birbirini ezmiyor.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.common import PROJECT_ROOT, load_config


def sh(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--algo", default="PPO")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    py = [sys.executable, "-m"]
    cfgarg = ["--config", args.config]                       # <-- DUZELTME
    eval_eps = ["--episodes", "5"] if args.quick else []

    # Senaryo adi: configs/hard.yaml -> "hard"; default.yaml -> "default"
    scenario = Path(args.config).stem
    tag = f"{args.algo.lower()}_{scenario}"

    # 1) Training ------------------------------------------------------------
    model_path = args.model
    if not args.skip_train:
        run_name = f"{args.algo.lower()}_{scenario}_seed{args.seed}{'_quick' if args.quick else ''}"
        cmd = py + ["src.train", *cfgarg, "--algo", args.algo,
                    "--seed", str(args.seed), "--run-name", run_name]
        if args.quick:
            cmd.append("--quick")
        sh(cmd)
        mdir = PROJECT_ROOT / cfg["paths"]["models"] / run_name
        best = mdir / "best_model.zip"
        model_path = str(best if best.exists() else mdir / "final_model.zip")
    assert model_path, "No model available; use --model with --skip-train."

    # 2) RL evaluation ---------------------------------------------------------
    sh(py + ["src.evaluate", *cfgarg, "--model", model_path, "--algo", args.algo,
             "--tag", tag, *eval_eps])

    # 3) Baselines ----------------------------------------------------------------
    sh(py + ["src.evaluate", *cfgarg, "--baseline", "apf",
             "--tag", f"apf_{scenario}", *eval_eps])
    sh(py + ["src.evaluate", *cfgarg, "--baseline", "astar",
             "--tag", f"astar_{scenario}", *eval_eps])

    # 4) Pareto sweep -----------------------------------------------------------
    sh(py + ["src.evaluate", *cfgarg, "--model", model_path, "--algo", args.algo,
             "--pareto", "--tag", tag, *eval_eps])

    # 5) Figures -------------------------------------------------------------------
    metrics = str(PROJECT_ROOT / cfg["paths"]["metrics"])
    pareto_csv = str(PROJECT_ROOT / cfg["paths"]["metrics"] / f"{tag}_pareto.csv")
    sh(py + ["src.plots", *cfgarg, "--compare", metrics])
    sh(py + ["src.plots", *cfgarg, "--pareto", pareto_csv])
    sh(py + ["src.plots", *cfgarg, "--trajectory", "--model", model_path, "--algo", args.algo])
    sh(py + ["src.plots", *cfgarg, "--learning", str(PROJECT_ROOT / cfg["paths"]["logs"])])

    print("\n[pipeline] complete. Figures -> results/figures, metrics -> results/metrics")


if __name__ == "__main__":
    main()
