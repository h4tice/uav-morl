"""Publication-quality figures (English labels, camera-ready).

    python -m src.plots --config configs/hard.yaml --compare results/metrics_final
    python -m src.plots --config configs/hard.yaml --pareto results/metrics/ppo_hard_v4_300_pareto.csv
    python -m src.plots --config configs/hard.yaml --trajectory --model <model.zip> --algo PPO
    python -m src.plots --config configs/hard.yaml --learning results/logs_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import PROJECT_ROOT, ensure_dirs, load_config, set_global_seed

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# Pretty names for known run tags (fallback: the raw stem)
LABELS = {
    "ppo_hard_v4": "PPO (proposed)",
    "ppo_hard_v4_s43": "PPO (seed 43)",
    "ppo_hard_v4_s44": "PPO (seed 44)",
    "apf_hard_v4": "APF (oracle)",
    "astar_hard_v4": "A* (replanning)",
    "sac_hard_v4": "SAC",
    "td3_hard_v4": "TD3",
    "ppo_hard_v4_seed42": "PPO seed 42",
    "ppo_hard_v4_seed43": "PPO seed 43",
    "ppo_hard_v4_seed44": "PPO seed 44",
}


def _label(stem: str) -> str:
    return LABELS.get(stem, stem)


# ------------------------------------------------------------------ trajectory
def plot_trajectory(cfg: dict, model_path: str, algo: str, seed: int, out_dir: Path) -> None:
    from src.evaluate import run_episodes, sb3_policy

    policy = sb3_policy(model_path, algo)
    rows, trajs = run_episodes(policy, cfg, n_episodes=4, seed=seed, save_trajectories=4)

    fig = plt.figure(figsize=(11, 9))
    for i, traj in enumerate(trajs):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#1f77b4", lw=1.6, label="UAV trajectory")
        ax.scatter(*traj[0], color="green", s=45, marker="o", label="Start")
        ax.scatter(*traj[-1], color="red", s=60, marker="*", label="End")
        r = rows[i]
        status = "Success" if r["success"] else ("Collision" if r["collision"] else "Timeout")
        ax.set_title(f"Episode {i+1}: {status} | E={r['energy_j']:.0f} J | L={r['path_length_m']:.1f} m",
                     fontsize=9)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        if i == 0:
            ax.legend(fontsize=7)
    out = out_dir / "trajectories_3d.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plots] -> {out}")


# ---------------------------------------------------------------------- pareto
def plot_pareto(csv_path: str, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    keys = ["w_goal", "w_energy", "w_safety"]
    pool = df[df["success"]] if df["success"].any() else df
    grouped = pool.groupby(keys).agg(
        energy=("energy_j", "mean"),
        path=("path_length_m", "mean"),
        clearance=("min_clearance_m", "mean"),
    ).reset_index()
    sr_df = df.groupby(keys)["success"].mean().reset_index(name="sr")
    grouped = grouped.merge(sr_df, on=keys, how="left")
    sr = grouped["sr"].values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc = axes[0].scatter(grouped["energy"], grouped["path"], c=sr, cmap="viridis",
                         s=70, edgecolors="k", vmin=0, vmax=1)
    axes[0].set_xlabel("Mean energy consumption [J]")
    axes[0].set_ylabel("Mean path length [m]")
    axes[0].set_title("Energy vs. path length")
    plt.colorbar(sc, ax=axes[0], label="Success rate")

    sc2 = axes[1].scatter(grouped["energy"], grouped["clearance"], c=grouped["w_safety"],
                          cmap="plasma", s=70, edgecolors="k")
    axes[1].set_xlabel("Mean energy consumption [J]")
    axes[1].set_ylabel("Mean min. obstacle clearance [m]")
    axes[1].set_title("Energy vs. safety")
    plt.colorbar(sc2, ax=axes[1], label=r"$w_{\mathrm{safety}}$")

    out = out_dir / "pareto_front.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plots] -> {out}")


# --------------------------------------------------------------------- compare
def plot_compare(metrics_dir: str, out_dir: Path) -> None:
    summaries = {}
    for f in sorted(Path(metrics_dir).glob("*_summary.json")):
        with open(f, encoding="utf-8") as fh:
            summaries[f.stem.replace("_summary", "")] = json.load(fh)
    if not summaries:
        print("[plots] no summaries found")
        return

    names = list(summaries)
    metrics = [
        ("success_rate", "Success rate", None),
        ("collision_rate", "Collision rate", None),
        ("energy_j_mean", "Energy [J]", "energy_j_std"),
        ("path_length_m_mean", "Path length [m]", "path_length_m_std"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (key, title, err_key) in zip(axes, metrics):
        vals = [summaries[n].get(key, 0.0) for n in names]
        errs = [summaries[n].get(err_key, 0.0) for n in names] if err_key else None
        ax.bar(range(len(names)), vals, yerr=errs, capsize=3,
               color=[plt.cm.tab10(i) for i in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([_label(n) for n in names], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    out = out_dir / "method_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plots] -> {out}")


# -------------------------------------------------------------------- learning
def plot_learning_curves(log_root: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    found = False
    for npz in sorted(Path(log_root).rglob("evaluations.npz")):
        data = np.load(npz)
        mean_r = data["results"].mean(axis=1)
        std_r = data["results"].std(axis=1)
        label = _label(npz.parent.name)
        ax.plot(data["timesteps"], mean_r, label=label)
        ax.fill_between(data["timesteps"], mean_r - std_r, mean_r + std_r, alpha=0.2)
        found = True
    if not found:
        print("[plots] no evaluations.npz found")
        return
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean evaluation return")
    ax.legend(fontsize=8)
    out = out_dir / "learning_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plots] -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--trajectory", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--algo", default="PPO")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--pareto", default=None)
    p.add_argument("--compare", default=None)
    p.add_argument("--learning", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_global_seed(args.seed)
    out_dir = PROJECT_ROOT / cfg["paths"]["figures"]

    if args.trajectory:
        assert args.model, "--trajectory requires --model"
        plot_trajectory(cfg, args.model, args.algo, args.seed, out_dir)
    if args.pareto:
        plot_pareto(args.pareto, out_dir)
    if args.compare:
        plot_compare(args.compare, out_dir)
    if args.learning:
        plot_learning_curves(args.learning, out_dir)


if __name__ == "__main__":
    main()
