"""Evaluate trained policies and classical baselines under identical conditions.

Produces per-episode CSVs, aggregate JSON summaries and (optionally) a
Pareto sweep over preference weight vectors for the preference-conditioned
policy.

Usage:
    python -m src.evaluate --model results/models/<run>/best_model.zip --algo PPO
    python -m src.evaluate --baseline apf
    python -m src.evaluate --baseline astar
    python -m src.evaluate --model ... --pareto        # weight sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.classical import AStarReplanner, PotentialFieldPlanner
from src.envs import UAVPathPlanningEnv
from src.utils.common import (
    PROJECT_ROOT,
    aggregate_metrics,
    episode_metrics,
    ensure_dirs,
    load_config,
    save_json,
    set_global_seed,
)


def _make_env(cfg: dict, weights: list[float] | None) -> UAVPathPlanningEnv:
    env_cfg = dict(cfg["env"])
    if weights is not None:
        env_cfg = {**env_cfg, "reward": {**env_cfg.get("reward", {}),
                                         "weights": list(weights),
                                         "randomize_weights": False}}
    return UAVPathPlanningEnv(env_cfg)


def run_episodes(policy_fn, cfg: dict, n_episodes: int, seed: int,
                 weights: list[float] | None = None,
                 reset_hook=None, save_trajectories: int = 0):
    env = _make_env(cfg, weights)
    rows, trajectories = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if reset_hook:
            reset_hook()
        terminated = truncated = False
        reason = None
        ep_reward_vec = np.zeros(3)
        while not (terminated or truncated):
            action = policy_fn(obs, env)
            obs, _, terminated, truncated, info = env.step(action)
            ep_reward_vec += info["reward_vector"]
            reason = info["terminated_reason"]
        row = episode_metrics(env, info, reason)
        row["episode"] = ep
        row["return_goal"], row["return_energy"], row["return_safety"] = ep_reward_vec
        rows.append(row)
        if ep < save_trajectories:
            trajectories.append(np.asarray(env.trajectory))
    return rows, trajectories


def sb3_policy(model_path: str, algo: str):
    from stable_baselines3 import PPO, SAC, TD3

    cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algo.upper()]
    model = cls.load(model_path, device="cpu")

    def _fn(obs, _env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return _fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default=None, help="Path to an SB3 .zip model")
    p.add_argument("--algo", default="PPO")
    p.add_argument("--baseline", choices=["apf", "astar"], default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--pareto", action="store_true",
                   help="Sweep preference weights (requires --model)")
    p.add_argument("--tag", default=None, help="Output name tag")
    args = p.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_global_seed(args.seed)
    n_eps = args.episodes or cfg["evaluation"]["n_episodes"]
    metrics_dir = PROJECT_ROOT / cfg["paths"]["metrics"]

    if args.baseline:
        if args.baseline == "apf":
            planner = PotentialFieldPlanner()
            policy_fn = lambda obs, env: planner.act(env)
            reset_hook = None
        else:
            planner = AStarReplanner()
            policy_fn = lambda obs, env: planner.act(env)
            reset_hook = planner.reset
        tag = args.tag or f"baseline_{args.baseline}"
        rows, _ = run_episodes(policy_fn, cfg, n_eps, args.seed, reset_hook=reset_hook)
        _export(rows, metrics_dir, tag)
        return

    if not args.model:
        raise SystemExit("Provide --model or --baseline.")

    policy_fn = sb3_policy(args.model, args.algo)
    tag = args.tag or f"{args.algo.lower()}_{Path(args.model).parent.name}"

    if args.pareto:
        all_rows = []
        for w in cfg["evaluation"]["pareto_weights"]:
            rows, _ = run_episodes(policy_fn, cfg, n_eps, args.seed, weights=w)
            for r in rows:
                r["w_goal"], r["w_energy"], r["w_safety"] = w
            agg = aggregate_metrics(rows)
            agg["weights"] = w
            print(f"[pareto] w={w} SR={agg['success_rate']:.2f} "
                  f"E={agg['energy_j_mean']:.0f}J L={agg['path_length_m_mean']:.1f}m")
            all_rows.extend(rows)
        df = pd.DataFrame(all_rows)
        out = metrics_dir / f"{tag}_pareto.csv"
        df.to_csv(out, index=False)
        print(f"[pareto] saved -> {out}")
    else:
        rows, _ = run_episodes(policy_fn, cfg, n_eps, args.seed)
        _export(rows, metrics_dir, tag)


def _export(rows: list[dict], metrics_dir: Path, tag: str) -> None:
    df = pd.DataFrame(rows)
    csv_path = metrics_dir / f"{tag}_episodes.csv"
    df.to_csv(csv_path, index=False)
    agg = aggregate_metrics(rows)
    save_json(agg, metrics_dir / f"{tag}_summary.json")
    print(f"[eval] {tag}: SR={agg['success_rate']:.2f} "
          f"CR={agg['collision_rate']:.2f} "
          f"E={agg['energy_j_mean']:.0f}±{agg['energy_j_std']:.0f} J "
          f"L={agg['path_length_m_mean']:.1f} m -> {csv_path}")


if __name__ == "__main__":
    main()
