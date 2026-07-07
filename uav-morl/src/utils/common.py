"""Shared utilities: configuration, seeding, environment factory, metrics."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_cfg: dict, seed: int | None = None, weights: list[float] | None = None):
    """Factory returning a thunk (needed by SB3 vectorised envs)."""
    from src.envs import UAVPathPlanningEnv

    def _thunk():
        cfg = dict(env_cfg)
        if weights is not None:
            cfg = json.loads(json.dumps(cfg))  # deep copy
            cfg.setdefault("reward", {})["weights"] = list(weights)
            cfg["reward"]["randomize_weights"] = False
        env = UAVPathPlanningEnv(cfg)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _thunk


def ensure_dirs(cfg: dict) -> None:
    for key in ("models", "logs", "figures", "metrics"):
        (PROJECT_ROOT / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


# --------------------------------------------------------------------- metrics
def episode_metrics(env, info: dict, terminated_reason: str | None) -> dict:
    """Summarise a finished evaluation episode."""
    straight = float(env.initial_goal_dist)
    length = env.path_length()
    return {
        "success": terminated_reason == "goal",
        "collision": terminated_reason == "collision",
        "timeout": terminated_reason in ("timeout", None),
        "battery_dead": terminated_reason == "battery",
        "path_length_m": length,
        "straight_line_m": straight,
        "path_efficiency": straight / length if length > 1e-6 else 0.0,
        "energy_j": float(env.energy_used),
        "energy_per_meter_j": float(env.energy_used) / max(length, 1e-6),
        "flight_time_s": env.steps * env.dt,
        "final_goal_distance_m": float(info["goal_distance"]),
        "min_clearance_m": float(info["min_clearance"]),
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    """Mean/std aggregation with success-conditioned path statistics."""
    out: dict[str, float] = {
        "n_episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "timeout_rate": float(np.mean([r["timeout"] for r in rows])),
    }
    succ = [r for r in rows if r["success"]]
    pool = succ if succ else rows
    for key in ("path_length_m", "path_efficiency", "energy_j", "energy_per_meter_j", "flight_time_s"):
        vals = np.array([r[key] for r in pool])
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    return out
