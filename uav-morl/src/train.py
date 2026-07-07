"""Train PPO/SAC policies on the multi-objective UAV environment.

Usage (from the project root, venv active):
    python -m src.train --config configs/default.yaml --algo PPO --seed 42
    python -m src.train --algo SAC --timesteps 500000

TensorBoard:
    tensorboard --logdir results/logs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.utils.common import PROJECT_ROOT, ensure_dirs, load_config, make_env, set_global_seed


def build_vec_env(cfg: dict, n_envs: int, seed: int, subproc: bool):
    thunks = []
    for i in range(n_envs):
        thunk = make_env(cfg["env"], seed=seed + i)
        thunks.append(lambda t=thunk: Monitor(t()))
    if subproc and n_envs > 1:
        return SubprocVecEnv(thunks, start_method="spawn")
    return DummyVecEnv(thunks)


def train(config_path: str, algo: str | None, seed: int | None, timesteps: int | None,
          n_envs: int | None, run_name: str | None, quick: bool) -> Path:
    cfg = load_config(config_path)
    ensure_dirs(cfg)

    algo = (algo or cfg["training"]["algorithm"]).upper()
    seed = seed if seed is not None else cfg["experiment"]["seed"]
    timesteps = timesteps or cfg["training"]["total_timesteps"]
    n_envs = n_envs or (cfg["training"]["n_envs"] if algo == "PPO" else 1)
    if quick:
        timesteps, n_envs = 8000, 2

    set_global_seed(seed)
    run_name = run_name or f"{algo.lower()}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = PROJECT_ROOT / cfg["paths"]["logs"] / run_name
    model_dir = PROJECT_ROOT / cfg["paths"]["models"] / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_vec_env(cfg, n_envs, seed, subproc=not quick)
    eval_env = build_vec_env(cfg, 1, seed + 10_000, subproc=False)

    common = dict(
        seed=seed,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=cfg["experiment"].get("device", "auto"),
    )
    if algo == "PPO":
        h = dict(cfg["training"]["ppo"])
        pk = h.pop("policy_kwargs", {})
        model = PPO("MlpPolicy", vec_env, policy_kwargs=dict(pk), **h, **common)
    elif algo == "SAC":
        h = dict(cfg["training"]["sac"])
        pk = h.pop("policy_kwargs", {})
        model = SAC("MlpPolicy", vec_env, policy_kwargs=dict(pk), **h, **common)
    elif algo == "TD3":
        import numpy as np
        from stable_baselines3.common.noise import NormalActionNoise
        h = dict(cfg["training"].get("td3", {
            "learning_rate": 3.0e-4, "buffer_size": 500000, "batch_size": 256,
            "gamma": 0.99, "tau": 0.005, "train_freq": 1, "gradient_steps": 1,
            "learning_starts": 10000,
            "policy_kwargs": {"net_arch": [256, 256]},
        }))
        pk = h.pop("policy_kwargs", {})
        n_act = vec_env.action_space.shape[-1]
        noise = NormalActionNoise(mean=np.zeros(n_act), sigma=0.1 * np.ones(n_act))
        model = TD3("MlpPolicy", vec_env, policy_kwargs=dict(pk),
                    action_noise=noise, **h, **common)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=max(cfg["training"]["eval_freq"] // n_envs, 1),
            n_eval_episodes=cfg["training"]["n_eval_episodes"],
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(cfg["training"]["checkpoint_freq"] // n_envs, 1),
            save_path=str(model_dir),
            name_prefix="ckpt",
        ),
    ]

    print(f"[train] algo={algo} seed={seed} timesteps={timesteps} n_envs={n_envs}")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=not quick)

    final_path = model_dir / "final_model.zip"
    model.save(final_path)
    print(f"[train] saved -> {final_path}")
    vec_env.close()
    eval_env.close()
    return final_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--algo", choices=["PPO", "SAC", "TD3", "ppo", "sac", "td3"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--quick", action="store_true", help="8k-step smoke test")
    args = p.parse_args()
    train(args.config, args.algo, args.seed, args.timesteps, args.n_envs, args.run_name, args.quick)


if __name__ == "__main__":
    main()
