"""Multi-objective UAV path-planning environment (Gymnasium API).

State space (continuous):
    - Relative goal vector (3), normalised by arena diagonal
    - UAV velocity (3), normalised by v_max
    - LiDAR-like ray distances (n_rays), normalised by sensor range
    - Remaining energy fraction (1)
    - Minimum obstacle clearance (1), normalised
    - Preference / scalarisation weight vector (3)  [optional, for
      preference-conditioned MORL]

Action space (continuous):
    - 3-D acceleration command in [-1, 1]^3, scaled by a_max.

Objectives (reward vector r = [r_goal, r_energy, r_safety]):
    r_goal   : progress towards the goal + terminal bonus
    r_energy : negative normalised propulsion energy of the step
    r_safety : penalty inside the danger zone + terminal collision penalty

The scalar reward returned to single-objective RL algorithms is the
weighted sum  w · r  (linear scalarisation). The full vector is exposed in
``info['reward_vector']`` so that Pareto analyses and alternative
scalarisations (e.g. Chebyshev) can be computed offline.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .energy import EnergyModelConfig, RotaryWingEnergyModel
from .obstacles import DynamicObstacleField


class UAVPathPlanningEnv(gym.Env):
    """3-D kinematic UAV navigation with dynamic obstacles and energy cost."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}

        # --- Arena & episode -------------------------------------------------
        size = cfg.get("arena_size", [60.0, 60.0, 20.0])
        self.bounds = np.array([[0.0, size[0]], [0.0, size[1]], [2.0, size[2]]])
        self.arena_diag = float(np.linalg.norm(self.bounds[:, 1] - self.bounds[:, 0]))
        self.dt = float(cfg.get("dt", 0.2))
        self.max_steps = int(cfg.get("max_steps", 400))
        self.goal_threshold = float(cfg.get("goal_threshold", 2.0))
        self.min_start_goal_dist = float(cfg.get("min_start_goal_dist", 40.0))

        # --- UAV dynamics ----------------------------------------------------
        self.uav_radius = float(cfg.get("uav_radius", 0.5))
        self.v_max = float(cfg.get("v_max", 8.0))
        self.a_max = float(cfg.get("a_max", 4.0))
        self.drag = float(cfg.get("drag", 0.05))

        # --- Sensors ----------------------------------------------------------
        self.n_rays = int(cfg.get("n_rays", 16))
        self.sensor_range = float(cfg.get("sensor_range", 15.0))
        self._ray_dirs = self._build_ray_directions(self.n_rays)

        # --- Obstacles ---------------------------------------------------------
        obs_cfg = cfg.get("obstacles", {})
        self._obstacle_kwargs = dict(
            n_obstacles=int(obs_cfg.get("count", 10)),
            radius_range=tuple(obs_cfg.get("radius_range", [1.5, 3.5])),
            speed_range=tuple(obs_cfg.get("speed_range", [0.5, 2.5])),
            dynamic_fraction=float(obs_cfg.get("dynamic_fraction", 0.7)),
        )

        # --- Energy -------------------------------------------------------------
        e_cfg = cfg.get("energy", {})
        self.energy_model = RotaryWingEnergyModel(
            EnergyModelConfig(**{k: v for k, v in e_cfg.items() if k in EnergyModelConfig.__dataclass_fields__})
        )
        self.battery_capacity = float(cfg.get("battery_capacity_j", 60000.0))

        # --- Reward -----------------------------------------------------------
        r_cfg = cfg.get("reward", {})
        self.w = np.array(r_cfg.get("weights", [0.6, 0.2, 0.2]), dtype=np.float64)
        self.w = self.w / (self.w.sum() + 1e-12)
        self.randomize_weights = bool(r_cfg.get("randomize_weights", False))
        self.include_weights_in_obs = bool(r_cfg.get("include_weights_in_obs", True))
        self.k_progress = float(r_cfg.get("k_progress", 1.0))
        self.k_energy = float(r_cfg.get("k_energy", 1.0))
        self.k_safety = float(r_cfg.get("k_safety", 1.0))
        # DUZELTME: enerji odul olcegi artik konfigurasyondan geliyor
        # (eski surumde 0.1 sabitti; odunlesimin gorunur olmasi icin varsayilan 0.3)
        self.energy_scale = float(r_cfg.get("energy_scale", 0.3))
        self.danger_radius = float(r_cfg.get("danger_radius", 3.0))
        self.r_goal_bonus = float(r_cfg.get("goal_bonus", 50.0))
        self.r_collision = float(r_cfg.get("collision_penalty", -50.0))
        self.r_timeout = float(r_cfg.get("timeout_penalty", -10.0))
        self.step_penalty = float(r_cfg.get("step_penalty", 0.02))

        # --- Spaces -------------------------------------------------------------
        # DUZELTME v3: onceki isin taramasi gozleme ekleniyor (ray_history).
        # Tek karelik tarama engel HIZINI tasiyamaz; iki ardisik tarama
        # farkindan ajan hareketli engellerin yonunu cikartabilir.
        self.ray_history = bool(cfg.get("ray_history", True))
        base_dim = 3 + 3 + self.n_rays * (2 if self.ray_history else 1) + 1 + 1
        obs_dim = base_dim + (3 if self.include_weights_in_obs else 0)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        # --- Runtime state -------------------------------------------------------
        self.rng = np.random.default_rng()
        self.field: DynamicObstacleField | None = None
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.goal = np.zeros(3)
        self.energy_used = 0.0
        self.steps = 0
        self.trajectory: list[np.ndarray] = []

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _build_ray_directions(n_rays: int) -> np.ndarray:
        """n_rays-2 horizontal rays plus one up and one down."""
        n_h = max(n_rays - 2, 4)
        angles = np.linspace(0.0, 2 * np.pi, n_h, endpoint=False)
        horiz = np.stack([np.cos(angles), np.sin(angles), np.zeros(n_h)], axis=1)
        vertical = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        dirs = np.vstack([horiz, vertical])[:n_rays]
        return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    def _sample_start_goal(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(200):
            start = np.array([self.rng.uniform(lo + 2, hi - 2) for lo, hi in self.bounds])
            goal = np.array([self.rng.uniform(lo + 2, hi - 2) for lo, hi in self.bounds])
            if np.linalg.norm(goal - start) >= self.min_start_goal_dist:
                return start, goal
        return start, goal  # fall back to last sample

    # ------------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        if "weights" in options:
            self.w = np.asarray(options["weights"], dtype=np.float64)
            self.w = self.w / (self.w.sum() + 1e-12)
        elif self.randomize_weights:
            raw = self.rng.dirichlet(np.ones(3))
            self.w = raw

        self.pos, self.goal = self._sample_start_goal()
        self.vel = np.zeros(3)
        self.energy_used = 0.0
        self.steps = 0

        self.field = DynamicObstacleField(self.rng, self.bounds, **self._obstacle_kwargs)
        self.field.reset(keep_clear=[(self.pos, 5.0), (self.goal, 5.0)])

        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))
        self.initial_goal_dist = self.prev_goal_dist
        self.trajectory = [self.pos.copy()]
        # Ilk adimda "onceki tarama" = simdiki tarama
        self._prev_rays = self.field.ray_distances(self.pos, self._ray_dirs, self.sensor_range)

        return self._observation(), self._info(np.zeros(3), terminated_reason=None)

    def step(self, action: np.ndarray):
        assert self.field is not None, "reset() must be called before step()"
        self.steps += 1

        # --- Kinematics ------------------------------------------------------
        acc = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * self.a_max
        self.vel = (1.0 - self.drag) * self.vel + acc * self.dt
        speed = np.linalg.norm(self.vel)
        if speed > self.v_max:
            self.vel *= self.v_max / speed
        self.pos = self.pos + self.vel * self.dt
        out_of_bounds = bool(
            np.any(self.pos < self.bounds[:, 0]) or np.any(self.pos > self.bounds[:, 1])
        )
        self.pos = np.clip(self.pos, self.bounds[:, 0], self.bounds[:, 1])
        self.trajectory.append(self.pos.copy())

        # --- World update ------------------------------------------------------
        self.field.step(self.dt)

        # --- Energy -------------------------------------------------------------
        e_step = self.energy_model.step_energy(self.vel, self.dt)
        self.energy_used += e_step
        e_norm = e_step / (self.energy_model.hover_power * self.dt)  # ~1 at hover

        # --- Objective components ------------------------------------------------
        goal_dist = float(np.linalg.norm(self.goal - self.pos))
        progress = self.prev_goal_dist - goal_dist
        self.prev_goal_dist = goal_dist

        clearance = self.field.min_distance(self.pos)
        collided = clearance <= self.uav_radius
        reached = goal_dist <= self.goal_threshold
        battery_dead = self.energy_used >= self.battery_capacity
        timeout = self.steps >= self.max_steps

        r_goal = self.k_progress * progress - self.step_penalty
        r_energy = -self.k_energy * e_norm * self.energy_scale
        if clearance < self.danger_radius:
            proximity = 1.0 - clearance / self.danger_radius
            r_safety = -self.k_safety * proximity**2
        else:
            r_safety = 0.0
        if out_of_bounds:
            r_safety -= 1.0

        terminated_reason = None
        if reached:
            r_goal += self.r_goal_bonus
            terminated_reason = "goal"
        elif collided:
            r_safety += self.r_collision
            terminated_reason = "collision"
        elif battery_dead:
            r_energy += self.r_timeout
            terminated_reason = "battery"
        elif timeout:
            r_goal += self.r_timeout * 0.5
            terminated_reason = "timeout"

        reward_vec = np.array([r_goal, r_energy, r_safety], dtype=np.float64)
        reward = float(self.w @ reward_vec)

        terminated = reached or collided or battery_dead
        truncated = timeout and not terminated

        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(reward_vec, terminated_reason),
        )

    # ------------------------------------------------------------------- internals
    def _observation(self) -> np.ndarray:
        rel_goal = (self.goal - self.pos) / self.arena_diag
        vel_n = self.vel / self.v_max
        rays = self.field.ray_distances(self.pos, self._ray_dirs, self.sensor_range)
        rays_n = 2.0 * rays / self.sensor_range - 1.0
        prev_rays_n = 2.0 * self._prev_rays / self.sensor_range - 1.0
        self._prev_rays = rays
        energy_frac = 1.0 - 2.0 * min(self.energy_used / self.battery_capacity, 1.0)
        clearance = min(self.field.min_distance(self.pos), self.sensor_range)
        clearance_n = 2.0 * clearance / self.sensor_range - 1.0

        parts = [rel_goal * 3.0, vel_n, rays_n]
        if self.ray_history:
            parts.append(prev_rays_n)
        parts += [[energy_frac], [clearance_n]]
        if self.include_weights_in_obs:
            parts.append(2.0 * self.w - 1.0)
        obs = np.concatenate([np.atleast_1d(np.asarray(p)) for p in parts])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _info(self, reward_vec: np.ndarray, terminated_reason: str | None) -> dict:
        return {
            "reward_vector": reward_vec,
            "weights": self.w.copy(),
            "position": self.pos.copy(),
            "goal": self.goal.copy(),
            "goal_distance": float(np.linalg.norm(self.goal - self.pos)),
            "energy_used_j": self.energy_used,
            "min_clearance": float(self.field.min_distance(self.pos)) if self.field else np.inf,
            "steps": self.steps,
            "terminated_reason": terminated_reason,
            "obstacles": self.field.state_snapshot() if self.field else None,
        }

    # -------------------------------------------------------------------- extras
    def path_length(self) -> float:
        traj = np.asarray(self.trajectory)
        if len(traj) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
