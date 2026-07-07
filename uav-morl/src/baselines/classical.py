"""Classical planning baselines for comparison against the RL policies.

1. ``AStarReplanner`` — 3-D grid A* replanned at a fixed interval against the
   current obstacle snapshot (a standard "frozen-world replanning" baseline
   for dynamic environments).
2. ``PotentialFieldPlanner`` — Artificial Potential Field (APF) with
   attractive goal and repulsive obstacle forces (Khatib, 1986), a purely
   reactive baseline.

Both baselines output acceleration commands in the same normalised action
space as the RL policies, so they run through the *identical* environment
and metric pipeline.
"""

from __future__ import annotations

import heapq

import numpy as np


class PotentialFieldPlanner:
    """Reactive APF controller producing normalised accelerations."""

    def __init__(self, k_att: float = 1.0, k_rep: float = 30.0, rho0: float = 6.0):
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho0 = rho0

    def act(self, env) -> np.ndarray:
        to_goal = env.goal - env.pos
        d_goal = np.linalg.norm(to_goal) + 1e-9
        f = self.k_att * to_goal / d_goal  # unit attractive force

        for ob in env.field.obstacles:
            diff = env.pos - ob.center
            rho = np.linalg.norm(diff) - ob.radius
            if 1e-6 < rho < self.rho0:
                f += (
                    self.k_rep
                    * (1.0 / rho - 1.0 / self.rho0)
                    / rho**2
                    * diff
                    / (np.linalg.norm(diff) + 1e-9)
                )

        # Velocity damping keeps the controller stable near the goal.
        desired_vel = np.clip(f, -1.0, 1.0) * env.v_max
        acc = (desired_vel - env.vel) / (env.a_max * env.dt + 1e-9)
        return np.clip(acc, -1.0, 1.0).astype(np.float32)


class AStarReplanner:
    """Grid-based A* replanned every ``replan_every`` steps."""

    def __init__(self, resolution: float = 2.0, replan_every: int = 5, inflate: float = 1.0):
        self.res = resolution
        self.replan_every = replan_every
        self.inflate = inflate
        self._path: list[np.ndarray] = []
        self._since_replan = 10**9

    # -------------------------------------------------------------- grid utils
    def _to_grid(self, p: np.ndarray, lo: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.round((p - lo) / self.res).astype(int))

    def _to_world(self, c: tuple[int, int, int], lo: np.ndarray) -> np.ndarray:
        return lo + np.array(c, dtype=float) * self.res

    def _blocked(self, p: np.ndarray, env) -> bool:
        margin = env.uav_radius + self.inflate
        return env.field.min_distance(p) <= margin

    def _plan(self, env) -> list[np.ndarray]:
        lo = env.bounds[:, 0]
        hi = env.bounds[:, 1]
        dims = np.ceil((hi - lo) / self.res).astype(int) + 1
        start = self._to_grid(env.pos, lo)
        goal = self._to_grid(env.goal, lo)

        moves = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]

        def h(c):
            return self.res * np.linalg.norm(np.array(c) - np.array(goal))

        open_heap = [(h(start), 0.0, start, None)]
        came: dict = {}
        g_cost = {start: 0.0}
        closed = set()
        expansions = 0

        while open_heap and expansions < 60000:
            _, g, cur, parent = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            came[cur] = parent
            expansions += 1
            if np.linalg.norm(np.array(cur) - np.array(goal)) * self.res <= env.goal_threshold:
                path = [cur]
                while came[path[-1]] is not None:
                    path.append(came[path[-1]])
                return [self._to_world(c, lo) for c in reversed(path)]
            for m in moves:
                nxt = (cur[0] + m[0], cur[1] + m[1], cur[2] + m[2])
                if any(n < 0 or n >= d for n, d in zip(nxt, dims)):
                    continue
                if nxt in closed:
                    continue
                wp = self._to_world(nxt, lo)
                if self._blocked(wp, env):
                    continue
                ng = g + self.res * np.linalg.norm(m)
                if ng < g_cost.get(nxt, np.inf):
                    g_cost[nxt] = ng
                    heapq.heappush(open_heap, (ng + h(nxt), ng, nxt, cur))
        return []  # planning failed

    # ------------------------------------------------------------------ control
    def act(self, env) -> np.ndarray:
        self._since_replan += 1
        if self._since_replan >= self.replan_every or not self._path:
            self._path = self._plan(env)
            self._since_replan = 0

        target = env.goal
        if self._path:
            # Track the first waypoint that is sufficiently far ahead.
            while len(self._path) > 1 and np.linalg.norm(self._path[0] - env.pos) < self.res:
                self._path.pop(0)
            target = self._path[0]

        direction = target - env.pos
        dist = np.linalg.norm(direction) + 1e-9
        desired_speed = min(env.v_max, 2.0 * dist)
        desired_vel = direction / dist * desired_speed
        acc = (desired_vel - env.vel) / (env.a_max * env.dt + 1e-9)
        return np.clip(acc, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        self._path = []
        self._since_replan = 10**9
