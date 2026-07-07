"""Dynamic obstacle field for the UAV environment.

Obstacles are spheres whose centres follow one of three motion patterns:
linear (bouncing inside the arena), sinusoidal, or circular. A ray-casting
sensor provides LiDAR-like range measurements used in the observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obstacle:
    center: np.ndarray                # (3,) current position
    radius: float
    motion: str                       # 'static' | 'linear' | 'sinusoidal' | 'circular'
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    anchor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    amplitude: float = 0.0
    omega: float = 0.0
    phase: float = 0.0
    axis: int = 0


class DynamicObstacleField:
    """Manages a set of moving spherical obstacles inside a box arena."""

    def __init__(
        self,
        rng: np.random.Generator,
        bounds: np.ndarray,
        n_obstacles: int,
        radius_range: tuple[float, float] = (1.0, 3.0),
        speed_range: tuple[float, float] = (0.5, 2.5),
        dynamic_fraction: float = 0.7,
    ) -> None:
        self.rng = rng
        self.bounds = bounds  # shape (3, 2): [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        self.n_obstacles = n_obstacles
        self.radius_range = radius_range
        self.speed_range = speed_range
        self.dynamic_fraction = dynamic_fraction
        self.obstacles: list[Obstacle] = []
        self.time = 0.0

    # ------------------------------------------------------------------ setup
    def reset(self, keep_clear: list[tuple[np.ndarray, float]]) -> None:
        """Sample a new obstacle configuration.

        Parameters
        ----------
        keep_clear:
            List of (point, clearance) pairs — typically the start and goal —
            around which no obstacle may be spawned.
        """
        self.obstacles = []
        self.time = 0.0
        attempts = 0
        while len(self.obstacles) < self.n_obstacles and attempts < 500:
            attempts += 1
            radius = self.rng.uniform(*self.radius_range)
            center = np.array(
                [self.rng.uniform(lo + radius, hi - radius) for lo, hi in self.bounds]
            )
            if any(np.linalg.norm(center - p) < c + radius for p, c in keep_clear):
                continue

            if self.rng.random() < self.dynamic_fraction:
                motion = self.rng.choice(["linear", "sinusoidal", "circular"])
            else:
                motion = "static"

            speed = self.rng.uniform(*self.speed_range)
            direction = self.rng.normal(size=3)
            direction[2] *= 0.3  # mostly planar motion, as for ground-clutter traffic
            direction /= np.linalg.norm(direction) + 1e-9

            self.obstacles.append(
                Obstacle(
                    center=center.copy(),
                    radius=radius,
                    motion=str(motion),
                    velocity=speed * direction,
                    anchor=center.copy(),
                    amplitude=self.rng.uniform(1.0, 4.0),
                    omega=self.rng.uniform(0.3, 1.2),
                    phase=self.rng.uniform(0.0, 2 * np.pi),
                    axis=int(self.rng.integers(0, 2)),
                )
            )

    # ----------------------------------------------------------------- update
    def step(self, dt: float) -> None:
        self.time += dt
        for ob in self.obstacles:
            if ob.motion == "static":
                continue
            if ob.motion == "linear":
                ob.center += ob.velocity * dt
                for k in range(3):
                    lo, hi = self.bounds[k]
                    if ob.center[k] - ob.radius < lo or ob.center[k] + ob.radius > hi:
                        ob.velocity[k] *= -1.0
                        ob.center[k] = np.clip(ob.center[k], lo + ob.radius, hi - ob.radius)
            elif ob.motion == "sinusoidal":
                ob.center = ob.anchor.copy()
                ob.center[ob.axis] += ob.amplitude * np.sin(ob.omega * self.time + ob.phase)
            elif ob.motion == "circular":
                ob.center = ob.anchor.copy()
                ob.center[0] += ob.amplitude * np.cos(ob.omega * self.time + ob.phase)
                ob.center[1] += ob.amplitude * np.sin(ob.omega * self.time + ob.phase)

    # ---------------------------------------------------------------- queries
    def min_distance(self, point: np.ndarray) -> float:
        """Distance from ``point`` to the closest obstacle surface."""
        if not self.obstacles:
            return np.inf
        return min(
            float(np.linalg.norm(point - ob.center) - ob.radius) for ob in self.obstacles
        )

    def collides(self, point: np.ndarray, uav_radius: float) -> bool:
        return self.min_distance(point) <= uav_radius

    def ray_distances(
        self, origin: np.ndarray, directions: np.ndarray, max_range: float
    ) -> np.ndarray:
        """Ray-cast against all spheres; returns clipped hit distances.

        Parameters
        ----------
        origin: (3,) sensor origin.
        directions: (R, 3) unit direction vectors.
        max_range: sensor range; misses are reported as ``max_range``.
        """
        dists = np.full(directions.shape[0], max_range, dtype=np.float64)
        for ob in self.obstacles:
            oc = origin - ob.center                       # (3,)
            b = directions @ oc                           # (R,)
            c = float(oc @ oc) - ob.radius**2
            disc = b**2 - c
            mask = disc > 0.0
            if not np.any(mask):
                continue
            sqrt_disc = np.sqrt(disc[mask])
            t = -b[mask] - sqrt_disc                      # nearest intersection
            t = np.where(t > 0.0, t, np.inf)
            dists[mask] = np.minimum(dists[mask], t)
        return np.clip(dists, 0.0, max_range)

    def state_snapshot(self) -> np.ndarray:
        """(N, 4) array of obstacle centres and radii (for logging/plots)."""
        if not self.obstacles:
            return np.zeros((0, 4))
        return np.array([[*ob.center, ob.radius] for ob in self.obstacles])
