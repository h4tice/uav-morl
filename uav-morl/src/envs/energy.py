"""Rotary-wing UAV propulsion energy model.

Implements the analytical power consumption model for rotary-wing UAVs
proposed by Zeng, Xu & Zhang (2019), "Energy Minimization for Wireless
Communication with Rotary-Wing UAV", IEEE TWC. The model expresses the
required propulsion power as a function of the horizontal flight speed,
with an additional term for climb/descent power.

P(V) = P0 * (1 + 3V^2 / U_tip^2)                      (blade profile power)
     + Pi * ( sqrt(1 + V^4/(4 v0^4)) - V^2/(2 v0^2) )^(1/2)   (induced power)
     + 0.5 * d0 * rho * s * A * V^3                    (parasite power)

Climb power is approximated as m * g * Vz for Vz > 0 (ascending);
a partial recovery factor is applied when descending.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyModelConfig:
    """Physical parameters of the rotary-wing energy model.

    Default values follow Zeng & Zhang (2019), Table I.
    """

    p0: float = 79.86      # Blade profile power in hover [W]
    pi: float = 88.63      # Induced power in hover [W]
    u_tip: float = 120.0   # Rotor blade tip speed [m/s]
    v0: float = 4.03       # Mean rotor induced velocity in hover [m/s]
    d0: float = 0.6        # Fuselage drag ratio [-]
    rho: float = 1.225     # Air density [kg/m^3]
    s: float = 0.05        # Rotor solidity [-]
    a: float = 0.503       # Rotor disc area [m^2]
    mass: float = 2.0      # UAV mass [kg]
    g: float = 9.81        # Gravitational acceleration [m/s^2]
    descent_recovery: float = 0.3  # Fraction of potential power recovered when descending


class RotaryWingEnergyModel:
    """Computes instantaneous propulsion power and per-step energy."""

    def __init__(self, config: EnergyModelConfig | None = None) -> None:
        self.cfg = config or EnergyModelConfig()
        # Hover power used for normalisation of the energy objective.
        self.hover_power = self.power(np.zeros(3))

    def power(self, velocity: np.ndarray) -> float:
        """Instantaneous propulsion power [W] for a 3-D velocity vector."""
        c = self.cfg
        v_h = float(np.linalg.norm(velocity[:2]))  # horizontal speed
        v_z = float(velocity[2]) if velocity.shape[0] > 2 else 0.0

        blade_profile = c.p0 * (1.0 + 3.0 * v_h**2 / c.u_tip**2)
        induced = c.pi * np.sqrt(
            max(np.sqrt(1.0 + v_h**4 / (4.0 * c.v0**4)) - v_h**2 / (2.0 * c.v0**2), 0.0)
        )
        parasite = 0.5 * c.d0 * c.rho * c.s * c.a * v_h**3

        if v_z >= 0.0:
            climb = c.mass * c.g * v_z
        else:
            climb = c.descent_recovery * c.mass * c.g * v_z  # negative => small saving

        return float(blade_profile + induced + parasite + max(climb, -0.5 * blade_profile))

    def step_energy(self, velocity: np.ndarray, dt: float) -> float:
        """Energy consumed [J] over one control step of duration ``dt``."""
        return self.power(velocity) * dt
