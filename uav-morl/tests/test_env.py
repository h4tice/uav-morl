"""Unit tests for the UAV MORL environment. Run with:  pytest -q"""

import numpy as np
import pytest

from src.envs import EnergyModelConfig, RotaryWingEnergyModel, UAVPathPlanningEnv
from src.utils.common import load_config


@pytest.fixture
def env():
    cfg = load_config()["env"]
    return UAVPathPlanningEnv(cfg)


def test_gymnasium_api(env):
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)


def test_reset_determinism(env):
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    assert np.allclose(obs1, obs2)


def test_observation_bounds(env):
    obs, _ = env.reset(seed=3)
    for _ in range(50):
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_reward_vector_and_scalarisation(env):
    env.reset(seed=1)
    _, r, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    vec = info["reward_vector"]
    assert vec.shape == (3,)
    assert abs(r - float(env.w @ vec)) < 1e-9


def test_energy_model_monotone_hover_vs_fast():
    m = RotaryWingEnergyModel(EnergyModelConfig())
    p_hover = m.power(np.zeros(3))
    p_opt = m.power(np.array([10.0, 0.0, 0.0]))
    p_fast = m.power(np.array([30.0, 0.0, 0.0]))
    # Rotary-wing curve: an intermediate speed is cheaper than hover,
    # very high speed is more expensive again (U-shaped power curve).
    assert p_opt < p_hover
    assert p_fast > p_opt
    # Climbing costs more than level flight.
    assert m.power(np.array([5.0, 0.0, 2.0])) > m.power(np.array([5.0, 0.0, 0.0]))


def test_collision_terminates(env):
    env.reset(seed=5)
    # Teleport the UAV inside the first obstacle and step.
    ob = env.field.obstacles[0]
    env.pos = ob.center.copy()
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated and info["terminated_reason"] == "collision"


def test_goal_reaching_terminates(env):
    env.reset(seed=9)
    env.pos = env.goal - np.array([0.5, 0.0, 0.0])
    env.prev_goal_dist = 0.5
    _, r, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated and info["terminated_reason"] == "goal"
    assert r > 0


def test_weight_conditioning(env):
    obs_a, _ = env.reset(seed=2, options={"weights": [0.8, 0.1, 0.1]})
    obs_b, _ = env.reset(seed=2, options={"weights": [0.1, 0.8, 0.1]})
    # Same world seed, different preference encoding => observations differ
    assert not np.allclose(obs_a, obs_b)
