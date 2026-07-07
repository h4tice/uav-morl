# UAV-MORL — An End-to-End Multi-Objective Reinforcement Learning Framework for UAV Path Planning with Dynamic Obstacle Avoidance and Energy Efficiency

This repository contains the complete, reproducible implementation of a
**preference-conditioned multi-objective reinforcement learning (MORL)**
framework for UAV path planning in 3-D environments with **fast-moving
obstacles**, jointly optimising **goal reaching**, **propulsion energy**, and
**safety**. It accompanies the manuscript *"An End-to-End Multi-Objective
Reinforcement Learning Framework for Unmanned Aerial Vehicle Path Planning
with Dynamic Obstacle Avoidance and Energy Efficiency"*.

## Highlights

1. **Physics-informed energy objective.** The analytical rotary-wing
   propulsion power model of Zeng, Xu & Zhang (2019) — blade-profile,
   induced, parasite, and climb power — is embedded directly in the reward.
2. **Fully dynamic obstacle field.** 20 spherical obstacles with linear
   (bouncing), sinusoidal, and circular motion, perceived through a
   LiDAR-like 16-ray sensor. The observation includes the **previous ray
   scan**, letting the policy infer obstacle motion from consecutive range
   measurements (`ray_history`, can be disabled for ablation).
3. **Preference-conditioned MORL.** The reward is a vector
   `r = [r_goal, r_energy, r_safety]`. During training the scalarisation
   weight `w ~ Dirichlet(1,1,1)` is resampled every episode and appended to
   the observation, so a **single trained policy** traces an approximate
   Pareto front at test time simply by changing `w`.
4. **Algorithm-agnostic.** Instantiated with **PPO**, **SAC**, and **TD3**
   (Stable-Baselines3). Off-policy learners achieve the highest success
   rates; on-policy PPO yields the most energy-efficient flight.
5. **Fair, statistically rigorous benchmarking.** A full-state
   **APF oracle** and a **replanning 3-D A\*** baseline run through the
   *identical* environment interface and metric pipeline. A **paired
   evaluation protocol** (every preference vector tested on identical world
   instances) with Wilcoxon/Welch/Mann–Whitney/Fisher tests is included.

## Key results (hard scenario: 20 dynamic obstacles, v_max = 25 m/s)

| Method | SR ↑ | CR ↓ | Energy [J] ↓ | Path [m] |
|---|---|---|---|---|
| SAC (ours, sensor-based) | **0.91** | **0.09** | 769 ± 182 | 53.6 |
| TD3 (ours, sensor-based) | 0.87 | 0.13 | 754 ± 177 | 55.0 |
| PPO (ours, sensor-based) | 0.80 ± 0.07 | 0.20 ± 0.07 | **657 ± 22** | 50.0 ± 1.8 |
| APF (full-state oracle) | 0.82 | 0.18 | 676 ± 208 | 51.8 |
| Replanning A\* | 0.59 | 0.24 | 1363 ± 181 | 47.8 |

PPO values are mean ± std over three seeds (42/43/44, 3M steps each);
SAC/TD3 are single-seed at 1M steps. Shifting the preference vector of the
SAC policy from goal-dominant to energy-dominant yields a **paired energy
saving of 103 ± 196 J (13.0%, Wilcoxon p < 10⁻⁴)** at constant success
rate. The goal–energy trade-off exists only when `v_max` exceeds the
maximum-range speed of the power curve (≈18.3 m/s); below it the objectives
align and the Pareto front collapses (verified analytically and
empirically).

## Project structure

```
uav-morl/
├── configs/
│   ├── default.yaml        # Easy scenario (10 obstacles, 70% dynamic, v_max=8)
│   └── hard.yaml           # Paper scenario (20 obstacles, 100% dynamic, v_max=25)
├── src/
│   ├── envs/
│   │   ├── uav_env.py      # Gymnasium multi-objective UAV environment
│   │   ├── obstacles.py    # Dynamic obstacle field + ray-casting sensor
│   │   └── energy.py       # Rotary-wing energy model (Zeng & Zhang, 2019)
│   ├── baselines/classical.py  # APF oracle and replanning A*
│   ├── train.py            # PPO / SAC / TD3 training (parallel envs, callbacks)
│   ├── evaluate.py         # Evaluation + Pareto preference sweep
│   ├── analyze.py          # Significance tests + paired weight-effect analysis
│   ├── plots.py            # Publication-quality figures (English, 300 DPI)
│   ├── run_experiment.py   # End-to-end pipeline (single command)
│   └── utils/common.py     # Config, seeding, metric helpers
├── tests/test_env.py       # 8 unit tests (physics, API, determinism)
├── scripts/*.bat           # Windows 11 helper scripts
├── paper/                  # Manuscript drafts and LaTeX skeleton
└── results/                # models / logs / metrics / figures
```

## Installation (Windows 11 + VS Code)

1. Install **Python 3.10–3.12** (check "Add python.exe to PATH").
2. Open the project folder in VS Code and run in the terminal:

```bat
scripts\setup_venv.bat
pip install scipy
```

3. Select `.venv\Scripts\python.exe` as the interpreter
   (`Ctrl+Shift+P → Python: Select Interpreter`).
4. Verify — all 8 tests must pass:

```bat
.venv\Scripts\activate
python -m pytest tests -q
```

> **GPU (optional):** `pip install torch --index-url https://download.pytorch.org/whl/cu121`

Linux/macOS: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt scipy`.

## Usage

### Quick smoke test (~5 min)
```bat
python -m src.run_experiment --quick --config configs\hard.yaml
```

### Full pipeline (train → evaluate → baselines → Pareto → figures)
```bat
python -m src.run_experiment --algo PPO --seed 42 --config configs\hard.yaml
```
The `--config` argument propagates to **every** stage; run names and metric
tags automatically include the scenario (e.g. `ppo_hard_seed42`), so
different scenarios never overwrite each other.

### Step by step
```bat
:: Training (PPO | SAC | TD3)
python -m src.train --config configs\hard.yaml --algo PPO --seed 42 --timesteps 3000000 --run-name ppo_hard_seed42
python -m src.train --config configs\hard.yaml --algo SAC --seed 42 --timesteps 1000000 --run-name sac_hard_seed42
python -m src.train --config configs\hard.yaml --algo TD3 --seed 42 --timesteps 1000000 --run-name td3_hard_seed42

:: Monitoring
tensorboard --logdir results\logs

:: Evaluation (100 episodes) and baselines
python -m src.evaluate --config configs\hard.yaml --model results\models\ppo_hard_seed42\best_model.zip --algo PPO --tag ppo_hard
python -m src.evaluate --config configs\hard.yaml --baseline apf --tag apf_hard
python -m src.evaluate --config configs\hard.yaml --baseline astar --tag astar_hard

:: Pareto preference sweep (8 weight vectors × 300 shared worlds)
python -m src.evaluate --config configs\hard.yaml --model results\models\sac_hard_seed42\best_model.zip --algo SAC --pareto --tag sac_hard_300 --episodes 300

:: Statistical analysis
python -m src.analyze --compare results\metrics\ppo_hard_episodes.csv results\metrics\apf_hard_episodes.csv results\metrics\astar_hard_episodes.csv
python -m src.analyze --pareto results\metrics\sac_hard_300_pareto.csv

:: Figures (English labels, 300 DPI)
python -m src.plots --config configs\hard.yaml --compare results\metrics
python -m src.plots --config configs\hard.yaml --pareto results\metrics\sac_hard_300_pareto.csv
python -m src.plots --config configs\hard.yaml --trajectory --model results\models\ppo_hard_seed42\best_model.zip --algo PPO
python -m src.plots --config configs\hard.yaml --learning results\logs
```

## Metrics

Reported per method (`results/metrics/*_summary.json`): success rate,
collision rate, per-episode propulsion energy [J], energy per metre [J/m],
path length [m], path efficiency (straight-line/flown), minimum obstacle
clearance [m], flight time [s]. `analyze.py` adds Welch t / Mann–Whitney U
(energy), Fisher exact (success), and paired Wilcoxon signed-rank tests
(preference control on identical worlds).

## Problem formulation (summary)

MOMDP `M = (S, A, P, r, γ)` with `r ∈ R³ = [r_goal, r_energy, r_safety]`.
**Observation (43-D):** relative goal (3) + velocity (3) + 16-ray range scan
(16) + previous scan (16) + battery fraction (1) + min. clearance (1) +
preference vector `w` (3). **Action:** continuous 3-D acceleration in
`[-1,1]³` (scale 8 m/s²). **Scalarisation:** linear `w·r`,
`w ~ Dirichlet(1)` during training. **Termination:** goal (< 3 m),
collision, battery depletion (60 kJ), or 400 steps. Full details in the
manuscript and `configs/hard.yaml`.

## Reproducibility

- All hyperparameters in `configs/*.yaml`; seeds fixed via `set_global_seed`
- Environment determinism covered by unit tests
- Raw per-episode CSVs retained for statistical re-analysis
- Ablation switches: `ray_history: false` (temporal sensing),
  `v_max` (objective alignment), reward weights/scales

## Citation

If you use this code, please cite the accompanying manuscript (under
review) and:

- Y. Zeng, J. Xu, R. Zhang, "Energy Minimization for Wireless Communication
  with Rotary-Wing UAV," *IEEE Trans. Wireless Commun.*, 18(4), 2019.
- A. Raffin et al., "Stable-Baselines3," *JMLR*, 22(268), 2021.
- M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement
  Learning Environments," arXiv:2407.17032, 2024.

## License

MIT
