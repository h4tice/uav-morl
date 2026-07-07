@echo off
call .venv\Scripts\activate.bat
REM Makale icin 3 bagimsiz tohumla PPO ve SAC egitimi
for %%S in (42 43 44) do (
    python -m src.train --algo PPO --seed %%S --run-name ppo_seed%%S
    python -m src.train --algo SAC --seed %%S --run-name sac_seed%%S
)
