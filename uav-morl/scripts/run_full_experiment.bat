@echo off
call .venv\Scripts\activate.bat
REM Tam deney: PPO egitimi (1M adim) + degerlendirme + baseline + Pareto + figurler
python -m src.run_experiment --algo PPO --seed 42
