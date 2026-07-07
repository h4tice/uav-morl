@echo off
call .venv\Scripts\activate.bat
REM ~5 dakikalik duman testi: tum boru hatti kucuk olcekte calisir
python -m src.run_experiment --quick
