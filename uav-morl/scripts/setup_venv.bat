@echo off
REM ============================================================
REM  UAV-MORL: Windows 11 kurulum betigi (VS Code terminalinde calistirin)
REM  Kullanim: proje kok dizininde  ->  scripts\setup_venv.bat
REM ============================================================
python --version || (echo Python bulunamadi! https://python.org adresinden 3.10+ kurun & exit /b 1)
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Kurulum tamam. Ortami etkinlestirmek icin:  .venv\Scripts\activate
echo Hizli test:  python -m pytest tests -q
