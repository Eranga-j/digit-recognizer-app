@echo off
cd /d "%~dp0"

call .venv\Scripts\activate

python gradio_app.py

pause
