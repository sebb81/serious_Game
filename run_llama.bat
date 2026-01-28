@echo off
:: Définir le dossier de cache DANS le dossier courant pour que ce soit portable
set HF_HOME=%~dp0\python\cache_huggingface
set TORCH_HOME=%~dp0\python\cache_torch

:: Lancer l'app
echo Lancement de l'Atelier IA Frugale...
echo Ne fermez pas cette fenetre.
call "python\python-3.12.6.amd64\python.exe" -m streamlit run app.py
pause