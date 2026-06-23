@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Herramienta de Apoyo a la Decision Clinica - TFM
echo ============================================================
echo   HERRAMIENTA DE APOYO A LA DECISION CLINICA  (TFM)
echo   Modelos optimos v2: CXR + ECG + Analiticas
echo ------------------------------------------------------------
echo   Iniciando... se abrira sola en tu navegador.
echo   Si no se abre: http://localhost:8501
echo   Para cerrarla: pulsa Ctrl+C en esta ventana.
echo ============================================================
echo.
"venv\Scripts\python.exe" -m streamlit run app_decision.py --server.address 127.0.0.1 --server.port 8501
echo.
echo La herramienta se ha cerrado. Puedes cerrar esta ventana.
pause
