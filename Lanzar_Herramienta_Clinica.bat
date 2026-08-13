@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Herramienta de Apoyo a la Decision Clinica - TFM
echo ============================================================
echo   HERRAMIENTA DE APOYO A LA DECISION CLINICA  (TFM)
echo   App React (fusion multimodal v3): CXR + ECG + Analiticas
echo ------------------------------------------------------------
echo   Sirviendo el build de produccion...
echo   Abrela en: http://localhost:5178
echo   Para cerrarla: pulsa Ctrl+C en esta ventana.
echo ============================================================
echo.
cd /d "%~dp0\..\herramienta"
call npm run preview -- --host --port 5178
echo.
echo La herramienta se ha cerrado. Puedes cerrar esta ventana.
pause
