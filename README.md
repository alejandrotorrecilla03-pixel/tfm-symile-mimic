# TFM · Apoyo multimodal a la decisión clínica (Symile-MIMIC-IV)

Trabajo Fin de Máster (Universidad de Salamanca). Sistema **multimodal** que, para un mismo
ingreso hospitalario, combina **radiografía de tórax (CXR)**, **electrocardiograma (ECG)** y
**analítica de sangre (LABS)** y predice seis hallazgos radiográficos —Atelectasia,
Cardiomegalia, Edema, Opacidad pulmonar, Derrame pleural y «Sin hallazgo»— como **apoyo al
triaje, nunca como decisor autónomo**.

> **Datos (DUA de MIMIC-IV).** El repositorio contiene **solo código**. Los datos, los volúmenes
> `.npy`, los modelos entrenados y las salidas derivadas (con `hadm_id`) están **excluidos por
> `.gitignore`**: no pueden redistribuirse públicamente según el acuerdo de uso de MIMIC-IV.

## Resultados (test, n = 464)

| Modelo | Métrica | Valor |
|---|---|---|
| CXR (DenseNet121/CheXNet) v2 | AUC-ROC | **0,747** — la modalidad más fuerte |
| Fusión multimodal (stacking v2) | AUC-PR macro | **0,594** vs 0,545 monomodal |
| Modelo embarcado en la app | Fusión tardía **v3** (media ponderada por patología) | — |

Pipeline **congelado**: modelos = *train* · calibración isotónica, pesos de fusión y umbrales =
*validación* · *test* se reporta una sola vez.

## Estructura del repositorio

```
tfm_multimodal_clinico/            (este repo — solo código)
├── notebooks/                     11 notebooks del pipeline (v1/v2/v3 por modalidad + stacking)
│   └── eda/                       EDA reproducible como script (imagen/ECG + tabular)
├── scripts/
│   ├── herramienta/               genera los datos/activos de la app clínica (data.json, Grad-CAM, ECG, analíticas, QR)
│   ├── fusion/                    experimentos de fusión tardía (v3 embarcado, v4/v5 explorados)
│   ├── informes/                  tablas y figuras para el Informe de Modelos
│   └── legacy/                    scripts superados (prefijo OBSOLETO_): se conservan para reproducir
├── data/build_datasets.py         construcción del conjunto de trabajo (data/ y salidas/ ignorados)
├── run_nb.py                      ejecuta un notebook como script (evita el OOM del kernel de Jupyter)
├── RUNBOOK.md                     orden de ejecución del pipeline y avisos
└── Lanzar_Herramienta_Clinica.bat lanzador de la app (sirve el build de producción)
```

### Componentes hermanos (fuera de este repo, deliverables separados)

- **`../herramienta/`** — la app clínica (React + Vite, estática, precomputada). Ver `herramienta/DESPLIEGUE.md`.
- **`../_build_tfm/`** — builders de la documentación en Word/PDF (docx-js).
- **`../DOCUMENTACIÓN/`** — informes finales (EDA, Modelos, Herramienta).

## Cómo ejecutar

```bash
# 1) crear el entorno
python -m venv venv && venv/Scripts/pip install -r requirements.txt

# 2) ejecutar un notebook del pipeline (ver orden en RUNBOOK.md)
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe run_nb.py "notebooks/03_LABS_Tabular_v1.ipynb" > logs/labs_v1.log 2>&1

# 3) regenerar los datos/activos de la app clínica
venv/Scripts/python.exe scripts/herramienta/_herramienta_datos.py
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/herramienta/_herramienta_fase1.py
```

Detalles de hardware, memoria y verificación de resultados en [RUNBOOK.md](RUNBOOK.md).
