# RUNBOOK — Ejecución del pipeline TFM Symile-MIMIC

Guía para ejecutar los 11 notebooks ya adaptados al **etiquetado FINAL** y a la métrica **AUC-PR**.

---

## ⚠️ Lee esto antes de lanzar nada

### 1. Usa `run_nb.py`, no `jupyter nbconvert`
En este equipo **el kernel de Jupyter muere por falta de memoria** (`DeadKernelError: Kernel died`),
incluso en notebooks que como script plano consumen menos de 1 GB. Se comprobó midiendo celda a celda:
ECG v1 murió con nbconvert pero se ejecutó entero como script con un pico de **987 MB**.

```bash
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe run_nb.py "notebooks/<NOTEBOOK>.ipynb" > logs/<NOTEBOOK>.log 2>&1
echo "EXIT=$?"
```

**Contrapartida:** el `.ipynb` **no queda con las salidas incrustadas**. Los CSV, JSON y figuras sí se
escriben, que es lo que consumen el stacking y la documentación. Si necesitas el notebook con sus
outputs para la entrega, ábrelo y ejecútalo en Jupyter cuando tengas RAM libre.

### 2. Nunca uses un pipe para comprobar el resultado
`comando | tail` devuelve el código de salida de `tail`, **no el del comando**. Eso ya nos hizo dar
por buena una ejecución que había fallado. Redirige a fichero y comprueba `$?`.

### 3. Cómo verificar de verdad que algo se ejecutó
Un `EXIT=0` **no basta**. Comprueba que se escribieron los ficheros de salida con fecha reciente:

```bash
ls -la salidas/<modalidad>/<version>/
```

### 4. Memoria
El equipo tiene 16,8 GB pero suele haber **solo ~2 GB libres**. Cierra aplicaciones antes de las
ejecuciones largas. **Lanza un notebook cada vez**: dos en paralelo garantizan el OOM y además la CPU
(4 núcleos físicos) ya se satura con uno.

---

## Orden de ejecución

> ⚠️ **En imagen el orden se invierte: CXR v2 va ANTES que v1**, porque v1 no procesa imágenes: carga
> los embeddings que genera v2 y aborta con `assert` si no existen.

| # | Notebook | Estado | Coste | Notas |
|---|---|---|---|---|
| 1 | `03_LABS_Tabular_v1` | ✅ **hecho** | minutos | macro AP = **0,3968** [0,368–0,436] |
| 2 | `03_LABS_Tabular_v2` | ✅ **hecho** | ~2 h | macro AP = **0,3910** [0,362–0,429] |
| 3 | `03_LABS_Tabular_v3` | ⬜ | ~2 h | TabPFN solo si defines `TABPFN_TOKEN` |
| 4 | `02_ECG_ResNet1D_v1` | ✅ **hecho** | ~30 min | macro AP = **0,3667** [0,339–0,405] |
| 5 | `02_ECG_ResNet1D_v2` | ⬜ | ~1–2 h | |
| 6 | `02_ECG_ResNet1D_v3` | ⬜ | horas | señal 2500, K=5 |
| 7 | **`01_CXR_DenseNet121_v2`** | ⬜ | **horas** | **Genera los embeddings.** Va antes que v1 |
| 8 | `01_CXR_DenseNet121_v1` | ⬜ | minutos | Solo si el paso 7 terminó |
| 9 | `01_CXR_DenseNet121_v3` | ⬜ | horas | 2 backbones + TTA + bagging |
| 10 | `04_STACKING_Multimodal_v1` | ⬜ | minutos | Necesita las 3 modalidades |
| 11 | `04_STACKING_Multimodal_v2` | ⬜ | ~30 min | Necesita las 3 modalidades |

El **stacking va el último**: lee los CSV `*_pred_{oof_train,val,test}.csv` de las tres modalidades
desde `salidas/01_cxr/v2`, `salidas/02_ecg/v2` y `salidas/03_labs/v2`.

---

## Qué produce cada notebook

En su carpeta `salidas/<modalidad>/<version>/`:

- `metrics_per_label_*.csv` — AP, AUC, IC bootstrap y prevalencia por etiqueta
- `puntos_operacion_test.csv` — Se/Sp/VPP/VPN en los tres puntos (F1, cribado, confirmación)
- `calibracion_brier.csv` — Brier antes y después de calibrar
- `estratificacion_subgrupos.csv` — rendimiento por sexo, etnia, ingreso y proyección AP/PA
- `preregistro_regla_decision.json` — la regla congelada **antes** de tocar test
- `<mod>_pred_{oof_train,val,test}.csv` — predicciones para el stacking
- `figuras/` — curva de fiabilidad, importancia, AP vs prevalencia

---

## Avisos sobre los resultados

**El AUC baja respecto a las cifras antiguas y es correcto.** El etiquetado final convierte los NaN en
negativos, multiplicando por ~4 el número de negativos e incluyendo tórax anormales por otra causa.
La tarea es más difícil pero clínicamente honesta. **Las cifras anteriores no son comparables.**

**En imagen baja además por otro motivo:** `cxr_view` ya no entra como predictor. Era un proxy de
gravedad (la placa AP se hace al paciente encamado) y parte del rendimiento previo venía de ese atajo.

**LABS v2 no superó a v1** (0,3910 vs 0,3968, con IC solapados). Es el techo de la señal tabular que
anticipaba el EDA, no un fallo.

**LABS v2 se ejecutó con los *cluster scores* aún dentro.** El A/B posterior demostró que no aportan
(diferencia +0,0005, IC [−0,0010, +0,0019]) y se retiraron del código. La diferencia es ruido, así que
no invalida el resultado; reejecuta v2 solo si quieres coherencia estricta entre código y cifras.

---

## Si algo falla

| Síntoma | Causa probable | Solución |
|---|---|---|
| `DeadKernelError: Kernel died` | OOM del kernel de Jupyter | Usa `run_nb.py` |
| Muere sin traceback en el tuning | RAM insuficiente | Cierra aplicaciones |
| CXR v1 aborta con `assert` | Faltan los embeddings | Ejecuta antes CXR v2 |
| Se queda colgado sin avanzar | TabPFN pidiendo login | Define `TABPFN_TOKEN` o déjalo desactivado |
| `EXIT=0` pero sin resultados | Usaste un pipe | Redirige a fichero y comprueba `$?` |
