# scripts/

Utilidades de generación (fuera de los notebooks). **Se ejecutan desde la raíz del repositorio**,
para que las rutas relativas (`salidas/…`, `data/clean/…`, `../herramienta/public`) resuelvan bien.

| Carpeta | Contenido | Estado |
|---|---|---|
| `herramienta/` | `_herramienta_datos.py` (data.json) · `_herramienta_fase1.py` (Grad-CAM/Rx/ECG + curados) · `_herramienta_gradcam.py` (Grad-CAM por patología) · `_herramienta_labs.py` (analíticas por relevancia clínica) · `_gen_qr.py` (QR del enlace) | ✅ vigente |
| `fusion/` | `_fusion_v3.py` (fusión tardía v3, la embarcada) · `_fusion_v4v5.py` (fusiones exploradas, no batieron a v2) | ✅ vigente / experimental |
| `informes/` | `_prep_informe_modelos.py` · `_figs_informe_modelos.py` (tablas y figuras del Informe de Modelos) | ✅ vigente |
| `legacy/` | `OBSOLETO_*` — parches puntuales ya aplicados a los informes | ⚠️ superado |

> **`legacy/`**: no se borra porque documenta cómo se generaron partes de la entrega y sirve para
> reproducirla. El prefijo `OBSOLETO_` avisa de que no forma parte del flujo actual.
