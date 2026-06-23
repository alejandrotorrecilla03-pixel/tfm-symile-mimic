# -*- coding: utf-8 -*-
# Consolida los 4 stacking en 2: borra v1 (regenera) y v3 (subconjunto de v4); renumera v2->v1 (base) y v4->v2 (avanzado).
import os, shutil, json
from pathlib import Path
NB=Path(r"C:\TFM\1.Opción - Symile Mimic\tfm_multimodal_clinico\notebooks")

# 1) BORRAR superados (notebooks + carpetas + generadores obsoletos)
for f in ["04_STACKING_Multimodal_v1.ipynb","04_STACKING_Multimodal_v3.ipynb","_gen_v3.py","_gen_v4.py"]:
    p=NB/f
    if p.exists(): p.unlink(); print("borrado:",f)
for d in ["outputs_stacking","outputs_stacking_v3"]:
    p=NB/d
    if p.exists(): shutil.rmtree(p); print("borrada carpeta:",d)

# 2) RENOMBRAR notebooks + carpetas (orden: v2->v1 libera 'v2' para v4->v2)
def mv(a,b):
    pa,pb=NB/a,NB/b
    if pa.exists() and not pb.exists(): os.rename(pa,pb); print("ren:",a,"->",b)
    elif pb.exists(): print("ya existe:",b)
mv("04_STACKING_Multimodal_v2.ipynb","04_STACKING_Multimodal_v1.ipynb")   # base
mv("outputs_stacking_v2","outputs_stacking_v1")
mv("04_STACKING_Multimodal_v4.ipynb","04_STACKING_Multimodal_v2.ipynb")   # avanzado
mv("outputs_stacking_v4","outputs_stacking_v2")

# 3) ACTUALIZAR referencias internas a las carpetas de salida
def repl(fn, a, b):
    p=NB/fn
    if not p.exists(): return
    s=p.read_text(encoding="utf-8")
    if a in s: p.write_text(s.replace(a,b),encoding="utf-8"); print("refs:",fn,a,"->",b)
repl("04_STACKING_Multimodal_v1.ipynb","outputs_stacking_v2","outputs_stacking_v1")
repl("04_STACKING_Multimodal_v2.ipynb","outputs_stacking_v4","outputs_stacking_v2")
print("=== CONSOLIDACIÓN OK ===")
