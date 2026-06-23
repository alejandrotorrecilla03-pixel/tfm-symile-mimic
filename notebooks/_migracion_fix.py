# -*- coding: utf-8 -*-
# Arregla colisiones de carpetas: mueve las BUENAS a su sitio (v3_2->v3, v2_2->v2) tras quitar las obsoletas,
# y actualiza referencias que quedaban (stacking v1, generadores v3/v4). Verifica antes de borrar.
import os, shutil, json, glob
from pathlib import Path
NB=Path(r"C:\TFM\1.Opción - Symile Mimic\tfm_multimodal_clinico\notebooks")

def is_obsolete_ecg_v3(d):
    js=glob.glob(str(d/"summary*.json"))
    if not js: return True
    try: s=json.load(open(js[0],encoding="utf-8")); mp=s.get("test_macro_path",1)
    except: return True
    return mp is not None and mp < 0.63   # la vieja v3 era 0.626

# --- LABS: outputs_labs_tabular_v2 (vieja, vacía) -> borrar; v2_2 -> v2 ---
old=NB/"outputs_labs_tabular_v2"; good=NB/"outputs_labs_tabular_v2_2"
if good.exists():
    n=len(list(old.rglob("*"))) if old.exists() else 0
    print(f"LABS vieja v2: {n} ficheros (se espera 0)")
    if old.exists() and n==0: shutil.rmtree(old)
    if not old.exists(): os.rename(good,old); print("LABS: v2_2 -> v2 (datos movidos)")
    else: print("LABS: la vieja v2 NO estaba vacía, no toco nada")

# --- ECG: outputs_ecg_resnet1d_v3 (vieja 0.626) -> borrar; v3_2 -> v3 ---
old=NB/"outputs_ecg_resnet1d_v3"; good=NB/"outputs_ecg_resnet1d_v3_2"
if good.exists():
    obs=is_obsolete_ecg_v3(old) if old.exists() else True
    print(f"ECG vieja v3 obsoleta (0.626)? {obs}")
    if old.exists() and obs: shutil.rmtree(old)
    if not old.exists(): os.rename(good,old); print("ECG: v3_2 -> v3 (datos movidos)")
    else: print("ECG: la vieja v3 NO parece obsoleta, no toco nada")

# --- Actualizar refs que quedaban ---
ren={"outputs_cxr_densenet121_v5":"outputs_cxr_densenet121_v2",
     "outputs_ecg_resnet1d_v3_1":"outputs_ecg_resnet1d_v2",
     "outputs_ecg_resnet1d_v3_2":"outputs_ecg_resnet1d_v3",
     "outputs_labs_tabular_v2_2":"outputs_labs_tabular_v2"}
for fn in ["04_STACKING_Multimodal_v1.ipynb","_gen_v3.py","_gen_v4.py","_test_v4.py","analisis_fusion.py"]:
    t=NB/fn
    if not t.exists(): continue
    s=t.read_text(encoding="utf-8"); o=s
    for a,b in ren.items(): s=s.replace(a,b)
    if s!=o: t.write_text(s,encoding="utf-8"); print("refs actualizadas:",fn)
print("=== FIX OK ===")
