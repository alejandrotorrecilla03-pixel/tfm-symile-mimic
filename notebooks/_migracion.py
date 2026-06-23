# -*- coding: utf-8 -*-
# Migración a esquema limpio v1/v2/v3: renombra los MEJORES notebooks y sus carpetas de salida
# (conservando outputs) y actualiza todas las referencias downstream. NO borra nada.
import os
from pathlib import Path
NB=Path(r"C:\TFM\1.Opción - Symile Mimic\tfm_multimodal_clinico\notebooks"); ROOT=NB.parent

nb_ren={
 "01_CXR_DenseNet121_v5.ipynb":"01_CXR_DenseNet121_v2.ipynb",
 "02_ECG_ResNet1D_v3_1.ipynb":"02_ECG_ResNet1D_v2.ipynb",
 "02_ECG_ResNet1D_12lead_v3_2.ipynb":"02_ECG_ResNet1D_v3.ipynb",
 "03_LABS_Tabular_v2_2.ipynb":"03_LABS_Tabular_v2.ipynb",
}
dir_ren={
 "outputs_cxr_densenet121_v5":"outputs_cxr_densenet121_v2",
 "outputs_ecg_resnet1d_v3_1":"outputs_ecg_resnet1d_v2",
 "outputs_ecg_resnet1d_v3_2":"outputs_ecg_resnet1d_v3",
 "outputs_labs_tabular_v2_2":"outputs_labs_tabular_v2",
}
for a,b in nb_ren.items():
    pa,pb=NB/a,NB/b
    if pa.exists() and not pb.exists(): os.rename(pa,pb); print("notebook:",a,"->",b)
    elif pb.exists(): print("notebook ya existe:",b)
for a,b in dir_ren.items():
    pa,pb=NB/a,NB/b
    if pa.exists() and not pb.exists(): os.rename(pa,pb); print("carpeta :",a,"->",b)
    elif pb.exists(): print("carpeta ya existe:",b)

# Actualizar referencias a las carpetas en notebooks/scripts downstream
targets=[NB/x for x in [
 "01_CXR_DenseNet121_v2.ipynb","02_ECG_ResNet1D_v2.ipynb","02_ECG_ResNet1D_v3.ipynb","03_LABS_Tabular_v2.ipynb",
 "04_STACKING_Multimodal_v2.ipynb","04_STACKING_Multimodal_v3.ipynb","04_STACKING_Multimodal_v4.ipynb",
 "05_HERRAMIENTA_DECISION_v1.ipynb","_explore_tool.py","_gen_informe_tool.py","_gen_tool.py","_test_v4.py","analisis_fusion.py"]]
targets.append(ROOT/"app_decision.py")
for t in targets:
    if not t.exists(): continue
    s=t.read_text(encoding="utf-8"); o=s
    for a,b in dir_ren.items(): s=s.replace(a,b)
    if s!=o: t.write_text(s,encoding="utf-8"); print("refs actualizadas:",t.name)
print("=== MIGRACION OK ===")
