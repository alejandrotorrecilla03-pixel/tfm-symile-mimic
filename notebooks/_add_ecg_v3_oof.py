# -*- coding: utf-8 -*-
# Añade captura de OOF + celda de export al notebook ECG v3 (pesado), sin tocar el resto.
import json
from pathlib import Path
P=Path("02_ECG_ResNet1D_v3.ipynb")
nb=json.load(open(P,encoding="utf-8"))
def src(c): return "".join(c["source"]) if isinstance(c["source"],list) else c["source"]

kf_i=cal_i=None
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    s=src(c)
    if "val_pred_raw=acc_val/K_FOLDS" in s and "fold_rows=[]" in s: kf_i=i
    if "val_pred=apply_cal(val_pred_raw)" in s: cal_i=i
assert kf_i is not None and cal_i is not None, f"no encuentro celdas (kf={kf_i}, cal={cal_i})"

# 1) Captura de OOF en la celda K-fold
s=src(nb["cells"][kf_i])
s=s.replace("fold_rows=[]; acc_val=",
            "fold_rows=[]; oof_train=np.zeros((len(df_train),N_LABELS),np.float32); acc_val=")
s=s.replace("mm=multilabel_metrics(predict(model,X_train[va],meta_train[va]),y_train[va],m_train[va])",
            "pva=predict(model,X_train[va],meta_train[va]); oof_train[va]=pva\n    mm=multilabel_metrics(pva,y_train[va],m_train[va])")
nb["cells"][kf_i]["source"]=s
assert "oof_train[va]=pva" in s, "no se inyectó la captura de OOF"

# 2) Celda de export tras la calibración
export=(
"# CELDA 13b · EXPORTAR OOF/val/test PARA EL STACKING (hadm_id, ecg_<label>, ecg_<label>_cal)\n"
"def save_predictions(df, raw, cal, name):\n"
"    cols={\"hadm_id\": df[\"hadm_id\"].to_numpy()}\n"
"    for j,l in enumerate(LABELS):\n"
"        key=l.replace(\" \",\"_\"); cols[f\"ecg_{key}\"]=raw[:,j]; cols[f\"ecg_{key}_cal\"]=cal[:,j]\n"
"    out=pd.DataFrame(cols); p=OUTPUT_DIR/f\"ecg_pred_{name}.csv\"; out.to_csv(p,index=False); print(\"   guardado\",p.name)\n"
"oof_cal=apply_cal(oof_train)\n"
"save_predictions(df_train, oof_train,     oof_cal,   \"oof_train\")\n"
"save_predictions(df_val,   val_pred_raw,  val_pred,  \"val\")\n"
"save_predictions(df_test,  test_pred_raw, test_pred, \"test\")\n"
"print(\"OOF/val/test del ECG v3 exportados -> los carga el stacking.\")")
cell={"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":export}
nb["cells"].insert(cal_i+1, cell)

json.dump(nb, open(P,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"OK · K-fold celda {kf_i}, export insertada tras {cal_i} · total celdas {len(nb['cells'])}")
