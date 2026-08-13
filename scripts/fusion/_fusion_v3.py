"""STACKING v3 — nuevas familias de fusión, selección por etiqueta.
Explora familias que v2 no enfatizaba: media aritmética, media geométrica, media de rangos,
PESOS NO-NEGATIVOS ÓPTIMOS por AP en validación (búsqueda en el símplex) y meta-LogReg regularizada.
Protocolo del tutor: ajusta/selecciona en VAL, reporta UNA vez en TEST (AP primaria + IC bootstrap).
Ligero (sin matplotlib). Salida: salidas/04_stacking/v3/.
"""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

np.random.seed(42)
LAB   = ["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
LAB_U = [l.replace(" ","_") for l in LAB]
PATH  = [l for l in LAB if l != "No Finding"]          # 5 patologías (métrica primaria)
MODS  = ["CXR","ECG","LABS"]
OUT   = "salidas/04_stacking/v3"; os.makedirs(OUT, exist_ok=True)

def targets(df):
    raw = df[LAB].to_numpy(dtype=float)
    y = (raw == 1.0).astype(np.float32)                # POS=1 ; NEG = 0 + NaN
    m = (raw != -1.0).astype(np.float32)               # −1 ENMASCARADO (U-Ignore)
    return y, m

def load(split_bp, split_cl):
    bp = pd.read_csv(f"salidas/04_stacking/v1/base_preds_{split_bp}.csv")
    cl = pd.read_csv(f"data/clean/{split_cl}_clean.csv", sep=";")
    df = bp.merge(cl[["hadm_id"] + LAB], on="hadm_id", how="inner")
    y, m = targets(df)
    # X[modalidad] -> matriz (N, 6) de probabilidades
    X = {mod: df[[f"{mod}_{u}" for u in LAB_U]].to_numpy(dtype=float) for mod in MODS}
    return X, y, m

Xv, yv, mv = load("val", "val")
Xt, yt, mt = load("test", "test")
print(f"val N={len(yv)} · test N={len(yt)}")

def ap(y, p, msk):
    s = msk == 1
    if s.sum() < 2 or y[s].sum() == 0: return np.nan
    return average_precision_score(y[s], p[s])

def col(X, j):  # (p_cxr, p_ecg, p_labs) para la etiqueta j
    return X["CXR"][:, j], X["ECG"][:, j], X["LABS"][:, j]

# ── familias de fusión (cada una: (probs_val, probs_test) para la etiqueta j) ──
def f_cxr(j):   return Xv["CXR"][:, j], Xt["CXR"][:, j]                 # referencia mono
def f_avg(j):
    return sum(col(Xv,j))/3, sum(col(Xt,j))/3
def f_geo(j):
    cv=col(Xv,j); ct=col(Xt,j); e=1e-6
    return (np.clip(cv[0],e,1)*np.clip(cv[1],e,1)*np.clip(cv[2],e,1))**(1/3), \
           (np.clip(ct[0],e,1)*np.clip(ct[1],e,1)*np.clip(ct[2],e,1))**(1/3)
def f_rank(j):
    cv=col(Xv,j); ct=col(Xt,j)
    rv=sum(rankdata(c)/len(c) for c in cv)/3
    rt=sum(rankdata(c)/len(c) for c in ct)/3
    return rv, rt
def f_wopt(j):
    cv=np.stack(col(Xv,j),1); ct=np.stack(col(Xt,j),1)
    grid=[(a/10,b/10,(10-a-b)/10) for a in range(11) for b in range(11-a)]
    best,bw=-1,(1,0,0)
    for w in grid:
        a=ap(yv[:,j], cv@np.array(w), mv[:,j])
        if not np.isnan(a) and a>best: best,bw=a,w
    return cv@np.array(bw), ct@np.array(bw), bw
def f_logreg(j):
    s=mv[:,j]==1
    Xtr=np.stack(col(Xv,j),1); Xte=np.stack(col(Xt,j),1)
    lr=LogisticRegression(C=0.5,max_iter=1000).fit(Xtr[s], yv[s,j])
    return lr.predict_proba(Xtr)[:,1], lr.predict_proba(Xte)[:,1]

FUSERS = {"CXR_mono":f_cxr, "media":f_avg, "geometrica":f_geo,
          "rangos":f_rank, "pesos_opt_AP":f_wopt, "meta_logreg":f_logreg}

# ── selección POR ETIQUETA por AP en validación; reporte en test ──
rows=[]; test_pred=np.zeros((len(yt),6),np.float32); elegido={}; pesos={}
for j,lab in enumerate(LAB):
    mejores=None; best_val=-1
    fila={"etiqueta":lab}
    for name,fn in FUSERS.items():
        out=fn(j); pv,pt=out[0],out[1]
        av=ap(yv[:,j],pv,mv[:,j]); at=ap(yt[:,j],pt,mt[:,j])
        fila[f"{name}_val"]=round(float(av),4); fila[f"{name}_test"]=round(float(at),4)
        if not np.isnan(av) and av>best_val:
            best_val=av; mejores=(name,pt);
            if name=="pesos_opt_AP": pesos[lab]=out[2]
    elegido[lab]=mejores[0]; test_pred[:,j]=mejores[1]; rows.append(fila)

# macro AP test sobre las 5 patologías
def macro(pred):
    aps=[ap(yt[:,LAB.index(l)],pred[:,LAB.index(l)],mt[:,LAB.index(l)]) for l in PATH]
    return float(np.nanmean(aps)), aps
macro_v3,_=macro(test_pred)
macro_cxr,_=macro(np.stack([Xt["CXR"][:,j] for j in range(6)],1))

# IC bootstrap de la macro-AP (test)
def boot(pred,n=1000):
    N=len(yt); out=[]
    for _ in range(n):
        idx=np.random.randint(0,N,N)
        aps=[]
        for l in PATH:
            j=LAB.index(l); s=(mt[idx,j]==1)
            if s.sum()<2 or yt[idx,j][s].sum()==0: continue
            aps.append(average_precision_score(yt[idx,j][s],pred[idx,j][s]))
        if aps: out.append(np.mean(aps))
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5))
ic_v3=boot(test_pred); ic_cxr=boot(np.stack([Xt["CXR"][:,j] for j in range(6)],1))

pd.DataFrame(rows).to_csv(f"{OUT}/comparativa_fusion_v3.csv",index=False)
V2=0.5942
res={"macro_ap_v3":round(macro_v3,4),"ic_v3":[round(x,4) for x in ic_v3],
     "macro_ap_cxr_mono":round(macro_cxr,4),"ic_cxr":[round(x,4) for x in ic_cxr],
     "macro_ap_v2_referencia":V2,
     "mejora_vs_mono":round(macro_v3-macro_cxr,4),"mejora_vs_v2":round(macro_v3-V2,4),
     "ic_solapan_con_mono":not (ic_v3[0]>ic_cxr[1] or ic_cxr[0]>ic_v3[1]),
     "fuser_elegido_por_etiqueta":elegido,"pesos_opt":pesos}
json.dump(res,open(f"{OUT}/veredicto_v3.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
print("FIN OK")
