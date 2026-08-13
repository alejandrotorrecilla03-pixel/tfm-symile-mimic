"""Datos nuevos para los arreglos del informe (ejecutar DESPUÉS de _prep_informe_modelos.py):
  C1 · Prevalencia media por subgrupo -> reescribe 09_estratificacion (con columna prevalencia).
  C2 · Puntos de operación del FUSOR v3 (modelo final): umbrales fijados en VAL, reportados en TEST.
  C3 · Guarda las diferencias bootstrap emparejadas para la figura del test emparejado.
La fusión v3 = media ponderada por patología (pesos de veredicto_v3.json).
"""
import numpy as np, pandas as pd, json, csv, os
from sklearn.metrics import average_precision_score, roc_auc_score

SEED=42; rng=np.random.RandomState(SEED)
S="salidas"; TAB=f"{S}/_informe_modelos/tablas"
LAB=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
LABU=[l.replace(" ","_") for l in LAB]; DIS=[0,1,2,3,5]
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema",
    "Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgo","Pleural Effusion":"Derrame pleural"}
MOD=["CXR","ECG","LABS"]
def f(x,n=3):
    if x is None or (isinstance(x,float) and np.isnan(x)): return ""
    return f"{float(x):.{n}f}".replace(".",",")

def load(sp):
    d=pd.read_csv(f"data/clean/{sp}_clean.csv",sep=";")
    raw=d[LAB].to_numpy(float)
    hid=d["hadm_id"].to_numpy()
    fn={"train":"oof_train","val":"val","test":"test"}[sp]
    bp=pd.read_csv(f"{S}/04_stacking/v1/base_preds_{fn}.csv").set_index("hadm_id").loc[hid].reset_index()
    Pm={m:bp[[f"{m}_{u}" for u in LABU]].to_numpy(float) for m in MOD}
    return d,hid,(raw==1).astype(float),(raw!=-1).astype(float),Pm

W=json.load(open(f"{S}/04_stacking/v3/veredicto_v3.json"))["pesos_opt"]
def fus_v3(Pm):
    P=np.zeros_like(Pm["CXR"])
    for j,lab in enumerate(LAB):
        w=W[lab]; P[:,j]=w[0]*Pm["CXR"][:,j]+w[1]*Pm["ECG"][:,j]+w[2]*Pm["LABS"][:,j]
    return P

dte,hid_te,Yte,Mte,Pm_te=load("test"); Pfus_te=fus_v3(Pm_te)
dva,hid_va,Yva,Mva,Pm_va=load("val");  Pfus_va=fus_v3(Pm_va)
Pcxr_te=Pm_te["CXR"]

def macro(Y,M,P,idx=DIS):
    a=[average_precision_score(Y[M[:,j]==1,j],P[M[:,j]==1,j]) for j in idx if Y[M[:,j]==1,j].sum()>0]
    return float(np.mean(a))

# ═══════════════ C1 · PREVALENCIA POR SUBGRUPO ═══════════════
es=pd.read_csv(f"{S}/01_cxr/v2/estratificacion_subgrupos.csv")
VARMAP={"gender":"Sexo","race":"Etnia","admission_type":"Tipo de ingreso","cxr_view":"Proyección"}
GRP={"ASIAN":"Asiática","BLACK":"Negra","HISPANIC_LATINO":"Hispana/Latina","UNKNOWN":"Desconocida",
     "WHITE":"Blanca","EMERGENCY":"Urgencias","OBSERVATION":"Observación","SCHEDULED":"Programado",
     "URGENT":"Urgente","AP":"AP (portátil)","PA":"PA (de pie)"}
def prev_sub(var,grp):
    col=dte[var].astype(str); sel=(col==str(grp)).to_numpy()
    if sel.sum()==0: return np.nan
    ps=[]
    for j in DIS:
        m=(Mte[sel,j]==1);
        if m.sum()>0: ps.append(Yte[sel,j][m].mean())
    return float(np.mean(ps)) if ps else np.nan
rows=[]
for _,r in es.iterrows():
    if pd.isna(r.get("macro_AP")): continue
    g=str(r["grupo"])
    gname=("Femenino" if g=="0" else "Masculino") if r["variable"]=="gender" else GRP.get(g,g)
    rows.append([VARMAP.get(r["variable"],r["variable"]),gname,int(r["n"]),
                 f(prev_sub(r["variable"],g)),f(r["macro_AP"]),f(r["macro_AUC"])])
with open(f"{TAB}/09_estratificacion_cxr_v2.csv","w",encoding="utf-8",newline="") as fh:
    w=csv.writer(fh); w.writerow(["Variable","Subgrupo","n","Prev. media","AUC-PR macro","AUC-ROC macro"]); w.writerows(rows)
print("C1 · estratificación con prevalencia:",len(rows),"filas")

# ═══════════════ C2 · PUNTOS DE OPERACIÓN DEL FUSOR v3 (umbrales en VAL) ═══════════════
grid=np.round(np.arange(0.02,0.96,0.01),2)
def metrics(y,p,thr):
    pred=(p>=thr).astype(int)
    TP=int(((pred==1)&(y==1)).sum()); TN=int(((pred==0)&(y==0)).sum())
    FP=int(((pred==1)&(y==0)).sum()); FN=int(((pred==0)&(y==1)).sum())
    Se=TP/max(TP+FN,1); Sp=TN/max(TN+FP,1); VPP=TP/max(TP+FP,1); VPN=TN/max(TN+FN,1)
    return Se,Sp,VPP,VPN,FN,FP
prows=[]
for j,lab in enumerate(LAB):
    sv=Mva[:,j]==1; yv,pv=Yva[sv,j],Pfus_va[sv,j]
    st=Mte[:,j]==1; yt,pt=Yte[st,j],Pfus_te[st,j]
    # F1 (val)
    bestf1,thr_f1=-1,0.5
    for thr in grid:
        Se,Sp,VPP,_,_,_=metrics(yv,pv,thr); f1=2*VPP*Se/max(VPP+Se,1e-9)
        if f1>bestf1: bestf1,thr_f1=f1,thr
    # cribado Se>=0.90 (val): mayor umbral con Se>=0.90
    thr_cri=grid[0]
    for thr in grid:
        Se,_,_,_,_,_=metrics(yv,pv,thr)
        if Se>=0.90: thr_cri=thr
    # confirmación Sp>=0.90 (val): menor umbral con Sp>=0.90
    thr_con=grid[-1]
    for thr in grid[::-1]:
        _,Sp,_,_,_,_=metrics(yv,pv,thr)
        if Sp>=0.90: thr_con=thr
    for nom,thr in [("f1",thr_f1),("cribado_Se>=0.90",thr_cri),("confirmacion_Sp>=0.90",thr_con)]:
        Se,Sp,VPP,VPN,FN,FP=metrics(yt,pt,thr)
        prows.append([nom,ES[lab],f(thr,2),f(Se),f(Sp),f(VPP),f(VPN),FN,FP])
with open(f"{TAB}/27_puntos_fusion_v3.csv","w",encoding="utf-8",newline="") as fh:
    w=csv.writer(fh); w.writerow(["Punto","Hallazgo","Umbral","Sensib.","Especif.","VPP","VPN","FN","FP"]); w.writerows(prows)
print("C2 · puntos de operación fusor v3:",len(prows),"filas")

# ═══════════════ C3 · diferencias bootstrap emparejadas (para la figura) ═══════════════
N=len(Yte); diffs=[]
for _ in range(3000):
    b=rng.randint(0,N,N); diffs.append(macro(Yte[b],Mte[b],Pfus_te[b])-macro(Yte[b],Mte[b],Pcxr_te[b]))
np.save(f"{S}/04_stacking/diffs_pareado.npy",np.array(diffs))
print("C3 · diffs guardadas · media=%.4f IC=[%.4f, %.4f]"%(np.mean(diffs),np.percentile(diffs,2.5),np.percentile(diffs,97.5)))
print("fusor v3 macro AP test = %.4f (control)"%macro(Yte,Mte,Pfus_te))
