"""Explora DOS fusiones nuevas y las compara honestamente con mono (CXR) y con la fusión v2 (0,594):
  A · Fusión ponderada por CONFIANZA (por paciente): peso de cada modalidad ∝ prior_por_etiqueta · (certeza)^γ.
  B · Fusión TEMPRANA (feature-level): [embedding CXR 1024 + features analíticas + prob ECG] -> LogReg por etiqueta.
Métrica primaria: AUC-PR macro sobre las 5 patologías (excluye «No Finding»), con IC bootstrap.
Salida: salidas/04_stacking/veredicto_v4v5.json + tablas curadas para el informe.
"""
import numpy as np, pandas as pd, os, json, itertools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

SEED=42; rng=np.random.RandomState(SEED)
S="salidas"; TABOUT=f"{S}/_informe_modelos/tablas"; os.makedirs(TABOUT,exist_ok=True)
LAB=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
DIS=[0,1,2,3,5]                         # las 5 patologías (sin «No Finding») para el macro
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema",
    "Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgo","Pleural Effusion":"Derrame pleural"}
MOD=["CXR","ECG","LABS"]
LABU=[l.replace(" ","_") for l in LAB]      # nombres de columna en base_preds (con guion bajo)

def labels(sp):
    d=pd.read_csv(f"data/clean/{sp}_clean.csv",sep=";")
    raw=d[LAB].to_numpy(float)
    return d["hadm_id"].to_numpy(), (raw==1).astype(float), (raw!=-1).astype(float), d

def base(sp):   # probs calibradas de las 3 modalidades, alineadas por hadm_id
    f={"train":"oof_train","val":"val","test":"test"}[sp]
    return pd.read_csv(f"{S}/04_stacking/v1/base_preds_{f}.csv")

def macro_ap(Y,M,P,idx=DIS):
    aps=[]
    for j in idx:
        s=M[:,j]==1
        if Y[s,j].sum()==0: continue
        aps.append(average_precision_score(Y[s,j],P[s,j]))
    return float(np.mean(aps))

def boot_ic(Y,M,P,idx=DIS,n=1000):
    N=len(Y); vals=[]
    for _ in range(n):
        bs=rng.randint(0,N,N); vals.append(macro_ap(Y[bs],M[bs],P[bs],idx))
    return float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))

# ── cargar ──
res={}
data={}
for sp in ["train","val","test"]:
    hid,Y,M,dfc=labels(sp); bp=base(sp)
    bp=bp.set_index("hadm_id").loc[hid].reset_index()      # alinear al orden de las etiquetas
    Pm={m:bp[[f"{m}_{l}" for l in LABU]].to_numpy(float) for m in MOD}
    data[sp]=dict(hid=hid,Y=Y,M=M,dfc=dfc,Pm=Pm)

# baseline de referencia
mono_test=macro_ap(data["test"]["Y"],data["test"]["M"],data["test"]["Pm"]["CXR"])

# ══════════════ A · PONDERADA POR CONFIANZA ══════════════
def conf(p): return np.abs(2*p-1)                          # certeza 0..1
grid=[w for w in itertools.product(np.arange(0,1.01,0.2),repeat=3) if abs(sum(w)-1)<1e-6]
gammas=[0.0,0.5,1.0,2.0]
def fuse_conf(Pm,prior,gamma):
    out=np.zeros_like(Pm["CXR"])
    for jcol in range(6):
        ws=[]
        for k,m in enumerate(MOD):
            c=conf(Pm[m][:,jcol]); ws.append(prior[k]*(c+1e-3)**gamma)
        W=np.stack(ws,0); W=W/ (W.sum(0,keepdims=True)+1e-9)
        out[:,jcol]=sum(W[k]*Pm[m][:,jcol] for k,m in enumerate(MOD))
    return out
# elegir prior por etiqueta + gamma en VAL
Yv,Mv,Pv=data["val"]["Y"],data["val"]["M"],data["val"]["Pm"]
Yt,Mt,Pt=data["test"]["Y"],data["test"]["M"],data["test"]["Pm"]
best_prior=[None]*6; best_g=[0]*6
for jcol in range(6):
    bestv=-1
    for g in gammas:
        for w in grid:
            ws=[]
            for k,m in enumerate(MOD):
                c=conf(Pv[m][:,jcol]); ws.append(w[k]*(c+1e-3)**g)
            W=np.stack(ws,0); W=W/(W.sum(0,keepdims=True)+1e-9)
            pf=sum(W[k]*Pv[m][:,jcol] for k,m in enumerate(MOD))
            s=Mv[:,jcol]==1
            if Yv[s,jcol].sum()==0: continue
            ap=average_precision_score(Yv[s,jcol],pf[s])
            if ap>bestv: bestv=ap; best_prior[jcol]=w; best_g[jcol]=g
PA=np.zeros_like(Pt["CXR"])
for jcol in range(6):
    ws=[]
    for k,m in enumerate(MOD):
        c=conf(Pt[m][:,jcol]); ws.append(best_prior[jcol][k]*(c+1e-3)**best_g[jcol])
    W=np.stack(ws,0); W=W/(W.sum(0,keepdims=True)+1e-9)
    PA[:,jcol]=sum(W[k]*Pt[m][:,jcol] for k,m in enumerate(MOD))
apA=macro_ap(Yt,Mt,PA); icA=boot_ic(Yt,Mt,PA)
gamma_medio=float(np.mean([best_g[j] for j in DIS]))

# ══════════════ B · FUSIÓN TEMPRANA (feature-level) ══════════════
NONLAB=["hadm_id"]+LAB
NUMCOLS=None
def feats(sp):
    global NUMCOLS
    d=data[sp]["dfc"]; hid=data[sp]["hid"]
    emb=np.load(f"{S}/01_cxr/v2/embeddings/emb_{sp}.npy")            # (N,1024) en orden del clean
    if NUMCOLS is None:                                              # solo columnas numéricas (analíticas), fijadas en train
        num=data["train"]["dfc"].select_dtypes(include="number")
        NUMCOLS=[c for c in num.columns if c not in NONLAB]
    Xlab=d[NUMCOLS].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    assert len(emb)==len(d), f"emb {len(emb)} != clean {len(d)} en {sp}"
    ecg=data[sp]["Pm"]["ECG"]                                        # 6 probs ECG (OOF en train)
    return np.hstack([emb,Xlab,ecg]), NUMCOLS
Xtr,labcols=feats("train"); Xva,_=feats("val"); Xte,_=feats("test")
# imputar NaN (mediana de train) + estandarizar
med=np.nanmedian(Xtr,0);
for X in (Xtr,Xva,Xte):
    inds=np.where(np.isnan(X)); X[inds]=np.take(med,inds[1])
sc=StandardScaler().fit(Xtr); Xtr,Xva,Xte=sc.transform(Xtr),sc.transform(Xva),sc.transform(Xte)
Ytr,Mtr=data["train"]["Y"],data["train"]["M"]
PB=np.zeros((len(Xte),6))
for jcol in range(6):
    s=Mtr[:,jcol]==1
    bestC,bestv,bestp=1,-1,None
    for C in [0.01,0.05,0.1,0.5]:
        clf=LogisticRegression(C=C,max_iter=2000,class_weight="balanced")
        clf.fit(Xtr[s],Ytr[s,jcol])
        sv=Mv[:,jcol]==1; pv=clf.predict_proba(Xva[sv])[:,1]
        if Yv[sv,jcol].sum()==0: continue
        ap=average_precision_score(Yv[sv,jcol],pv)
        if ap>bestv: bestv=ap; bestC=C; bestp=clf.predict_proba(Xte)[:,1]
    PB[:,jcol]=bestp
apB=macro_ap(Yt,Mt,PB); icB=boot_ic(Yt,Mt,PB)

# ══════════════ RESULTADOS ══════════════
V2=0.5942
out={"mono_CXR":round(mono_test,4),"fusion_v2":V2,
 "A_confianza":{"macro_ap":round(apA,4),"ic":[round(icA[0],4),round(icA[1],4)],
   "gamma_medio":gamma_medio,"vs_mono":round(apA-mono_test,4),"vs_v2":round(apA-V2,4)},
 "B_temprana":{"macro_ap":round(apB,4),"ic":[round(icB[0],4),round(icB[1],4)],
   "vs_mono":round(apB-mono_test,4),"vs_v2":round(apB-V2,4)}}
json.dump(out,open(f"{S}/04_stacking/veredicto_v4v5.json","w"),indent=1,ensure_ascii=False)

def f(x,n=3): return f"{x:.{n}f}".replace(".",",")
rows=[
 ["CXR en solitario (referencia)",f(mono_test),"—","—"],
 ["Fusión v2 (avanzada, la mejor)",f(V2),"+"+f(V2-mono_test),"—"],
 ["A · Ponderada por confianza",f(apA)+f" [{f(icA[0])}–{f(icA[1])}]","+"+f(apA-mono_test),f(apA-V2)],
 ["B · Fusión temprana",f(apB)+f" [{f(icB[0])}–{f(icB[1])}]","+"+f(apB-mono_test),f(apB-V2)],
]
import csv
with open(f"{TABOUT}/22_fusiones_nuevas.csv","w",encoding="utf-8",newline="") as fh:
    w=csv.writer(fh); w.writerow(["Enfoque","AUC-PR macro (IC 95 %)","Δ vs CXR mono","Δ vs v2"]); w.writerows(rows)
print(json.dumps(out,indent=1,ensure_ascii=False))
print("gamma por patología (A):",{ES[LAB[j]]:best_g[j] for j in DIS})
