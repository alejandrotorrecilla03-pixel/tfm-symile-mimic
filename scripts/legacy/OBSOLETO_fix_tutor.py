"""Tres arreglos del tutor (sin reabrir modelos):
  1) Significación EMPAREJADA de la fusión v3 vs CXR (bootstrap sobre los mismos 464 pacientes).
  2) Baseline SOLO edad+sexo: comprobar que ECG y analíticas aportan AP por encima.
La fusión v3 = media ponderada por patología (pesos de veredicto_v3.json), reconstruida de base_preds.
Salida: salidas/04_stacking/veredicto_pareado.json + tablas curadas.
"""
import numpy as np, pandas as pd, json, csv, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

SEED=42; rng=np.random.RandomState(SEED)
S="salidas"; TAB=f"{S}/_informe_modelos/tablas"
LAB=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
LABU=[l.replace(" ","_") for l in LAB]; DIS=[0,1,2,3,5]   # 5 patologías (sin «No Finding»)
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema",
    "Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgo","Pleural Effusion":"Derrame pleural"}
MOD=["CXR","ECG","LABS"]
def f(x,n=3): return f"{x:.{n}f}".replace(".",",")

def labels(sp):
    d=pd.read_csv(f"data/clean/{sp}_clean.csv",sep=";")
    raw=d[LAB].to_numpy(float); return d["hadm_id"].to_numpy(),(raw==1).astype(float),(raw!=-1).astype(float),d

def base(sp):
    fn={"train":"oof_train","val":"val","test":"test"}[sp]
    return pd.read_csv(f"{S}/04_stacking/v1/base_preds_{fn}.csv")

hid,Y,M,dfc=labels("test")
bp=base("test").set_index("hadm_id").loc[hid].reset_index()
Pm={m:bp[[f"{m}_{u}" for u in LABU]].to_numpy(float) for m in MOD}

def macro(Y,M,P,idx=DIS):
    a=[average_precision_score(Y[M[:,j]==1,j],P[M[:,j]==1,j]) for j in idx if Y[M[:,j]==1,j].sum()>0]
    return float(np.mean(a))

# ── fusión v3 reconstruida (media ponderada por patología) ──
W=json.load(open(f"{S}/04_stacking/v3/veredicto_v3.json"))["pesos_opt"]  # [CXR,ECG,LABS]
Pfus=np.zeros((len(Y),6))
for j,lab in enumerate(LAB):
    w=W[lab]; Pfus[:,j]=w[0]*Pm["CXR"][:,j]+w[1]*Pm["ECG"][:,j]+w[2]*Pm["LABS"][:,j]
Pcxr=Pm["CXR"]
ap_fus=macro(Y,M,Pfus); ap_cxr=macro(Y,M,Pcxr)

# ── 1) BOOTSTRAP EMPAREJADO de la diferencia ──
N=len(Y); diffs=[]
for _ in range(2000):
    b=rng.randint(0,N,N); diffs.append(macro(Y[b],M[b],Pfus[b])-macro(Y[b],M[b],Pcxr[b]))
diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5]); pval=float((diffs<=0).mean())
sig=lo>0
# per-patología (las 6): AP_fus vs AP_cxr
per=[]
n_mejora=0
for j,lab in enumerate(LAB):
    s=M[:,j]==1
    apf=average_precision_score(Y[s,j],Pfus[s,j]); apc=average_precision_score(Y[s,j],Pcxr[s,j])
    per.append((ES[lab],apc,apf,apf-apc)); n_mejora+= (apf>apc)

# ── 2) BASELINE edad+sexo vs ECG vs LABS ──
_,Ytr,Mtr,dtr=labels("train")
def dem(d): return d[["age","gender"]].apply(pd.to_numeric,errors="coerce").to_numpy(float)
Xtr=dem(dtr); Xte=dem(dfc)
med=np.nanmedian(Xtr,0)
for X in (Xtr,Xte):
    inds=np.where(np.isnan(X)); X[inds]=np.take(med,inds[1])
sc=StandardScaler().fit(Xtr); Xtr2,Xte2=sc.transform(Xtr),sc.transform(Xte)
Pbase=np.zeros((len(Y),6))
for j in range(6):
    s=Mtr[:,j]==1
    clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr2[s],Ytr[s,j])
    Pbase[:,j]=clf.predict_proba(Xte2)[:,1]
ap_base=macro(Y,M,Pbase); ap_ecg=macro(Y,M,Pm["ECG"]); ap_labs=macro(Y,M,Pm["LABS"])

out={
 "pareado":{"ap_cxr":round(ap_cxr,4),"ap_fus_v3":round(ap_fus,4),"diff":round(ap_fus-ap_cxr,4),
   "ic_diff":[round(lo,4),round(hi,4)],"excluye_cero":bool(sig),"p_una_cola":round(pval,4),
   "n_patologias_mejoran":int(n_mejora)},
 "baseline_edad_sexo":{"ap_edad_sexo":round(ap_base,4),"ap_ecg":round(ap_ecg,4),"ap_labs":round(ap_labs,4),
   "ecg_supera":bool(ap_ecg>ap_base),"labs_supera":bool(ap_labs>ap_base)},
}
json.dump(out,open(f"{S}/04_stacking/veredicto_pareado.json","w"),indent=1,ensure_ascii=False)

# tablas curadas
with open(f"{TAB}/24_veredicto_pareado.csv","w",encoding="utf-8",newline="") as fh:
    w=csv.writer(fh); w.writerow(["Comparación","AUC-PR","Diferencia emparejada","IC 95 % de la diferencia","¿Excluye el 0?"])
    w.writerow(["CXR en solitario",f(ap_cxr),"—","—","—"])
    w.writerow(["Fusión v3 (final)",f(ap_fus),f"+{f(ap_fus-ap_cxr)}",f"[{f(lo)} – {f(hi)}]","Sí (concluyente)" if sig else "No"])
with open(f"{TAB}/25_pareado_por_patologia.csv","w",encoding="utf-8",newline="") as fh:
    w=csv.writer(fh); w.writerow(["Hallazgo","AP CXR","AP fusión v3","Δ","¿Mejora?"])
    for nom,apc,apf,dd in per: w.writerow([nom,f(apc),f(apf),("+" if dd>=0 else "")+f(dd),"Sí" if apf>apc else "No"])
with open(f"{TAB}/26_baseline_edadsexo.csv","w",encoding="utf-8",newline="") as fh:
    w=csv.writer(fh); w.writerow(["Modelo","AUC-PR macro","¿Supera a edad+sexo?"])
    w.writerow(["Baseline edad + sexo",f(ap_base),"— (referencia)"])
    w.writerow(["ECG (v2)",f(ap_ecg),"Sí" if ap_ecg>ap_base else "No"])
    w.writerow(["Analíticas (LABS v1)",f(ap_labs),"Sí" if ap_labs>ap_base else "No"])

print(json.dumps(out,indent=1,ensure_ascii=False))
print("6 patologías:", [(n, "sube" if d>0 else "baja") for n,_,_,d in per])
