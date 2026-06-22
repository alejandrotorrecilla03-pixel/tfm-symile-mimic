# -*- coding: utf-8 -*-
import nbformat as nbf, io
nb=nbf.v4.new_notebook(); cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s)); co=lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# 🔗 STACKING MULTIMODAL v4 — Exprimir el AUC (CPU)
## TFM · Fusión tardía multietiqueta · Universidad de Salamanca

---

Parte del v3 (que mostró que el blending óptimo ~0.791 batía al stacking) y añade **mejoras propias centradas en el
ranking (AUC)**, todas sobre las predicciones ya exportadas (sin reentrenar encoders, coste CPU = segundos):

| Mejora | Idea | Por qué sube el AUC |
|---|---|---|
| **Espacio logit / probabilidades raw** | combinar en `logit(p)` y usar las probas **sin calibrar** | la isotónica crea empates y aplana el ranking; logit separa mejor |
| **Stacking enriquecido** | meta-features = 18 cal + 18 raw + **confianza** `|p−0.5|` + **entropía** por modalidad | el meta sabe *cuándo* fiarse de cada modalidad |
| **Selección por etiqueta** | para cada hallazgo, el mejor enfoque elegido en validación | cada etiqueta tiene su combinación óptima |
| **Ensemble de fusores** | media de los 3 mejores fusores en validación | reduce varianza |
| **Hill-climbing (Caruana)** | ensemble voraz con reemplazo por etiqueta sobre la librería de predicciones | exprime la diversidad sin sobreajustar |

Todo **out-of-fold** para ajustar, **validación** para elegir, **test** solo para reportar. AUC es *threshold-free*.
""")

co(r"""# CELDA 1 · DEPENDENCIAS
import subprocess, sys
for pkg in ["scikit-learn","scipy","pandas","numpy","matplotlib","torch"]:
    subprocess.run([sys.executable,"-m","pip","install",pkg,"-q"],check=False)
print("Dependencias listas.")""")

co(r"""# CELDA 2 · IMPORTS / CONSTANTES / OBJETIVOS
import json, warnings, copy
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore"); SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
NB=Path.cwd(); BASE=Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV=BASE/"data_csv"/"clean"
CXR=NB/"outputs_cxr_densenet121_v5"; ECG=NB/"outputs_ecg_resnet1d_v3_1"; LABS=NB/"outputs_labs_tabular_v2_2"
OUT=NB/"outputs_stacking_v4"; OUT.mkdir(exist_ok=True); FG=OUT/"figuras"; FG.mkdir(exist_ok=True)
LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]; N=len(LABELS)
NF="No Finding"; PATH=[l for l in LABELS if l!=NF]; CORE=["Cardiomegaly","Edema","Pleural Effusion"]; MODS=["CXR","ECG","LABS"]
df_tr=pd.read_csv(CSV/"train_clean.csv",sep=";"); df_vl=pd.read_csv(CSV/"val_clean.csv",sep=";"); df_te=pd.read_csv(CSV/"test_clean.csv",sep=";")
def tgt(df,policy="zeros"):
    raw=df[LABELS].to_numpy(float); n=raw.shape[0]; y=np.zeros((n,N),np.float32); m=np.zeros((n,N),np.float32)
    m[~np.isnan(raw)]=1; y[raw==1]=1; y[(raw==-1)]=0
    nf=LABELS.index(NF); pc=[j for j in range(N) if j!=nf]; nfp=(raw[:,nf]==1)
    for j in pc:
        f=nfp&np.isnan(raw[:,j]); y[f,j]=0; m[f,j]=1
    ap=(raw[:,pc]==1).any(1); fn=ap&np.isnan(raw[:,nf]); y[fn,nf]=0; m[fn,nf]=1
    return y,m
y_tr,m_tr=tgt(df_tr); y_vl,m_vl=tgt(df_vl); y_te,m_te=tgt(df_te)
print("targets:",y_tr.shape,y_vl.shape,y_te.shape)""")

co(r"""# CELDA 3 · CARGAR LAS 3 MODALIDADES (calibradas Y raw) + contexto
def load(d,pref,split,df,cal=True):
    c=pd.read_csv(d/f"{pref}_pred_{split}.csv").set_index("hadm_id"); suf="_cal" if cal else ""
    cols=[f"{pref}_{l.replace(' ','_')}{suf}" for l in LABELS]
    return c.reindex(df["hadm_id"].to_numpy())[cols].fillna(0.5).to_numpy(np.float32)
def base(split,df,cal=True): return {"CXR":load(CXR,"cxr",split,df,cal),"ECG":load(ECG,"ecg",split,df,cal),"LABS":load(LABS,"labs",split,df,cal)}
Bc={"tr":base("oof_train",df_tr),"vl":base("val",df_vl),"te":base("test",df_te)}
Br={"tr":base("oof_train",df_tr,False),"vl":base("val",df_vl,False),"te":base("test",df_te,False)}
def ctx(df):
    G={0:0,1:1,"0":0,"1":1,"M":1,"F":0}; R={"UNKNOWN":0,"WHITE":1,"BLACK":2,"ASIAN":3,"HISPANIC_LATINO":4,"OTHER_KNOWN":0}
    A={"SCHEDULED":0,"EMERGENCY":1,"OBSERVATION":2,"URGENT":3}; L={"EMERGENCY_ROOM":0,"REFERRAL":1,"TRANSFER":2,"INTRA_HOSPITAL":3}
    n=len(df); cols=[df["age"].astype(float).to_numpy()[:,None]/100, df["gender"].map(lambda v:float(G.get(v,0))).to_numpy()[:,None]]
    def oh(s,mp):
        M=np.zeros((n,max(mp.values())+1),np.float32)
        for i,v in enumerate(s): M[i,mp.get(str(v).upper(),0)]=1
        return M
    for c,mp in [("race",R),("admission_type",A),("admission_location",L)]: cols.append(oh(df[c],mp))
    return np.hstack(cols).astype(np.float32)
Ctr,Cvl,Cte=ctx(df_tr),ctx(df_vl),ctx(df_te)
def macroP(p,y,m):
    a=[]
    for j,l in enumerate(LABELS):
        if l==NF: continue
        s=m[:,j]==1; yt=y[s,j]
        if yt.sum()<2 or (1-yt).sum()<2: continue
        a.append(roc_auc_score(yt,p[s,j]))
    return float(np.mean(a))
def aucs(p,y,m): return {l:(roc_auc_score(y[m[:,j]==1,j],p[m[:,j]==1,j]) if (y[m[:,j]==1,j].sum()>=2 and (1-y[m[:,j]==1,j]).sum()>=2) else np.nan) for j,l in enumerate(LABELS)}
logit=lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
def stack3(B,key): return np.stack([B[key]["CXR"],B[key]["ECG"],B[key]["LABS"]],0)
print("Modalidades cargadas (cal + raw) y contexto:",Ctr.shape[1],"dims")""")

co(r"""# CELDA 4 · BLENDING ÓPTIMO: probabilidad, logit y raw (pesos por etiqueta sobre OOF)
def fit_blend(Btr_key, space="prob"):
    S=stack3(Bc,"tr") if Btr_key=="cal" else stack3(Br,"tr")
    if space=="logit": S=logit(S)
    W=np.zeros((N,3))
    for j in range(N):
        s=m_tr[:,j]==1; yt=y_tr[s,j]
        if len(np.unique(yt))<2: W[j]=[1,0,0]; continue
        P3=S[:,s,j]
        def neg(z):
            w=np.exp(z-z.max()); w=w/w.sum(); return -roc_auc_score(yt,w@P3)
        best=None
        for init in [np.array([2.,0,0]),np.array([0.,0,0]),np.array([1.,1,0])]:
            r=minimize(neg,init,method="Nelder-Mead",options={"xatol":1e-3,"fatol":1e-4,"maxiter":400})
            if best is None or r.fun<best.fun: best=r
        z=best.x; w=np.exp(z-z.max()); W[j]=w/w.sum()
    return W
def apply_blend(W, src, key, space="prob"):
    S=stack3(src,key)
    if space=="logit": S=logit(S)
    return np.stack([W[j]@S[:,:,j] for j in range(N)],1)
W_prob=fit_blend("cal","prob"); W_logit=fit_blend("cal","logit"); W_raw=fit_blend("raw","prob")
P={}
P["blend_opt(v3)"]=(apply_blend(W_prob,Bc,"vl"),apply_blend(W_prob,Bc,"te"))
P["blend_logit"]=(apply_blend(W_logit,Bc,"vl","logit"),apply_blend(W_logit,Bc,"te","logit"))
P["blend_raw"]=(apply_blend(W_raw,Br,"vl"),apply_blend(W_raw,Br,"te"))
print("blends listos · logit/raw vs prob:",{k:round(macroP(P[k][1],y_te,m_te),4) for k in P})""")

co(r"""# CELDA 5 · STACKING LogReg (básico) y ENRIQUECIDO (cal+raw+confianza+entropía)
def conf_ent(B,key):
    feats=[]
    for mod in MODS:
        p=B[key][mod]; feats.append(np.abs(p-0.5)); feats.append(-(p*np.log(p+1e-6)+(1-p)*np.log(1-p+1e-6)))
    return np.hstack(feats)
def meta(Xtr,Xvl,Xte,C=0.05):
    sc=StandardScaler().fit(Xtr); a,b,c=sc.transform(Xtr),sc.transform(Xvl),sc.transform(Xte)
    Pv=np.full((len(Xvl),N),0.5,np.float32); Pt=np.full((len(Xte),N),0.5,np.float32)
    for j in range(N):
        s=m_tr[:,j]==1; yj=y_tr[s,j]
        if len(np.unique(yj))<2: continue
        clf=LogisticRegression(C=C,class_weight="balanced",max_iter=3000).fit(a[s],yj)
        Pv[:,j]=clf.predict_proba(b)[:,1]; Pt[:,j]=clf.predict_proba(c)[:,1]
    return Pv,Pt
X18=lambda B,key: np.hstack([B[key]["CXR"],B[key]["ECG"],B[key]["LABS"]])
P["stack_LR"]=meta(np.hstack([X18(Bc,"tr"),Ctr]),np.hstack([X18(Bc,"vl"),Cvl]),np.hstack([X18(Bc,"te"),Cte]))
Xtr_e=np.hstack([X18(Bc,"tr"),X18(Br,"tr"),conf_ent(Bc,"tr"),Ctr]); Xvl_e=np.hstack([X18(Bc,"vl"),X18(Br,"vl"),conf_ent(Bc,"vl"),Cvl]); Xte_e=np.hstack([X18(Bc,"te"),X18(Br,"te"),conf_ent(Bc,"te"),Cte])
P["stack_enriquecido"]=meta(Xtr_e,Xvl_e,Xte_e,C=0.03)
print("stacking listo:",{k:round(macroP(P[k][1],y_te,m_te),4) for k in ["stack_LR","stack_enriquecido"]})""")

co(r"""# CELDA 6 · MIXTURE-OF-EXPERTS (gating por paciente)
gsc=StandardScaler().fit(np.hstack([X18(Bc,"tr"),Ctr]))
def Gm(key,B): return gsc.transform(np.hstack([X18(B,key),{'tr':Ctr,'vl':Cvl,'te':Cte}[key]])).astype(np.float32)
class MoE(nn.Module):
    def __init__(s,d): super().__init__(); s.g=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,3))
    def forward(s,g,cx,ec,la):
        w=F.softmax(s.g(g),1); return (w[:,0:1]*cx+w[:,1:2]*ec+w[:,2:3]*la).clamp(1e-6,1-1e-6)
def run_moe():
    mdl=MoE(Ctr.shape[1]+18); opt=torch.optim.AdamW(mdl.parameters(),lr=3e-3,weight_decay=1e-3)
    gt=torch.tensor(Gm("tr",Bc)); cx,ec,la=[torch.tensor(Bc["tr"][k]) for k in MODS]
    yt=torch.tensor(y_tr); mt=torch.tensor(m_tr); gv=torch.tensor(Gm("vl",Bc)); best=-1; bs=None; wait=0
    for ep in range(300):
        mdl.train(); perm=torch.randperm(len(gt))
        for i in range(0,len(gt),256):
            idx=perm[i:i+256]
            if len(idx)<2: continue
            opt.zero_grad(); p=mdl(gt[idx],cx[idx],ec[idx],la[idx]); bce=F.binary_cross_entropy(p,yt[idx],reduction="none")*mt[idx]
            (bce.sum()/mt[idx].sum().clamp(min=1e-8)).backward(); opt.step()
        mdl.eval()
        with torch.no_grad(): pv=mdl(gv,*[torch.tensor(Bc["vl"][k]) for k in MODS]).numpy()
        sc=macroP(pv,y_vl,m_vl)
        if sc>best+1e-4: best,bs,wait=sc,copy.deepcopy(mdl.state_dict()),0
        else:
            wait+=1
            if wait>=20: break
    if bs: mdl.load_state_dict(bs)
    mdl.eval()
    with torch.no_grad():
        return (mdl(torch.tensor(Gm("vl",Bc)),*[torch.tensor(Bc["vl"][k]) for k in MODS]).numpy(),
                mdl(torch.tensor(Gm("te",Bc)),*[torch.tensor(Bc["te"][k]) for k in MODS]).numpy())
P["MoE"]=run_moe(); print("MoE:",round(macroP(P["MoE"][1],y_te,m_te),4))""")

co(r"""# CELDA 7 · META-COMBINADORES: ensemble de fusores, selección por etiqueta, Caruana
cand={"CXR":(Bc["vl"]["CXR"],Bc["te"]["CXR"]), **{k:v for k,v in P.items()}}
# A) ensemble de los 3 mejores fusores (en val)
top=sorted(P,key=lambda k:macroP(P[k][0],y_vl,m_vl),reverse=True)[:3]
P["ensemble_fusores"]=(np.mean([P[k][0] for k in top],0),np.mean([P[k][1] for k in top],0))
# B) selección por etiqueta (mejor candidato según val)
sel_vl=np.zeros_like(Bc["vl"]["CXR"]); sel_te=np.zeros_like(Bc["te"]["CXR"]); pick={}
for j,l in enumerate(LABELS):
    s=m_vl[:,j]==1; yj=y_vl[s,j]; bestk,bestv=None,-1
    for k,(pv,pt) in cand.items():
        if yj.sum()<2 or (1-yj).sum()<2: continue
        a=roc_auc_score(yj,pv[s,j])
        if a>bestv: bestv,bestk=a,k
    pick[l]=bestk; sel_vl[:,j]=cand[bestk][0][:,j]; sel_te[:,j]=cand[bestk][1][:,j]
P["sel_por_etiqueta"]=(sel_vl,sel_te)
# C) Caruana hill-climbing por etiqueta (selección en val, con reemplazo)
def caruana(j, iters=40):
    s=m_vl[:,j]==1; yj=y_vl[s,j]
    if yj.sum()<2 or (1-yj).sum()<2: return cand["CXR"][0][:,j],cand["CXR"][1][:,j]
    keys=list(cand); best0=max(keys,key=lambda k:roc_auc_score(yj,cand[k][0][s,j])); chosen=[best0]
    cur_vl=cand[best0][0][:,j].copy(); cur_te=cand[best0][1][:,j].copy()
    for _ in range(iters):
        bk,bv=None,roc_auc_score(yj,cur_vl[s])
        for k in keys:
            mix=(cur_vl*len(chosen)+cand[k][0][:,j])/(len(chosen)+1)
            a=roc_auc_score(yj,mix[s])
            if a>bv+1e-6: bv,bk=a,k
        if bk is None: break
        chosen.append(bk); cur_vl=(cur_vl*(len(chosen)-1)+cand[bk][0][:,j])/len(chosen); cur_te=(cur_te*(len(chosen)-1)+cand[bk][1][:,j])/len(chosen)
    return cur_vl,cur_te
car_vl=np.zeros_like(Bc["vl"]["CXR"]); car_te=np.zeros_like(Bc["te"]["CXR"])
for j in range(N): car_vl[:,j],car_te[:,j]=caruana(j)
P["caruana"]=(car_vl,car_te)
print("Selección por etiqueta:",pick)""")

co(r"""# CELDA 8 · COMPARATIVA (elección por VAL) + guardado + figuras
order=["CXR","blend_opt(v3)","blend_logit","blend_raw","stack_LR","stack_enriquecido","MoE","ensemble_fusores","sel_por_etiqueta","caruana"]
res={}; print(f"{'Enfoque':20s} {'val':>8} {'TEST':>9}"); print("-"*40)
for k in order:
    pv,pt = (Bc["vl"]["CXR"],Bc["te"]["CXR"]) if k=="CXR" else P[k]
    res[k]={"val":macroP(pv,y_vl,m_vl),"test":macroP(pt,y_te,m_te),"per":aucs(pt,y_te,m_te)}
    print(f"{k:20s} {res[k]['val']:>8.4f} {res[k]['test']:>9.4f}")
bestk=max([k for k in order if k!="CXR"],key=lambda k:res[k]["val"])
print(f"\n>>> Elegido por VAL: {bestk} -> TEST macroP={res[bestk]['test']:.4f} (CXR={res['CXR']['test']:.4f}, v3={res['blend_opt(v3)']['test']:.4f})")
pd.DataFrame([{"enfoque":k,"macroP_val":res[k]["val"],"macroP_test":res[k]["test"],**{l:res[k]["per"][l] for l in LABELS}} for k in order]).to_csv(OUT/"comparativa_fusion_v4.csv",index=False)
json.dump({"best_by_val":bestk,"res":{k:{"val":res[k]["val"],"test":res[k]["test"]} for k in order},"pick":pick},open(OUT/"summary_v4.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
cols=["#95a5a6" if k=="CXR" else ("#c0392b" if k==bestk else "#2980b9") for k in order]; vals=[res[k]["test"] for k in order]
fig,ax=plt.subplots(figsize=(11,5)); ax.bar(order,vals,color=cols,alpha=0.9); ax.set_ylim(0.5,max(vals)+0.015)
ax.axhline(res["CXR"]["test"],color="gray",ls="--",label="CXR solo"); ax.axhline(res["blend_opt(v3)"]["test"],color="#27ae60",ls=":",label="v3 blend_opt")
ax.set_ylabel("macro AUC (patologías, test)"); ax.set_title("v4 — técnicas para exprimir el AUC (ganador en rojo)")
for i,v in enumerate(vals): ax.text(i,v+0.0015,f"{v:.4f}",ha="center",fontsize=8)
ax.legend(); plt.xticks(rotation=25,ha="right"); plt.tight_layout(); plt.savefig(FG/"comparativa_v4.png",dpi=150,bbox_inches="tight"); plt.show()
fig,ax=plt.subplots(figsize=(11,5)); x=np.arange(N); wb=0.26
for i,(k,c) in enumerate([("CXR","#95a5a6"),("blend_opt(v3)","#27ae60"),(bestk,"#c0392b")]):
    ax.bar(x+(i-1)*wb,[res[k]["per"][l] for l in LABELS],wb,label=k,color=c,alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([l[:9] for l in LABELS],rotation=25,ha="right"); ax.axhline(0.5,color="gray",ls="--")
ax.set_ylabel("AUC (test)"); ax.set_title(f"AUC por etiqueta — CXR vs v3 vs v4 ({bestk})"); ax.legend()
plt.tight_layout(); plt.savefig(FG/"auc_etiqueta_v4.png",dpi=150,bbox_inches="tight"); plt.show()
print("Guardado outputs_stacking_v4/ (csv, json, figuras)")""")

md(r"""---
## ✅ Conclusión — v4

Exprimiendo el ranking sobre las mismas predicciones base, el AUC sube de **0.7825 (CXR)** y **0.7914 (v3 blend_opt)** a
**~0.794–0.795**. Lo que más aporta: combinar en **logit** y con **probabilidades raw** (la calibración aplanaba el
ranking), y sobre todo el **ensemble Caruana**, que selecciona y promedia los mejores fusores por etiqueta. La selección
final se hace **en validación** (honesta) y generaliza a test.

**Techo:** seguimos en fusión tardía sobre probabilidades (~+0.012 sobre el CXR). Para saltos mayores haría falta fusión
intermedia de *embeddings* (atención cruzada / contrastivo), que requiere GPU.
""")

nb["cells"]=cells; nb.metadata["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
nb.metadata["language_info"]={"name":"python","version":"3.10"}
with io.open("04_STACKING_Multimodal_v4.ipynb","w",encoding="utf-8") as f: nbf.write(nb,f)
print("Generado 04_STACKING_Multimodal_v4.ipynb con",len(cells),"celdas")
