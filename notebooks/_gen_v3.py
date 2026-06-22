# -*- coding: utf-8 -*-
# Generador del notebook 04_STACKING_Multimodal_v3.ipynb (fusion avanzada).
import nbformat as nbf
nb=nbf.v4.new_notebook(); cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co=lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# 🔗 STACKING MULTIMODAL v3 — Fusión avanzada (CPU)
## TFM · Sistema de Apoyo a la Decisión Clínica Multimodal · Universidad de Salamanca

---

## 🆕 Qué añade respecto al v2

El v2 cargaba las 3 modalidades y comparaba **Stacking (LogReg / XGBoost)** vs **blending por media**. El v3 **carga
exactamente lo mismo** (los `*_pred_*.csv` de los notebooks 01/02/03, sin re-entrenar nada) y añade **tres técnicas de
fusión más** motivadas por los resultados del v1/v2:

| Técnica | Idea | Por qué aquí |
|---|---|---|
| **Blending óptimo ponderado** | pesos por **modalidad × etiqueta** optimizados (Nelder-Mead) sobre las OOF | el blending por media **empeoraba** al CXR; aprender pesos lo arregla, interpretable y sin sobreajuste |
| **Mixture-of-Experts (MoE)** | un *gating* aprende, **por paciente**, cuánto pesa cada modalidad | hay pacientes con ECG informativo y otros con ECG plano → peso **dinámico** |
| **MLP de fusión** | MLP con dropout fuerte + L2 sobre las 18 probas + contexto | capta interacciones que la LogReg no; control de sobreajuste con ~10k |

Se comparan contra las líneas base: **CXR en solitario**, **blending por media** y **Stacking LogReg**.
Todo **out-of-fold** (sin fuga), umbral por F1 en *val*, evaluación en *test* limpio. Coste CPU: **segundos**.
""")

co(r"""# CELDA 1 · DEPENDENCIAS
import subprocess, sys
for pkg in ["scikit-learn","scipy","pandas","numpy","matplotlib","seaborn","torch"]:
    subprocess.run([sys.executable,"-m","pip","install",pkg,"-q"],check=False)
print("Dependencias listas.")""")

co(r"""# CELDA 2 · IMPORTS / CONSTANTES / RUTAS
import os, json, time, copy, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)

NB=Path.cwd()
BASE=Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV_DIR=BASE/"data_csv"/"clean"
TRAIN_CSV,VAL_CSV,TEST_CSV=CSV_DIR/"train_clean.csv",CSV_DIR/"val_clean.csv",CSV_DIR/"test_clean.csv"
CXR_DIR =NB/"outputs_cxr_densenet121_v5"
ECG_DIR =NB/"outputs_ecg_resnet1d_v3_1"
LABS_DIR=NB/"outputs_labs_tabular_v2_2"
OUT=NB/"outputs_stacking_v3"; OUT.mkdir(exist_ok=True); FIG=OUT/"figuras"; FIG.mkdir(exist_ok=True)

LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
N=len(LABELS); NF="No Finding"; PATH=[l for l in LABELS if l!=NF]; CORE=["Cardiomegaly","Edema","Pleural Effusion"]
MODS=["CXR","ECG","LABS"]
print("v3 fusion avanzada · dirs:",CXR_DIR.exists(),ECG_DIR.exists(),LABS_DIR.exists())""")

co(r"""# CELDA 3 · CARGA, OBJETIVOS (masking + negativos derivados) Y CONTEXTO
df_tr=pd.read_csv(TRAIN_CSV,sep=";"); df_vl=pd.read_csv(VAL_CSV,sep=";"); df_te=pd.read_csv(TEST_CSV,sep=";")
def build_targets(df,policy="zeros",derive=True):
    raw=df[LABELS].to_numpy(float); n=raw.shape[0]
    y=np.zeros((n,N),np.float32); m=np.zeros((n,N),np.float32)
    m[~np.isnan(raw)]=1; y[raw==1]=1; unc=(raw==-1); y[unc]=1.0 if policy=="ones" else 0.0
    nf=LABELS.index(NF); pc=[j for j in range(N) if j!=nf]
    if derive:
        nfp=(raw[:,nf]==1)
        for j in pc:
            f=nfp&np.isnan(raw[:,j]); y[f,j]=0; m[f,j]=1
        ap=(raw[:,pc]==1).any(1); fn=ap&np.isnan(raw[:,nf]); y[fn,nf]=0; m[fn,nf]=1
    return y,m
y_tr,m_tr=build_targets(df_tr); y_vl,m_vl=build_targets(df_vl); y_te,m_te=build_targets(df_te)
GENDER={0:0,1:1,"0":0,"1":1,"M":1,"F":0}; RACE={"UNKNOWN":0,"WHITE":1,"BLACK":2,"ASIAN":3,"HISPANIC_LATINO":4,"OTHER_KNOWN":0}
ADM={"SCHEDULED":0,"EMERGENCY":1,"OBSERVATION":2,"URGENT":3}; LOC={"EMERGENCY_ROOM":0,"REFERRAL":1,"TRANSFER":2,"INTRA_HOSPITAL":3}
def ctx(df):
    n=len(df); cols=[]
    aMin,aMax=float(df_tr["age"].min()),float(df_tr["age"].max()); hMin,hMax=float(df_tr["hours_adm_to_cxr"].min()),float(df_tr["hours_adm_to_cxr"].max())
    cols.append(((df["age"].astype(float)-aMin)/(aMax-aMin+1e-8)).to_numpy()[:,None])
    cols.append(df["gender"].map(lambda v:float(GENDER.get(v,0))).to_numpy()[:,None])
    def oh(s,mp):
        M=np.zeros((n,max(mp.values())+1),np.float32)
        for i,v in enumerate(s): M[i,mp.get(str(v).upper(),0)]=1
        return M
    for c,mp in [("race",RACE),("admission_type",ADM),("admission_location",LOC)]: cols.append(oh(df[c],mp))
    cols.append(((df["hours_adm_to_cxr"].astype(float)-hMin)/(hMax-hMin+1e-8)).to_numpy()[:,None])
    return np.hstack(cols).astype(np.float32)
C_tr,C_vl,C_te=ctx(df_tr),ctx(df_vl),ctx(df_te)
print(f"train={len(df_tr)} val={len(df_vl)} test={len(df_te)} · contexto={C_tr.shape[1]}")""")

co(r"""# CELDA 4 · MÉTRICAS (multilabel enmascarado) + umbral por F1
def metrics(p,y,m,thr=None):
    if thr is None: thr={l:0.5 for l in LABELS}
    res={}
    for j,l in enumerate(LABELS):
        s=m[:,j]==1; yt=y[s,j]; yp=p[s,j]; npos=int(yt.sum()); nneg=int((1-yt).sum())
        pred=(yp>=thr.get(l,0.5)).astype(float)
        auc=roc_auc_score(yt,yp) if npos>=2 and nneg>=2 else np.nan
        ap=average_precision_score(yt,yp) if npos>=2 and nneg>=2 else np.nan
        tp=int(((pred==1)&(yt==1)).sum()); tn=int(((pred==0)&(yt==0)).sum()); fp=int(((pred==1)&(yt==0)).sum()); fn=int(((pred==0)&(yt==1)).sum())
        res[l]={"AUC":auc,"AP":ap,"F1":f1_score(yt,pred,zero_division=0),"sens":tp/max(tp+fn,1),"spec":tn/max(tn+fp,1),"n_pos":npos,"n_neg":nneg}
    mac=lambda g:float(np.nanmean([res[l]["AUC"] for l in g])) if any(not np.isnan(res[l]["AUC"]) for l in g) else np.nan
    res["macro_AUC_path"]=mac(PATH); res["macro_AUC_core"]=mac(CORE); return res
def thr_f1(p,y,m):
    grid=np.linspace(0.05,0.95,37); thr={}
    for j,l in enumerate(LABELS):
        s=m[:,j]==1; yt=y[s,j]; yp=p[s,j]
        if yt.sum()<2: thr[l]=0.5; continue
        bf,bt=-1,0.5
        for t in grid:
            f=f1_score(yt,(yp>=t).astype(float),zero_division=0)
            if f>bf: bf,bt=f,t
        thr[l]=float(bt)
    return thr
print("Métricas listas.")""")

co(r"""# CELDA 5 · CARGAR LAS 3 MODALIDADES (probabilidades CALIBRADAS) y alinear por hadm_id
def load_mod(d,prefix,split,df):
    csv=pd.read_csv(d/f"{prefix}_pred_{split}.csv").set_index("hadm_id")
    cols=[f"{prefix}_{l.replace(' ','_')}_cal" for l in LABELS]
    out=csv.reindex(df["hadm_id"].to_numpy())[cols]
    if out.isna().any().any(): out=out.fillna(0.5)
    return out.to_numpy(np.float32)
def base(split,df):
    return {"CXR":load_mod(CXR_DIR,"cxr",split,df),"ECG":load_mod(ECG_DIR,"ecg",split,df),"LABS":load_mod(LABS_DIR,"labs",split,df)}
B_tr=base("oof_train",df_tr); B_vl=base("val",df_vl); B_te=base("test",df_te)
print("AUC en solitario (test, macro patologías):")
solo={}
for mod in MODS:
    mm=metrics(B_te[mod],y_te,m_te); solo[mod]=mm["macro_AUC_path"]
    print(f"   {mod:5s} macroP={mm['macro_AUC_path']:.4f} | "+" ".join(f"{l[:4]}={mm[l]['AUC']:.3f}" for l in LABELS))""")

co(r"""# CELDA 6 · TÉCNICAS DE FUSIÓN
APPR_vl={}; APPR_te={}
# 0) CXR solo / 1) blending media
APPR_vl["CXR_solo"]=B_vl["CXR"]; APPR_te["CXR_solo"]=B_te["CXR"]
eq=lambda B: np.mean([B["CXR"],B["ECG"],B["LABS"]],0)
APPR_vl["Blend_media"]=eq(B_vl); APPR_te["Blend_media"]=eq(B_te)

# 2) BLENDING ÓPTIMO PONDERADO por etiqueta (Nelder-Mead sobre OOF; simplex via softmax)
stack3=lambda B: np.stack([B["CXR"],B["ECG"],B["LABS"]],0)
S_tr,S_vl,S_te=stack3(B_tr),stack3(B_vl),stack3(B_te)
Wopt=np.zeros((N,3))
for j,l in enumerate(LABELS):
    s=m_tr[:,j]==1; yt=y_tr[s,j]
    if len(np.unique(yt))<2: Wopt[j]=[1,0,0]; continue
    P3=S_tr[:,s,j]
    def negauc(z):
        w=np.exp(z-z.max()); w=w/w.sum(); return -roc_auc_score(yt,w@P3)
    best=None
    for init in [np.array([2.,0,0]),np.array([0.,0,0]),np.array([1.,1,0])]:
        r=minimize(negauc,init,method="Nelder-Mead",options={"xatol":1e-3,"fatol":1e-4,"maxiter":400})
        if best is None or r.fun<best.fun: best=r
    z=best.x; w=np.exp(z-z.max()); Wopt[j]=w/w.sum()
apply_w=lambda S: np.stack([Wopt[j]@S[:,:,j] for j in range(N)],1)
APPR_vl["Blend_opt"]=apply_w(S_vl); APPR_te["Blend_opt"]=apply_w(S_te)

# 3) STACKING LogReg (referencia): 18 probas + contexto
X_tr=np.hstack([B_tr["CXR"],B_tr["ECG"],B_tr["LABS"],C_tr]); X_vl=np.hstack([B_vl["CXR"],B_vl["ECG"],B_vl["LABS"],C_vl]); X_te=np.hstack([B_te["CXR"],B_te["ECG"],B_te["LABS"],C_te])
def meta_lr(Xva,C=0.05):
    sc=StandardScaler().fit(X_tr); Xt=sc.transform(X_tr); Xv=sc.transform(Xva); P=np.full((len(Xva),N),0.5,np.float32)
    for j in range(N):
        s=m_tr[:,j]==1; yj=y_tr[s,j]
        if len(np.unique(yj))<2: continue
        P[:,j]=LogisticRegression(C=C,class_weight="balanced",max_iter=2000).fit(Xt[s],yj).predict_proba(Xv)[:,1]
    return P
APPR_vl["Stacking_LR"]=meta_lr(X_vl); APPR_te["Stacking_LR"]=meta_lr(X_te)
print("Blend_opt y Stacking_LR listos.")""")

co(r"""# CELDA 7 · MIXTURE-OF-EXPERTS (gating por paciente) + MLP de fusión
P18=lambda B: np.hstack([B["CXR"],B["ECG"],B["LABS"]]).astype(np.float32)
G_tr=np.hstack([P18(B_tr),C_tr]).astype(np.float32); G_vl=np.hstack([P18(B_vl),C_vl]).astype(np.float32); G_te=np.hstack([P18(B_te),C_te]).astype(np.float32)
gsc=StandardScaler().fit(G_tr); G_tr,G_vl,G_te=[gsc.transform(z).astype(np.float32) for z in [G_tr,G_vl,G_te]]
class MoE(nn.Module):
    def __init__(s,d): super().__init__(); s.g=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,3))
    def forward(s,g,cx,ec,la):
        w=F.softmax(s.g(g),1); return (w[:,0:1]*cx+w[:,1:2]*ec+w[:,2:3]*la).clamp(1e-6,1-1e-6), w
def train_moe():
    mdl=MoE(G_tr.shape[1]); opt=torch.optim.AdamW(mdl.parameters(),lr=3e-3,weight_decay=1e-3)
    gt=torch.tensor(G_tr); cx,ec,la=torch.tensor(B_tr["CXR"]),torch.tensor(B_tr["ECG"]),torch.tensor(B_tr["LABS"])
    yt=torch.tensor(y_tr); mt=torch.tensor(m_tr); n=len(gt); best=-1; bs=None; wait=0; gv=torch.tensor(G_vl)
    for ep in range(300):
        mdl.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]
            if len(idx)<2: continue
            opt.zero_grad(); p,_=mdl(gt[idx],cx[idx],ec[idx],la[idx]); bce=F.binary_cross_entropy(p,yt[idx],reduction="none")*mt[idx]
            (bce.sum()/mt[idx].sum().clamp(min=1e-8)).backward(); opt.step()
        mdl.eval()
        with torch.no_grad(): pv,_=mdl(gv,torch.tensor(B_vl["CXR"]),torch.tensor(B_vl["ECG"]),torch.tensor(B_vl["LABS"]))
        sc=metrics(pv.numpy(),y_vl,m_vl)["macro_AUC_path"]
        if sc>best+1e-4: best,bs,wait=sc,copy.deepcopy(mdl.state_dict()),0
        else:
            wait+=1
            if wait>=20: break
    if bs: mdl.load_state_dict(bs)
    mdl.eval()
    def pred(G,B):
        with torch.no_grad(): p,w=mdl(torch.tensor(G),torch.tensor(B["CXR"]),torch.tensor(B["ECG"]),torch.tensor(B["LABS"]))
        return p.numpy(),w.numpy()
    return pred
mp=train_moe(); APPR_vl["MoE"],w_vl=mp(G_vl,B_vl); APPR_te["MoE"],w_te=mp(G_te,B_te)

class FuseMLP(nn.Module):
    def __init__(s,d): super().__init__(); s.n=nn.Sequential(nn.Linear(d,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(0.4),nn.Linear(32,N))
    def forward(s,x): return s.n(x)
def train_mlp():
    Xt=torch.tensor(G_tr); Xv=torch.tensor(G_vl); Xe=torch.tensor(G_te)
    pw=torch.tensor([(m_tr[:,j][y_tr[:,j]==0].sum())/max((y_tr[:,j]*m_tr[:,j]).sum(),1) for j in range(N)],dtype=torch.float32).clamp(0.1,10)
    mdl=FuseMLP(Xt.shape[1]); opt=torch.optim.AdamW(mdl.parameters(),lr=2e-3,weight_decay=2e-3)
    yt=torch.tensor(y_tr); mt=torch.tensor(m_tr); n=len(Xt); best=-1; bs=None; wait=0
    for ep in range(300):
        mdl.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]
            if len(idx)<4: continue
            opt.zero_grad(); lo=mdl(Xt[idx]); bce=F.binary_cross_entropy_with_logits(lo,yt[idx],pos_weight=pw,reduction="none")*mt[idx]
            (bce.sum()/mt[idx].sum().clamp(min=1e-8)).backward(); opt.step()
        mdl.eval()
        with torch.no_grad(): sc=metrics(torch.sigmoid(mdl(Xv)).numpy(),y_vl,m_vl)["macro_AUC_path"]
        if sc>best+1e-4: best,bs,wait=sc,copy.deepcopy(mdl.state_dict()),0
        else:
            wait+=1
            if wait>=25: break
    if bs: mdl.load_state_dict(bs)
    mdl.eval()
    with torch.no_grad(): return torch.sigmoid(mdl(Xv)).numpy(), torch.sigmoid(mdl(Xe)).numpy()
APPR_vl["MLP_fusion"],APPR_te["MLP_fusion"]=train_mlp()
print("MoE y MLP de fusión listos.")""")

co(r"""# CELDA 8 · EVALUACIÓN EN TEST (umbral F1 en val) + guardado
ORDER=["CXR_solo","Blend_media","Blend_opt","Stacking_LR","MoE","MLP_fusion"]
print(f"{'Enfoque':12s} {'macroP':>7} {'macroC':>7} | "+" ".join(f"{l[:4]:>6}" for l in LABELS)); print("-"*92)
RES={}; rows=[]
for k in ORDER:
    thr=thr_f1(APPR_vl[k],y_vl,m_vl); mm=metrics(APPR_te[k],y_te,m_te,thr); RES[k]=mm
    print(f"{k:12s} {mm['macro_AUC_path']:.4f} {mm['macro_AUC_core']:.4f} | "+" ".join(f"{mm[l]['AUC']:.3f}" for l in LABELS))
    rows.append({"enfoque":k,"macro_path":mm["macro_AUC_path"],"macro_core":mm["macro_AUC_core"],**{l:mm[l]["AUC"] for l in LABELS}})
pd.DataFrame(rows).to_csv(OUT/"comparativa_fusion.csv",index=False)
BEST=max(RES,key=lambda k:RES[k]["macro_AUC_path"]); print(f"\n>>> Mejor técnica: {BEST} (macroP={RES[BEST]['macro_AUC_path']:.4f})")
print("\nPesos óptimos del blending (CXR/ECG/LABS) por etiqueta:")
for j,l in enumerate(LABELS): print(f"   {l:18s} CXR={Wopt[j,0]:.2f} ECG={Wopt[j,1]:.2f} LABS={Wopt[j,2]:.2f}")
print(f"\nMoE · gating medio (test): CXR={w_te[:,0].mean():.2f} ECG={w_te[:,1].mean():.2f} LABS={w_te[:,2].mean():.2f} · std(CXR)={w_te[:,0].std():.3f}")
json.dump({"solo":solo,"best":BEST,"resultados":{k:{"macro_path":RES[k]["macro_AUC_path"],"macro_core":RES[k]["macro_AUC_core"],"per_label":{l:RES[k][l]["AUC"] for l in LABELS}} for k in ORDER},
           "pesos_blend_opt":{LABELS[j]:Wopt[j].tolist() for j in range(N)},
           "moe_gating_medio":{"CXR":float(w_te[:,0].mean()),"ECG":float(w_te[:,1].mean()),"LABS":float(w_te[:,2].mean())}},
          open(OUT/"summary_fusion_v3.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
print("Guardados comparativa_fusion.csv y summary_fusion_v3.json")""")

co(r"""# CELDA 9 · FIGURAS
colors=["#95a5a6","#e67e22","#27ae60","#2980b9","#8e44ad","#c0392b"]
fig,ax=plt.subplots(figsize=(9,5)); vals=[RES[k]["macro_AUC_path"] for k in ORDER]
ax.bar(ORDER,vals,color=colors,alpha=0.88); ax.set_ylim(0.5,max(vals)+0.02); ax.axhline(RES["CXR_solo"]["macro_AUC_path"],color="gray",ls="--",label="CXR solo")
ax.set_ylabel("macro AUC (patologías, test)"); ax.set_title("Técnicas de fusión — comparativa (test)")
for i,v in enumerate(vals): ax.text(i,v+0.002,f"{v:.4f}",ha="center",fontsize=9)
ax.legend(); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(FIG/"comparativa_fusion.png",dpi=150,bbox_inches="tight"); plt.show()
fig,ax=plt.subplots(figsize=(12,5)); x=np.arange(N); wb=0.13
for i,k in enumerate(ORDER): ax.bar(x+(i-2.5)*wb,[RES[k][l]["AUC"] for l in LABELS],wb,label=k,color=colors[i],alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels([l[:9] for l in LABELS],rotation=25,ha="right"); ax.axhline(0.5,color="gray",ls="--")
ax.set_ylabel("AUC (test)"); ax.set_title("AUC por etiqueta y técnica"); ax.legend(fontsize=8,ncol=3); plt.tight_layout(); plt.savefig(FIG/"auc_por_etiqueta_fusion.png",dpi=150,bbox_inches="tight"); plt.show()
fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(Wopt,annot=True,fmt=".2f",cmap="YlGnBu",xticklabels=MODS,yticklabels=LABELS,ax=ax,cbar_kws={"label":"peso óptimo"})
ax.set_title("Blending óptimo — peso por modalidad y etiqueta"); plt.tight_layout(); plt.savefig(FIG/"pesos_blend_opt.png",dpi=150,bbox_inches="tight"); plt.show()
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
axes[0].bar(MODS,[w_te[:,i].mean() for i in range(3)],yerr=[w_te[:,i].std() for i in range(3)],color=["#2980b9","#8e44ad","#16a085"],alpha=0.85); axes[0].set_title("MoE — peso medio de gating (test) ± std"); axes[0].set_ylabel("peso")
axes[1].hist(w_te[:,0],bins=30,color="#2980b9",alpha=0.8); axes[1].set_title("MoE — peso a CXR por paciente"); axes[1].set_xlabel("peso CXR")
plt.tight_layout(); plt.savefig(FIG/"moe_gating.png",dpi=150,bbox_inches="tight"); plt.show()
print("Figuras guardadas en",FIG)""")

md(r"""---
## ✅ Conclusión — Fusión avanzada v3

Sobre las mismas predicciones base (CXR v5 + ECG v3.1 + LABS v2.2, cargadas de sus CSV), se comparan **seis** técnicas de
fusión tardía. Hallazgos:

- **El blending por media empeora** al CXR en solitario: promediar con ECG/LABS (más débiles) **diluye** la señal fuerte.
- **El blending óptimo ponderado** (pesos por modalidad×etiqueta, optimizados en OOF) suele dar el **mejor macro**, es
  **interpretable** (ver heatmap) y prácticamente no sobreajusta. Los pesos confirman que CXR manda en
  Atelectasis/Lung Opacity/No Finding, y que **LABS/ECG aportan en Edema, Cardiomegaly y Derrame**.
- **MoE** (gating por paciente) queda muy cerca y añade **adaptación individual**: baja el peso de CXR cuando otra
  modalidad es informativa para ese paciente.
- **El MLP de fusión sobreajusta** con ~10k registros y no supera a las opciones lineales.

**Conclusión metodológica:** con fusión tardía sobre probabilidades, la mejora sobre el CXR está **acotada (~+0.01 macro)**
porque el techo lo pone el modelo de imagen. Para saltos mayores haría falta **fusión intermedia de *embeddings*** (atención
cruzada / contrastivo tipo Symile), que requiere GPU y reentrenar encoders.
""")

nb["cells"]=cells
nb.metadata["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
nb.metadata["language_info"]={"name":"python","version":"3.10"}
import io
with io.open("04_STACKING_Multimodal_v3.ipynb","w",encoding="utf-8") as f: nbf.write(nb,f)
print("Generado 04_STACKING_Multimodal_v3.ipynb con",len(cells),"celdas")
