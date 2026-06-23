# -*- coding: utf-8 -*-
# Generador de 01_CXR_DenseNet121_v3 (PESADO): DenseNet121 + TTA(flip) + 2º backbone ResNet50@512 (con fallback)
# + cabeza MLP con bagging multisemilla + ensemble con LogReg. Todo congelado (embeddings). <24 h en CPU.
import nbformat as nbf, io
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); co=lambda s:C.append(nbf.v4.new_code_cell(s))

md(r"""# 🫁 CXR DenseNet121 — **v3 PESADO** (CPU · máximo nivel · modelo independiente)
## TFM · Módulo de Imagen (Radiografía de tórax) · Universidad de Salamanca

---

## 🎯 Qué es esta versión (la más compleja de las 3)
Como las etiquetas son radiográficas, el backbone congelado **DenseNet121** ya marca el techo y el *fine-tuning* agresivo
**empeoraba** (lo vimos en versiones previas). Por eso el "máximo nivel" aquí **no** es descongelar, sino **diversidad de
representación + robustez**, manteniendo todo congelado (viable en CPU):

| Ingrediente | Qué aporta |
|---|---|
| **DenseNet121@224** (CheXNet) | embeddings base (reutiliza la caché del v2) |
| **TTA — Test-Time Augmentation** | se promedia el embedding de la imagen original y su **espejo horizontal** → más estable |
| **2º backbone ResNet50@512** (xrv) | representación complementaria; se concatena. *Con fallback*: si falla, el notebook sigue solo con DenseNet |
| **Cabeza MLP + metadatos + correlación de etiquetas** | la mejor cabeza del v2 |
| **Bagging multisemilla** | se entrena la cabeza con varias semillas × K-fold y se promedia |
| **Ensemble con LogReg** | se combina con una cabeza lineal (descorrelaciona) |

## ⏱️ Coste (CPU, sin GPU)
Pasadas de backbone congelado: DenseNet@224 ×2 (orig+flip) + ResNet50@512 ×2 ≈ varias horas (la de 512 es la lenta).
Las cabezas sobre embeddings son segundos. **Diseñado para caber en < 24 h.** Si quieres acortar, pon `USE_RESNET=False`.

## 🛡️ Bases del proyecto (intactas)
Masking de NaN/−1 · **negativos derivados por exclusividad de *No Finding*** · `pos_weight` dinámico · **calibración
isotónica** · **OOF sin fuga** para el stacking · salidas ricas.
""")

co(r"""# CELDA 1 · DEPENDENCIAS
import subprocess, sys
try: import torch  # noqa
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","torch","--index-url","https://download.pytorch.org/whl/cpu","-q"],check=True)
for p in ["torchxrayvision","scikit-learn","pandas","numpy","matplotlib","seaborn","tqdm"]:
    subprocess.run([sys.executable,"-m","pip","install",p,"-q"],check=False)
print("Dependencias listas.")""")

co(r"""# CELDA 2 · IMPORTS, RUTAS Y CONSTANTES
import os, gc, json, time, copy, random, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchxrayvision as xrv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
warnings.filterwarnings("ignore")
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4); DEVICE=torch.device("cpu")

BASE=Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV=BASE/"data_csv"/"clean"; NPY=BASE/"data_npy"
TRAIN_CSV,VAL_CSV,TEST_CSV=CSV/"train_clean.csv",CSV/"val_clean.csv",CSV/"test_clean.csv"
CXR_NPY={"train":NPY/"train"/"cxr_train.npy","val":NPY/"val"/"cxr_val.npy","test":NPY/"test"/"cxr_test.npy"}
HADM_NPY={"train":NPY/"train"/"hadm_id_train.npy","val":NPY/"val"/"hadm_id_val.npy","test":NPY/"test"/"hadm_id_test.npy"}
DENSE_CACHE=Path("outputs_cxr_densenet121_v2")/"embeddings"      # embeddings DenseNet@224 ya cacheados por el v2
OUTPUT_DIR=Path("outputs_cxr_densenet121_v3"); OUTPUT_DIR.mkdir(exist_ok=True)
EMB_DIR=OUTPUT_DIR/"embeddings"; EMB_DIR.mkdir(exist_ok=True); FIG=OUTPUT_DIR/"figuras"; FIG.mkdir(exist_ok=True)

LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
N_LABELS=len(LABELS); NO_FINDING="No Finding"
PATHOLOGY=[l for l in LABELS if l!=NO_FINDING]; CORE=["Cardiomegaly","Edema","Pleural Effusion"]
XRV_RANGE=1024.0; EMB_BATCH=16
DENSE_W="densenet121-res224-all"; DENSE_SIZE=224
RESNET_W="resnet50-res512-all"; RESNET_SIZE=512
USE_TTA=True; USE_RESNET=True               # pon USE_RESNET=False para acortar (~horas menos)
K_FOLDS=5; BAG_SEEDS=[42,1,7]               # bagging multisemilla de la cabeza
EPOCHS,PATIENCE=40,6
# metadatos (18 dims)
GENDER={0:0,1:1,"0":0,"1":1,"M":1,"F":0}; RACE={"UNKNOWN":0,"WHITE":1,"BLACK":2,"ASIAN":3,"HISPANIC_LATINO":4,"OTHER_KNOWN":0}
ADM={"SCHEDULED":0,"EMERGENCY":1,"OBSERVATION":2,"URGENT":3}; LOC={"EMERGENCY_ROOM":0,"REFERRAL":1,"TRANSFER":2,"INTRA_HOSPITAL":3}
VIEW={"AP":0,"PA":1}; META_DIM=1+1+5+4+4+1+2
print("Salidas en", OUTPUT_DIR, "· USE_TTA", USE_TTA, "· USE_RESNET", USE_RESNET)""")

co(r"""# CELDA 3 · CARGA Y ALINEACIÓN CSV <-> imagen (join por hadm_id; .npy mapeado en disco)
def load_split(csv,cxr_npy,hadm,name):
    df=pd.read_csv(csv,sep=";"); h=np.load(hadm,allow_pickle=True); h2i={int(x):i for i,x in enumerate(h)}
    df["_npy_idx"]=df["hadm_id"].map(lambda z:h2i.get(int(z),-1)); n0=len(df)
    df=df[df["_npy_idx"]>=0].reset_index(drop=True); a=np.load(cxr_npy,mmap_mode="r")
    print(f"{name:5s}: CSV={n0:,} con imagen={len(df):,} · {a.shape}"); return df,a
df_train,cxr_train=load_split(TRAIN_CSV,CXR_NPY["train"],HADM_NPY["train"],"train")
df_val,cxr_val=load_split(VAL_CSV,CXR_NPY["val"],HADM_NPY["val"],"val")
df_test,cxr_test=load_split(TEST_CSV,CXR_NPY["test"],HADM_NPY["test"],"test")""")

co(r"""# CELDA 4 · OBJETIVOS (masking + negativos derivados) y METADATOS (18 dims)
def build_targets(df, uncertainty_policy="zeros", derive=True, verbose=False):
    raw=df[LABELS].to_numpy(dtype=float); N=raw.shape[0]
    labels=np.zeros((N,N_LABELS),np.float32); mask=np.zeros((N,N_LABELS),np.float32)
    mask[~np.isnan(raw)]=1.0; labels[raw==1]=1.0
    unc=(raw==-1); labels[unc]=1.0 if uncertainty_policy=="ones" else 0.0
    nf=LABELS.index(NO_FINDING); pc=[j for j in range(N_LABELS) if j!=nf]; ndp=ndn=0
    if derive:
        nfp=(raw[:,nf]==1)
        for j in pc:
            f=nfp&np.isnan(raw[:,j]); labels[f,j]=0.0; mask[f,j]=1.0; ndp+=int(f.sum())
        ap=(raw[:,pc]==1).any(1); fn=ap&np.isnan(raw[:,nf]); labels[fn,nf]=0.0; mask[fn,nf]=1.0; ndn=int(fn.sum())
    if verbose: print(f"   negativos derivados -> patologías={ndp:,} · No Finding={ndn:,}")
    return labels,mask
AGE_MIN,AGE_MAX=float(df_train["age"].min()),float(df_train["age"].max())
HRS_MIN,HRS_MAX=float(df_train["hours_adm_to_cxr"].min()),float(df_train["hours_adm_to_cxr"].max())
def build_metadata(df):
    N=len(df); M=np.zeros((N,META_DIM),np.float32)
    for i,(_,r) in enumerate(df.iterrows()):
        o=0
        M[i,o]=(float(r.get("age",AGE_MIN))-AGE_MIN)/(AGE_MAX-AGE_MIN+1e-8); o+=1
        M[i,o]=float(GENDER.get(r.get("gender",0),0)); o+=1
        M[i,o+RACE.get(str(r.get("race","UNKNOWN")).upper(),0)]=1.0; o+=5
        M[i,o+ADM.get(str(r.get("admission_type","EMERGENCY")).upper(),1)]=1.0; o+=4
        M[i,o+LOC.get(str(r.get("admission_location","EMERGENCY_ROOM")).upper(),0)]=1.0; o+=4
        M[i,o]=(float(r.get("hours_adm_to_cxr",HRS_MIN))-HRS_MIN)/(HRS_MAX-HRS_MIN+1e-8); o+=1
        M[i,o+VIEW.get(str(r.get("cxr_view","AP")).upper(),0)]=1.0; o+=2
    return M
y_train,m_train=build_targets(df_train,verbose=True); y_val,m_val=build_targets(df_val); y_test,m_test=build_targets(df_test)
meta_train,meta_val,meta_test=build_metadata(df_train),build_metadata(df_val),build_metadata(df_test)
print("Metadatos:",meta_train.shape)""")

co(r"""# CELDA 5 · DATASET de imagen (preprocesado xrv) con flip opcional + extractor de embeddings
class CXRDataset(Dataset):
    def __init__(s,df,cxr,size,flip=False): s.idx=df["_npy_idx"].to_numpy(); s.cxr=cxr; s.size=size; s.flip=flip
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        arr=np.asarray(s.cxr[int(s.idx[i])],np.float32); g=arr[0] if arr.ndim==3 else arr
        if s.flip: g=g[:,::-1].copy()                                  # espejo horizontal (TTA)
        gmin,gmax=float(g.min()),float(g.max()); g=(g-gmin)/(gmax-gmin+1e-8); g=(2.0*g-1.0)*XRV_RANGE
        t=torch.from_numpy(g)[None,None]; t=F.interpolate(t,size=(s.size,s.size),mode="bilinear",align_corners=False)
        return t[0]
@torch.no_grad()
def extract(model, df, cxr, size, flip, desc):
    loader=DataLoader(CXRDataset(df,cxr,size,flip),batch_size=EMB_BATCH,shuffle=False,num_workers=0); out=[]
    for x in tqdm(loader,desc=desc,leave=False):
        f=model.features(x.to(DEVICE)); f=F.relu(f,inplace=True)
        f=F.adaptive_avg_pool2d(f,(1,1)).reshape(f.shape[0],-1); out.append(f.cpu().numpy())
    return np.concatenate(out,0).astype(np.float32)
print("Dataset y extractor listos.")""")

co(r"""# CELDA 6 · EMBEDDINGS: DenseNet@224 (caché + TTA flip) + ResNet50@512 (con fallback). Cacheados.
def cached(name, fn):
    p=EMB_DIR/f"{name}.npy"
    if p.exists(): print(f"   caché {name}"); return np.load(p)
    a=fn(); np.save(p,a); return a
# --- DenseNet base: reutiliza la caché del v2 si está; si no, la calcula ---
def dense_base(split, df, cxr):
    shared=DENSE_CACHE/f"emb_{split}.npy"
    if shared.exists() and len(np.load(shared,mmap_mode="r"))==len(df): return np.load(shared)
    m=xrv.models.DenseNet(weights=DENSE_W).to(DEVICE).eval()
    e=extract(m,df,cxr,DENSE_SIZE,False,f"dense {split}"); del m; gc.collect(); return e
emb={}
for split,df,cxr in [("train",df_train,cxr_train),("val",df_val,cxr_val),("test",df_test,cxr_test)]:
    base=cached(f"dense_{split}", lambda df=df,cxr=cxr,split=split: dense_base(split,df,cxr))
    parts=[base]
    if USE_TTA:
        mdl=xrv.models.DenseNet(weights=DENSE_W).to(DEVICE).eval()
        flip=cached(f"dense_flip_{split}", lambda df=df,cxr=cxr,mdl=mdl,split=split: extract(mdl,df,cxr,DENSE_SIZE,True,f"dense-flip {split}"))
        del mdl; gc.collect(); parts=[ (base+flip)/2.0 ]                # TTA: promedio orig+flip
    if USE_RESNET:
        try:
            rm=xrv.models.ResNet(weights=RESNET_W).to(DEVICE).eval()
            r=cached(f"resnet_{split}", lambda df=df,cxr=cxr,rm=rm,split=split: extract(rm,df,cxr,RESNET_SIZE,False,f"resnet {split}"))
            parts.append(r); del rm; gc.collect()
        except Exception as ex:
            print("   ⚠ ResNet50@512 no disponible, sigo solo con DenseNet:",str(ex)[:80])
    emb[split]=np.hstack(parts).astype(np.float32)
EMB_DIM=emb["train"].shape[1]
print("EMB_DIM =",EMB_DIM,"(DenseNet 1024" + (" + ResNet 2048" if emb['train'].shape[1]>1024 else "") + ")")""")

co(r"""# CELDA 7 · MÉTRICAS, pos_weight, pérdida, cabeza MLP (img+meta+correlación) y entrenamiento
def multilabel_metrics(probs,labels,mask,thresholds=None):
    if thresholds is None: thresholds={l:0.5 for l in LABELS}
    res={}
    for j,l in enumerate(LABELS):
        s=mask[:,j]==1; yt=labels[s,j]; yp=probs[s,j]; npos=int(yt.sum()); nneg=int((1-yt).sum())
        pred=(yp>=thresholds.get(l,0.5)).astype(float)
        auc=roc_auc_score(yt,yp) if npos>=2 and nneg>=2 else float("nan")
        ap=average_precision_score(yt,yp) if npos>=2 and nneg>=2 else float("nan")
        tp=int(((pred==1)&(yt==1)).sum()); tn=int(((pred==0)&(yt==0)).sum())
        fp=int(((pred==1)&(yt==0)).sum()); fn=int(((pred==0)&(yt==1)).sum())
        res[l]={"AUC":auc,"AP":ap,"F1":f1_score(yt,pred,zero_division=0),"sens":tp/max(tp+fn,1),"spec":tn/max(tn+fp,1),
                "n_pos":npos,"n_neg":nneg,"TP":tp,"TN":tn,"FP":fp,"FN":fn,"thr":thresholds.get(l,0.5)}
    mac=lambda g:float(np.nanmean([res[l]["AUC"] for l in g])) if any(not np.isnan(res[l]["AUC"]) for l in g) else float("nan")
    res["macro_AUC_core"]=mac(CORE); res["macro_AUC_path"]=mac(PATHOLOGY); return res
def best_thresholds_by_f1(probs,labels,mask):
    grid=np.linspace(0.05,0.95,37); thr={}
    for j,l in enumerate(LABELS):
        s=mask[:,j]==1; yt=labels[s,j]; yp=probs[s,j]
        if yt.sum()<2: thr[l]=0.5; continue
        bf,bt=-1,0.5
        for t in grid:
            f=f1_score(yt,(yp>=t).astype(float),zero_division=0)
            if f>bf: bf,bt=f,t
        thr[l]=float(bt)
    return thr
def pos_weights(y,m,clip=10.0):
    w=np.ones(N_LABELS,np.float32)
    for j in range(N_LABELS):
        s=m[:,j]==1; pos=(y[s,j]==1).sum(); neg=(y[s,j]==0).sum(); w[j]=np.clip(neg/max(pos,1),1/clip,clip)
    return torch.tensor(w)
class LabelCorr(nn.Module):
    def __init__(s,n=N_LABELS): super().__init__(); s.lin=nn.Linear(n,n,bias=False); nn.init.eye_(s.lin.weight); s.lin.weight.data*=0.1; s.norm=nn.LayerNorm(n)
    def forward(s,x): return s.norm(x+s.lin(x))
class CXRHead(nn.Module):
    def __init__(s,emb_dim,dropout=0.3):
        super().__init__()
        s.img=nn.Sequential(nn.Linear(emb_dim,512),nn.BatchNorm1d(512),nn.ReLU(inplace=True))
        s.meta=nn.Sequential(nn.Linear(META_DIM,64),nn.BatchNorm1d(64),nn.ReLU(inplace=True))
        s.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(512+64,128),nn.BatchNorm1d(128),nn.ReLU(inplace=True),nn.Dropout(dropout*0.5),nn.Linear(128,N_LABELS))
        s.corr=LabelCorr()
    def forward(s,e,meta): z=torch.cat([s.img(e),s.meta(meta)],1); return s.corr(s.head(z))
@torch.no_grad()
def predict_head(h,E,Me,bs=256):
    h.eval(); out=[]
    for i in range(0,len(E),bs):
        out.append(torch.sigmoid(h(torch.from_numpy(E[i:i+bs]),torch.from_numpy(Me[i:i+bs]))).cpu().numpy())
    return np.concatenate(out,0)
def train_head(Etr,Mtr,ytr,mtr,Eva,Mva,yva,mva,seed=42,epochs=EPOCHS,patience=PATIENCE):
    torch.manual_seed(seed); pw=pos_weights(ytr,mtr); h=CXRHead(Etr.shape[1]).to(DEVICE)
    opt=torch.optim.AdamW(h.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,1e-7)
    Et=torch.from_numpy(Etr); Mt=torch.from_numpy(Mtr); Yt=torch.from_numpy(ytr); Kt=torch.from_numpy(mtr); n=len(Et); best,bs,wait=-1,None,0
    for ep in range(epochs):
        h.train(); perm=torch.randperm(n)
        for i in range(0,n,128):
            idx=perm[i:i+128]
            if len(idx)<2: continue
            opt.zero_grad(); lo=h(Et[idx],Mt[idx])
            bce=F.binary_cross_entropy_with_logits(lo,Yt[idx],pos_weight=pw,reduction="none")*Kt[idx]
            (bce.sum()/Kt[idx].sum().clamp(min=1e-8)).backward(); opt.step()
        sch.step(); sc=multilabel_metrics(predict_head(h,Eva,Mva),yva,mva)["macro_AUC_path"]
        if not np.isnan(sc) and sc>best+1e-4: best,bs,wait=sc,copy.deepcopy(h.state_dict()),0
        else:
            wait+=1
            if wait>=patience: break
    if bs is not None: h.load_state_dict(bs)
    return h
print("Cabeza y entrenamiento listos.")""")

co(r"""# CELDA 8 · K-FOLD + BAGGING MULTISEMILLA -> OOF (sin fuga) + val/test ; ENSEMBLE con LogReg
sc_=StandardScaler().fit(emb["train"]); Etr=sc_.transform(emb["train"]).astype(np.float32)
Eva=sc_.transform(emb["val"]).astype(np.float32); Ete=sc_.transform(emb["test"]).astype(np.float32)
def logreg_ovr(Xtr,ytr,mtr,Xva):
    P=np.full((len(Xva),N_LABELS),0.5,np.float32)
    for j in range(N_LABELS):
        s=mtr[:,j]==1; yj=ytr[s,j]
        if len(np.unique(yj))<2: P[:,j]=float(yj.mean()) if len(yj) else 0.5; continue
        P[:,j]=LogisticRegression(C=0.5,class_weight="balanced",max_iter=2000).fit(Xtr[s],yj).predict_proba(Xva)[:,1]
    return P
kf=KFold(K_FOLDS,shuffle=True,random_state=SEED)
oof_mlp=np.zeros((len(df_train),N_LABELS),np.float32); oof_lr=np.zeros_like(oof_mlp)
acc_val_mlp=np.zeros((len(df_val),N_LABELS),np.float32); acc_test_mlp=np.zeros((len(df_test),N_LABELS),np.float32)
acc_val_lr=np.zeros_like(acc_val_mlp); acc_test_lr=np.zeros_like(acc_test_mlp); t0=time.time()
for k,(tr,va) in enumerate(kf.split(np.arange(len(df_train)))):
    # MLP con bagging multisemilla
    pv=np.zeros((len(va),N_LABELS),np.float32); pvl=np.zeros((len(df_val),N_LABELS),np.float32); pte=np.zeros((len(df_test),N_LABELS),np.float32)
    for sd in BAG_SEEDS:
        h=train_head(Etr[tr],meta_train[tr],y_train[tr],m_train[tr],Etr[va],meta_train[va],y_train[va],m_train[va],seed=sd)
        pv+=predict_head(h,Etr[va],meta_train[va]); pvl+=predict_head(h,Eva,meta_val); pte+=predict_head(h,Ete,meta_test); del h; gc.collect()
    oof_mlp[va]=pv/len(BAG_SEEDS); acc_val_mlp+=pvl/len(BAG_SEEDS); acc_test_mlp+=pte/len(BAG_SEEDS)
    # LogReg
    oof_lr[va]=logreg_ovr(Etr[tr],y_train[tr],m_train[tr],Etr[va])
    acc_val_lr+=logreg_ovr(Etr,y_train,m_train,Eva); acc_test_lr+=logreg_ovr(Etr,y_train,m_train,Ete)
    print(f"Fold {k+1}/{K_FOLDS} ({(time.time()-t0)/60:.1f} min)")
val_mlp=acc_val_mlp/K_FOLDS; test_mlp=acc_test_mlp/K_FOLDS; val_lr=acc_val_lr/K_FOLDS; test_lr=acc_test_lr/K_FOLDS
# ENSEMBLE: media de MLP (bagged) y LogReg
oof_train=0.5*oof_mlp+0.5*oof_lr; val_pred_raw=0.5*val_mlp+0.5*val_lr; test_pred_raw=0.5*test_mlp+0.5*test_lr
print("OOF macro_path: MLP=%.4f LogReg=%.4f Ensemble=%.4f" % (
    multilabel_metrics(oof_mlp,y_train,m_train)["macro_AUC_path"],
    multilabel_metrics(oof_lr,y_train,m_train)["macro_AUC_path"],
    multilabel_metrics(oof_train,y_train,m_train)["macro_AUC_path"]))""")

co(r"""# CELDA 9 · CALIBRACIÓN ISOTÓNICA (val) + umbrales por F1
calibrators={}
for j,l in enumerate(LABELS):
    s=m_val[:,j]==1; yt=y_val[s,j]; yp=val_pred_raw[s,j]
    calibrators[l]=None if len(np.unique(yt))<2 else IsotonicRegression(out_of_bounds="clip").fit(yp,yt)
def apply_cal(P):
    O=P.copy()
    for j,l in enumerate(LABELS):
        if calibrators[l] is not None: O[:,j]=calibrators[l].predict(P[:,j])
    return O
oof_cal=apply_cal(oof_train); val_pred=apply_cal(val_pred_raw); test_pred=apply_cal(test_pred_raw)
thr_val=best_thresholds_by_f1(val_pred,y_val,m_val); print("Calibrado y umbrales listos.")""")

co(r"""# CELDA 10 · EVALUACIÓN EN TEST + tabla + summary
M=multilabel_metrics(test_pred,y_test,m_test,thresholds=thr_val); rows=[]
print(f"{'Etiqueta':18s} {'AUC':>7} {'AP':>7} {'F1':>7} {'N+':>5} {'N-':>5}"); print("-"*54)
for l in LABELS:
    m=M[l]; auc=f"{m['AUC']:.4f}" if not np.isnan(m['AUC']) else "  N/A"
    print(f"{l:18s} {auc:>7} {m['AP']:7.4f} {m['F1']:7.4f} {m['n_pos']:5d} {m['n_neg']:5d}")
    rows.append({"label":l,**{k:m[k] for k in ['AUC','AP','F1','sens','spec','n_pos','n_neg']}})
print("-"*54); print(f"MACRO core={M['macro_AUC_core']:.4f} · MACRO patol.={M['macro_AUC_path']:.4f}")
pd.DataFrame(rows).to_csv(OUTPUT_DIR/"metrics_per_label_v3.csv",index=False)
json.dump({"version":"CXR v3 pesado (DenseNet+TTA+ResNet50 + MLP bagging + LogReg)","emb_dim":int(EMB_DIM),
           "test_macro_core":M["macro_AUC_core"],"test_macro_path":M["macro_AUC_path"],"test_per_label":{l:M[l] for l in LABELS}},
          open(OUTPUT_DIR/"summary_v3.json","w",encoding="utf-8"),indent=2,default=str,ensure_ascii=False)
print("Guardados metrics_per_label_v3.csv y summary_v3.json")""")

co(r"""# CELDA 11 · MATRICES DE CONFUSIÓN + ROC/PR + AUC por etiqueta
fig,axes=plt.subplots(2,3,figsize=(13,8)); fig.suptitle("CXR v3 — Matrices de confusión (test)",fontweight="bold")
for j,l in enumerate(LABELS):
    ax=axes[j//3,j%3]; s=m_test[:,j]==1; yt=y_test[s,j]; yp=(test_pred[s,j]>=thr_val.get(l,0.5)).astype(int)
    if len(yt)==0: ax.axis("off"); continue
    sns.heatmap(confusion_matrix(yt,yp,labels=[0,1]),annot=True,fmt="d",cmap="Blues",cbar=False,ax=ax,xticklabels=["P0","P1"],yticklabels=["R0","R1"]); ax.set_title(l,fontsize=10)
plt.tight_layout(); plt.savefig(FIG/"confusion_v3.png",dpi=150,bbox_inches="tight"); plt.show()
fig,axes=plt.subplots(1,2,figsize=(14,5))
for j,l in enumerate(LABELS):
    s=m_test[:,j]==1; yt=y_test[s,j]; yp=test_pred[s,j]
    if len(np.unique(yt))<2: continue
    fpr,tpr,_=roc_curve(yt,yp); axes[0].plot(fpr,tpr,label=f"{l} ({M[l]['AUC']:.3f})")
    pr,rc,_=precision_recall_curve(yt,yp); axes[1].plot(rc,pr,label=f"{l} ({M[l]['AP']:.3f})")
axes[0].plot([0,1],[0,1],"k--",alpha=0.4); axes[0].set_title("ROC (test)"); axes[0].legend(fontsize=8)
axes[1].set_title("Precisión-Recall (test)"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"roc_pr_v3.png",dpi=150,bbox_inches="tight"); plt.show()""")

co(r"""# CELDA 12 · EXPORTAR OOF/val/test PARA EL STACKING (hadm_id, cxr_<label>, cxr_<label>_cal)
def save_predictions(df, raw, cal, name):
    cols={"hadm_id":df["hadm_id"].to_numpy()}
    for j,l in enumerate(LABELS):
        key=l.replace(" ","_"); cols[f"cxr_{key}"]=raw[:,j]; cols[f"cxr_{key}_cal"]=cal[:,j]
    out=pd.DataFrame(cols); p=OUTPUT_DIR/f"cxr_pred_{name}.csv"; out.to_csv(p,index=False); print("   guardado",p.name)
save_predictions(df_train, oof_train, oof_cal, "oof_train")
save_predictions(df_val, val_pred_raw, val_pred, "val")
save_predictions(df_test, test_pred_raw, test_pred, "test")
print("OOF/val/test del CXR v3 exportados.")""")

md(r"""---
## ✅ Resumen — CXR v3 (pesado)
Diversidad de representación con **todo congelado** (viable en CPU): **DenseNet121 + TTA(flip) + ResNet50@512**
(concatenados), cabeza **MLP+metadatos+correlación** con **bagging multisemilla**, **ensemble con LogReg**, calibración y
**OOF sin fuga**. Si el AUC no supera al **v2 (0.809)**, confirma que para estas etiquetas el DenseNet congelado ya está
en el techo — la conclusión honesta del módulo de imagen.""")

nb["cells"]=C; nb.metadata["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
nb.metadata["language_info"]={"name":"python","version":"3.10"}
io.open("01_CXR_DenseNet121_v3.ipynb","w",encoding="utf-8").write(nbf.writes(nb))
print("Generado 01_CXR_DenseNet121_v3.ipynb con",len(C),"celdas")
