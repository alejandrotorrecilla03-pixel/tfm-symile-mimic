# -*- coding: utf-8 -*-
# Motor de explicabilidad ENRIQUECIDA + figuras para el dosier de la herramienta de decisión.
# Añade: aportación de cada prueba a la decisión, contrafactual (quitar una prueba), fiabilidad por confianza,
# acuerdo/discrepancia y aviso de baja certeza (abstención). Selecciona casos representativos y genera figuras + JSON.
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); np.random.seed(42)

NB=Path.cwd(); BASE=Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV=BASE/"data_csv"/"clean"
DIRS={"CXR":(NB/"outputs_cxr_densenet121_v2","cxr"),"ECG":(NB/"outputs_ecg_resnet1d_v2","ecg"),"LABS":(NB/"outputs_labs_tabular_v2","labs")}
OUT=NB/"outputs_herramienta_decision"; FG=OUT/"dosier"; FG.mkdir(parents=True,exist_ok=True)
LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]; N=len(LABELS)
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema","Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgos","Pleural Effusion":"Derrame pleural"}
MODS=["CXR","ECG","LABS"]; MOD_ES={"CXR":"Radiografía","ECG":"ECG","LABS":"Analíticas"}; COL={"CXR":"#2980b9","ECG":"#8e44ad","LABS":"#16a085"}
df_vl=pd.read_csv(CSV/"val_clean.csv",sep=";"); df_te=pd.read_csv(CSV/"test_clean.csv",sep=";")
def tgt(df):
    raw=df[LABELS].to_numpy(float); n=raw.shape[0]; y=np.zeros((n,N),np.float32); m=np.zeros((n,N),np.float32)
    m[~np.isnan(raw)]=1; y[raw==1]=1; y[(raw==-1)]=0
    nf=LABELS.index("No Finding"); pc=[j for j in range(N) if j!=nf]; nfp=(raw[:,nf]==1)
    for j in pc:
        f=nfp&np.isnan(raw[:,j]); y[f,j]=0; m[f,j]=1
    ap=(raw[:,pc]==1).any(1); fn=ap&np.isnan(raw[:,nf]); y[fn,nf]=0; m[fn,nf]=1
    return y,m
y_vl,m_vl=tgt(df_vl); y_te,m_te=tgt(df_te)
def load(split,df):
    P={}
    for mod,(d,pref) in DIRS.items():
        c=pd.read_csv(d/f"{pref}_pred_{split}.csv").set_index("hadm_id"); cols=[f"{pref}_{l.replace(' ','_')}_cal" for l in LABELS]
        P[mod]=c.reindex(df["hadm_id"].to_numpy())[cols].fillna(0.5).to_numpy(np.float32)
    return P
Bvl=load("val",df_vl); Bte=load("test",df_te)
skill=np.zeros((3,N))
for mi,mod in enumerate(MODS):
    for j in range(N):
        s=m_vl[:,j]==1; yt=y_vl[s,j]
        if yt.sum()>=2 and (1-yt).sum()>=2: skill[mi,j]=max(roc_auc_score(yt,Bvl[mod][s,j])-0.5,0.0)
def reliability(mod,j,conf):
    s=m_vl[:,j]==1; p=Bvl[mod][s,j]; yt=y_vl[s,j]; c=np.abs(p-0.5)*2; sel=(c>=max(conf-0.15,0))&(c<=min(conf+0.15,1.0))
    if sel.sum()<10: sel=np.ones_like(c,bool)
    return float(((p[sel]>=0.5).astype(int)==yt[sel]).mean())

def explain(idx, j):
    probs={mo:float(Bte[mo][idx,j]) for mo in MODS}; conf={mo:abs(probs[mo]-0.5)*2 for mo in MODS}
    w={mo:skill[mi,j]*max(conf[mo],0.05) for mi,mo in enumerate(MODS)}; Z=sum(w.values())+1e-9
    wn={mo:w[mo]/Z for mo in MODS}; comb=sum(wn[mo]*probs[mo] for mo in MODS); lead=max(MODS,key=lambda mo:w[mo])
    aport={mo:wn[mo]*probs[mo] for mo in MODS}                      # suma = comb
    push={mo:wn[mo]*(probs[mo]-0.5) for mo in MODS}                 # empuje hacia SÍ(+)/NO(-); suma = comb-0.5
    cf={}                                                           # contrafactual: decisión SIN cada prueba
    for mo in MODS:
        otros=[o for o in MODS if o!=mo]; zz=sum(w[o] for o in otros)+1e-9
        cf[mo]=round(sum((w[o]/zz)*probs[o] for o in otros),3)
    influ={mo:round(comb-cf[mo],3) for mo in MODS}                  # cuánto cambia la decisión esa prueba
    skilled=[mo for mi,mo in enumerate(MODS) if skill[mi,j]>0.05]
    disagree=[mo for mo in skilled if (probs[mo]>=0.5)!=(probs[lead]>=0.5)]
    abst = max(conf.values())<0.25                                  # ninguna prueba se moja -> baja certeza
    return {"label":LABELS[j],"hallazgo":ES[LABELS[j]],"probs":probs,"conf":conf,
            "skill":{mo:float(skill[mi,j]) for mi,mo in enumerate(MODS)},"w":wn,
            "aport":aport,"push":push,"contrafactual":cf,"influencia":influ,
            "comb":round(comb,3),"lead":lead,"rel_lead":round(reliability(lead,j,conf[lead]),2),
            "disagree":disagree,"abstencion":bool(abst),"y":(int(y_te[idx,j]) if m_te[idx,j]==1 else None)}

def hadm(idx): return int(df_te["hadm_id"].iloc[idx])
def idx_of(h): return int(df_te.index[df_te["hadm_id"]==h][0])

# -------- figura de barras por hallazgo (la del producto) --------
def fig_finding(rep_list, title, path):
    fig,axes=plt.subplots(len(rep_list),1,figsize=(8.2,1.7*len(rep_list)+0.6))
    if len(rep_list)==1: axes=[axes]
    for ax,f in zip(axes,rep_list):
        for k,mo in enumerate(MODS):
            ax.barh(k,f["probs"][mo],color=COL[mo],alpha=1.0 if mo==f["lead"] else 0.5,
                    edgecolor="black" if mo==f["lead"] else "none",linewidth=1.8)
            ax.text(f["probs"][mo]+0.012,k,f"{f['probs'][mo]:.0%}  ·  peso {f['w'][mo]:.0%}",va="center",fontsize=8)
        ax.axvline(0.5,color="gray",ls="--",lw=1); ax.axvline(f["comb"],color="#c0392b",lw=2.2)
        ax.text(f["comb"],2.75,f"decisión {f['comb']:.0%}",color="#c0392b",fontsize=8,ha="center")
        ax.set_yticks(range(3)); ax.set_yticklabels([MOD_ES[m] for m in MODS],fontsize=8.5); ax.set_xlim(0,1.34); ax.set_ylim(-0.6,3.1)
        extra=" · ⚠ baja certeza" if f["abstencion"] else (" · ⚠ discrepancia" if f["disagree"] else "")
        ax.set_title(f"{f['hallazgo']} — fíate de: {MOD_ES[f['lead']]}{extra}",fontsize=10,loc="left")
    plt.suptitle(title,fontweight="bold",fontsize=11); plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

# -------- figura LEYENDA ANOTADA (limpia: marcadores ①-⑤ + caja lateral) --------
def fig_leyenda(f, path):
    fig,ax=plt.subplots(figsize=(10.2,4.4))
    lead_idx=MODS.index(f["lead"])
    for k,mo in enumerate(MODS):
        ax.barh(k,f["probs"][mo],color=COL[mo],alpha=1.0 if mo==f["lead"] else 0.5,edgecolor="black" if mo==f["lead"] else "none",linewidth=2.0)
        ax.text(f["probs"][mo]+0.012,k,f"{f['probs'][mo]:.0%} · peso {f['w'][mo]:.0%}",va="center",fontsize=9)
    ax.axvline(0.5,color="gray",ls="--",lw=1.3); ax.axvline(f["comb"],color="#c0392b",lw=2.6)
    ax.set_yticks(range(3)); ax.set_yticklabels([MOD_ES[m] for m in MODS],fontsize=10); ax.set_xlim(0,2.15); ax.set_ylim(-0.7,3.0)
    ax.set_xlabel("probabilidad de que el hallazgo esté presente (0 = no, 1 = sí)")
    def tag(x,y,n):
        ax.text(x,y,n,fontsize=10.5,fontweight="bold",color="white",ha="center",va="center",zorder=5,
                bbox=dict(boxstyle="circle,pad=0.15",fc="#c0392b" if n in ("3","4") else "#34495e",ec="white",lw=0.5))
    longest=max(MODS,key=lambda mo:f["probs"][mo])
    tag(f["probs"][longest]/2, MODS.index(longest), "1")        # dentro de la barra más larga
    tag(min(f["probs"]["CXR"]+0.42,1.15), 0, "2")               # junto al texto "peso" de la 1ª barra
    tag(0.5, 2.75, "3")                                          # línea gris 50%
    tag(f["comb"], 2.75, "4")                                    # línea roja decisión
    tag(0.02, lead_idx, "5")                                     # borde negro = líder
    leg=("①  LONGITUD de la barra = probabilidad que da esa prueba\n"
         "②  PESO = cuánto se le hace caso (= habilidad × confianza)\n"
         "③  línea gris = 50% (umbral sí / no)\n"
         f"④  línea roja = DECISIÓN combinada ({f['comb']:.0%})\n"
         "⑤  borde negro = prueba LÍDER (en la que fiarse)")
    ax.text(1.30,1.5,leg,fontsize=9.2,va="center",ha="left",
            bbox=dict(boxstyle="round,pad=0.5",fc="#f7f7f5",ec="#cccccc"))
    ax.set_title(f"Cómo leer la herramienta — ejemplo: {f['hallazgo']} (paciente {f['_hadm']})",fontweight="bold",fontsize=11)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

# ================== SELECCIÓN DE CASOS REPRESENTATIVOS ==================
casos=[]; usados=set()
def first(cond):
    for idx in range(len(df_te)):
        if hadm(idx) in usados: continue
        for j in range(N):
            if LABELS[j]=="No Finding" or m_te[idx,j]!=1: continue
            f=explain(idx,j)
            if cond(f): return idx,j,f
    return None
defs=[("A","Radiografía clara y fiable: guíate por ella",
        lambda f: f["lead"]=="CXR" and f["conf"]["CXR"]>=0.55 and (f["comb"]>=0.5)==(f["y"]==1) and not f["disagree"]),
      ("B","La radiografía duda y las analíticas aciertan",
        lambda f: f["lead"]=="LABS" and f["y"] is not None and (f["comb"]>=0.5)==(f["y"]==1)),
      ("C","Las pruebas se contradicen: manda la más fiable",
        lambda f: len(f["disagree"])>=1 and f["conf"]["CXR"]>=0.4),
      ("D","Un ECG muy seguro pero poco hábil: se le baja el peso",
        lambda f: f["conf"]["ECG"]>=0.6 and f["lead"]!="ECG" and f["w"]["ECG"]<0.25),
      ("E","Baja certeza global: ninguna prueba se moja (precaución)",
        lambda f: f["abstencion"])]
for code,desc,cond in defs:
    r=first(cond)
    if r is None: print("(sin caso para)",code); continue
    idx,j,f=r; usados.add(hadm(idx)); f["_hadm"]=hadm(idx); f["_caso"]=code; f["_desc"]=desc
    fig_finding([f],f"Caso {code} — paciente {hadm(idx)}",FG/f"caso_{code}.png")
    casos.append(f)
# Caso F: paciente con varios hallazgos
for idx in range(len(df_te)):
    js=[j for j in range(N) if LABELS[j]!="No Finding" and m_te[idx,j]==1]
    if len(js)>=4 and hadm(idx) not in usados:
        fl=[explain(idx,j) for j in sorted(js,key=lambda j:-np.mean([Bte[mo][idx,j] for mo in MODS]))[:4]]
        for f in fl: f["_hadm"]=hadm(idx)
        fig_finding(fl,f"Caso F (varios hallazgos) — paciente {hadm(idx)}",FG/"caso_F.png")
        casos.append({"_caso":"F","_desc":"Paciente con varios hallazgos a la vez","_hadm":hadm(idx),"multi":fl}); break

# Leyenda anotada con el caso A
lf=dict(casos[0]); lf["_hadm"]=casos[0]["_hadm"]; fig_leyenda(lf,FG/"leyenda_anotada.png")

json.dump({"casos":casos,"skill":{MODS[mi]:{ES[LABELS[j]]:round(float(skill[mi,j]),3) for j in range(N) if LABELS[j]!='No Finding'} for mi in range(3)}},
          open(OUT/"dosier_casos.json","w",encoding="utf-8"),indent=2,ensure_ascii=False,default=float)
print("Casos generados:",[c["_caso"] for c in casos]); print("Figuras en",FG)
