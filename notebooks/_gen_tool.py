# -*- coding: utf-8 -*-
import nbformat as nbf, io
nb=nbf.v4.new_notebook(); cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s)); co=lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# 🩺 HERRAMIENTA DE APOYO A LA DECISIÓN CLÍNICA (multimodal, explicable)
## TFM · ¿En qué prueba fiarse para cada paciente y hallazgo? · Universidad de Salamanca

---

## 🎯 Qué hace (y por qué, no es "otro stacking")

Las 6 etiquetas son **hallazgos radiográficos** (extraídos del informe del CXR), así que el CXR es casi su propio oráculo
y la fusión sube poco el AUC. **El valor real no es el AUC: es orientar al médico sobre QUÉ PRUEBA mirar** para cada
paciente y hallazgo, con una explicación en lenguaje natural.

Para cada hallazgo y paciente, la herramienta combina tres señales que **ya tenemos**:

1. **Probabilidad calibrada** de cada prueba (CXR / ECG / analíticas) → *qué dice cada una*.
2. **Confianza** de cada prueba = `|p − 0.5|` → *cuán segura está en este paciente*.
3. **Habilidad** de cada prueba para ese hallazgo = `AUC(validación) − 0.5` → *en qué prueba fiarse para ese hallazgo*.

**Peso(prueba) = habilidad × confianza.** La prueba con más peso es "la que mirar"; comparando se genera la frase
("la radiografía duda → fíate de las analíticas"). Todo se estima en **validación**; el test solo ilustra. Es **decisión
de APOYO**: surfacea información y la justifica, no sustituye al clínico.
""")

co(r"""# CELDA 1 · DEPENDENCIAS
import subprocess, sys
for p in ["scikit-learn","pandas","numpy","matplotlib"]:
    subprocess.run([sys.executable,"-m","pip","install",p,"-q"],check=False)
print("listo")""")

co(r"""# CELDA 2 · DATOS + PREDICCIONES CALIBRADAS POR MODALIDAD
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); np.random.seed(42)
NB=Path.cwd(); BASE=Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV=BASE/"data_csv"/"clean"
DIRS={"CXR":(NB/"outputs_cxr_densenet121_v5","cxr"),"ECG":(NB/"outputs_ecg_resnet1d_v3_1","ecg"),"LABS":(NB/"outputs_labs_tabular_v2_2","labs")}
OUT=NB/"outputs_herramienta_decision"; OUT.mkdir(exist_ok=True); (OUT/"figuras").mkdir(exist_ok=True)
LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]; N=len(LABELS)
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema","Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgos","Pleural Effusion":"Derrame pleural"}
MODS=["CXR","ECG","LABS"]; MOD_ES={"CXR":"la radiografía","ECG":"el ECG","LABS":"las analíticas"}
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
        c=pd.read_csv(d/f"{pref}_pred_{split}.csv").set_index("hadm_id")
        cols=[f"{pref}_{l.replace(' ','_')}_cal" for l in LABELS]
        P[mod]=c.reindex(df["hadm_id"].to_numpy())[cols].fillna(0.5).to_numpy(np.float32)
    return P
Bvl=load("val",df_vl); Bte=load("test",df_te)
print("Pacientes val/test:",len(df_vl),len(df_te))""")

co(r"""# CELDA 3 · HABILIDAD (AUC-0.5 en val) y FIABILIDAD por confianza
skill=np.zeros((3,N))
for mi,mod in enumerate(MODS):
    for j in range(N):
        s=m_vl[:,j]==1; yt=y_vl[s,j]
        if yt.sum()>=2 and (1-yt).sum()>=2: skill[mi,j]=max(roc_auc_score(yt,Bvl[mod][s,j])-0.5,0.0)
def reliability(mod,j,conf):    # P(acierto | confianza) estimada en val (ventana ±0.15)
    s=m_vl[:,j]==1; p=Bvl[mod][s,j]; yt=y_vl[s,j]; c=np.abs(p-0.5)*2
    sel=(c>=max(conf-0.15,0))&(c<=min(conf+0.15,1.0))
    if sel.sum()<10: sel=np.ones_like(c,bool)
    return float(((p[sel]>=0.5).astype(int)==yt[sel]).mean())
fig,ax=plt.subplots(figsize=(7,3.2))
im=ax.imshow(skill,cmap="YlGnBu",aspect="auto",vmin=0,vmax=0.45)
ax.set_xticks(range(N)); ax.set_xticklabels([ES[l] for l in LABELS],rotation=25,ha="right",fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels(MODS)
for mi in range(3):
    for j in range(N): ax.text(j,mi,f"{skill[mi,j]:.2f}",ha="center",va="center",fontsize=8)
ax.set_title("Habilidad por prueba y hallazgo (AUC−0.5, validación)"); plt.colorbar(im,fraction=0.025)
plt.tight_layout(); plt.savefig(OUT/"figuras"/"habilidad_prueba_hallazgo.png",dpi=150,bbox_inches="tight"); plt.show()
print("CXR es la prueba más hábil en todos los hallazgos; LABS es la complementaria; ECG la más débil.")""")

co(r"""# CELDA 4 · MOTOR DE EXPLICACIÓN POR PACIENTE
def explain_patient(idx, B, y, m, finding=None, topk=3):
    out={"hadm":int(df_te["hadm_id"].iloc[idx]),"findings":[]}
    cand=[j for j in range(N) if LABELS[j]!="No Finding" and m[idx,j]==1]
    base_p={j:np.mean([B[mod][idx,j] for mod in MODS]) for j in cand}
    order=sorted(cand,key=lambda j:-base_p[j])[:topk] if finding is None else [LABELS.index(finding)]
    for j in order:
        probs={mod:float(B[mod][idx,j]) for mod in MODS}; conf={mod:abs(probs[mod]-0.5)*2 for mod in MODS}
        w={mod:skill[mi,j]*max(conf[mod],0.05) for mi,mod in enumerate(MODS)}; Z=sum(w.values())+1e-9
        wn={mod:w[mod]/Z for mod in MODS}; comb=sum(wn[mod]*probs[mod] for mod in MODS)
        lead=max(MODS,key=lambda mod:w[mod]); rel=reliability(lead,j,conf[lead])
        disagree=[mod for mi,mod in enumerate(MODS) if skill[mi,j]>0.05 and (probs[mod]>=0.5)!=(probs[lead]>=0.5)]
        out["findings"].append({"label":LABELS[j],"probs":probs,"conf":conf,"w":wn,"comb":comb,"lead":lead,"rel_lead":rel,"disagree":disagree,"y":int(y[idx,j])})
    return out
def narrate(f):
    l=ES[f["label"]]; lead=f["lead"]; cxr=f["probs"]["CXR"]; comb=f["comb"]; lead_es=MOD_ES[lead]
    s=[f"• {l}: decisión combinada {comb:.0%}."]
    if lead=="CXR" and f["conf"]["CXR"]>=0.4:
        s.append(f"  La radiografía es clara y fiable aquí ({cxr:.0%}); guíate sobre todo por ella.")
    elif lead!="CXR":
        s.append(f"  ⚠ La radiografía está dudosa para {l} ({cxr:.0%}, baja confianza); {lead_es} aportan más en este "
                 f"hallazgo ({f['probs'][lead]:.0%}) y aciertan ~{f['rel_lead']:.0%} cuando están así de seguras → "
                 f"ten en cuenta sobre todo {lead_es}.")
    else:
        s.append(f"  La radiografía es la referencia ({cxr:.0%}) con confianza media; contrasta con las demás si la clínica no encaja.")
    if f["disagree"]:
        s.append(f"  Discrepancia: {' y '.join(MOD_ES[d] for d in f['disagree'])} no coinciden con {lead_es}; "
                 f"la recomendación se inclina por la prueba más fiable.")
    return "\n".join(s)
def figure_patient(rep, path=None):
    F=rep["findings"]; fig,axes=plt.subplots(len(F),1,figsize=(8,1.7*len(F)+0.5))
    if len(F)==1: axes=[axes]
    cols={"CXR":"#2980b9","ECG":"#8e44ad","LABS":"#16a085"}
    for ax,f in zip(axes,F):
        for k,mod in enumerate(MODS):
            ax.barh(k,f["probs"][mod],color=cols[mod],alpha=1.0 if mod==f["lead"] else 0.55,
                    edgecolor="black" if mod==f["lead"] else "none",linewidth=1.6)
            ax.text(f["probs"][mod]+0.01,k,f"{f['probs'][mod]:.0%} · peso {f['w'][mod]:.0%}",va="center",fontsize=8)
        ax.axvline(0.5,color="gray",ls="--",lw=1); ax.axvline(f["comb"],color="red",lw=2)
        ax.text(f["comb"],2.7,f"decisión {f['comb']:.0%}",color="red",fontsize=8,ha="center")
        ax.set_yticks(range(3)); ax.set_yticklabels([MOD_ES[m] for m in MODS],fontsize=8); ax.set_xlim(0,1.3)
        ax.set_title(f"{ES[f['label']]}  —  fíate de: {MOD_ES[f['lead']]}",fontsize=10,loc="left")
    plt.suptitle(f"Paciente hadm={rep['hadm']} — apoyo a la decisión por prueba",fontweight="bold")
    plt.tight_layout()
    if path: plt.savefig(path,dpi=150,bbox_inches="tight")
    plt.show()
def explicar(hadm_id):
    # Punto de entrada para el médico: pasa un hadm_id del test y obtén informe + figura
    m=df_te.index[df_te["hadm_id"]==hadm_id]
    if len(m)==0: print("hadm_id no encontrado en test"); return
    rep=explain_patient(int(m[0]),Bte,y_te,m_te)
    print(f"===== Paciente hadm={hadm_id} =====")
    for f in rep["findings"]: print(narrate(f))
    figure_patient(rep)
    return rep
print("Motor listo. Usa explicar(<hadm_id>) para cualquier paciente del test.")""")

co(r"""# CELDA 5 · DEMOSTRACIÓN: pacientes donde NO manda el CXR (el caso interesante)
ejemplos=[]
for idx in range(len(df_te)):
    rep=explain_patient(idx,Bte,y_te,m_te)
    if any(f["lead"]!="CXR" for f in rep["findings"]) and len(rep["findings"])>=2: ejemplos.append((idx,rep))
    if len(ejemplos)>=3: break
for k,(idx,rep) in enumerate(ejemplos):
    print(f"\n--- Ejemplo {k+1} (hadm={rep['hadm']}) ---")
    for f in rep["findings"]: print(narrate(f)); print(f"     [verdad: {'SÍ' if f['y']==1 else 'no'} {ES[f['label']]}]")
    figure_patient(rep,OUT/"figuras"/f"paciente_{k+1}_hadm{rep['hadm']}.png")
print("\nFiguras guardadas en",OUT/"figuras")""")

md(r"""---
## ✅ Cómo se usa y qué aporta

- `explicar(hadm_id)` devuelve, para un paciente, **qué dice cada prueba**, **cuánto fiarse de cada una** (peso = habilidad×confianza),
  la **decisión combinada** y una **explicación en lenguaje natural** del tipo *"la radiografía duda → fíate de las analíticas"*.
- La recomendación es **trazable**: cada peso se descompone en habilidad (validada por hallazgo) × confianza (de este paciente).

## ⚠️ Honestidad clínica
- Es **apoyo**, no diagnóstico: surfacea y justifica información; el médico decide.
- Como las etiquetas vienen del CXR, la herramienta **suele recomendar la radiografía**; su valor está en los casos
  donde el CXR es **dudoso** y otra prueba (sobre todo analíticas en Edema/Derrame/Cardiomegalia) es **más fiable**.
- La fiabilidad por confianza se estima en validación (n limitado): tómese como orientación, no como garantía.

## 🔭 Siguiente nivel de explicabilidad (si se dispone de GPU)
Atribución por **SHAP** sobre el meta-modelo y **contrafactuales** ("si quito el CXR, la decisión pasa de X a Y"),
y atención cruzada sobre *embeddings* para señalar **qué región/derivación/analítica** concreta dispara la recomendación.
""")

nb["cells"]=cells; nb.metadata["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
nb.metadata["language_info"]={"name":"python","version":"3.10"}
with io.open("05_HERRAMIENTA_DECISION_v1.ipynb","w",encoding="utf-8") as f: nbf.write(nb,f)
print("Generado 05_HERRAMIENTA_DECISION_v1.ipynb con",len(cells),"celdas")
