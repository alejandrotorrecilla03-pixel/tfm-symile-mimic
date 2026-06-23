# -*- coding: utf-8 -*-
# Exploración para el informe de la herramienta: calibración, confianza-vs-acierto, habilidad y MUCHOS ejemplos.
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); np.random.seed(42)
NB=Path.cwd(); BASE=Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV=BASE/"data_csv"/"clean"
DIRS={"CXR":(NB/"outputs_cxr_densenet121_v2","cxr"),"ECG":(NB/"outputs_ecg_resnet1d_v2","ecg"),"LABS":(NB/"outputs_labs_tabular_v2","labs")}
LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]; N=len(LABELS)
PATHO=[l for l in LABELS if l!="No Finding"]
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema","Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgos","Pleural Effusion":"Derrame pleural"}
MODS=["CXR","ECG","LABS"]; MOD_ES={"CXR":"la radiografía","ECG":"el ECG","LABS":"las analíticas"}; COL={"CXR":"#2980b9","ECG":"#8e44ad","LABS":"#16a085"}
OUT=NB/"outputs_herramienta_decision"; FG=OUT/"figuras"; FG.mkdir(parents=True,exist_ok=True)
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

# ============ FIGURAS GLOBALES ============
# (1) habilidad heatmap
fig,ax=plt.subplots(figsize=(7.5,3.2)); im=ax.imshow(skill,cmap="YlGnBu",aspect="auto",vmin=0,vmax=0.45)
ax.set_xticks(range(N)); ax.set_xticklabels([ES[l] for l in LABELS],rotation=20,ha="right",fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels([MOD_ES[m] for m in MODS])
for mi in range(3):
    for j in range(N): ax.text(j,mi,f"{skill[mi,j]:.2f}",ha="center",va="center",fontsize=8)
ax.set_title("Habilidad por prueba y hallazgo  (AUC−0.5 en validación)"); plt.colorbar(im,fraction=0.025); plt.tight_layout(); plt.savefig(FG/"exp_habilidad.png",dpi=150,bbox_inches="tight"); plt.close()

# (2) calibración por modalidad (pooled patologías, test)
fig,ax=plt.subplots(figsize=(6.2,5)); bins=np.linspace(0,1,11)
calib_tbl={}
for mod in MODS:
    ps=[]; os=[]
    P=[]; Y=[]
    for j,l in enumerate(LABELS):
        if l=="No Finding": continue
        s=m_te[:,j]==1; P.append(Bte[mod][s,j]); Y.append(y_te[s,j])
    P=np.concatenate(P); Y=np.concatenate(Y); idx=np.digitize(P,bins)-1
    xs=[]; ys=[]
    for b in range(10):
        mb=idx==b
        if mb.sum()>=20: xs.append(P[mb].mean()); ys.append(Y[mb].mean())
    ax.plot(xs,ys,"o-",color=COL[mod],label=MOD_ES[mod],alpha=0.85); calib_tbl[mod]=list(zip([round(x,2) for x in xs],[round(v,2) for v in ys]))
ax.plot([0,1],[0,1],"k--",alpha=0.5,label="calibración perfecta"); ax.set_xlabel("probabilidad que dice la prueba"); ax.set_ylabel("frecuencia real de positivos")
ax.set_title("Calibración (test, patologías agregadas)"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(FG/"exp_calibracion.png",dpi=150,bbox_inches="tight"); plt.close()

# (3) histograma de confianza por modalidad (test, pooled)
fig,axes=plt.subplots(1,3,figsize=(13,3.4),sharey=True)
for ax,mod in zip(axes,MODS):
    C=[]
    for j,l in enumerate(LABELS):
        if l=="No Finding": continue
        s=m_te[:,j]==1; C.append(np.abs(Bte[mod][s,j]-0.5)*2)
    C=np.concatenate(C); ax.hist(C,bins=20,color=COL[mod],alpha=0.85); ax.set_title(f"Confianza de {MOD_ES[mod]}"); ax.set_xlabel("|p−0.5|·2  (0=duda, 1=segura)")
axes[0].set_ylabel("nº de casos"); plt.tight_layout(); plt.savefig(FG/"exp_confianza_hist.png",dpi=150,bbox_inches="tight"); plt.close()

# (4) acierto vs confianza por modalidad (val, pooled) -> demuestra que la confianza es útil
fig,ax=plt.subplots(figsize=(6.5,4.2)); cbins=[0,0.2,0.4,0.6,0.8,1.01]; accconf={}
for mod in MODS:
    P=[]; Y=[]
    for j,l in enumerate(LABELS):
        if l=="No Finding": continue
        s=m_vl[:,j]==1; P.append(Bvl[mod][s,j]); Y.append(y_vl[s,j])
    P=np.concatenate(P); Y=np.concatenate(Y); C=np.abs(P-0.5)*2; pred=(P>=0.5).astype(int); accs=[]; xs=[]
    for b in range(len(cbins)-1):
        mb=(C>=cbins[b])&(C<cbins[b+1])
        if mb.sum()>=20: accs.append((pred[mb]==Y[mb]).mean()); xs.append((cbins[b]+cbins[b+1])/2)
    ax.plot(xs,accs,"o-",color=COL[mod],label=MOD_ES[mod]); accconf[mod]=[(round(x,1),round(a,2)) for x,a in zip(xs,accs)]
ax.set_xlabel("confianza  |p−0.5|·2"); ax.set_ylabel("acierto (acc. a umbral 0.5)"); ax.set_title("A más confianza, más acierto (validación)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(FG/"exp_acierto_vs_confianza.png",dpi=150,bbox_inches="tight"); plt.close()

# ============ MOTOR DE EXPLICACIÓN ============
def explain(idx, topk=4):
    cand=[j for j in range(N) if LABELS[j]!="No Finding" and m_te[idx,j]==1]
    base={j:np.mean([Bte[mod][idx,j] for mod in MODS]) for j in cand}
    order=sorted(cand,key=lambda j:-base[j])[:topk]; F=[]
    for j in order:
        probs={mod:float(Bte[mod][idx,j]) for mod in MODS}; conf={mod:abs(probs[mod]-0.5)*2 for mod in MODS}
        w={mod:skill[mi,j]*max(conf[mod],0.05) for mi,mod in enumerate(MODS)}; Z=sum(w.values())+1e-9
        wn={mod:w[mod]/Z for mod in MODS}; comb=sum(wn[mod]*probs[mod] for mod in MODS); lead=max(MODS,key=lambda mod:w[mod])
        disagree=[mod for mi,mod in enumerate(MODS) if skill[mi,j]>0.05 and (probs[mod]>=0.5)!=(probs[lead]>=0.5)]
        F.append({"label":LABELS[j],"probs":probs,"conf":conf,"skill":{mod:float(skill[mi,j]) for mi,mod in enumerate(MODS)},
                  "w":wn,"comb":comb,"lead":lead,"rel_lead":reliability(lead,j,conf[lead]),"disagree":disagree,"y":int(y_te[idx,j])})
    return {"hadm":int(df_te["hadm_id"].iloc[idx]),"idx":idx,"findings":F}
def narrate(f):
    l=ES[f["label"]]; lead=f["lead"]; cxr=f["probs"]["CXR"]; lead_es=MOD_ES[lead]
    s=[f"{l}: decisión combinada {f['comb']:.0%} (verdad: {'SÍ' if f['y']==1 else 'no'})."]
    if lead=="CXR" and f["conf"]["CXR"]>=0.4: s.append(f"La radiografía es clara y fiable aquí ({cxr:.0%}); guíate sobre todo por ella.")
    elif lead!="CXR": s.append(f"La radiografía está dudosa ({cxr:.0%}, baja confianza); {lead_es} aportan más en este hallazgo ({f['probs'][lead]:.0%}) y aciertan ~{f['rel_lead']:.0%} así de seguras → ten en cuenta sobre todo {lead_es}.")
    else: s.append(f"La radiografía es la referencia ({cxr:.0%}) con confianza media; contrasta si la clínica no encaja.")
    if f["disagree"]: s.append(f"Discrepancia: {' y '.join(MOD_ES[d] for d in f['disagree'])} no coinciden con {lead_es}.")
    return " ".join(s)
def figure_patient(rep, path):
    F=rep["findings"]; fig,axes=plt.subplots(len(F),1,figsize=(8,1.55*len(F)+0.6))
    if len(F)==1: axes=[axes]
    for ax,f in zip(axes,F):
        for k,mod in enumerate(MODS):
            ax.barh(k,f["probs"][mod],color=COL[mod],alpha=1.0 if mod==f["lead"] else 0.5,edgecolor="black" if mod==f["lead"] else "none",linewidth=1.6)
            ax.text(f["probs"][mod]+0.01,k,f"{f['probs'][mod]:.0%} · peso {f['w'][mod]:.0%}",va="center",fontsize=7.5)
        ax.axvline(0.5,color="gray",ls="--",lw=1); ax.axvline(f["comb"],color="red",lw=2)
        ax.set_yticks(range(3)); ax.set_yticklabels([MOD_ES[m] for m in MODS],fontsize=8); ax.set_xlim(0,1.32)
        ax.set_title(f"{ES[f['label']]} — fíate de: {MOD_ES[f['lead']]} · decisión {f['comb']:.0%}",fontsize=9.5,loc="left")
    plt.suptitle(f"Paciente hadm={rep['hadm']}",fontweight="bold",fontsize=11); plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

# ============ SELECCIÓN DE MUCHOS EJEMPLOS DIVERSOS ============
buckets={"cxr_seguro_acierta":None,"defiere_labs_acierta":None,"defiere_labs_falla":None,"discrepancia":None,"ecg_ignorado":None,"multi_hallazgo":None}
for idx in range(len(df_te)):
    rep=explain(idx)
    if not rep["findings"]: continue
    for f in rep["findings"]:
        if buckets["cxr_seguro_acierta"] is None and f["lead"]=="CXR" and f["conf"]["CXR"]>=0.55 and (f["comb"]>=0.5)==(f["y"]==1):
            buckets["cxr_seguro_acierta"]=(rep,f["label"])
        if buckets["defiere_labs_acierta"] is None and f["lead"]=="LABS" and (f["comb"]>=0.5)==(f["y"]==1):
            buckets["defiere_labs_acierta"]=(rep,f["label"])
        if buckets["defiere_labs_falla"] is None and f["lead"]=="LABS" and (f["comb"]>=0.5)!=(f["y"]==1):
            buckets["defiere_labs_falla"]=(rep,f["label"])
        if buckets["discrepancia"] is None and f["disagree"]:
            buckets["discrepancia"]=(rep,f["label"])
        if buckets["ecg_ignorado"] is None and f["conf"]["ECG"]>=0.6 and f["lead"]!="ECG" and f["w"]["ECG"]<0.25:
            buckets["ecg_ignorado"]=(rep,f["label"])
    if buckets["multi_hallazgo"] is None and len(rep["findings"])>=4:
        buckets["multi_hallazgo"]=(rep,None)
ejemplos=[]; seen=set()
for name,val in buckets.items():
    if val is None: continue
    rep,focus=val
    if rep["hadm"] in seen:  # busca otro distinto para no repetir paciente
        continue
    seen.add(rep["hadm"]); path=FG/f"ej_{name}_hadm{rep['hadm']}.png"; figure_patient(rep,path)
    ejemplos.append({"bucket":name,"focus":focus,"fig":path.name,"hadm":rep["hadm"],
                     "findings":[{**{k:f[k] for k in ["label","probs","conf","skill","w","comb","lead","rel_lead","disagree","y"]},"narr":narrate(f)} for f in rep["findings"]]})
print(f"Ejemplos generados: {len(ejemplos)} -> {[e['bucket'] for e in ejemplos]}")

# ============ ESTADÍSTICAS GLOBALES (lead por hallazgo) ============
lead_count={l:{m:0 for m in MODS} for l in PATHO}; tot={l:0 for l in PATHO}
for idx in range(len(df_te)):
    for j,l in enumerate(LABELS):
        if l=="No Finding" or m_te[idx,j]!=1: continue
        probs={mod:Bte[mod][idx,j] for mod in MODS}; conf={mod:abs(probs[mod]-0.5)*2 for mod in MODS}
        w={mod:skill[mi,j]*max(conf[mod],0.05) for mi,mod in enumerate(MODS)}; lead=max(MODS,key=lambda mod:w[mod])
        lead_count[l][lead]+=1; tot[l]+=1
fig,ax=plt.subplots(figsize=(8.5,4)); x=np.arange(len(PATHO)); bottom=np.zeros(len(PATHO))
for mod in MODS:
    vals=[100*lead_count[l][mod]/max(tot[l],1) for l in PATHO]; ax.bar(x,vals,bottom=bottom,label=MOD_ES[mod],color=COL[mod],alpha=0.85); bottom+=vals
ax.set_xticks(x); ax.set_xticklabels([ES[l] for l in PATHO],rotation=20,ha="right",fontsize=8); ax.set_ylabel("% de pacientes donde lidera"); ax.set_title("¿Qué prueba 'manda' en cada hallazgo? (test)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FG/"exp_lead_distribucion.png",dpi=150,bbox_inches="tight"); plt.close()
lead_pct={l:{m:round(100*lead_count[l][m]/max(tot[l],1),1) for m in MODS} for l in PATHO}

json.dump({"skill":{MODS[mi]:{LABELS[j]:round(float(skill[mi,j]),3) for j in range(N)} for mi in range(3)},
           "calib":calib_tbl,"accconf":accconf,"lead_pct":lead_pct,"ejemplos":ejemplos},
          open(OUT/"exploracion.json","w",encoding="utf-8"),indent=2,ensure_ascii=False,default=float)
print("Guardado exploracion.json y figuras en",FG)
print("Lead % por hallazgo:"); [print(f"   {ES[l]:18s} "+" ".join(f"{m}={lead_pct[l][m]:.0f}%" for m in MODS)) for l in PATHO]
