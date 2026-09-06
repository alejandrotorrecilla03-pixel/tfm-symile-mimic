"""Genera figuras nuevas para el Informe de Modelos:
  - CXR v2 (mejor modelo): curvas ROC y Precisión-Exhaustividad por patología + barras AUC/AP.
  - LABS: representación del EJE DOMINANTE (Edad y RDW×Edad) sobre la importancia.
Salida: salidas/_informe_modelos/figuras/. Estilo azules USAL.
"""
import numpy as np, pandas as pd, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score

FIG = "salidas/_informe_modelos/figuras"; os.makedirs(FIG, exist_ok=True)
BLUE="#1F3864"; BLUE2="#2E5496"; ORANGE="#C55A11"; GREY="#595959"
plt.rcParams.update({"font.family":"Cambria","font.size":10,"axes.edgecolor":"#888",
                     "axes.titlecolor":BLUE,"figure.dpi":150})
ES = {"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema",
      "Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgo","Pleural Effusion":"Derrame pleural"}
LAB=list(ES.keys()); LAB_U=[l.replace(" ","_") for l in LAB]
def nf_last(df):
    # "No Finding" / "Sin hallazgo" siempre como ultima fila (convencion del TFM)
    key="label" if "label" in df.columns else df.columns[0]
    mask=df[key].isin(["No Finding","Sin hallazgo"])
    return pd.concat([df[~mask],df[mask]]).reset_index(drop=True)
PAL=["#1F3864","#2E5496","#5B8FCB","#C55A11","#7F7F7F","#2E8B57"]

# ── datos: predicciones calibradas CXR v2 + etiquetas FINAL ──
pred=pd.read_csv("salidas/01_cxr/v2/cxr_pred_test.csv")
cl=pd.read_csv("data/clean/test_clean.csv",sep=";")
df=pred.merge(cl[["hadm_id"]+LAB],on="hadm_id",how="inner")
raw=df[LAB].to_numpy(float); Y=(raw==1).astype(int); M=(raw!=-1)
P=df[[f"cxr_{u}_cal" for u in LAB_U]].to_numpy(float)

# ── FIG 1 · ROC + PR por patología ──
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4.4))
for j,lab in enumerate(LAB):
    s=M[:,j]
    if Y[s,j].sum()==0: continue
    fpr,tpr,_=roc_curve(Y[s,j],P[s,j]); auc=roc_auc_score(Y[s,j],P[s,j])
    a1.plot(fpr,tpr,color=PAL[j],lw=1.8,label=f"{ES[lab]} ({auc:.2f})")
    pr,rc,_=precision_recall_curve(Y[s,j],P[s,j]); ap=average_precision_score(Y[s,j],P[s,j])
    a2.plot(rc,pr,color=PAL[j],lw=1.8,label=f"{ES[lab]} ({ap:.2f})")
a1.plot([0,1],[0,1],"--",color=GREY,lw=1); a1.set_title("Curva ROC (AUC-ROC)")
a1.set_xlabel("1 − Especificidad"); a1.set_ylabel("Sensibilidad"); a1.legend(fontsize=7.5,loc="lower right")
a2.set_title("Curva Precisión–Exhaustividad (AUC-PR)"); a2.set_xlabel("Exhaustividad (recall)")
a2.set_ylabel("Precisión"); a2.legend(fontsize=7.5,loc="upper right")
for a in (a1,a2): a.set_xlim(0,1); a.set_ylim(0,1.02); a.grid(alpha=.25)
fig.suptitle("CXR v2 — rendimiento por patología (test)",color=BLUE,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{FIG}/01_cxr_v2_roc_pr.png",bbox_inches="tight"); plt.close(fig)

# ── FIG 2 · barras AUC-PR y AUC-ROC por patología (CXR v2) ──
m=nf_last(pd.read_csv("salidas/01_cxr/v2/metrics_per_label_v2.csv"))
labs=[ES.get(l,l) for l in m["label"]]; y=np.arange(len(labs))
fig,ax=plt.subplots(figsize=(8.2,4.2))
ax.barh(y-0.2,m["AP"],0.4,color=BLUE,label="AUC-PR")
ax.barh(y+0.2,m["AUC"],0.4,color=BLUE2,label="AUC-ROC")
for i,(ap,pv) in enumerate(zip(m["AP"],m["prevalencia"])):
    ax.plot([pv,pv],[i-0.4,i],color=ORANGE,lw=2.6,
            label=("Prevalencia (línea base solo de la AUC-PR)" if i==0 else None))
ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.set_xlim(0,1)
ax.set_xlabel("Valor"); ax.legend(loc="lower right",fontsize=8)
ax.set_title("CXR v2 — AUC-PR y AUC-ROC por patología (naranja = prevalencia, línea base solo de la AUC-PR)")
ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(f"{FIG}/02_cxr_v2_auc_etiqueta.png",bbox_inches="tight"); plt.close(fig)

# ── FIG 3 · EJE DOMINANTE: Edad y RDW×Edad ──
imp=pd.read_csv("salidas/03_labs/v1/importancia_mi_anova.csv").head(12).iloc[::-1]
def color(v):
    if v in ("Edad","RDW×Edad"): return ORANGE
    if str(v).startswith("falta:"): return "#C0392B"
    return BLUE2
cols=[color(v) for v in imp["variable"]]
fig,ax=plt.subplots(figsize=(8.4,5))
ax.barh(imp["variable"],imp["media"],color=cols)
ax.set_xlabel("Importancia media (Información mutua + ANOVA)")
ax.set_title("Eje dominante del módulo tabular: Edad + RDW×Edad\n(naranja = eje dominante · rojo = indicadores de ausencia «falta:X»)",
             color=BLUE,fontsize=10.5)
ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(f"{FIG}/21_labs_eje_dominante.png",bbox_inches="tight"); plt.close(fig)

# ── FIG · COMPARATIVA DE FUSORES (todos, la mejor destacada) ──
import json
cf=pd.read_csv("salidas/04_stacking/v2/comparativa_fusion_v4.csv")
NOM={"CXR":"CXR en solitario","blend_opt(v3)":"Mezcla ponderada","blend_logit":"Mezcla en logit",
 "blend_raw":"Mezcla sin calibrar","stack_LR":"Meta-regresión logística","stack_enriquecido":"Meta-modelo enriquecido",
 "MoE":"Mezcla de expertos (MoE)","ensemble_fusores":"Ensemble de fusores","sel_por_etiqueta":"Selección por patología",
 "caruana":"Selección voraz (Caruana)"}
vals={NOM[e]:float(cf[cf["enfoque"]==e]["macroP_test"].iloc[0]) for e in NOM}
vals["Pesos fijos interpretables"]=json.load(open("salidas/04_stacking/v3/veredicto_v3.json"))["macro_ap_v3"]
d45=json.load(open("salidas/04_stacking/veredicto_v4v5.json"))
vals["Ponderación por confianza"]=d45["A_confianza"]["macro_ap"]; vals["Fusión temprana"]=d45["B_temprana"]["macro_ap"]
mono=vals["CXR en solitario"]
items=sorted(((k,v) for k,v in vals.items() if k!="CXR en solitario"),key=lambda x:x[1])
names=[k for k,_ in items]; vv=[v for _,v in items]
best="Selección por patología"; interp="Pesos fijos interpretables"
cols=[ORANGE if n==best else ("#5B8FCB" if n==interp else BLUE) for n in names]
fig,ax=plt.subplots(figsize=(8.8,5.4))
bars=ax.barh(names,vv,color=cols)
ax.axvline(mono,color="#C0392B",ls="--",lw=1.6,label=f"CXR en solitario ({mono:.3f})")
for i,v in enumerate(vv): ax.text(v+0.002,i,f"{v:.3f}".replace(".",","),va="center",fontsize=8)
i_final=names.index(interp)
ax.annotate("MODELO FINAL\n(elegido por interpretabilidad,\nno por ser el mejor en test)",
            xy=(vv[i_final],i_final), xytext=(0.606,i_final+1.7),
            fontsize=8, color="#1F3864", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->",color="#1F3864",lw=1.3))
ax.set_xlim(0.53,0.63); ax.set_xlabel("AUC-PR macro (test)")
ax.set_title("Comparativa de fusores (naranja = el mejor EN VALIDACIÓN, no necesariamente en test · azul claro = modelo final)",color=BLUE,fontsize=9.5)
ax.legend(loc="lower right",fontsize=8.5); ax.grid(axis="x",alpha=.25)
fig.tight_layout(); fig.savefig(f"{FIG}/22_fusores_comparativa.png",bbox_inches="tight"); plt.close(fig)

# ── FIG · CXR vs MODELO FINAL (fusión v3) por patología ──  [coherente con la Tabla 25; Cardiomegalia NO mejora]
per25=pd.read_csv("salidas/_informe_modelos/tablas/25_pareado_por_patologia.csv")
tofl=lambda s:float(str(s).replace(",",".").replace("+",""))
labs6=list(per25["Hallazgo"]); yv=np.arange(len(labs6))
apc=[tofl(x) for x in per25["AP CXR"]]; apf=[tofl(x) for x in per25["AP fusión v3"]]
fig,ax=plt.subplots(figsize=(8.6,4.6))
ax.barh(yv+0.2,apc,0.4,color=BLUE2,label="CXR en solitario")
ax.barh(yv-0.2,apf,0.4,color=ORANGE,label="Fusión v3 (modelo final)")
ax.set_yticks(yv); ax.set_yticklabels(labs6); ax.invert_yaxis(); ax.set_xlim(0,0.8)
ax.set_xlabel("AUC-PR (test)"); ax.legend(loc="lower right",fontsize=8.5)
ax.set_title("CXR en solitario vs. el modelo final (fusión v3), por patología",color=BLUE,fontsize=10.5)
ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(f"{FIG}/23_cxr_vs_fusion.png",bbox_inches="tight"); plt.close(fig)

# ── FIG · PESOS del modelo final (fusión v3) por patología ──  [coincide celda a celda con la Tabla 9/19]
import json as _json
W=_json.load(open("salidas/04_stacking/v3/veredicto_v3.json"))["pesos_opt"]
wc=[W[l][0] for l in LAB]; we=[W[l][1] for l in LAB]; wl=[W[l][2] for l in LAB]
labs6b=[ES[l] for l in LAB]; yv=np.arange(6)
fig,ax=plt.subplots(figsize=(8.8,4.6))
ax.barh(yv,wc,color=BLUE,label="Imagen (CXR)")
ax.barh(yv,we,left=wc,color=ORANGE,label="ECG")
ax.barh(yv,wl,left=[a+b for a,b in zip(wc,we)],color="#2E8B57",label="Analíticas")
for i in range(6):
    if wc[i]>0.06: ax.text(wc[i]/2,i,f"{int(round(wc[i]*100))}%",ha="center",va="center",color="white",fontsize=8.5)
    if we[i]>0.06: ax.text(wc[i]+we[i]/2,i,f"{int(round(we[i]*100))}%",ha="center",va="center",color="white",fontsize=8.5)
    if wl[i]>0.06: ax.text(wc[i]+we[i]+wl[i]/2,i,f"{int(round(wl[i]*100))}%",ha="center",va="center",color="white",fontsize=8.5)
ax.set_yticks(yv); ax.set_yticklabels(labs6b); ax.invert_yaxis(); ax.set_xlim(0,1)
ax.set_xlabel("Peso (suma = 1 por patología)"); ax.legend(loc="lower right",fontsize=8,ncol=3)
ax.set_title("Modelo final (fusión v3): peso de cada modalidad por patología",color=BLUE,fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{FIG}/26_pesos_v3.png",bbox_inches="tight"); plt.close(fig)

# ── FIG · MATRICES DE CONFUSIÓN CXR v2 (punto F1) ──
po=pd.read_csv("salidas/01_cxr/v2/puntos_operacion_test.csv")
f1=po[po["punto"]=="f1"].set_index("etiqueta")
fig,axes=plt.subplots(2,3,figsize=(9.5,6))
for j,lab in enumerate(LAB):
    ax=axes[j//3,j%3]; s=M[:,j]
    thr=float(f1.loc[lab,"umbral"]); pred=(P[s,j]>=thr).astype(int); yt=Y[s,j]
    TN=int(((pred==0)&(yt==0)).sum()); FP=int(((pred==1)&(yt==0)).sum())
    FN=int(((pred==0)&(yt==1)).sum()); TP=int(((pred==1)&(yt==1)).sum())
    cm=np.array([[TN,FP],[FN,TP]])
    ax.imshow(cm,cmap="Blues"); ax.set_title(f"{ES[lab]} (umbral {thr:.2f})".replace(".",","),fontsize=9,color=BLUE)
    for a in range(2):
        for b in range(2):
            ax.text(b,a,cm[a,b],ha="center",va="center",fontsize=11,
                    color="white" if cm[a,b]>cm.max()*0.6 else "black")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred. sano","Pred. enfermo"],fontsize=7)
    ax.set_yticks([0,1]); ax.set_yticklabels(["Real sano","Real enfermo"],fontsize=7)
fig.suptitle("CXR v2 — matrices de confusión por patología (punto F1)",color=BLUE,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{FIG}/24_cxr_v2_confusion.png",bbox_inches="tight"); plt.close(fig)

# ── FIG · TEST EMPAREJADO (distribución de la diferencia + delta por patología) ──
diffs=np.load("salidas/04_stacking/diffs_pareado.npy")
lo,hi=np.percentile(diffs,[2.5,97.5]); me=diffs.mean()
per=pd.read_csv("salidas/_informe_modelos/tablas/25_pareado_por_patologia.csv")
dvals=[float(str(x).replace(",",".").replace("+","")) for x in per["Δ"]]
labs=list(per["Hallazgo"])
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.5,4.4))
a1.hist(diffs,bins=45,color="#9DB3CE",edgecolor="white")
a1.axvline(0,color="#C0392B",lw=2,label="Sin diferencia (0)")
a1.axvspan(lo,hi,color=ORANGE,alpha=.18,label=f"IC 95 %: [{lo:.3f} – {hi:.3f}]".replace(".",","))
a1.axvline(me,color=BLUE,lw=2,ls="--",label=f"Media +{me:.3f}".replace(".",","))
a1.set_title("Diferencia emparejada: AUC-PR (fusión − CXR)",color=BLUE,fontsize=10.5)
a1.set_xlabel("Diferencia de AUC-PR sobre los mismos pacientes"); a1.set_ylabel("Frecuencia (bootstrap)")
a1.legend(fontsize=8,loc="upper right"); a1.grid(alpha=.2)
yv=np.arange(len(labs)); cols=[ORANGE if d>0 else "#C0392B" for d in dvals]
a2.barh(yv,dvals,color=cols); a2.axvline(0,color="#555",lw=1)
a2.set_yticks(yv); a2.set_yticklabels(labs,fontsize=8); a2.invert_yaxis()
for i,d in enumerate(dvals): a2.text(d+(0.002 if d>=0 else -0.002),i,("+" if d>=0 else "")+f"{d:.3f}".replace(".",","),
                                     va="center",ha="left" if d>=0 else "right",fontsize=8)
a2.set_title("Mejora por patología (fusión − CXR)",color=BLUE,fontsize=10.5); a2.set_xlabel("Δ AUC-PR"); a2.grid(axis="x",alpha=.2)
fig.suptitle("La fusión mejora a la imagen de forma significativa (el IC de la diferencia excluye el 0)",color=BLUE,fontweight="bold",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{FIG}/25_test_emparejado.png",bbox_inches="tight"); plt.close(fig)

DIS=[0,1,2,3,5]
# ══════ ECG v2: figuras LIMPIAS (título correcto, mismo estilo que CXR) ══════
predE=pd.read_csv("salidas/02_ecg/v2/ecg_pred_test.csv").merge(cl[["hadm_id"]+LAB],on="hadm_id",how="inner")
rawE=predE[LAB].to_numpy(float); YE=(rawE==1).astype(int); ME=(rawE!=-1)
PE=predE[[f"ecg_{u}_cal" for u in LAB_U]].to_numpy(float)
# ROC + PR
figE,(e1,e2)=plt.subplots(1,2,figsize=(10,4.4))
for j,lab in enumerate(LAB):
    s=ME[:,j]
    if YE[s,j].sum()==0: continue
    fpr,tpr,_=roc_curve(YE[s,j],PE[s,j]); e1.plot(fpr,tpr,color=PAL[j],lw=1.8,label=f"{ES[lab]} ({roc_auc_score(YE[s,j],PE[s,j]):.2f})")
    pr,rc,_=precision_recall_curve(YE[s,j],PE[s,j]); e2.plot(rc,pr,color=PAL[j],lw=1.8,label=f"{ES[lab]} ({average_precision_score(YE[s,j],PE[s,j]):.2f})")
e1.plot([0,1],[0,1],"--",color=GREY,lw=1); e1.set_title("Curva ROC (AUC-ROC)"); e1.set_xlabel("1 − Especificidad"); e1.set_ylabel("Sensibilidad"); e1.legend(fontsize=7.5,loc="lower right")
e2.set_title("Curva Precisión–Exhaustividad (AUC-PR)"); e2.set_xlabel("Exhaustividad"); e2.set_ylabel("Precisión"); e2.legend(fontsize=7.5,loc="upper right")
for a in (e1,e2): a.set_xlim(0,1); a.set_ylim(0,1.02); a.grid(alpha=.25)
figE.suptitle("ECG v2 — rendimiento por patología (test)",color=BLUE,fontweight="bold")
figE.tight_layout(rect=[0,0,1,0.96]); figE.savefig(f"{FIG}/06_ecg_v2_curvas.png",bbox_inches="tight"); plt.close(figE)
# barras AUC-PR / AUC-ROC + prevalencia
mE=nf_last(pd.read_csv("salidas/02_ecg/v2/metrics_per_label_v2.csv")); labsE=[ES.get(l,l) for l in mE["label"]]; yE=np.arange(len(labsE))
figB,ax=plt.subplots(figsize=(8.2,4.2))
ax.barh(yE-0.2,mE["AP"],0.4,color=BLUE,label="AUC-PR"); ax.barh(yE+0.2,mE["AUC"],0.4,color=BLUE2,label="AUC-ROC")
for i,pv in enumerate(mE["prevalencia"]): ax.plot([pv,pv],[i-0.4,i],color=ORANGE,lw=2.6,label=("Prevalencia (línea base solo de la AUC-PR)" if i==0 else None))
ax.set_yticks(yE); ax.set_yticklabels(labsE); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel("Valor"); ax.legend(loc="lower right",fontsize=8)
ax.set_title("ECG v2 — AUC-PR y AUC-ROC por patología (naranja = prevalencia, línea base solo de la AUC-PR)"); ax.grid(axis="x",alpha=.25)
figB.tight_layout(); figB.savefig(f"{FIG}/05_ecg_v2_auc_etiqueta.png",bbox_inches="tight"); plt.close(figB)

# ── barras AUC-PR / AUC-ROC + prevalencia (LABS v1, la versión que entra en la fusión) ──
try:
    mL=nf_last(pd.read_csv("salidas/03_labs/v1/metrics_per_label_v1.csv")); labsL=[ES.get(l,l) for l in mL["label"]]; yL=np.arange(len(labsL))
    figL,ax=plt.subplots(figsize=(8.2,4.2))
    ax.barh(yL-0.2,mL["AP"],0.4,color=BLUE,label="AUC-PR"); ax.barh(yL+0.2,mL["AUC"],0.4,color=BLUE2,label="AUC-ROC")
    for i,pv in enumerate(mL["prevalencia"]): ax.plot([pv,pv],[i-0.4,i],color=ORANGE,lw=2.6,label=("Prevalencia (línea base solo de la AUC-PR)" if i==0 else None))
    ax.set_yticks(yL); ax.set_yticklabels(labsL); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel("Valor"); ax.legend(loc="lower right",fontsize=8)
    ax.set_title("Analíticas v1 — AUC-PR y AUC-ROC por patología (naranja = prevalencia, línea base solo de la AUC-PR)"); ax.grid(axis="x",alpha=.25)
    figL.tight_layout(); figL.savefig(f"{FIG}/20_labs_v1_auc_etiqueta.png",bbox_inches="tight"); plt.close(figL)
    print("   + 20_labs_v1_auc_etiqueta.png")
except Exception as e:
    print("   ! labs bar fig omitida:",e)

# ══════ Fiabilidad LIMPIA (CXR v2 y ECG v2), agrupando las 5 patologías ══════
def reliab(Pmat,Ymat,Mmat,titulo,fname):
    ps=[];ys=[]
    for j in DIS:
        s=Mmat[:,j]==1; ps+=list(Pmat[s,j]); ys+=list(Ymat[s,j])
    ps=np.array(ps);ys=np.array(ys); bins=np.linspace(0,1,11); idx=np.clip(np.digitize(ps,bins)-1,0,9)
    xs=[];obs=[]
    for b in range(10):
        m=idx==b
        if m.sum()>=8: xs.append(ps[m].mean()); obs.append(ys[m].mean())
    fig,ax=plt.subplots(figsize=(5.2,4.6)); ax.plot([0,1],[0,1],"--",color=GREY,label="Calibración perfecta")
    ax.plot(xs,obs,"o-",color=BLUE,lw=2,label="Modelo (tras isotónica)")
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel("Probabilidad predicha");ax.set_ylabel("Frecuencia real de positivos")
    ax.set_title(titulo,color=BLUE,fontsize=10.5);ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(f"{FIG}/{fname}",bbox_inches="tight");plt.close(fig)
reliab(P,Y,M,"CXR v2 — curva de fiabilidad (calibración)","04_cxr_v2_fiabilidad.png")
reliab(PE,YE,ME,"ECG v2 — curva de fiabilidad (calibración)","08_ecg_v2_fiabilidad.png")

# ══════ Matrices de confusión del MODELO FINAL (fusión v3) ══════
import json as _j2
Wv=_j2.load(open("salidas/04_stacking/v3/veredicto_v3.json"))["pesos_opt"]
def loadbp(sp):
    fn={"val":"val","test":"test"}[sp]; clx=pd.read_csv(f"data/clean/{sp}_clean.csv",sep=";")
    bp=pd.read_csv(f"salidas/04_stacking/v1/base_preds_{fn}.csv").set_index("hadm_id").loc[clx["hadm_id"].to_numpy()].reset_index()
    raw=clx[LAB].to_numpy(float); Pm={m:bp[[f"{m}_{u}" for u in LAB_U]].to_numpy(float) for m in ["CXR","ECG","LABS"]}
    Pf=np.zeros((len(raw),6))
    for j,lab in enumerate(LAB): w=Wv[lab]; Pf[:,j]=w[0]*Pm["CXR"][:,j]+w[1]*Pm["ECG"][:,j]+w[2]*Pm["LABS"][:,j]
    return (raw==1).astype(int),(raw!=-1),Pf
Yv,Mv,Pfv=loadbp("val"); Yt2,Mt2,Pft=loadbp("test")
# Umbrales de CRIBADO fijados en validación (idénticos a la Tabla 7.6 de la memoria
# y al modo "cribado" de la herramienta): min{umbral : Se(val) >= 0.90 aprox.}
THR_CRIBADO={"Atelectasis":0.24,"Cardiomegaly":0.30,"Edema":0.15,
             "Lung Opacity":0.25,"No Finding":0.08,"Pleural Effusion":0.28}
figC,axes=plt.subplots(2,3,figsize=(9.5,6))
for j,lab in enumerate(LAB):
    thr=THR_CRIBADO[lab]
    st=Mt2[:,j]==1; pred=(Pft[st,j]>=thr).astype(int); yt=Yt2[st,j]
    TN=int(((pred==0)&(yt==0)).sum());FP=int(((pred==1)&(yt==0)).sum());FN=int(((pred==0)&(yt==1)).sum());TP=int(((pred==1)&(yt==1)).sum())
    cm=np.array([[TN,FP],[FN,TP]]); ax=axes[j//3,j%3]; ax.imshow(cm,cmap="Blues")
    ax.set_title(f"{ES[lab]} (umbral {thr:.2f})".replace(".",","),fontsize=9,color=BLUE)
    for a in range(2):
        for bcol in range(2): ax.text(bcol,a,cm[a,bcol],ha="center",va="center",fontsize=11,color="white" if cm[a,bcol]>cm.max()*0.6 else "black")
    ax.set_xticks([0,1]);ax.set_xticklabels(["Pred. sano","Pred. enfermo"],fontsize=7);ax.set_yticks([0,1]);ax.set_yticklabels(["Real sano","Real enfermo"],fontsize=7)
figC.suptitle("Modelo final (fusión v3) — matrices de confusión por patología (modo CRIBADO)",color=BLUE,fontweight="bold")
figC.tight_layout(rect=[0,0,1,0.96]); figC.savefig(f"{FIG}/27_fusion_v3_confusion.png",bbox_inches="tight"); plt.close(figC)

print("OK · regeneradas ECG v2 (05,06,08), fiabilidad CXR/ECG (04,08), y matrices confusión fusión v3 (27)")
