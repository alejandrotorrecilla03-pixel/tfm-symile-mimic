# -*- coding: utf-8 -*-
"""
EDA DEFINITIVO — Symile-MIMIC-IV
================================
Objetivo del estudio: sistema multimodal de apoyo a la decisión clínica que predice 6
hallazgos del informe radiológico (Atelectasia, Cardiomegalia, Edema, Opacidad pulmonar,
Derrame pleural y «Sin hallazgo») a partir de radiografía, ECG y analíticas de sangre.
Las etiquetas provienen SOLO de la radiografía; ECG y labs anticipan esos hallazgos.

Objetivo del EDA: caracterizar los datos y extraer el MÁXIMO de decisiones PRE-MODELO
(definición de negativo, desbalanceo y pos_weight, métrica, qué señal aporta cada
modalidad, missingness/MNAR, selección de variables, equidad, protocolo train/val/test).

Paradigma de etiquetado (definición FINAL acordada con el tutor):
  POSITIVO = 1 ; NEGATIVO = 0 explícito + NaN→0 (no mencionado = ausente).
  −1 (incierto) = ENMASCARADO (fuera de pérdida y métrica; estrategia U-Ignore de CheXpert).
  «Sin hallazgo» NO se usa como negativo de las demás (evita sesgo de espectro).

Genera, para cada bloque:  FIGURA (PNG)  +  TABLA (CSV)  con los números.
Salida:  salidas/00_eda/figuras/*.png  y  salidas/00_eda/tablas/*.csv
Ejecutar: python notebooks/EDA_Symile_MIMIC.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif

BASE = Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV  = BASE / "data_csv" / "clean"
ROOT = Path(r"C:\TFM\1.Opción - Symile Mimic\tfm_multimodal_clinico\salidas\00_eda")
FIGD = ROOT / "figuras"; TABD = ROOT / "tablas"
FIGD.mkdir(parents=True, exist_ok=True); TABD.mkdir(parents=True, exist_ok=True)

# ───────── SISTEMA VISUAL UNIFICADO ─────────
INK, ACCENT, GRID, SUB = "#1F2A44", "#2E5496", "#D9DEE8", "#5A6678"
sns.set_theme(style="white")
mpl.rcParams.update({
    "figure.dpi":120,"savefig.dpi":160,"savefig.bbox":"tight","font.family":"Cambria","font.size":12,
    "axes.titlesize":15,"axes.titleweight":"bold","axes.titlecolor":INK,"axes.labelsize":12,"axes.labelcolor":INK,
    "axes.edgecolor":"#7A8499","axes.linewidth":1.0,"xtick.color":INK,"ytick.color":INK,"text.color":INK,
    "legend.fontsize":10,"legend.frameon":True,"legend.edgecolor":GRID,"legend.framealpha":0.95,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True})
LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema","Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgo","Pleural Effusion":"Derrame pleural"}
PATHO=[l for l in LABELS if l!="No Finding"]; NF=LABELS.index("No Finding"); PC=[j for j in range(6) if j!=NF]
# Paleta USAL unificada (coherente con el informe de modelos: marino + naranja institucionales)
LCOL={"Atelectasis":"#1F3864","Cardiomegaly":"#2E5496","Edema":"#5B8FCB","Lung Opacity":"#C55A11","No Finding":"#7F7F7F","Pleural Effusion":"#2E8B57"}
SCOL={"train":"#1F3864","val":"#C55A11","test":"#2E8B57"}; SES={"train":"Entrenamiento","val":"Validación","test":"Test"}
POSC,NEGC="#2E5496","#C55A11"; GENDER={0:0,1:1,"0":0,"1":1,"M":1,"F":0}; XS=np.arange(6)

def style(ax,title=None,sub=None,xlabel=None,ylabel=None,grid=True):
    if title: ax.set_title(title,pad=16 if sub else 10)
    if sub: ax.text(0.5,1.015,sub,transform=ax.transAxes,ha="center",va="bottom",fontsize=10.5,color=SUB,style="italic")
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    if not grid: ax.grid(False)
NF_=[0]; NT_=[0]
def savefig(fig,name):
    NF_[0]+=1; fig.savefig(FIGD/f"{NF_[0]:02d}_{name}.png"); plt.close(fig)
def savetab(df,name,index=True):
    NT_[0]+=1; df.to_csv(TABD/f"{NT_[0]:02d}_{name}.csv",index=index,encoding="utf-8-sig"); print(f"  fig+tab {name}")
def disp(c): return c.rsplit("_",1)[0].replace("_"," ").title()
def xlabels(ax): ax.set_xticks(XS); ax.set_xticklabels([ES[l] for l in LABELS],rotation=20,ha="right")

# ───────── CARGA ─────────
DF={s:pd.read_csv(CSV/f"{s}_clean.csv",sep=";") for s in ["train","val","test"]}; train=DF["train"]
DEMO=["subject_id","hadm_id","cxr_path","ecg_path","age","gender","race","admission_type","admission_location","cxr_view","hours_adm_to_cxr"]
RAW=[c for c in train.columns if c not in LABELS and c not in DEMO and "pctile" not in c]
PCT=[c for c in train.columns if "pctile" in c]; RAWD=[disp(c) for c in RAW]; raw_arr=train[RAW].values.astype(float); miss_raw=train[RAW].isna().mean().values
def POS(df,l): return (df[l]==1).values
def NEG(df,l): return ((df[l]==0)|(df[l].isna())).values   # 0 explícito + NaN→0 (el −1 NO entra)
def EXCL(df,l): return (df[l]==-1).values                  # incierto → enmascarado (excluido)
print("EDA definitivo → figuras y TABLAS en", ROOT)

# ════════ BLOQUE 1 · VISIÓN GENERAL ════════
rows=[]
for s in ["train","val","test"]:
    d=DF[s]; rows.append({"Partición":SES[s],"n":len(d),"Edad media":round(d["age"].mean(),1),"Edad mediana":d["age"].median(),
        "% Hombre":round((d["gender"].map(lambda v:GENDER.get(v,0))==1).mean()*100,1),
        "% Urgente":round((d["admission_type"].astype(str).str.upper()=="EMERGENCY").mean()*100,1),
        "Missingness medio labs %":round(d[RAW].isna().mean().mean()*100,1)})
resumen=pd.DataFrame(rows)
fig,ax=plt.subplots(figsize=(7,4.2)); sizes=[len(DF[s]) for s in ["train","val","test"]]
b=ax.bar([SES[s] for s in ["train","val","test"]],sizes,color=[SCOL[s] for s in ["train","val","test"]],width=0.6,edgecolor="white",lw=1.5)
for r,v in zip(b,sizes): ax.text(r.get_x()+r.get_width()/2,v+120,f"{v:,}",ha="center",fontweight="bold")
ax.set_ylim(0,max(sizes)*1.15); ax.set_yticks([]); ax.spines["left"].set_visible(False)
style(ax,"Tamaño de las particiones","11.214 pacientes · cobertura de las 3 modalidades = 100 %",ylabel="")
savefig(fig,"particiones"); savetab(resumen,"resumen_general",index=False)
fig,ax=plt.subplots(figsize=(7.5,4.4)); covm={}
for s in ["train","val","test"]:
    k=(DF[s][PATHO].to_numpy()==1).sum(1); vc=pd.Series(k).value_counts(normalize=True).sort_index()*100; covm[SES[s]]=vc
    ax.plot(vc.index,vc.values,"o-",color=SCOL[s],lw=2.2,ms=6,label=SES[s])
ax.set_xticks(range(0,6)); style(ax,"Número de patologías positivas por paciente","Problema multietiqueta: muchos co-diagnósticos",xlabel="nº de patologías positivas",ylabel="% de pacientes"); ax.legend()
savefig(fig,"positivas_por_paciente"); savetab(pd.DataFrame(covm).fillna(0).round(1),"patologias_por_paciente")

# ════════ BLOQUE 2 · DEMOGRAFÍA + TESTS ════════
fig,ax=plt.subplots(figsize=(8,4.6))
for s in ["train","val","test"]: sns.kdeplot(DF[s]["age"].dropna(),ax=ax,color=SCOL[s],lw=2.4,label=SES[s])
sk=stats.skew(train["age"].dropna()); W,pS=stats.shapiro(train["age"].dropna().sample(min(4000,len(train)),random_state=0))
style(ax,"Distribución de la edad por partición",f"Train: media 67,7 · asimetría {sk:+.2f} · Shapiro-Wilk p<0,001 (no gaussiana)",xlabel="Edad (años)",ylabel="Densidad"); ax.legend()
savefig(fig,"edad_distribucion")
fig,ax=plt.subplots(figsize=(5.6,5.2)); stats.probplot(train["age"].dropna(),dist="norm",plot=ax)
ax.get_lines()[0].set(marker="o",ms=3,markerfacecolor=ACCENT,markeredgecolor="none",alpha=0.5); ax.get_lines()[1].set(color=NEGC,lw=2)
style(ax,"QQ-plot de la edad (entrenamiento)","Apartamiento moderado de la normalidad",xlabel="Cuantiles teóricos",ylabel="Cuantiles observados"); savefig(fig,"edad_qqplot")
g=train["gender"].map(lambda v:GENDER.get(v,0)); a_m=train["age"][g==1].dropna(); a_f=train["age"][g==0].dropna()
Umw,pMW=stats.mannwhitneyu(a_m,a_f)
fig,ax=plt.subplots(figsize=(6.4,4.8)); parts=ax.violinplot([a_m,a_f],showmedians=True)
for pc,c in zip(parts["bodies"],[ACCENT,"#C55A11"]): pc.set_facecolor(c); pc.set_alpha(0.6)
parts["cmedians"].set_color(INK); ax.set_xticks([1,2]); ax.set_xticklabels([f"Hombre\nn={len(a_m):,}",f"Mujer\nn={len(a_f):,}"])
style(ax,"Edad por sexo","Mann-Whitney p<0,001 (mujeres ligeramente mayores)",ylabel="Edad (años)"); savefig(fig,"edad_genero")
order=["WHITE","BLACK","UNKNOWN","HISPANIC_LATINO","ASIAN","OTHER_KNOWN"]; elab={"WHITE":"Blanca","BLACK":"Negra","UNKNOWN":"Desconocida","HISPANIC_LATINO":"Hispana","ASIAN":"Asiática","OTHER_KNOWN":"Otra"}
data=[train["age"][train["race"]==o].dropna() for o in order]; Hkw,pKW=stats.kruskal(*[d for d in data if len(d)>5])
fig,ax=plt.subplots(figsize=(9,4.8)); bp=ax.boxplot(data,patch_artist=True,medianprops={"color":INK,"lw":2},widths=0.6,showfliers=False)
for box in bp["boxes"]: box.set_facecolor("#9FBCE0")
ax.set_xticklabels([elab[o] for o in order],rotation=20,ha="right"); style(ax,"Edad por grupo étnico",f"Kruskal-Wallis H={Hkw:.0f}, p<0,001",ylabel="Edad (años)"); savefig(fig,"edad_etnia")
demo_rows=[]
for s in ["train","val","test"]:
    d=DF[s]
    for o in order:
        demo_rows.append({"Partición":SES[s],"Grupo étnico":elab[o],"n":int((d["race"]==o).sum()),"%":round((d["race"]==o).mean()*100,1)})
savetab(pd.DataFrame(demo_rows),"etnia_por_split",index=False)
tests=pd.DataFrame([
    {"Test":"Shapiro-Wilk (normalidad edad, train)","Estadístico":round(W,4),"p-valor":"<0,001" if pS<1e-3 else round(pS,4)},
    {"Test":"Mann-Whitney (edad por sexo)","Estadístico":round(Umw,0),"p-valor":"<0,001" if pMW<1e-3 else round(pMW,4)},
    {"Test":"Kruskal-Wallis (edad por etnia)","Estadístico":round(Hkw,1),"p-valor":"<0,001" if pKW<1e-3 else round(pKW,4)},
])
w=0.26
fig,ax=plt.subplots(figsize=(6.6,5.6)); bins=np.arange(15,101,5); hmh,_=np.histogram(a_m,bins=bins); hfh,_=np.histogram(a_f,bins=bins); yc=(bins[:-1]+bins[1:])/2
ax.barh(yc,-hmh,height=4,color=ACCENT,label="Hombre"); ax.barh(yc,hfh,height=4,color="#C55A11",label="Mujer")
ax.set_xticks(ax.get_xticks()); ax.set_xticklabels([f"{abs(int(t))}" for t in ax.get_xticks()]); style(ax,"Pirámide de edad por sexo",xlabel="nº de pacientes",ylabel="Edad (años)"); ax.legend(); savefig(fig,"piramide_edad")
fig,ax=plt.subplots(figsize=(7,4.2)); male=[(DF[s]["gender"].map(lambda v:GENDER.get(v,0))==1).mean()*100 for s in ["train","val","test"]]; fem=[100-mm for mm in male]
ax.bar(range(3),male,0.6,label="Hombre",color=ACCENT,edgecolor="white"); ax.bar(range(3),fem,0.6,bottom=male,label="Mujer",color="#C55A11",edgecolor="white")
for i,(mm,ff) in enumerate(zip(male,fem)): ax.text(i,mm/2,f"{mm:.0f}%",ha="center",color="white",fontweight="bold"); ax.text(i,mm+ff/2,f"{ff:.0f}%",ha="center",color="white",fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels([SES[s] for s in ["train","val","test"]]); ax.set_yticks([]); ax.spines["left"].set_visible(False)
style(ax,"Distribución por sexo y partición",ylabel=""); ax.legend(ncol=2,loc="lower center",bbox_to_anchor=(0.5,-0.22)); savefig(fig,"genero_split")
fig,ax=plt.subplots(figsize=(9,4.6))
for k,s in enumerate(["train","val","test"]):
    vc=DF[s]["race"].value_counts(normalize=True)*100; ax.bar(np.arange(6)+(k-1)*w,[vc.get(o,0) for o in order],w,label=SES[s],color=SCOL[s],edgecolor="white")
ax.set_xticks(range(6)); ax.set_xticklabels([elab[o] for o in order],rotation=20,ha="right"); style(ax,"Composición étnica por partición","«Desconocida» es categoría registrada, no NaN",ylabel="% de pacientes"); ax.legend(); savefig(fig,"etnia_split")
adm=sorted(train["admission_type"].dropna().unique())
fig,ax=plt.subplots(figsize=(9,4.6))
for k,s in enumerate(["train","val","test"]):
    vc=DF[s]["admission_type"].value_counts(normalize=True)*100; ax.bar(np.arange(len(adm))+(k-1)*w,[vc.get(a,0) for a in adm],w,label=SES[s],color=SCOL[s],edgecolor="white")
ax.set_xticks(range(len(adm))); ax.set_xticklabels([a.title() for a in adm],rotation=20,ha="right"); style(ax,"Tipo de ingreso por partición",ylabel="% de pacientes"); ax.legend()
savefig(fig,"ingreso_split"); savetab(tests,"tests_demograficos",index=False)

# ════════ BLOQUE 3 · ETIQUETAS + CONTRASTE DE NEGATIVO ════════
est_rows=[]
for s in ["train","val","test"]:
    raw=DF[s][LABELS].to_numpy(float)
    for j,l in enumerate(LABELS):
        v=raw[:,j]; est_rows.append({"Partición":SES[s],"Hallazgo":ES[l],"Positivo(1)":int((v==1).sum()),"Negativo(0)":int((v==0).sum()),
            "Incierto(-1)":int((v==-1).sum()),"NaN":int(np.isnan(v).sum()),"% NaN":round(np.isnan(v).mean()*100,1),"% -1":round((v==-1).mean()*100,1)})
savetab(pd.DataFrame(est_rows),"etiquetas_estados",index=False)
fig,ax=plt.subplots(figsize=(9.5,4.8))
for k,s in enumerate(["train","val","test"]):
    prev=[(DF[s][l]==1).mean()*100 for l in LABELS]; ax.bar(XS+(k-1)*w,prev,w,label=SES[s],color=SCOL[s],edgecolor="white")
ax.axhline(50,color="#7A8499",ls="--",lw=1); xlabels(ax); style(ax,"Prevalencia de positivos por etiqueta y partición","Positivo = 1 sobre el total · estable entre splits",ylabel="% positivos"); ax.legend(); savefig(fig,"prevalencia_split")
raw=train[LABELS].to_numpy(float); n=len(train)
z0=[(raw[:,j]==0).mean()*100 for j in range(6)]; zm1=[(raw[:,j]==-1).mean()*100 for j in range(6)]; znan=[np.isnan(raw[:,j]).mean()*100 for j in range(6)]; zpos=[(raw[:,j]==1).mean()*100 for j in range(6)]
fig,ax=plt.subplots(figsize=(9.5,5.0)); b0=np.array(zpos); b1=b0+np.array(z0); b2=b1+np.array(znan)
ax.bar(XS,zpos,color="#2E8B57",label="Positivo (1)",edgecolor="white")
ax.bar(XS,z0,bottom=b0,color="#5B8FCB",label="Negativo: 0 explícito",edgecolor="white")
ax.bar(XS,znan,bottom=b1,color="#B5728E",label="Negativo: NaN → 0",edgecolor="white")
ax.bar(XS,zm1,bottom=b2,color="#B0B0B0",label="Excluido: −1 (enmascarado)",edgecolor="white")
xlabels(ax); ax.set_ylim(0,108); style(ax,"Composición de cada etiqueta (definición final)","Negativo = 0 + NaN→0; el −1 (incierto) se excluye de pérdida y métrica",ylabel="% de registros"); ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.34),ncol=2); savefig(fig,"composicion_negativo")
def contraste(df):
    rawc=df[LABELS].to_numpy(float); nn=len(df); rows=[]
    nfp=(rawc[:,NF]==1); ap=(rawc[:,PC]==1).any(1)
    for j,l in enumerate(LABELS):
        v=rawc[:,j]; pos=int((v==1).sum()); zero=int((v==0).sum()); unc=int((v==-1).sum())
        der=int(((nfp if l!="No Finding" else ap)&np.isnan(v)).sum())
        neg_fin=nn-pos-unc   # 0 + NaN->0 ; el -1 se excluye (enmascarado)
        rows.append({"Hallazgo":ES[l],"Positivos":pos,"Excluidos(-1)":unc,"Neg_previa(0)":zero,"Neg_miIdea(+NoFinding)":zero+der,
                     "Neg_final(NaN->0,-1 excl)":neg_fin,"pos_weight":round(neg_fin/max(pos,1),2)})
    return pd.DataFrame(rows)
for s in ["train","val","test"]: savetab(contraste(DF[s]),f"contraste_negativo_{s}",index=False)
ct=contraste(train); pos=ct["Positivos"].values; neg=ct["Neg_final(NaN->0,-1 excl)"].values
fig,ax=plt.subplots(figsize=(9.5,4.8))
ax.bar(XS-0.2,pos,0.4,label="Positivos",color="#2E8B57",edgecolor="white"); ax.bar(XS+0.2,neg,0.4,label="Negativos (0 + NaN→0)",color=ACCENT,edgecolor="white")
for i in range(6): ax.text(i,max(pos[i],neg[i])+120,f"{neg[i]/max(pos[i],1):.1f}:1",ha="center",fontsize=9,fontweight="bold")
xlabels(ax); style(ax,"Balance positivos vs negativos (−1 excluido)","Negativo = 0 + NaN→0; desbalanceo moderado (1,7:1–6,3:1), se resuelve con pos_weight",ylabel="nº de pacientes"); ax.legend(); savefig(fig,"balance_pos_neg")
fig,ax=plt.subplots(figsize=(9,4.6)); pw=ct["pos_weight"].values
bb=ax.bar(XS,pw,color=[LCOL[l] for l in LABELS],edgecolor="white",width=0.62)
for rr,v in zip(bb,pw): ax.text(rr.get_x()+rr.get_width()/2,v+0.06,f"{v:.2f}",ha="center",fontweight="bold",fontsize=10)
ax.axhline(1,color="#7A8499",ls="--",lw=1); xlabels(ax); style(ax,"pos_weight por etiqueta (nº neg / nº pos)","Lo aplican CXR/ECG (BCE) y el tabular (scale_pos_weight)",ylabel="pos_weight"); savefig(fig,"pos_weight")
posM=(train[LABELS].to_numpy()==1)
co=np.array([[int((posM[:,i]&posM[:,j]).sum()) for j in range(6)] for i in range(6)])
codf=pd.DataFrame(co,index=[ES[l] for l in LABELS],columns=[ES[l] for l in LABELS])
fig,ax=plt.subplots(figsize=(7.6,6.4)); sns.heatmap(co,mask=np.triu(np.ones_like(co,bool),1),annot=True,fmt=",d",cmap="Blues",cbar=False,ax=ax,xticklabels=[ES[l] for l in LABELS],yticklabels=[ES[l] for l in LABELS],linewidths=1,linecolor="white",annot_kws={"size":10})
ax.set_xticklabels(ax.get_xticklabels(),rotation=30,ha="right"); style(ax,"Co-ocurrencia de hallazgos positivos (train)","Diagonal = nº de positivos",grid=False); savefig(fig,"coocurrencia"); savetab(codf,"coocurrencia")
J=np.array([[(posM[:,i]&posM[:,j]).sum()/max((posM[:,i]|posM[:,j]).sum(),1) for j in range(6)] for i in range(6)])
Jdf=pd.DataFrame(np.round(J,3),index=[ES[l] for l in LABELS],columns=[ES[l] for l in LABELS])
fig,ax=plt.subplots(figsize=(7.6,6.4)); sns.heatmap(J,mask=np.triu(np.ones_like(J,bool),1),annot=True,fmt=".2f",cmap="YlOrRd",vmin=0,vmax=0.5,cbar_kws={"shrink":0.7,"label":"Jaccard"},ax=ax,xticklabels=[ES[l] for l in LABELS],yticklabels=[ES[l] for l in LABELS],linewidths=1,linecolor="white",annot_kws={"size":10})
ax.set_xticklabels(ax.get_xticklabels(),rotation=30,ha="right"); style(ax,"Similitud de Jaccard entre etiquetas","Todas ≤ 0,30 → poco redundantes",grid=False); savefig(fig,"jaccard"); savetab(Jdf,"jaccard")

# ════════ BLOQUE 4 · ANALÍTICAS ════════
miss_pct=train[PCT].isna().mean().values; sidx=np.argsort(miss_raw)[::-1]
missdf=pd.DataFrame({"Analítica":RAWD,"% ausente (train)":np.round(miss_raw*100,1)}).sort_values("% ausente (train)",ascending=False)
fig,ax=plt.subplots(figsize=(11,5.2)); xb=np.arange(len(RAW))
ax.bar(xb-0.2,miss_raw[sidx]*100,0.4,color=ACCENT,label="Raw"); ax.bar(xb+0.2,miss_pct[sidx]*100,0.4,color="#C55A11",label="Percentil")
ax.axhline(70,color=NEGC,ls="--",lw=1.2,label="Umbral 70%"); ax.set_xticks(xb); ax.set_xticklabels([RAWD[i] for i in sidx],rotation=90,fontsize=7)
style(ax,"Valores ausentes por analítica (raw vs percentil, train)","Ninguna supera el 70 %",ylabel="% ausente"); ax.legend(); savefig(fig,"missingness_raw_pct"); savetab(missdf,"missingness_analiticas",index=False)
top6=np.argsort(miss_raw)[:6]
fig,ax=plt.subplots(figsize=(8.5,4.8))
for i in top6:
    x=train[RAW[i]].dropna().values; p1,p99=np.nanpercentile(x,[1,99]); xn=(x-p1)/(p99-p1+1e-9); sns.kdeplot(np.clip(xn,0,1),ax=ax,lw=2,label=RAWD[i])
style(ax,"Distribución (raw) de las analíticas más frecuentes","Normalizadas P1–P99",xlabel="valor normalizado",ylabel="densidad"); ax.legend(fontsize=8); savefig(fig,"dist_raw")
fig,ax=plt.subplots(figsize=(8.5,4.8))
for i in top6: sns.kdeplot(train[PCT[i]].dropna().values,ax=ax,lw=2,label=disp(PCT[i].replace("pctile_","")))
style(ax,"Distribución (percentil) de las mismas analíticas","El percentil normaliza la escala 0–1",xlabel="percentil",ylabel="densidad"); ax.legend(fontsize=8); savefig(fig,"dist_pct")
obs_rank=train[RAW].notna().mean().sort_values(ascending=False).index[:20]; corr=train[obs_rank].corr()
fig,ax=plt.subplots(figsize=(9,7.6)); sns.heatmap(corr,mask=np.triu(np.ones_like(corr,bool),1),cmap="RdBu_r",center=0,vmin=-1,vmax=1,cbar_kws={"shrink":0.6,"label":"Pearson"},ax=ax,xticklabels=[disp(c) for c in obs_rank],yticklabels=[disp(c) for c in obs_rank],linewidths=0.4,linecolor="white")
ax.tick_params(labelsize=8); style(ax,"Matriz de correlación de las analíticas más observadas",grid=False); savefig(fig,"correlacion")

# ════════ BLOQUE 5 · ANALÍTICAS vs ETIQUETAS ════════
top3=np.argsort(miss_raw)[:3]
fig,axes=plt.subplots(1,3,figsize=(13,4.4))
for col_i,lc in enumerate(PATHO[:3]):
    ax=axes[col_i]; lab=top3[col_i%len(top3)]; x=raw_arr[:,lab]; p1,p99=np.nanpercentile(x,[1,99]); xn=(x-p1)/(p99-p1+1e-9)
    gp=xn[POS(train,lc)]; gp=gp[~np.isnan(gp)]; gn=xn[NEG(train,lc)]; gn=gn[~np.isnan(gn)]
    parts=ax.violinplot([gp,gn],showmedians=True)
    for pc,c in zip(parts["bodies"],[POSC,NEGC]): pc.set_facecolor(c); pc.set_alpha(0.55)
    parts["cmedians"].set_color(INK); ax.set_xticks([1,2]); ax.set_xticklabels(["Pos","Neg"]); style(ax,ES[lc],RAWD[lab],ylabel="valor norm." if col_i==0 else None)
fig.tight_layout(); savefig(fig,"violines_lab_etiqueta")
combos=[("Pleural Effusion","albumin_50862","Derrame pleural","Albúmina"),("Edema","urea_nitrogen_51006","Edema","Urea BUN"),("Cardiomegaly","inr_pt_51237","Cardiomegalia","INR"),("Pleural Effusion","creatinine_50912","Derrame pleural","Creatinina"),("Edema","hemoglobin_51222","Edema","Hemoglobina"),("Atelectasis","hematocrit_51221","Atelectasia","Hematocrito")]
fig,axes=plt.subplots(2,3,figsize=(13.5,7.5))
for i,(lc,lab_col,ld,lab_disp) in enumerate(combos):
    ax=axes[i//3,i%3]
    if lab_col not in RAW: ax.axis("off"); continue
    li=RAW.index(lab_col); gp=raw_arr[POS(train,lc),li]; gp=gp[~np.isnan(gp)]; gn=raw_arr[NEG(train,lc),li]; gn=gn[~np.isnan(gn)]
    if len(gp)<5 or len(gn)<5: ax.axis("off"); continue
    lo,hi=np.percentile(np.concatenate([gp,gn]),[2,98]); bins=np.linspace(lo,hi,34)
    ax.hist(np.clip(gp,lo,hi),bins=bins,color=POSC,alpha=0.6,density=True,label="Pos"); ax.hist(np.clip(gn,lo,hi),bins=bins,color=NEGC,alpha=0.5,density=True,label="Neg")
    style(ax,f"{lab_disp} · {ld}",ylabel="densidad" if i%3==0 else None); ax.legend(fontsize=8)
fig.tight_layout(); savefig(fig,"histogramas_comparativos")

# ════════ BLOQUE 6 · PODER DISCRIMINATIVO (AUC-MW) ════════
top20=np.argsort(miss_raw)[:20]; auc=pd.DataFrame(np.nan,index=[RAWD[i] for i in top20],columns=[ES[l] for l in PATHO]); star=auc.copy().astype(object)
for i in top20:
    for lc in PATHO:
        pv=raw_arr[POS(train,lc),i]; pv=pv[~np.isnan(pv)]; nv=raw_arr[NEG(train,lc),i]; nv=nv[~np.isnan(nv)]
        if len(pv)>=5 and len(nv)>=5:
            u,p=stats.mannwhitneyu(pv,nv); a=u/(len(pv)*len(nv)); auc.loc[RAWD[i],ES[lc]]=a; star.loc[RAWD[i],ES[lc]]=f"{a:.2f}{'*' if p<0.05 else ''}"
auc=auc.sort_values(by=ES["Pleural Effusion"],ascending=False); star=star.loc[auc.index]
fig,ax=plt.subplots(figsize=(8.5,7.5)); sns.heatmap(auc.values.astype(float),annot=star.values,fmt="",cmap="RdBu_r",center=0.5,vmin=0.40,vmax=0.62,cbar_kws={"shrink":0.6,"label":"AUC Mann-Whitney"},ax=ax,xticklabels=auc.columns,yticklabels=auc.index,linewidths=0.6,linecolor="white",annot_kws={"size":8})
ax.set_xticklabels(ax.get_xticklabels(),rotation=20,ha="right"); style(ax,"Poder discriminativo de analíticas (AUC-MW)","* significativo p<0,05 · ninguna supera 0,65",grid=False); savefig(fig,"auc_analiticas"); savetab(auc.round(3),"auc_analiticas")
def scatter_pair(c1,c2,lc,t):
    if c1 not in RAW or c2 not in RAW: return
    x=train[c1].values; y=train[c2].values
    fig,ax=plt.subplots(figsize=(6.4,5.6))
    for mask,c,nm in [(NEG(train,lc),NEGC,"Negativo"),(POS(train,lc),POSC,"Positivo")]:
        s=mask&~np.isnan(x)&~np.isnan(y); ax.scatter(x[s],y[s],s=10,alpha=0.4,color=c,label=nm,edgecolors="none")
    ax.legend(markerscale=2); style(ax,t,xlabel=disp(c1),ylabel=disp(c2)); savefig(fig,"scatter_"+c1.split('_')[0]+"_"+c2.split('_')[0])
scatter_pair("albumin_50862","urea_nitrogen_51006","Pleural Effusion","Albúmina vs Urea (color = Derrame pleural)")
scatter_pair("inr_pt_51237","hemoglobin_51222","Cardiomegaly","INR vs Hemoglobina (color = Cardiomegalia)")
auc_agg={}
for i,c in enumerate(RAW):
    s=0
    for lc in PATHO:
        pv=raw_arr[POS(train,lc),i]; pv=pv[~np.isnan(pv)]; nv=raw_arr[NEG(train,lc),i]; nv=nv[~np.isnan(nv)]
        if len(pv)>=5 and len(nv)>=5: u,_=stats.mannwhitneyu(pv,nv); s+=abs(u/(len(pv)*len(nv))-0.5)
    auc_agg[c]=s
topr=sorted(auc_agg,key=auc_agg.get,reverse=True)[:10]; cmin=train[topr].quantile(.05); cmax=train[topr].quantile(.95); crng=(cmax-cmin).replace(0,np.nan)
ang=np.linspace(0,2*np.pi,10,endpoint=False).tolist(); ang+=ang[:1]
fig,ax=plt.subplots(figsize=(7.2,7.2),subplot_kw=dict(polar=True))
for lc in PATHO:
    prof=[]
    for c in topr:
        v=train.loc[POS(train,lc),c].median(); prof.append(float((v-cmin[c])/crng[c]) if crng[c]==crng[c] else 0.5)
    prof+=prof[:1]; ax.plot(ang,prof,lw=2,color=LCOL[lc],label=ES[lc]); ax.fill(ang,prof,color=LCOL[lc],alpha=0.07)
ax.set_xticks(ang[:-1]); ax.set_xticklabels([disp(c) for c in topr],fontsize=8); ax.set_yticklabels([])
ax.set_title("Perfil analítico mediano por patología",fontweight="bold",pad=26,color=INK)
ax.text(0.5,1.06,"Perfiles solapados → se necesita imagen/ECG",transform=ax.transAxes,ha="center",fontsize=10,color=SUB,style="italic")
ax.legend(loc="upper right",bbox_to_anchor=(1.32,1.12),fontsize=8); savefig(fig,"radar_perfiles")
rk=sorted(auc_agg.items(),key=lambda x:x[1],reverse=True); rankdf=pd.DataFrame([{"Analítica":disp(c),"Suma|AUC-0.5|":round(v,3)} for c,v in rk])
fig,ax=plt.subplots(figsize=(8,6)); rk15=rk[:15][::-1]
ax.barh([disp(c) for c,_ in rk15],[v for _,v in rk15],color=ACCENT,edgecolor="white"); ax.tick_params(labelsize=9)
style(ax,"Ranking de analíticas por poder discriminativo agregado","Suma de |AUC−0,5| sobre las 5 patologías",xlabel="Σ |AUC − 0,5|"); savefig(fig,"ranking_auc"); savetab(rankdf,"ranking_analiticas",index=False)

# ════════ BLOQUE 7 · MNAR ════════
mnar_rows=[]; core=["Edema","Pleural Effusion","Cardiomegaly"]
fig,ax=plt.subplots(figsize=(9,4.8)); pm=[]; nm=[]
for l in core:
    p_=train[RAW][POS(train,l)].isna().mean().mean()*100; n_=train[RAW][NEG(train,l)].isna().mean().mean()*100; pm.append(p_); nm.append(n_)
    mnar_rows.append({"Hallazgo":ES[l],"% ausente positivos":round(p_,1),"% ausente negativos":round(n_,1),"Delta (neg-pos)":round(n_-p_,1)})
ax.bar(np.arange(3)-0.19,pm,0.38,label="Positivos",color=POSC,edgecolor="white"); ax.bar(np.arange(3)+0.19,nm,0.38,label="Negativos",color="#A9B4C9",edgecolor="white")
for i,(p_,nn_) in enumerate(zip(pm,nm)): ax.text(i-0.19,p_+0.3,f"{p_:.1f}",ha="center",fontsize=9); ax.text(i+0.19,nn_+0.3,f"{nn_:.1f}",ha="center",fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels([ES[l] for l in core]); style(ax,"Señal MNAR: ausencias según el estado del paciente","Los positivos tienen menos ausencias → incluir flags de missingness",ylabel="% medio ausente"); ax.legend()
savefig(fig,"mnar"); savetab(pd.DataFrame(mnar_rows),"mnar",index=False)
fig,ax=plt.subplots(figsize=(5.8,5.6)); mtr=train[RAW].isna().mean().values; mva=DF["val"][RAW].isna().mean().values; mte=DF["test"][RAW].isna().mean().values
ax.scatter(mtr*100,mva*100,s=22,color=SCOL["val"],alpha=0.8,label="Val"); ax.scatter(mtr*100,mte*100,s=22,color=SCOL["test"],alpha=0.8,label="Test"); ax.plot([0,100],[0,100],"--",color="#7A8499",lw=1)
Rv=np.corrcoef(mtr,mva)[0,1]; Rt=np.corrcoef(mtr,mte)[0,1]; style(ax,"Consistencia del missingness entre particiones",f"R(train,val)={Rv:.3f} · R(train,test)={Rt:.3f}",xlabel="% ausente (train)",ylabel="% ausente (val/test)"); ax.legend(); savefig(fig,"missingness_consistencia")
fig,ax=plt.subplots(figsize=(8,4.6)); dec=pd.qcut(train["age"],10,labels=False); mm=[np.isnan(raw_arr[(dec==d).values]).mean()*100 for d in range(10)]
ax.bar(range(10),mm,color=ACCENT,edgecolor="white"); ax.set_xticks(range(10)); ax.set_xticklabels([f"D{d+1}" for d in range(10)]); style(ax,"Missingness medio por decil de edad",xlabel="decil de edad",ylabel="% medio ausente"); savefig(fig,"missingness_edad")

# ════════ BLOQUE 8 · INFERENCIA ════════
top12=np.argsort(miss_raw)[:12]; pmat=pd.DataFrame(np.nan,index=[ES[l] for l in PATHO],columns=[RAWD[i] for i in top12]); dmed=pmat.copy()
for lc in PATHO:
    for i in top12:
        pv=raw_arr[POS(train,lc),i]; pv=pv[~np.isnan(pv)]; nv=raw_arr[NEG(train,lc),i]; nv=nv[~np.isnan(nv)]
        if len(pv)>=5 and len(nv)>=5:
            u,p=stats.mannwhitneyu(pv,nv); pmat.loc[ES[lc],RAWD[i]]=p
            allv=np.concatenate([pv,nv]); rng=np.nanpercentile(allv,99)-np.nanpercentile(allv,1)+1e-9; dmed.loc[ES[lc],RAWD[i]]=(np.median(pv)-np.median(nv))/rng
fig,ax=plt.subplots(figsize=(11,4.6)); sns.heatmap(-np.log10(pmat.values.astype(float)),annot=pmat.applymap(lambda x:"" if x!=x else ("***" if x<1e-3 else "**" if x<1e-2 else "*" if x<0.05 else "")).values,fmt="",cmap="Purples",cbar_kws={"shrink":0.7,"label":"-log10(p)"},ax=ax,xticklabels=[RAWD[i] for i in top12],yticklabels=[ES[l] for l in PATHO],linewidths=0.5,linecolor="white",annot_kws={"size":9,"color":"white"})
ax.set_xticklabels(ax.get_xticklabels(),rotation=30,ha="right"); style(ax,"Significación Mann-Whitney (analítica × patología)","* p<0,05 · ** p<0,01 · *** p<0,001",grid=False); savefig(fig,"inferencia_pvalores"); savetab(pmat.round(5),"inferencia_pvalores")
fig,ax=plt.subplots(figsize=(11,4.6)); sns.heatmap(dmed.values.astype(float),annot=True,fmt=".2f",cmap="RdBu_r",center=0,vmin=-0.12,vmax=0.12,cbar_kws={"shrink":0.7,"label":"Delta mediana norm."},ax=ax,xticklabels=[RAWD[i] for i in top12],yticklabels=[ES[l] for l in PATHO],linewidths=0.5,linecolor="white",annot_kws={"size":8})
ax.set_xticklabels(ax.get_xticklabels(),rotation=30,ha="right"); style(ax,"Tamaño del efecto: diferencia de medianas (positivo − negativo)","|Delta| ≤ 0,08 (moderado)",grid=False); savefig(fig,"inferencia_efecto"); savetab(dmed.round(3),"inferencia_efecto")

# ════════ BLOQUE 9 · MULTIVARIANTE ════════
imp=SimpleImputer(strategy="median"); Ximp=imp.fit_transform(train[RAW]); Xs=StandardScaler().fit_transform(Ximp); Z=linkage(Xs.T,method="ward")
fig,ax=plt.subplots(figsize=(11,5)); dendrogram(Z,labels=RAWD,ax=ax,color_threshold=0.7*max(Z[:,2]),leaf_font_size=8,above_threshold_color="#7A8499")
style(ax,"Agrupamiento jerárquico de analíticas (Ward)","Clusters: eritrocitario · renal · ácido-base · leucocitario",ylabel="distancia",grid=False); savefig(fig,"dendrograma")
dom=np.array([ES[PATHO[int(np.argmax(train.iloc[i][PATHO].values==1))]] if (train.iloc[i][PATHO].values==1).any() else "Sin patología" for i in range(len(train))])
pca=PCA(10,random_state=0); Z2=pca.fit_transform(Xs)
vardf=pd.DataFrame({"Componente":range(1,11),"Var. individual %":np.round(pca.explained_variance_ratio_*100,2),"Var. acumulada %":np.round(np.cumsum(pca.explained_variance_ratio_)*100,2)})
fig,ax=plt.subplots(figsize=(7.4,6.2))
for l in PATHO:
    s=dom==ES[l]; ax.scatter(Z2[s,0],Z2[s,1],s=8,alpha=0.5,color=LCOL[l],label=ES[l],edgecolors="none")
s=dom=="Sin patología"; ax.scatter(Z2[s,0],Z2[s,1],s=6,alpha=0.25,color="#B8B8B8",label="Sin patología",edgecolors="none")
ax.legend(markerscale=2,fontsize=9); style(ax,"PCA de las analíticas — por patología dominante",f"PC1 {pca.explained_variance_ratio_[0]*100:.1f}% + PC2 {pca.explained_variance_ratio_[1]*100:.1f}%",xlabel="PC1",ylabel="PC2",grid=False); savefig(fig,"pca_etiqueta")
fig,ax=plt.subplots(figsize=(7.4,6.2)); gg=train["gender"].map(lambda v:GENDER.get(v,0)).to_numpy()
for val,lab,c in [(1,"Hombre",ACCENT),(0,"Mujer","#C55A11")]:
    s=gg==val; ax.scatter(Z2[s,0],Z2[s,1],s=8,alpha=0.45,color=c,label=lab,edgecolors="none")
ax.legend(markerscale=2); style(ax,"PCA de las analíticas — por sexo",xlabel="PC1",ylabel="PC2",grid=False); savefig(fig,"pca_sexo")
fig,ax=plt.subplots(figsize=(8,4.6)); pf=PCA(min(30,len(RAW)),random_state=0).fit(Xs); ev=pf.explained_variance_ratio_*100; cum=np.cumsum(ev)
ax.bar(range(1,len(ev)+1),ev,color=ACCENT,alpha=0.85,edgecolor="white"); ax2=ax.twinx(); ax2.plot(range(1,len(ev)+1),cum,color="#C55A11",marker="o",ms=4,lw=2); ax2.axhline(80,color=NEGC,ls="--",lw=1); ax2.set_ylabel("acumulada (%)"); ax2.grid(False)
ax.set_xlabel("componente principal"); ax.set_ylabel("varianza individual (%)"); style(ax,"Scree plot — varianza explicada por PCA","~19 componentes para el 80 %"); savefig(fig,"scree"); savetab(vardf,"pca_varianza",index=False)
idx=np.random.RandomState(0).choice(len(Xs),min(3000,len(Xs)),replace=False); Zt=TSNE(2,perplexity=30,init="pca",random_state=0).fit_transform(Xs[idx])
fig,ax=plt.subplots(figsize=(7.4,6.4))
for l in PATHO:
    s=dom[idx]==ES[l]; ax.scatter(Zt[s,0],Zt[s,1],s=8,alpha=0.55,color=LCOL[l],label=ES[l],edgecolors="none")
ax.legend(markerscale=2,fontsize=9); style(ax,"t-SNE de las analíticas — por patología dominante","Sin clusters separables → fusión multimodal necesaria",xlabel="t-SNE 1",ylabel="t-SNE 2",grid=False); savefig(fig,"tsne")

# ════════ BLOQUE 10 · ASOCIACIONES + RAW/PCT ════════
spear=pd.DataFrame(np.nan,index=[ES[l] for l in PATHO],columns=[RAWD[i] for i in top12])
for lc in PATHO:
    yb=(train[lc]==1).astype(int)
    for i in top12:
        m=train[RAW[i]].notna()
        if m.sum()>30: spear.loc[ES[lc],RAWD[i]]=stats.spearmanr(yb[m],train.loc[m,RAW[i]])[0]
fig,ax=plt.subplots(figsize=(11,4.6)); sns.heatmap(spear.values.astype(float),annot=True,fmt=".2f",cmap="RdBu_r",center=0,vmin=-0.25,vmax=0.25,cbar_kws={"shrink":0.7,"label":"Spearman rho"},ax=ax,xticklabels=[RAWD[i] for i in top12],yticklabels=[ES[l] for l in PATHO],linewidths=0.5,linecolor="white",annot_kws={"size":8})
ax.set_xticklabels(ax.get_xticklabels(),rotation=30,ha="right"); style(ax,"Correlación de Spearman: analíticas × etiquetas","Cardiomegalia y Edema, las más correladas",grid=False); savefig(fig,"spearman"); savetab(spear.round(3),"spearman")
def cramers_v(a,b):
    ct2=pd.crosstab(a,b); chi2=stats.chi2_contingency(ct2)[0]; nn=ct2.sum().sum(); rr,kk=ct2.shape; return np.sqrt(chi2/(nn*(min(rr,kk)-1)+1e-9))
genrows=[]
fig,ax=plt.subplots(figsize=(8,4.6)); pm=[]; pf2=[]
for l in LABELS:
    yb=(train[l]==1).astype(int); ph=(yb[gg==1]==1).mean()*100; pmf=(yb[gg==0]==1).mean()*100; pm.append(ph); pf2.append(pmf)
    genrows.append({"Hallazgo":ES[l],"% pos Hombre":round(ph,1),"% pos Mujer":round(pmf,1),"V de Cramer":round(cramers_v(gg,yb),3)})
ax.bar(XS-0.19,pm,0.38,label="Hombre",color=ACCENT,edgecolor="white"); ax.bar(XS+0.19,pf2,0.38,label="Mujer",color="#C55A11",edgecolor="white")
xlabels(ax); style(ax,"Prevalencia de positivos por sexo","Diferencias significativas pero V de Cramér ≤ 0,05",ylabel="% positivos"); ax.legend(); savefig(fig,"genero_etiqueta"); savetab(pd.DataFrame(genrows),"genero_etiqueta",index=False)
etn_rows=[]
fig,ax=plt.subplots(figsize=(10,4.8)); w2=0.13
for k,o in enumerate(order):
    sub=train[train["race"]==o]; prev=[(sub[l]==1).mean()*100 for l in PATHO]
    for l,pv in zip(PATHO,prev): etn_rows.append({"Etnia":elab[o],"Hallazgo":ES[l],"% positivos":round(pv,1)})
    ax.bar(np.arange(5)+(k-2.5)*w2,prev,w2,label=elab[o],color=plt.cm.tab10(k))
ax.set_xticks(range(5)); ax.set_xticklabels([ES[l] for l in PATHO],rotation=15,ha="right"); style(ax,"Prevalencia de patologías por grupo étnico","Monitorizar equidad",ylabel="% positivos"); ax.legend(fontsize=8,ncol=2); savefig(fig,"etnia_etiqueta"); savetab(pd.DataFrame(etn_rows),"etnia_etiqueta",index=False)
rp=[]
for c in RAW:
    itemid=c.rsplit("_",1)[1]; pc=[p for p in PCT if p.endswith(itemid)]
    if pc:
        m=train[c].notna()&train[pc[0]].notna()
        if m.sum()>30: rp.append((disp(c),abs(stats.spearmanr(train.loc[m,c],train.loc[m,pc[0]])[0])))
rpdf=pd.DataFrame([{"Analítica":nn,"|Spearman| raw-percentil":round(v,3)} for nn,v in sorted(rp,key=lambda x:-x[1])])
rp25=sorted(rp,key=lambda x:x[1])[:25]
fig,ax=plt.subplots(figsize=(9,5)); ax.barh([nn for nn,_ in rp25],[v for _,v in rp25],color=ACCENT,edgecolor="white"); ax.axvline(0.9,color=NEGC,ls="--",lw=1.2,label="r=0,90"); ax.tick_params(labelsize=8)
style(ax,"Correlación entre valor raw y percentil por analítica","Alta correlación → redundantes (r>0,90 mayoría)",xlabel="|Spearman| raw vs percentil"); ax.legend(); savefig(fig,"raw_vs_percentil"); savetab(rpdf,"raw_vs_percentil",index=False)

# ════════ BLOQUE 11 · IVF ════════
print("  ... IVF (RF + MI + ANOVA), ~1-2 min")
FEAT=["age","gender"]+RAW; FEATN=["Edad","Sexo"]+RAWD
Xf=SimpleImputer(strategy="mean").fit_transform(train[FEAT].apply(lambda c:pd.to_numeric(c,errors="coerce")))
ivf={}
for l in LABELS:
    obs=(train[l]!=-1).values; y=(train.loc[obs,l]==1).astype(int).values; Xl=Xf[obs]   # excluye −1
    if y.sum()<20 or (1-y).sum()<20: continue
    rf=RandomForestClassifier(n_estimators=250,max_depth=6,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1).fit(Xl,y)
    mi=mutual_info_classif(Xl,y,random_state=42); fs=np.nan_to_num(f_classif(Xl,y)[0]); ivf[l]={"rf":rf.feature_importances_,"mi":mi,"f":fs}
glob=np.zeros(len(FEAT)); cnt=0
for l,rr in ivf.items():
    for key in ["rf","mi","f"]: v=rr[key]; v=v/(v.max()+1e-9); glob+=v; cnt+=1
glob/=cnt; topf=np.argsort(glob)[::-1]
ivfdf=pd.DataFrame({"Variable":[FEATN[i] for i in topf],"Importancia media (RF+MI+ANOVA)":np.round(glob[topf],3)})
fig,ax=plt.subplots(figsize=(8.5,6.2)); t15=topf[:15]; cols=["#E1A33A" if FEATN[i] in ["Edad","Sexo"] else ACCENT for i in t15][::-1]
ax.barh([FEATN[i] for i in t15][::-1],[glob[i] for i in t15][::-1],color=cols,edgecolor="white")
style(ax,"IVF — Ranking global de importancia de variables","Media de RF-Gini + Información Mutua + ANOVA F · la Edad domina",xlabel="importancia media normalizada"); savefig(fig,"ivf_ranking"); savetab(ivfdf,"ivf_importancia",index=False)
hm=pd.DataFrame(index=FEATN,columns=[ES[l] for l in ivf])
for l,rr in ivf.items(): v=rr["rf"]; hm[ES[l]]=v/(v.max()+1e-9)
hm=hm.astype(float); hm["m"]=hm.mean(1); hm=hm.sort_values("m",ascending=False).drop(columns="m").head(15)
fig,ax=plt.subplots(figsize=(8,6.5)); sns.heatmap(hm,annot=True,fmt=".2f",cmap="YlOrRd",vmin=0,vmax=1,cbar_kws={"shrink":0.6,"label":"importancia norm."},ax=ax,linewidths=0.4,linecolor="white",annot_kws={"size":8})
ax.set_xticklabels(ax.get_xticklabels(),rotation=20,ha="right"); style(ax,"IVF — Importancia (RF-Gini) por etiqueta","Top-15 features",grid=False); savefig(fig,"ivf_heatmap")

# ════════ TABLA FINAL · DECISIONES PRE-MODELO ════════
dec=pd.DataFrame([
    ["Definicion de negativo","Positivo=1; negativo=0 explicito + NaN->0. El -1 (incierto) se ENMASCARA (fuera de perdida y metrica, U-Ignore). 'Sin hallazgo' NO como negativo."],
    ["Particiones","Oficiales train/val/test (sin solape de pacientes). No re-repartir."],
    ["Desbalanceo","Moderado (1,7:1-6,3:1). pos_weight=n_neg/n_pos por etiqueta (BCE) o scale_pos_weight (arboles). No SMOTE."],
    ["pos_weight (train)","Atelectasia 2,28 / Cardiomegalia 1,85 / Edema 3,43 / Opacidad 2,15 / Sin hallazgo 6,31 / Derrame 1,66"],
    ["Metrica primaria","AUC-PR (Average Precision) por patologia + IC bootstrap. Secundaria: AUC-ROC. No comparar AP entre prevalencias."],
    ["Calibracion","Platt/isotonica en VAL; verificar con Brier score y curva de fiabilidad; recomprobar tras fusion."],
    ["Puntos de operacion","Fijar en VAL: alta sensibilidad (cribado) y alta especificidad (confirmacion). Reportar Se/Sp/VPP/VPN."],
    ["Protocolo","TRAIN entrena; VAL calibra+fusion+umbrales+seleccion; TEST una sola vez. Pre-registrar la regla de decision."],
    ["Fusion","Late fusion binary relevance: meta-LR por patologia sobre probabilidades calibradas. Comparar vs mejor mono."],
    ["Features tabulares","Incluir flags de missingness (MNAR). Priorizar Edad, Urea, Albumina, RDW, hemograma. Raw==percentil (r>0,90): usar una."],
    ["Por modalidad","Imagen aprende Atelectasia/Opacidad; tabular aporta en Edema/Derrame/Cardiomegalia; el tabular solo NO separa diagnosticos."],
    ["Equidad","Evaluar por sexo y etnia (diferencias pequenas pero presentes)."],
], columns=["Decision","Detalle"])
savetab(dec,"DECISIONES_premodelo",index=False)

print(f"\nListo: {NF_[0]} figuras en {FIGD}")
print(f"       {NT_[0]} tablas (CSV) en {TABD}")
