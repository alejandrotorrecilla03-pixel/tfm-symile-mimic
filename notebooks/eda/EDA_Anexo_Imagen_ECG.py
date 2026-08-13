# -*- coding: utf-8 -*-
"""
ANEXO AL EDA — Señal cruda de IMAGEN (CXR) y ECG
=================================================
El EDA principal (EDA_Symile_MIMIC.py) es 100 % tabular (etiquetas, demografía, labs).
Este anexo caracteriza las DOS modalidades de señal que el EDA no tocaba, directamente
sobre los tensores preprocesados que consumen los backbones:

  CXR : data_npy/<split>/cxr_<split>.npy   → (N, 3, 320, 320) float32  (normalización ImageNet)
  ECG : data_npy/<split>/ecg_<split>.npy   → (N, 1, 5000, 12) float32  (12 derivaciones, ~10 s @500 Hz)

Objetivo: justificar decisiones de PREPROCESADO por modalidad y detectar problemas de
calidad (imágenes degeneradas/en blanco, derivaciones planas, NaN) antes de reentrenar.
Trabaja por MUESTREO vía mmap (equipo CPU-only): nunca carga los 12 GB de CXR en RAM.

Salida:  salidas/00_eda/anexo_img_ecg/figuras/*.png  y  .../tablas/*.csv
Ejecutar: python notebooks/EDA_Anexo_Imagen_ECG.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns

# ───────── RUTAS ─────────
BASE = Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
NPY  = BASE / "data_npy"
CSV  = BASE / "data_csv" / "clean"
ROOT = Path(r"C:\TFM\1.Opción - Symile Mimic\tfm_multimodal_clinico\salidas\00_eda\anexo_img_ecg")
FIGD = ROOT / "figuras"; TABD = ROOT / "tablas"
FIGD.mkdir(parents=True, exist_ok=True); TABD.mkdir(parents=True, exist_ok=True)

# ───────── SISTEMA VISUAL (idéntico al EDA principal) ─────────
INK, ACCENT, GRID, SUB = "#1F2A44", "#2E5496", "#D9DEE8", "#5A6678"
POSC, NEGC = "#2E5496", "#C44E52"
sns.set_theme(style="white")
mpl.rcParams.update({
    "figure.dpi":120,"savefig.dpi":160,"savefig.bbox":"tight","font.family":"DejaVu Sans","font.size":12,
    "axes.titlesize":15,"axes.titleweight":"bold","axes.titlecolor":INK,"axes.labelsize":12,"axes.labelcolor":INK,
    "axes.edgecolor":"#7A8499","axes.linewidth":1.0,"xtick.color":INK,"ytick.color":INK,"text.color":INK,
    "legend.fontsize":10,"legend.frameon":True,"legend.edgecolor":GRID,"legend.framealpha":0.95,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True})
LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema","Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgo","Pleural Effusion":"Derrame pleural"}
LCOL={"Atelectasis":"#4C72B0","Cardiomegaly":"#DD8452","Edema":"#55A868","Lung Opacity":"#C44E52","No Finding":"#8C8C8C","Pleural Effusion":"#937DB8"}

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
    NF_[0]+=1; fig.savefig(FIGD/f"{NF_[0]:02d}_{name}.png"); plt.close(fig); print(f"  fig  {NF_[0]:02d}_{name}")
def savetab(df,name,index=True):
    NT_[0]+=1; df.to_csv(TABD/f"{NT_[0]:02d}_{name}.csv",index=index,encoding="utf-8-sig"); print(f"  tab  {NT_[0]:02d}_{name}")

# Parámetros de muestreo (CPU-only): subconjuntos suficientes para caracterizar
RNG = np.random.RandomState(0)
N_CXR_STATS = 1500      # imágenes muestreadas para estadísticos de intensidad/calidad
N_ECG_STATS = 2000      # ECG muestreados para estadísticos por derivación/calidad
IMAGENET_MEAN = np.array([0.485,0.456,0.406]); IMAGENET_STD = np.array([0.229,0.224,0.225])
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]  # orden estándar 12 derivaciones

print("ANEXO EDA (imagen + ECG) → figuras y tablas en", ROOT)

# ───────── CARGA (mmap + etiquetas alineadas por orden de fila) ─────────
df = pd.read_csv(CSV/"train_clean.csv", sep=";")
cxr = np.load(NPY/"train"/"cxr_train.npy", mmap_mode="r")   # (N,3,320,320)
ecg = np.load(NPY/"train"/"ecg_train.npy", mmap_mode="r")   # (N,1,5000,12)
N = cxr.shape[0]
assert len(df)==N, f"desalineación csv({len(df)}) vs npy({N})"
def POS(l): return (df[l]==1).to_numpy()

# ════════════════════════════════════════════════════════════════════════
#  PARTE A · RADIOGRAFÍA (CXR)
# ════════════════════════════════════════════════════════════════════════
idx_c = np.sort(RNG.choice(N, min(N_CXR_STATS,N), replace=False))
# Acumuladores incrementales (no se guardan todos los píxeles)
BINS = np.linspace(-2.8, 2.8, 113); hist_ch = [np.zeros(len(BINS)-1) for _ in range(3)]
img_mean = np.zeros(len(idx_c)); img_std = np.zeros(len(idx_c)); img_min=np.zeros(len(idx_c)); img_max=np.zeros(len(idx_c))
ch_stats = np.zeros((3,4))  # mean,std,min,max por canal (promediado sobre muestra)
nan_imgs = 0
for k,i in enumerate(idx_c):
    im = np.asarray(cxr[i]).astype(np.float32)           # (3,320,320)
    if np.isnan(im).any(): nan_imgs += 1
    img_mean[k]=im.mean(); img_std[k]=im.std(); img_min[k]=im.min(); img_max[k]=im.max()
    for c in range(3):
        h,_=np.histogram(im[c].ravel(),bins=BINS); hist_ch[c]+=h
        ch_stats[c,0]+=im[c].mean(); ch_stats[c,1]+=im[c].std(); ch_stats[c,2]+=im[c].min(); ch_stats[c,3]+=im[c].max()
ch_stats/=len(idx_c)
centers=(BINS[:-1]+BINS[1:])/2

# A1 · Histograma de intensidad por canal (confirma normalización ImageNet)
fig,ax=plt.subplots(figsize=(8.4,4.8))
for c,col,nm in zip(range(3),["#C44E52","#55A868","#4C72B0"],["Canal R","Canal G","Canal B"]):
    ax.plot(centers,hist_ch[c]/hist_ch[c].sum(),lw=2,color=col,label=f"{nm} (μ={ch_stats[c,0]:+.2f})")
ax.axvline(0,color="#7A8499",ls="--",lw=1)
style(ax,"Distribución de intensidad del CXR por canal",
      "Normalización ImageNet: media≈0, mínimo≈−2,12 · listo para DenseNet-121",
      xlabel="valor de píxel normalizado",ylabel="densidad"); ax.legend()
savefig(fig,"cxr_intensidad_canal")
savetab(pd.DataFrame(ch_stats,columns=["media","std","min","max"],index=["R","G","B"]).round(3),"cxr_stats_canal")

# A2 · Media de intensidad por imagen (detección de imágenes atípicas/en blanco)
fig,ax=plt.subplots(figsize=(8,4.6))
ax.hist(img_mean,bins=40,color=ACCENT,edgecolor="white")
lo,hi=np.percentile(img_mean,[1,99])
ax.axvline(lo,color=NEGC,ls="--",lw=1.2,label=f"P1={lo:.2f}"); ax.axvline(hi,color=NEGC,ls="--",lw=1.2,label=f"P99={hi:.2f}")
style(ax,"Media de intensidad por imagen (muestra CXR)",
      f"n={len(idx_c)} · variación de brillo entre estudios",xlabel="media de píxel (normalizado)",ylabel="nº de imágenes"); ax.legend()
savefig(fig,"cxr_media_por_imagen")

# A3 · Detección de imágenes degeneradas (std anormalmente bajo = casi constante/en blanco)
thr = 0.15  # std normalizado por debajo del cual la imagen apenas tiene contraste
degen = int((img_std<thr).sum()); pct_degen=100*degen/len(idx_c)
fig,ax=plt.subplots(figsize=(8,4.6))
ax.hist(img_std,bins=40,color="#55A868",edgecolor="white")
ax.axvline(thr,color=NEGC,ls="--",lw=1.4,label=f"umbral degenerado={thr}")
style(ax,"Contraste por imagen (desviación típica de píxel)",
      f"Imágenes casi sin contraste: {degen}/{len(idx_c)} ({pct_degen:.1f}%) → posible .npy corrupto",
      xlabel="std de píxel (normalizado)",ylabel="nº de imágenes"); ax.legend()
savefig(fig,"cxr_contraste_degenerado")

# A4 · Montage de ejemplos por hallazgo (de-normalizado a escala de grises)
def denorm(im):  # (3,H,W) normalizado → grises 0..1
    d=np.stack([im[c]*IMAGENET_STD[c]+IMAGENET_MEAN[c] for c in range(3)],0)
    return np.clip(d.mean(0),0,1)
fig,axes=plt.subplots(len(LABELS),4,figsize=(9.5,13.5))
for r,l in enumerate(LABELS):
    pos=np.where(POS(l))[0][:4]
    for cix in range(4):
        ax=axes[r,cix]; ax.axis("off")
        if cix<len(pos):
            ax.imshow(denorm(np.asarray(cxr[pos[cix]]).astype(np.float32)),cmap="gray",vmin=0,vmax=1)
        if cix==0: ax.set_title(ES[l],loc="left",fontsize=11,color=LCOL[l],fontweight="bold",pad=4)
fig.suptitle("Ejemplos de radiografía por hallazgo (train)",fontsize=15,fontweight="bold",color=INK,y=0.995)
fig.text(0.5,0.975,"4 estudios positivos por etiqueta · de-normalizados para visualización",ha="center",fontsize=10.5,color=SUB,style="italic")
fig.tight_layout(rect=[0,0,1,0.965]); savefig(fig,"cxr_montage_por_hallazgo")

# A5 · Distribución de la proyección (cxr_view) y su relación con prevalencia
if "cxr_view" in df.columns:
    vc=df["cxr_view"].fillna("Desconocida").value_counts()
    fig,ax=plt.subplots(figsize=(7.6,4.4))
    b=ax.bar([str(x) for x in vc.index],vc.values,color=ACCENT,edgecolor="white",width=0.6)
    for r,v in zip(b,vc.values): ax.text(r.get_x()+r.get_width()/2,v+40,f"{v:,}",ha="center",fontweight="bold",fontsize=9)
    ax.set_ylim(0,vc.values.max()*1.15)
    style(ax,"Proyección radiográfica (cxr_view)","AP domina (paciente encamado) · condiciona la apariencia de cardiomegalia",ylabel="nº de estudios")
    savefig(fig,"cxr_view_dist")
    vrows=[]
    for v in vc.index:
        sub=df[df["cxr_view"].fillna("Desconocida")==v]
        row={"Proyección":str(v),"n":len(sub)}
        for l in LABELS: row[ES[l]]=round((sub[l]==1).mean()*100,1)
        vrows.append(row)
    savetab(pd.DataFrame(vrows),"cxr_view_prevalencia",index=False)

# ════════════════════════════════════════════════════════════════════════
#  PARTE B · ELECTROCARDIOGRAMA (ECG)
# ════════════════════════════════════════════════════════════════════════
idx_e = np.sort(RNG.choice(N, min(N_ECG_STATS,N), replace=False))
lead_amp=[[] for _ in range(12)]       # amplitud (percentiles) por derivación
lead_std=np.zeros((len(idx_e),12)); nan_ecg=0; flat_any=np.zeros(len(idx_e))
for k,i in enumerate(idx_e):
    e=np.asarray(ecg[i,0]).astype(np.float32)   # (5000,12)
    if np.isnan(e).any(): nan_ecg+=1; e=np.nan_to_num(e)
    s=e.std(0); lead_std[k]=s; flat_any[k]=int((s<1e-4).any())
    for c in range(12): lead_amp[c].append(e[:,c])
# muestra reducida de amplitudes para boxplot (concat de un subconjunto)
amp_sample=[np.concatenate(a[:200]) for a in lead_amp]

# B1 · Amplitud por derivación (boxplot 12 derivaciones)
fig,ax=plt.subplots(figsize=(10.5,4.8))
bp=ax.boxplot(amp_sample,patch_artist=True,showfliers=False,widths=0.6,medianprops={"color":INK,"lw":1.6})
for box in bp["boxes"]: box.set_facecolor("#A9C0E0")
ax.set_xticks(range(1,13)); ax.set_xticklabels(LEADS)
style(ax,"Amplitud de señal por derivación ECG (muestra)","Valores normalizados a [−1, 1] · derivaciones precordiales (V) más amplias",xlabel="derivación",ylabel="amplitud normalizada")
savefig(fig,"ecg_amplitud_derivacion")

# B2 · Amplitud media (std) por derivación — ORDENADA de mayor a menor
# El objetivo es que se VEA cuáles son las derivaciones de mayor variabilidad. Se ordenan de mayor a
# menor y se colorean por magnitud (más oscuro = más std). El % de señales planas va en TEXTO, no en
# el color (antes las precordiales V2-V4, las de MÁS std, salían naranjas por algún registro plano
# aislado, lo que contradecía el propósito del gráfico).
mean_std=lead_std.mean(0); frac_flat=(lead_std<1e-4).mean(0)*100
orden=np.argsort(mean_std)[::-1]                       # de mayor a menor std
labs_ord=[LEADS[i] for i in orden]; std_ord=mean_std[orden]
import matplotlib.cm as _cm
norm=(std_ord-std_ord.min())/(std_ord.max()-std_ord.min()+1e-9)
colors=_cm.get_cmap("Blues")(0.35+0.6*norm)            # gradiente: barra más oscura = más variabilidad
fig,ax=plt.subplots(figsize=(10.5,4.6))
b=ax.bar(range(12),std_ord,color=colors,edgecolor="white",width=0.66)
for r,v in zip(b,std_ord): ax.text(r.get_x()+r.get_width()/2,v+0.002,f"{v:.3f}",ha="center",fontsize=8,color=INK)
ax.set_xticks(range(12)); ax.set_xticklabels(labs_ord)
n_flat=int((frac_flat>0).sum())
style(ax,"Amplitud media de la señal por derivación (ordenada de mayor a menor)",
      "Barra más oscura = mayor variabilidad · las precordiales V2–V4 (más cercanas al corazón) dominan",
      xlabel="derivación (ordenadas por std)",ylabel="std medio")
savefig(fig,"ecg_std_derivacion")
savetab(pd.DataFrame({"Derivación":LEADS,"std medio":np.round(mean_std,4),"% señales planas":np.round(frac_flat,2)})
        .sort_values("std medio",ascending=False),"ecg_stats_derivacion",index=False)

# B3 · Calidad global: NaN y señales con alguna derivación plana
qual=pd.DataFrame([
    {"Métrica":"ECG muestreados","Valor":len(idx_e)},
    {"Métrica":"Con algún NaN","Valor":int(nan_ecg)},
    {"Métrica":"Con alguna derivación plana (std<1e-4)","Valor":int(flat_any.sum())},
    {"Métrica":"% con derivación plana","Valor":round(100*flat_any.mean(),2)},
    {"Métrica":"Longitud (muestras)","Valor":ecg.shape[2]},
    {"Métrica":"Derivaciones","Valor":ecg.shape[3]},
])
savetab(qual,"ecg_calidad",index=False)

# B4 · Trazado 12-derivaciones de un caso (ejemplo representativo)
i0=int(idx_e[0]); e0=np.asarray(ecg[i0,0]).astype(np.float32)
t=np.arange(e0.shape[0])/500.0   # 500 Hz → segundos
fig,axes=plt.subplots(6,2,figsize=(12,9),sharex=True)
for c in range(12):
    ax=axes[c%6,c//6]; ax.plot(t,e0[:,c],lw=0.6,color=INK); ax.set_ylabel(LEADS[c],rotation=0,ha="right",va="center",fontsize=9)
    ax.grid(True,color=GRID,lw=0.5); ax.tick_params(length=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
for c in (5,11): axes[c%6,c//6].set_xlabel("tiempo (s)")
fig.suptitle("Trazado ECG de 12 derivaciones — ejemplo (train)",fontsize=14,fontweight="bold",color=INK,y=0.995)
fig.tight_layout(rect=[0,0,1,0.97]); savefig(fig,"ecg_trazado_12deriv")

# B5 · Derivación II: caso con patología vs sin patología (misma escala)
il_pos=np.where(POS("Pleural Effusion"))[0]; il_nf=np.where(POS("No Finding"))[0]
if len(il_pos) and len(il_nf):
    ep=np.asarray(ecg[int(il_pos[0]),0,:, 1]).astype(np.float32); en=np.asarray(ecg[int(il_nf[0]),0,:, 1]).astype(np.float32)
    fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True,sharey=True)
    axes[0].plot(t,ep,lw=0.7,color=POSC); axes[0].set_title("Derrame pleural (positivo)",loc="left",fontsize=11,color=POSC)
    axes[1].plot(t,en,lw=0.7,color="#8C8C8C"); axes[1].set_title("Sin hallazgo",loc="left",fontsize=11,color=SUB)
    for ax in axes:
        ax.grid(True,color=GRID,lw=0.5); ax.tick_params(length=0)
        for s in ("top","right"): ax.spines[s].set_visible(False)
    axes[1].set_xlabel("tiempo (s)"); axes[0].set_ylabel("deriv. II"); axes[1].set_ylabel("deriv. II")
    fig.suptitle("Derivación II — comparación de un caso por estado",fontsize=13,fontweight="bold",color=INK)
    fig.tight_layout(); savefig(fig,"ecg_deriv_ii_comparacion")

# ───────── RESUMEN DEL ANEXO ─────────
resumen=pd.DataFrame([
    ["CXR forma",str(cxr.shape)],
    ["CXR normalización","ImageNet (media 0, min≈−2,12); 3 canales afines a partir de escala de grises"],
    ["CXR muestreado (stats)",str(len(idx_c))],
    ["CXR imágenes degeneradas (std<0,15)",f"{degen} ({pct_degen:.1f}%)"],
    ["CXR con NaN",str(nan_imgs)],
    ["ECG forma",str(ecg.shape)],
    ["ECG normalización","[−1, 1]; 12 derivaciones; 5000 muestras (~10 s @500 Hz)"],
    ["ECG muestreado (stats)",str(len(idx_e))],
    ["ECG con NaN",str(int(nan_ecg))],
    ["ECG con derivación plana",f"{int(flat_any.sum())} ({100*flat_any.mean():.2f}%)"],
], columns=["Aspecto","Valor"])
savetab(resumen,"RESUMEN_anexo",index=False)

print(f"\nListo: {NF_[0]} figuras en {FIGD}")
print(f"       {NT_[0]} tablas (CSV) en {TABD}")
