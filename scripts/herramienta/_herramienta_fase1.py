"""FASE 1 · Activos visuales para un conjunto CURADO de pacientes:
  - Radiografía con Grad-CAM (CheXNet/torchxrayvision) para la patología relevante.
  - Trazado del ECG (12 derivaciones).
  - Analíticas clave.
Salida: salidas/_herramienta/assets/*.png + curados.json (lista + metadatos).
"""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

S="salidas"; OUT=f"{S}/_herramienta"; AST=f"{OUT}/assets"; os.makedirs(AST,exist_ok=True)
BASE="C:/TFM/1.Opción - Symile Mimic/symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0"
LAB=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema","Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgo","Pleural Effusion":"Derrame pleural"}
XRV={"Atelectasis":"Atelectasis","Cardiomegaly":"Cardiomegaly","Edema":"Edema","Lung Opacity":"Lung Opacity","Pleural Effusion":"Effusion"}
data=json.load(open(f"{OUT}/data.json",encoding="utf-8"))
cl=pd.read_csv("data/clean/test_clean.csv",sep=";"); cl_idx={int(h):i for i,h in enumerate(cl["hadm_id"])}
by_hid={p["hadm_id"]:p for p in data["pacientes"]}
# volúmenes preprocesados (imagen y ECG), indexados por hadm_id
NPY=f"{BASE}/data_npy/test"
cxrV=np.load(f"{NPY}/cxr_test.npy",mmap_mode="r"); ecgV=np.load(f"{NPY}/ecg_test.npy",mmap_mode="r")
hids=np.load(f"{NPY}/hadm_id_test.npy"); ROW={}
for k,h in enumerate(hids): ROW.setdefault(int(h),k)   # primera fila de cada paciente

# ── selección CURADA (casos didácticos) ──
def prob(p,l): return p["hallazgos"][ES[l]]["fusion"]
def real(p,l): return p["hallazgos"][ES[l]]["real"]
P=data["pacientes"]
sel=[]
def add(hid,motivo,foco):
    if hid not in [s[0] for s in sel]: sel.append((hid,motivo,foco))
# Derrame alta fiabilidad, positivo real y alta prob
c=sorted([p for p in P if real(p,"Pleural Effusion")==1], key=lambda p:-prob(p,"Pleural Effusion"))
if c: add(c[0]["hadm_id"],"Derrame pleural claro (fiabilidad alta)","Pleural Effusion")
# Cardiomegalia positiva con proyección AP (aviso)
c=[p for p in P if real(p,"Cardiomegaly")==1 and p["proyeccion"]=="AP"]
c=sorted(c,key=lambda p:-prob(p,"Cardiomegaly"))
if c: add(c[0]["hadm_id"],"Cardiomegalia con proyección AP (aviso de magnificación)","Cardiomegaly")
# Edema positivo (complementariedad ECG)
c=sorted([p for p in P if real(p,"Edema")==1],key=lambda p:-prob(p,"Edema"))
if c: add(c[0]["hadm_id"],"Edema (el ECG aporta por el nexo cardíaco)","Edema")
# Falso negativo honesto: real positivo pero fusión baja
c=[p for p in P if real(p,"Pleural Effusion")==1 and prob(p,"Pleural Effusion")<0.3]
if c: add(c[0]["hadm_id"],"Falso negativo honesto (el sistema no lo detecta)","Pleural Effusion")
# Opacidad (baja fiabilidad): caso LIMPIO cribado sí / confirmación no y exclusivo (23561270)
c=[p for p in P if p["hadm_id"]==23561270]
if not c: c=sorted([p for p in P if real(p,"Lung Opacity")==1],key=lambda p:-prob(p,"Lung Opacity"))
if c: add(c[0]["hadm_id"],"Opacidad pulmonar (fiabilidad baja: cribado sí, confirmación no)","Lung Opacity")
# Sano (Sin hallazgo alto): chequeo de coherencia
c=sorted([p for p in P if real(p,"No Finding")==1],key=lambda p:-prob(p,"No Finding"))
if c: add(c[0]["hadm_id"],"Paciente sin hallazgos (chequeo de coherencia)","Pleural Effusion")
# Atelectasia positiva
c=sorted([p for p in P if real(p,"Atelectasis")==1],key=lambda p:-prob(p,"Atelectasis"))
if c: add(c[0]["hadm_id"],"Atelectasia","Atelectasis")
# Dos hallazgos confirmados, acierto en AMBOS modos (didáctico, exclusivo)
c=[p for p in P if p["hadm_id"]==27617935]
if not c: c=sorted(P,key=lambda p:-sum(1 for l in LAB[:-1] if real(p,l)==1))
if c: add(c[0]["hadm_id"],"Dos hallazgos (derrame + atelectasia): acierto en ambos modos","Pleural Effusion")
print(f"Seleccionados {len(sel)} pacientes curados")

# ── modelo Grad-CAM (CheXNet) ──
import torch, torchvision, torchxrayvision as xrv
model=xrv.models.DenseNet(weights="densenet121-res224-all"); model.eval()
resize=xrv.datasets.XRayResizer(224)
def load_img(hid):
    arr=np.asarray(cxrV[ROW[hid]],dtype=np.float32)   # (3,320,320) normalización ImageNet
    g=arr[0]
    disp=np.clip(g*0.229+0.485,0,1)                   # de-normaliza -> [0,1] para mostrar
    gg=(g-g.min())/(g.max()-g.min()+1e-8)
    xrvimg=resize(((gg*2-1)*1024.0)[None,...]).astype("float32")  # (1,224,224) rango torchxrayvision
    return xrvimg, disp
def gradcam(t_img, target):
    t=torch.from_numpy(t_img)[None,...].float().requires_grad_(True)
    acts={}
    def fhook(m,i,o): o.retain_grad(); acts["v"]=o
    h=model.features.register_forward_hook(fhook)
    out=model(t); idx=model.pathologies.index(target); model.zero_grad(); out[0,idx].backward()
    h.remove()
    A=acts["v"].detach()[0]; G=acts["v"].grad[0]; w=G.mean((1,2))
    cam=torch.relu((w[:,None,None]*A).sum(0)).numpy()
    cam=cam/(cam.max()+1e-8)
    import scipy.ndimage as ndi
    cam=ndi.zoom(cam, 224/cam.shape[0], order=1)
    return cam, float(torch.sigmoid(out[0,idx]))

curados=[]
for hid,motivo,foco in sel:
    p=by_hid[hid]; i=cl_idx[hid]; assets={}
    try:
        timg,disp=load_img(hid)
        # Grad-CAM de la patología foco (si xrv la tiene)
        if foco in XRV and XRV[foco] in model.pathologies:
            cam,_=gradcam(timg, XRV[foco])
            fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(disp,cmap="gray"); ax.imshow(cam,cmap="jet",alpha=0.42)
            ax.set_title(f"{ES[foco]} — dónde mira el modelo",fontsize=11); ax.axis("off")
            fn=f"{hid}_gradcam.png"; fig.savefig(f"{AST}/{fn}",bbox_inches="tight",dpi=95); plt.close(fig); assets["gradcam"]=fn
        # Radiografía sola
        fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(disp,cmap="gray"); ax.axis("off"); ax.set_title("Radiografía",fontsize=11)
        fn=f"{hid}_rx.png"; fig.savefig(f"{AST}/{fn}",bbox_inches="tight",dpi=95); plt.close(fig); assets["rx"]=fn
    except Exception as e:
        print(f"  ⚠ Rx/GradCAM {hid}: {e}")
    # ECG (12 derivaciones) desde el prep
    try:
        sig=np.asarray(ecgV[ROW[hid]])[0].T   # (12,5000): 12 derivaciones × 5000 muestras
        leads=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        fig,axes=plt.subplots(6,2,figsize=(9,7),sharex=True)
        for k in range(12):
            a=axes[k%6,k//6]; a.plot(sig[k],lw=0.6,color="#1F3864"); a.set_ylabel(leads[k],fontsize=8,rotation=0,labelpad=12); a.set_yticks([]); a.grid(alpha=.2)
        fig.suptitle("Electrocardiograma (12 derivaciones)",fontsize=11); fig.tight_layout()
        fn=f"{hid}_ecg.png"; fig.savefig(f"{AST}/{fn}",bbox_inches="tight",dpi=90); plt.close(fig); assets["ecg"]=fn
    except Exception as e:
        print(f"  ⚠ ECG {hid}: {e}")
    # Analíticas clave
    keylabs=["age","Urea Nitrogen","Creatinine","Albumin","RDW","Hemoglobin","Lymphocytes Pct","Neutrophils Pct"]
    labs={c:(None if pd.isna(cl[c].iloc[i]) else round(float(cl[c].iloc[i]),2)) for c in keylabs if c in cl.columns}
    curados.append({"hadm_id":hid,"motivo":motivo,"foco":ES[foco],"assets":assets,"analiticas_clave":labs})
    print(f"  ✓ {hid} · {motivo} · assets={list(assets)}")

json.dump(curados,open(f"{OUT}/curados.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
# copiar todos los assets (rx/ecg/gradcam) a la carpeta public de la app
import shutil
PUB="../herramienta/public/assets"
os.makedirs(PUB,exist_ok=True)
for f in os.listdir(AST):
    if f.endswith(".png"): shutil.copy(f"{AST}/{f}",f"{PUB}/{f}")
shutil.copy(f"{OUT}/curados.json","../herramienta/public/curados.json")
print(f"\nOK · {len(curados)} curados · assets en {AST} + copiados a public/")
