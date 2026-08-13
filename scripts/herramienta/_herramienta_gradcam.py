"""Genera Grad-CAM POR CADA PATOLOGÍA (no solo el foco) para los curados.
Así, al elegir un hallazgo en la app, el mapa de calor corresponde a ESE hallazgo.
Salida: assets/{hid}_gc_{slug}.png  +  curados.json['gradcam_por_patologia'] = {ES: fichero}.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

S="salidas"; OUT=f"{S}/_herramienta"; AST=f"{OUT}/assets"; os.makedirs(AST,exist_ok=True)
BASE="C:/TFM/1.Opción - Symile Mimic/symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0"
NPY=f"{BASE}/data_npy/test"
cxrV=np.load(f"{NPY}/cxr_test.npy",mmap_mode="r")
hids=np.load(f"{NPY}/hadm_id_test.npy"); ROW={}
for k,h in enumerate(hids): ROW.setdefault(int(h),k)

# ES -> objetivo torchxrayvision (Sin hallazgo no tiene mapa)
XRV={"Atelectasia":"Atelectasis","Cardiomegalia":"Cardiomegaly","Edema":"Edema",
     "Opacidad pulmonar":"Lung Opacity","Derrame pleural":"Effusion"}
SLUG={"Atelectasia":"atel","Cardiomegalia":"cardio","Edema":"edema",
      "Opacidad pulmonar":"opac","Derrame pleural":"derrame"}

import torch, torchxrayvision as xrv
model=xrv.models.DenseNet(weights="densenet121-res224-all"); model.eval()
resize=xrv.datasets.XRayResizer(224)
import scipy.ndimage as ndi

def load_img(hid):
    arr=np.asarray(cxrV[ROW[hid]],dtype=np.float32); g=arr[0]
    disp=np.clip(g*0.229+0.485,0,1)
    gg=(g-g.min())/(g.max()-g.min()+1e-8)
    xrvimg=resize(((gg*2-1)*1024.0)[None,...]).astype("float32")
    return xrvimg, disp

def gradcam(t_img, target):
    t=torch.from_numpy(t_img)[None,...].float().requires_grad_(True)
    acts={}
    def fhook(m,i,o): o.retain_grad(); acts["v"]=o
    h=model.features.register_forward_hook(fhook)
    out=model(t); idx=model.pathologies.index(target); model.zero_grad(); out[0,idx].backward()
    h.remove()
    A=acts["v"].detach()[0]; G=acts["v"].grad[0]; w=G.mean((1,2))
    cam=torch.relu((w[:,None,None]*A).sum(0)).numpy(); cam=cam/(cam.max()+1e-8)
    return ndi.zoom(cam,224/cam.shape[0],order=1)

cur=json.load(open(f"{OUT}/curados.json",encoding="utf-8"))
for c in cur:
    hid=c["hadm_id"]
    if hid not in ROW: continue
    timg,disp=load_img(hid); gpp={}
    for es,tgt in XRV.items():
        if tgt not in model.pathologies: continue
        cam=gradcam(timg,tgt)
        fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(disp,cmap="gray"); ax.imshow(cam,cmap="jet",alpha=0.42)
        ax.axis("off"); fn=f"{hid}_gc_{SLUG[es]}.png"
        fig.savefig(f"{AST}/{fn}",bbox_inches="tight",dpi=95,pad_inches=0); plt.close(fig); gpp[es]=fn
    c["gradcam_por_patologia"]=gpp
    print(f"  ✓ {hid} · {len(gpp)} mapas")

json.dump(cur,open(f"{OUT}/curados.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
import shutil; shutil.copy(f"{OUT}/curados.json","../herramienta/public/curados.json")
for f in os.listdir(AST):
    if "_gc_" in f: shutil.copy(f"{AST}/{f}",f"../herramienta/public/assets/{f}")
print("OK · Grad-CAM por patología generados y copiados a public/")
