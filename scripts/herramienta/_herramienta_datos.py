"""FASE 0 · Contrato de datos de la herramienta clínica.
Precomputa salidas/_herramienta/data.json con TODO lo que la web necesita:
  - meta: modelo final (fusión v3), rendimiento, límites, fiabilidad y umbrales por patología/modo.
  - pacientes: los 464 del test con, por hallazgo, probs por modalidad, fusión, pesos, decisión y etiqueta real.
Pipeline congelado: modelos=train, calibración/fusión/umbrales=val, test solo se reporta.
"""
import pandas as pd, numpy as np, json, os
S="salidas"; OUT=f"{S}/_herramienta"; os.makedirs(OUT,exist_ok=True)
LAB=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
LU=[l.replace(" ","_") for l in LAB]
ES={"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema",
    "Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgo","Pleural Effusion":"Derrame pleural"}
fl=lambda s:float(str(s).replace(",",".").replace("+","")) if str(s) not in ("—","","nan") else None

# ── modelo final: pesos v3 ──
V3=json.load(open(f"{S}/04_stacking/v3/veredicto_v3.json")); W=V3["pesos_opt"]
PAR=json.load(open(f"{S}/04_stacking/veredicto_pareado.json"))

# ── datos del test ──
cl=pd.read_csv("data/clean/test_clean.csv",sep=";")
bp=pd.read_csv(f"{S}/04_stacking/v1/base_preds_test.csv").set_index("hadm_id").loc[cl["hadm_id"]].reset_index()
Pm={m:bp[[f"{m}_{u}" for u in LU]].to_numpy(float) for m in ["CXR","ECG","LABS"]}
Pf=np.zeros((len(cl),6))
for j,l in enumerate(LAB):
    w=W[l]; Pf[:,j]=w[0]*Pm["CXR"][:,j]+w[1]*Pm["ECG"][:,j]+w[2]*Pm["LABS"][:,j]
raw=cl[LAB].to_numpy(float); Yreal=(raw==1); Mreal=(raw!=-1)

# ── fiabilidad por patología (tabla 18) ──
t18=pd.read_csv(f"{S}/_informe_modelos/tablas/18_sintesis_por_patologia.csv")
FIAB={r["Hallazgo"]:r["Fiabilidad en la herramienta"] for _,r in t18.iterrows()}

# ── umbrales + Se/Sp/VPP/VPN por patología y modo (tabla 27, fusión v3) ──
t27=pd.read_csv(f"{S}/_informe_modelos/tablas/27_puntos_fusion_v3.csv")
PUNTO={"f1":"equilibrio","cribado_Se>=0.90":"cribado","confirmacion_Sp>=0.90":"confirmacion"}
umbrales={}
for _,r in t27.iterrows():
    hall=r["Hallazgo"]; modo=PUNTO.get(r["Punto"],r["Punto"])
    umbrales.setdefault(hall,{})[modo]={"umbral":fl(r["Umbral"]),"Se":fl(r["Sensib."]),
        "Sp":fl(r["Especif."]),"VPP":fl(r["VPP"]),"VPN":fl(r["VPN"])}

# ── META ──
meta={
 "modelo":"Fusión multimodal v3 — media ponderada por patología (imagen + ECG + analíticas)",
 "n_test":int(len(cl)),
 "rendimiento":{"auc_pr_macro":PAR["pareado"]["ap_fus_v3"],"vs_imagen":PAR["pareado"]["diff"],
    "ic_diferencia":PAR["pareado"]["ic_diff"],"significativa":PAR["pareado"]["excluye_cero"],
    "patologias_mejoran":f'{PAR["pareado"]["n_patologias_mejoran"]}/6'},
 "pesos_por_patologia":{ES[l]:{"imagen":round(W[l][0]*100),"ecg":round(W[l][1]*100),"analiticas":round(W[l][2]*100)} for l in LAB},
 "fiabilidad_por_patologia":{ES[l]:FIAB.get(ES[l],"—") for l in LAB},
 "umbrales_por_patologia":umbrales,
 "avisos":{
   "Cardiomegalia":"Si la radiografía es AP (portátil), la silueta cardíaca puede verse magnificada; valorar con cautela.",
   "Opacidad pulmonar":"Fiabilidad limitada: es el hallazgo más difícil del sistema; usar preferentemente en modo cribado.",
   "Sin hallazgo":"No es un diagnóstico, sino un chequeo de coherencia: si sube, deberían bajar las patologías."},
 "limitaciones":[
   "Las etiquetas provienen del informe radiológico; el ECG y las analíticas ANTICIPAN hallazgos radiográficos, no tienen verdad propia.",
   "La mejora de la fusión sobre la imagen es estadísticamente significativa pero MODESTA (+0,038) y con solo 464 pacientes de test.",
   "Herramienta de APOYO, TRIAJE y PRIORIZACIÓN — nunca decisor autónomo. Su fortaleza es descartar con confianza (VPN alto).",
   "No debe subirse la probabilidad por el mero hecho de que existan analíticas.",
   "Pipeline congelado: modelos (train), calibración/fusión/umbrales (validación); el test solo se reportó una vez."],
}

# ── PACIENTES ──
pacientes=[]
for i in range(len(cl)):
    hall={}
    for j,l in enumerate(LAB):
        nomes=ES[l]; pf=float(Pf[i,j]); um=umbrales.get(nomes,{})
        def dec(modo):
            u=um.get(modo,{}).get("umbral");
            return None if u is None else ("positivo" if pf>=u else "negativo")
        hall[nomes]={
            "imagen":round(float(Pm["CXR"][i,j]),3),"ecg":round(float(Pm["ECG"][i,j]),3),
            "analiticas":round(float(Pm["LABS"][i,j]),3),"fusion":round(pf,3),
            "pesos":meta["pesos_por_patologia"][nomes],"fiabilidad":FIAB.get(nomes,"—"),
            "decision":{"cribado":dec("cribado"),"confirmacion":dec("confirmacion")},
            "real":(int(Yreal[i,j]) if Mreal[i,j] else None)}
    pacientes.append({"hadm_id":int(cl["hadm_id"].iloc[i]),
        "edad":int(cl["age"].iloc[i]) if pd.notna(cl["age"].iloc[i]) else None,
        "sexo":("Femenino" if cl["gender"].iloc[i]==0 else "Masculino"),
        "proyeccion":str(cl["cxr_view"].iloc[i]),"hallazgos":hall})

json.dump({"meta":meta,"pacientes":pacientes},open(f"{OUT}/data.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
sz=os.path.getsize(f"{OUT}/data.json")/1024
print(f"OK · data.json · {len(pacientes)} pacientes · {sz:.0f} KB")
print("Ejemplo paciente[0]:",json.dumps(pacientes[0]["hallazgos"]["Derrame pleural"],ensure_ascii=False))
