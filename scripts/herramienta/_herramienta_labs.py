"""Selección de ANALÍTICAS CLAVE por paciente — método documentado.

MÉTODO (acordado con el usuario):
  1. Se identifican los hallazgos CONFIRMADOS por el radiólogo (real==1). Si no hay
     ninguno, se usan los que el sistema marca positivos en cribado.
  2. Para cada hallazgo se toma su conjunto de analíticas CLÍNICAMENTE RELEVANTES
     (mapa FINDING_LABS, con justificación fisiológica por analítica).
  3. Se calcula el valor y su estado (bajo/normal/alto) frente al rango de referencia.
     El rango se AJUSTA POR SEXO en las analíticas que lo requieren (hemoglobina,
     hematocrito, creatinina); el resto usan rango de adulto sexo-neutro. La edad no
     se ajusta: para estas analíticas los rangos de adulto son estables con la edad.
  4. Se ORDENAN priorizando las ANORMALES (mayor desviación fuera de rango primero) y
     se muestran hasta 6, indicando para qué hallazgo son relevantes.
La edad/sexo NO son analíticas: van en la cabecera del paciente, no aquí.
Salida: curados.json['analiticas_clave'] = [{label,valor,unidad,estado,rango,por_que,relevante_para}]
"""
import pandas as pd, json, shutil
S="salidas"; OUT=f"{S}/_herramienta"
cl=pd.read_csv("data/clean/test_clean.csv",sep=";").set_index("hadm_id")
data=json.load(open(f"{OUT}/data.json",encoding="utf-8"))
pMap={p["hadm_id"]:p for p in data["pacientes"]}
PATOLOGIAS=["Cardiomegalia","Edema","Derrame pleural","Atelectasia","Opacidad pulmonar","Sin hallazgo"]

# Rangos de referencia que SÍ dependen del sexo (adulto). El resto son sexo-neutros.
# Fuente: intervalos de referencia estándar de laboratorio de adultos (manuales clínicos).
SEXR={
 "hemoglobin_51222":{"Masculino":(13.5,17.5),"Femenino":(12.0,15.5)},
 "hematocrit_51221":{"Masculino":(40,52),"Femenino":(36,46)},
 "creatinine_50912":{"Masculino":(0.7,1.3),"Femenino":(0.6,1.1)},
}
# catálogo: col -> (label, unidad, (lo,hi) por defecto, por qué importa)
LAB={
 "hemoglobin_51222":("Hemoglobina","g/dL",(12,17),"La anemia (baja) reduce el transporte de oxígeno y agrava la disnea y la insuficiencia cardíaca."),
 "hematocrit_51221":("Hematocrito","%",(36,50),"Proporción de glóbulos rojos; acompaña a la hemoglobina en la anemia."),
 "urea_nitrogen_51006":("Urea (BUN)","mg/dL",(7,20),"Se eleva en la congestión del síndrome cardiorrenal; marcador de sobrecarga de líquidos."),
 "creatinine_50912":("Creatinina","mg/dL",(0.6,1.3),"Función renal; su deterioro se asocia a retención de líquidos (edema, congestión)."),
 "sodium_50983":("Sodio","mmol/L",(135,145),"La hiponatremia aparece en la insuficiencia cardíaca avanzada por retención de agua."),
 "albumin_50862":("Albúmina","g/dL",(3.5,5.0),"La hipoalbuminemia baja la presión oncótica y favorece el edema y el derrame (trasudado)."),
 "wbc_count_51301":("Leucocitos","10⁹/L",(4.5,11),"La leucocitosis sugiere infección/inflamación (p. ej. neumonía detrás de una opacidad)."),
 "neutrophils_pct_51256":("Neutrófilos","%",(40,70),"Su elevación acompaña a las infecciones bacterianas (relevante en opacidad/neumonía)."),
 "lactate_50813":("Lactato","mmol/L",(0.5,2.2),"Sube en la hipoperfusión y la sepsis; señala gravedad en cuadros infecciosos."),
 "potassium_50971":("Potasio","mmol/L",(3.5,5.1),"Sus alteraciones favorecen arritmias (se reflejan en el ECG); frecuente con diuréticos."),
}
FINDING_LABS={
 "Cardiomegalia":["urea_nitrogen_51006","creatinine_50912","sodium_50983","hemoglobin_51222","potassium_50971"],
 "Edema":["urea_nitrogen_51006","creatinine_50912","sodium_50983","albumin_50862","hemoglobin_51222"],
 "Derrame pleural":["albumin_50862","urea_nitrogen_51006","creatinine_50912","hemoglobin_51222"],
 "Opacidad pulmonar":["wbc_count_51301","neutrophils_pct_51256","lactate_50813"],
 "Atelectasia":["wbc_count_51301","hemoglobin_51222"],
 "Sin hallazgo":[],
}
def estado(v,lo,hi): return "bajo" if v<lo else ("alto" if v>hi else "normal")
def desv(v,lo,hi): return (lo-v)/(hi-lo) if v<lo else ((v-hi)/(hi-lo) if v>hi else 0.0)

def compute(p,row):
    if p is None or row is None: return []
    foci=[es for es in PATOLOGIAS if es!="Sin hallazgo" and p["hallazgos"][es]["real"]==1]
    if not foci: foci=[es for es in PATOLOGIAS if es!="Sin hallazgo" and p["hallazgos"][es]["decision"]["cribado"]=="positivo"]
    cols=[]
    for f in foci:
        for col in FINDING_LABS.get(f,[]):
            if col not in cols: cols.append(col)
    sexo=p.get("sexo","")
    items=[]
    for col in cols:
        if col in cl.columns and pd.notna(row[col]):
            label,uni,(dlo,dhi),why=LAB[col]; v=round(float(row[col]),2)
            lo,hi=SEXR.get(col,{}).get(sexo,(dlo,dhi))   # rango ajustado por sexo si procede
            aj=col in SEXR and sexo in SEXR[col]
            rel=[f for f in foci if col in FINDING_LABS.get(f,[])]
            items.append({"label":label,"valor":v,"unidad":uni,"estado":estado(v,lo,hi),
                          "rango":f"{lo}–{hi}","rango_sexo":aj,"por_que":why,
                          "relevante_para":rel,"_d":desv(v,lo,hi)})
    items.sort(key=lambda x:-x["_d"])               # anormales primero (más desviación)
    for it in items: it.pop("_d")
    return items[:6]

# TODOS los pacientes -> data.json (para que la Explicabilidad tenga analíticas en los 464)
n=0
for p in data["pacientes"]:
    hid=p["hadm_id"]; row=cl.loc[hid] if hid in cl.index else None
    p["analiticas_clave"]=compute(p,row); n+=1
json.dump(data,open(f"{OUT}/data.json","w",encoding="utf-8"),ensure_ascii=False)
shutil.copy(f"{OUT}/data.json","../herramienta/public/data.json")

# curados.json (retrocompat: copia las analíticas ya calculadas)
cur=json.load(open(f"{OUT}/curados.json",encoding="utf-8"))
for c in cur: c["analiticas_clave"]=pMap[c["hadm_id"]].get("analiticas_clave",[])
json.dump(cur,open(f"{OUT}/curados.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
shutil.copy(f"{OUT}/curados.json","../herramienta/public/curados.json")
print(f"OK · analíticas para {n} pacientes en data.json + {len(cur)} curados")
print("OK · analíticas por relevancia+anormalidad. Ejemplo (primer curado):")
for L in cur[0]["analiticas_clave"]:
    print(f"  {L['label']:12s} {L['valor']} {L['unidad']:6s} [{L['estado']}]  rel:{','.join(L['relevante_para'])}")
