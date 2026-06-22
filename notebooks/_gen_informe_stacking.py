# -*- coding: utf-8 -*-
# Regenera el informe de stacking con resultados reales (v1/v2/v3), figuras y conclusiones.
import json, csv
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

NBO=Path(r"C:\TFM\1.Opción - Symile Mimic\tfm_multimodal_clinico\notebooks")
DOC=Path(r"C:\TFM\1.Opción - Symile Mimic\DOCUMENTACIÓN\Documentación Informe Stacking Multimodal.docx")
V2F=NBO/"outputs_stacking_v2"/"figuras"; FAF=NBO/"outputs_fusion_avanzada"/"figuras"

BLUE=RGBColor(0x1F,0x4E,0x79); LBLUE=RGBColor(0x2E,0x75,0xB6); GREEN=RGBColor(0x37,0x56,0x23); GRAY=RGBColor(0x59,0x59,0x59)
def jload(p): return json.load(open(p,encoding="utf-8"))
def cload(p):
    with open(p,encoding="utf-8") as f: return list(csv.DictReader(f))
sumf=jload(NBO/"outputs_fusion_avanzada"/"summary_fusion.json")
v1=cload(NBO/"outputs_stacking"/"comparativa_enfoques.csv")
v2=cload(NBO/"outputs_stacking_v2"/"comparativa_enfoques.csv")
v3=cload(NBO/"outputs_fusion_avanzada"/"comparativa_fusion.csv")
v4=cload(NBO/"outputs_stacking_v4"/"comparativa_fusion_v4.csv")
sum4=jload(NBO/"outputs_stacking_v4"/"summary_v4.json")
V4F=NBO/"outputs_stacking_v4"/"figuras"
LABELS=["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]

doc=Document()
st=doc.styles["Normal"]; st.font.name="Calibri"; st.font.size=Pt(10.5)

def shade(cell,hexc):
    sh=OxmlElement("w:shd"); sh.set(qn("w:val"),"clear"); sh.set(qn("w:fill"),hexc)
    cell._tc.get_or_add_tcPr().append(sh)
def H(txt,size=15,color=BLUE,before=14,after=6):
    p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(before); p.paragraph_format.space_after=Pt(after)
    r=p.add_run(txt); r.bold=True; r.font.size=Pt(size); r.font.color.rgb=color; return p
def P(txt,size=10.5,it=False,sp=4):
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(sp)
    r=p.add_run(txt); r.font.size=Pt(size); r.italic=it; return p
def bullet(txt):
    p=doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after=Pt(2); p.add_run(txt)
def img(path,width=15.5,cap=None):
    if not Path(path).exists(): P(f"[falta figura: {Path(path).name}]",it=True); return
    doc.add_picture(str(path),width=Cm(width)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    if cap:
        c=doc.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
        r=c.add_run(cap); r.italic=True; r.font.size=Pt(9); r.font.color.rgb=GRAY
def table(headers,rows,widths=None,best_row=None,small=False):
    t=doc.add_table(rows=1,cols=len(headers)); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    fs=Pt(8.5) if small else Pt(9.5)
    hc=t.rows[0].cells
    for i,h in enumerate(headers):
        shade(hc[i],"1F4E79"); pr=hc[i].paragraphs[0]; pr.alignment=WD_ALIGN_PARAGRAPH.CENTER
        run=pr.add_run(h); run.bold=True; run.font.color.rgb=RGBColor(0xFF,0xFF,0xFF); run.font.size=fs
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,v in enumerate(row):
            pr=cells[i].paragraphs[0]; pr.alignment=WD_ALIGN_PARAGRAPH.CENTER if i>0 else WD_ALIGN_PARAGRAPH.LEFT
            run=pr.add_run(str(v)); run.font.size=fs
            if best_row is not None and ri==best_row: run.bold=True
            if best_row is not None and ri==best_row: shade(cells[i],"E2EFDA")
    return t

# ===== PORTADA =====
ti=doc.add_paragraph(); ti.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=ti.add_run("Informe — Stacking y Fusión Multimodal"); r.bold=True; r.font.size=Pt(20); r.font.color.rgb=BLUE
sb=doc.add_paragraph(); sb.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=sb.add_run("CXR + ECG + Laboratorio · Fusión tardía multietiqueta · CPU"); r.font.size=Pt(12); r.font.color.rgb=LBLUE
sb2=doc.add_paragraph(); sb2.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=sb2.add_run("TFM · Sistema de Apoyo a la Decisión Clínica Multimodal · Universidad de Salamanca"); r.font.size=Pt(10); r.font.color.rgb=GRAY

# ===== 1. MÉTODO =====
H("1. Objetivo y método")
P("Se combinan las predicciones de los tres modelos por modalidad —imagen (CXR, DenseNet121 v5), "
  "electrocardiograma (ECG ResNet1D v3.1) y analíticas de sangre (LABS, blend v2.2)— mediante fusión tardía. "
  "Un meta-modelo aprende cuánto pesa cada modalidad para cada hallazgo. Se respetan todas las reglas de las modalidades: "
  "masking de NaN/−1, negativos derivados por exclusividad de No Finding, desequilibrio por etiqueta y AUC-ROC macro.")
P("Rigor anti-fuga: las predicciones de entrenamiento son out-of-fold (cada muestra la predice un modelo que no la vio); "
  "el test queda 100 % limpio; el umbral de decisión se fija por F1 en validación. Métrica principal: macro AUC de las "
  "5 patologías (No Finding se reporta aparte). Tres iteraciones del notebook de fusión:")
bullet("v1 — regenera ECG/LABS por dentro del propio stacking (lento, ~2 h).")
bullet("v2 — cada notebook de modalidad exporta su OOF/val/test; el stacking solo los carga (segundos).")
bullet("v3 — añade técnicas de fusión avanzadas sobre las mismas predicciones base.")

# ===== 2. TECHO DE LAS MODALIDADES =====
H("2. Techo de cada modalidad (AUC en solitario, test)")
P("Antes de fusionar conviene ver qué aporta cada fuente por separado. El CXR domina con diferencia; "
  "el ECG es la modalidad más débil y las analíticas quedan en un punto intermedio.")
solo=sumf["solo"]
table(["Modalidad","macro AUC (patologías)","Lectura"],
      [["CXR (imagen)",f"{solo['CXR']:.4f}","Modalidad fuerte; marca el techo del sistema"],
       ["LABS (analíticas)",f"{solo['LABS']:.4f}","Señal complementaria (fluidos, función renal/cardíaca)"],
       ["ECG (señal)",f"{solo['ECG']:.4f}","Débil para hallazgos radiográficos; aporta poco"]],
      best_row=0)

# ===== 3. RESULTADOS v1/v2 =====
H("3. Resultados del stacking (v1 y v2)")
def gv(rows,enf,key="macro_path"):
    for r in rows:
        if r["enfoque"]==enf: return float(r[key])
    return float("nan")
enfs=["CXR_solo","Blending","Stacking_LR","Stacking_XGB"]
rows=[]
for e in enfs:
    rows.append([e.replace("_"," "), f"{gv(v1,e):.4f}", f"{gv(v2,e):.4f}"])
table(["Enfoque","macro AUC v1","macro AUC v2"],rows)
P("Dos hallazgos claros y reproducidos en ambas versiones:")
bullet("El blending por media (0.776 / 0.774) queda POR DEBAJO del CXR en solitario (0.7825): promediar con ECG/LABS, "
       "mucho más débiles, diluye la señal fuerte de la imagen.")
bullet("El stacking (LogReg/XGBoost) supera al CXR solo, pero por un margen MODESTO (~+0.005 macro). v1 y v2 dan "
       "prácticamente el mismo resultado (0.787), lo que valida que el v2 (solo carga) reproduce fielmente al v1.")
img(V2F/"comparativa_enfoques.png",16,"Figura 1. Comparativa de enfoques de fusión (stacking v2): barras macro y AUC por etiqueta.")
img(V2F/"pesos_modalidad_enfermedad.png",11,"Figura 2. Peso de cada modalidad por enfermedad (coeficientes del meta-modelo LogReg).")

# ===== 4. FUSIÓN AVANZADA v3 =====
H("4. Fusión avanzada (v3): técnicas adicionales")
P("Motivados por lo anterior (el blending ingenuo perjudica y el stacking gana poco), se comparan seis técnicas de "
  "fusión sobre las mismas predicciones base. Resultados en test (macro AUC patologías y por etiqueta):")
order=["CXR_solo","Blend_media","Blend_opt","Stacking_LR","MoE","MLP_fusion"]
nice={"CXR_solo":"CXR solo","Blend_media":"Blending media","Blend_opt":"Blending óptimo","Stacking_LR":"Stacking LogReg","MoE":"Mixture-of-Experts","MLP_fusion":"MLP de fusión"}
d3={r["enfoque"]:r for r in v3}
best_e=sumf["best"]; best_idx=order.index(best_e)
rows=[]
for e in order:
    r=d3[e]; rows.append([nice[e], f"{float(r['macro_path']):.4f}"]+[f"{float(r[l]):.3f}" for l in LABELS])
table(["Técnica","macroP"]+[l[:8] for l in LABELS],rows,best_row=best_idx,small=True)
P(f"Ganador: {nice[best_e]} (macro {float(d3[best_e]['macro_path']):.4f}), por delante del Stacking LogReg "
  f"({float(d3['Stacking_LR']['macro_path']):.4f}) y del CXR solo ({float(d3['CXR_solo']['macro_path']):.4f}).")
img(FAF/"comparativa_fusion.png",15,"Figura 3. Comparativa de las seis técnicas de fusión (test).")
img(FAF/"auc_por_etiqueta_fusion.png",16,"Figura 4. AUC por etiqueta y técnica.")
P("Conclusiones por técnica:")
bullet("Blending óptimo ponderado (Nelder-Mead, pesos por modalidad×etiqueta sobre OOF): el mejor macro, interpretable y "
       "sin apenas sobreajuste. Es la opción recomendada.")
bullet("Mixture-of-Experts (gating por paciente): muy cerca del anterior y con valor añadido: adapta el peso de cada "
       "modalidad a cada paciente.")
bullet("MLP de fusión: sobreajusta con ~10k registros y no supera a las opciones lineales.")
img(FAF/"pesos_blend_opt.png",11,"Figura 5. Blending óptimo — peso por modalidad y etiqueta.")
g=sumf["moe_gating_medio"]
img(FAF/"moe_gating.png",16,f"Figura 6. MoE — gating medio (CXR={g['CXR']:.2f}, ECG={g['ECG']:.2f}, LABS={g['LABS']:.2f}) y su variación por paciente.")
P("Los pesos óptimos son clínicamente coherentes: en Atelectasis, Lung Opacity y No Finding manda casi por completo el "
  "CXR; en Edema, Cardiomegalia y Derrame pleural las analíticas (y algo el ECG) aportan señal complementaria —"
  "justo donde el líquido y la función cardíaca dejan huella en sangre.")

# ===== 5. v4 EXPRIMIR EL AUC =====
H("5. v4 — Exprimir el AUC (mejoras adicionales)")
P("Como el blending óptimo (0.7914) ya batía al stacking, se añaden mejoras centradas en el ranking (AUC), todas sobre "
  "las predicciones ya exportadas: combinar en espacio logit y con probabilidades sin calibrar, stacking enriquecido "
  "(cal+raw+confianza+entropía), selección del mejor enfoque por etiqueta, ensemble de fusores y un ensemble voraz tipo "
  "Caruana. El ajuste de pesos se hace en OOF, la elección entre técnicas en validación, y el test solo se reporta.")
d4={r["enfoque"]:r for r in v4}
order4=["CXR","blend_opt(v3)","blend_logit","blend_raw","stack_enriquecido","MoE","sel_por_etiqueta","caruana"]
nice4={"CXR":"CXR solo","blend_opt(v3)":"Blending óptimo (v3)","blend_logit":"Blending en logit","blend_raw":"Blending con raw",
       "stack_enriquecido":"Stacking enriquecido","MoE":"Mixture-of-Experts","sel_por_etiqueta":"Selección por etiqueta","caruana":"Ensemble Caruana"}
best4=sum4["best_by_val"]
rows=[[nice4[e],f"{float(d4[e]['macroP_val']):.4f}",f"{float(d4[e]['macroP_test']):.4f}"] for e in order4]
table(["Técnica","macro AUC val","macro AUC test"],rows,best_row=order4.index(best4))
P(f"Elegido por validación (sin mirar test): {nice4[best4]}, con macro AUC en test = {float(d4[best4]['macroP_test']):.4f} "
  f"— frente a 0.7914 del blending óptimo (v3) y 0.7825 del CXR solo. Dos lecciones útiles: combinar en logit y usar las "
  f"probabilidades sin calibrar rankea mejor (la isotónica aplanaba), y el ensemble Caruana exprime la diversidad sin sobreajustar.")
img(V4F/"comparativa_v4.png",16,"Figura 7. v4 — comparativa de técnicas para exprimir el AUC (ganador en rojo, líneas: CXR y v3).")
img(V4F/"auc_etiqueta_v4.png",16,"Figura 8. AUC por etiqueta — CXR vs v3 (blending óptimo) vs v4 (Caruana).")

# ===== 6. CONCLUSIONES =====
H("6. Conclusiones")
bullet("El CXR es el techo del sistema (0.783). La fusión tardía sobre probabilidades aporta una mejora real pero "
       f"acotada (~+0.012 macro): de 0.7825 (CXR) a 0.7914 (blending óptimo, v3) y hasta {float(d4[best4]['macroP_test']):.4f} "
       "(ensemble Caruana, v4).")
bullet("Promediar a ciegas hace daño; ponderar (estático por etiqueta o dinámico por paciente) es lo que recupera y "
       "supera al CXR. La interpretabilidad del blending óptimo y del MoE es un valor para el TFM.")
bullet("ECG es la modalidad limitante; su contribución a la fusión es marginal salvo en Cardiomegalia.")
bullet("El v2/v3 (carga de OOF) hace que re-correr la fusión sea instantáneo, separando el coste pesado (entrenar "
       "modalidades) de la experimentación de fusión.")

# ===== 7. MEJORAS PROPUESTAS =====
H("7. Mejoras propuestas")
P("Ordenadas por relación valor/coste en CPU:")
bullet("[HECHO en v3/v4] Blending óptimo ponderado, espacio logit + probabilidades raw, MoE y ensemble Caruana como "
       "fusión por defecto. Mejor resultado (Caruana, v4). No requiere modelos nuevos.")
bullet("[Fusión, barato] Optimizar pesos por subgrupo clínico (sexo, tipo de ingreso), no solo por etiqueta; y "
       "promediar varias semillas del MoE/Caruana para reducir varianza.")
bullet("[Modelo base, medio] El cuello de botella es el CXR (techo) y el ECG (débil). La vía con más recorrido es "
       "mejorar el encoder de imagen; el ECG difícilmente subirá. Si se aborda, se crearía una nueva versión del "
       "notebook 01 (CXR) y, en consecuencia, un stacking que la consuma.")
bullet("[Avanzado, requiere GPU] Fusión intermedia de embeddings con atención cruzada o preentrenamiento contrastivo "
       "tipo Symile/CLIP: pesos dinámicos por paciente sobre representaciones latentes, no sobre probabilidades. "
       "Es la única vía para saltos grandes, pero exige reentrenar encoders (inviable solo-CPU).")
P("Nota: las mejoras de fusión (v3 y v4) NO han necesitado crear nuevas versiones de CXR/ECG/LABS, porque operan sobre "
  "las predicciones ya exportadas. Solo se han creado los notebooks de fusión 04_STACKING_Multimodal_v3 y _v4.", it=True)

try:
    doc.save(str(DOC)); print("Informe guardado en:", DOC)
except PermissionError:
    alt=DOC.with_name("Documentación Informe Stacking Multimodal (v4).docx"); doc.save(str(alt))
    print("ORIGINAL BLOQUEADO (abierto en Word). Guardado en:", alt)
print("Figuras embebidas:", sum(1 for f in [V2F/"comparativa_enfoques.png",V2F/"pesos_modalidad_enfermedad.png",FAF/"comparativa_fusion.png",FAF/"auc_por_etiqueta_fusion.png",FAF/"pesos_blend_opt.png",FAF/"moe_gating.png"] if f.exists()),"de 6")
