# -*- coding: utf-8 -*-
# Proto TFM_Memoria_v2: misma estructura académica que v1 pero ACTUALIZADA (DenseNet121, 3 niveles, fusión, herramienta).
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
NBO=Path(r"C:\TFM\1.Opción - Symile Mimic\tfm_multimodal_clinico\notebooks")
FG=NBO/"outputs_herramienta_decision"/"figuras"
DOC=Path(r"C:\TFM\1.Opción - Symile Mimic\DOCUMENTACIÓN\TFM_Memoria_v2.docx")
GRAY=RGBColor(0x59,0x59,0x59); GREEN=RGBColor(0x2E,0x5A,0x1E); REDC=RGBColor(0xB0,0x2A,0x2A)
d=Document(); d.styles["Normal"].font.name="Calibri"; d.styles["Normal"].font.size=Pt(11)
def shade(c,h): sh=OxmlElement("w:shd"); sh.set(qn("w:val"),"clear"); sh.set(qn("w:fill"),h); c._tc.get_or_add_tcPr().append(sh)
def Hh(t,lvl=1): d.add_heading(t,level=lvl)
def P(t,sp=6,it=False,color=None,bold=False):
    p=d.add_paragraph(); p.paragraph_format.space_after=Pt(sp); p.paragraph_format.line_spacing=1.3
    for i,seg in enumerate(t.split("**")):
        r=p.add_run(seg); r.bold=bold or (i%2==1); r.italic=it
        if color: r.font.color.rgb=color
def bullet(t):
    p=d.add_paragraph(style="List Bullet"); p.paragraph_format.space_after=Pt(2)
    for i,seg in enumerate(t.split("**")): r=p.add_run(seg); r.bold=(i%2==1)
def img(path,w=14.0,cap=None):
    path=Path(path)
    if not path.exists(): P(f"[figura: {path.name}]",it=True,color=REDC); return
    d.add_picture(str(path),width=Cm(w)); d.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    if cap:
        c=d.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=c.add_run(cap); r.italic=True; r.font.size=Pt(9); r.font.color.rgb=GRAY
def table(headers,rows,best=None):
    t=d.add_table(rows=1,cols=len(headers)); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i]; shade(c,"1F4E79"); pr=c.paragraphs[0]; pr.alignment=WD_ALIGN_PARAGRAPH.CENTER; run=pr.add_run(h); run.bold=True; run.font.color.rgb=RGBColor(255,255,255); run.font.size=Pt(9.5)
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,v in enumerate(row):
            pr=cells[i].paragraphs[0]; pr.alignment=WD_ALIGN_PARAGRAPH.CENTER if i>0 else WD_ALIGN_PARAGRAPH.LEFT; run=pr.add_run(str(v)); run.font.size=Pt(9.5)
            if best is not None and ri==best: run.bold=True; shade(cells[i],"E2EFDA")
    d.add_paragraph().paragraph_format.space_after=Pt(2)

# ---- PORTADA ----
for txt,sz,b in [("UNIVERSIDAD DE SALAMANCA",13,True),("TRABAJO DE FIN DE MÁSTER",12,True),("",6,False),
                 ("Sistema Multimodal de Apoyo a la Decisión Clínica",17,True),
                 ("Radiografía de tórax, ECG y analíticas de sangre (Symile-MIMIC-IV)",12,False),
                 ("",6,False),("Memoria — versión 2 (borrador actualizado)",11,False)]:
    p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run(txt); r.bold=b; r.font.size=Pt(sz)
d.add_page_break()

# ---- 1. RESUMEN ----
Hh("1. Resumen",1)
P("Este trabajo desarrolla un sistema multimodal para predecir seis hallazgos cardiopulmonares (Atelectasia, "
  "Cardiomegalia, Edema, Opacidad pulmonar, No Finding y Derrame pleural) a partir de tres pruebas del mismo paciente —"
  " radiografía de tórax, electrocardiograma de 12 derivaciones y analíticas de sangre— enlazadas en Symile-MIMIC-IV. "
  "Todo el desarrollo se realiza en CPU. Además de los modelos por modalidad y de su fusión tardía (stacking), la "
  "aportación principal es una **herramienta de apoyo a la decisión** que, para cada paciente y hallazgo, indica en qué "
  "prueba conviene fijarse y lo justifica en lenguaje natural. El mejor modelo individual es la imagen (AUC-ROC macro "
  "0.809); la fusión alcanza 0.795. Se documentan las decisiones metodológicas que hicieron viable el problema "
  "(enmascarado multietiqueta, negativos derivados por exclusividad de No Finding y corrección del preprocesado de imagen).")

# ---- 2. INTRODUCCIÓN ----
Hh("2. Introducción",1)
Hh("2.1 Contexto del problema",2)
P("El diagnóstico clínico integra de forma natural varias fuentes (imagen, señales, laboratorio). Los sistemas de ayuda "
  "suelen centrarse en una sola modalidad; combinarlas de forma rigurosa e interpretable sigue siendo un reto, "
  "especialmente con etiquetas parciales y recursos de cómputo limitados.")
Hh("2.2 Motivación",2)
P("Más que maximizar una métrica global, interesa una herramienta que oriente al clínico: ante un paciente concreto, "
  "¿qué prueba es más fiable para cada hallazgo y por qué? Esta perspectiva convierte una limitación (la fusión aporta "
  "poco AUC) en el valor del sistema (triaje explicable).")
Hh("2.3 Importancia para la investigación y el sector",2)
P("El enfoque es reproducible en hardware modesto (CPU) y honesto en sus límites, lo que facilita su adopción y su "
  "extensión a otros conjuntos multimodales.")
Hh("2.4 Estructura del documento",2)
P("Tras el estado del arte (3) y los objetivos (4), la sección 5 describe los datos y su exploración; la 6 detalla la "
  "metodología y las decisiones críticas; la 7 presenta los resultados por modalidad y de la fusión; la 8 la herramienta "
  "de decisión; y las 9-10 la discusión y las conclusiones.")

# ---- 3. ESTADO DEL ARTE ----
Hh("3. Estado del arte",1)
bullet("**Diagnóstico por imagen**: redes preentrenadas en radiografías (CheXNet/DenseNet121, torchxrayvision) como estándar de facto.")
bullet("**Modelos multimodales en medicina**: fusión temprana (embeddings), tardía (probabilidades) y contrastiva (tipo Symile).")
bullet("**Multietiqueta con etiquetas parciales**: enmascarado de no observados y tratamiento de la incertidumbre.")
bullet("**Limitaciones**: pocos trabajos combinan rigor (sin fuga), viabilidad en CPU e interpretabilidad clínica por paciente — el hueco que cubre este TFM.")

# ---- 4. OBJETIVOS ----
Hh("4. Objetivos",1)
Hh("4.1 Objetivo general",2)
P("Construir un sistema multimodal reproducible (CPU) que prediga los seis hallazgos y, sobre todo, asista al clínico "
  "indicando de forma explicable en qué prueba fiarse para cada paciente y hallazgo.")
Hh("4.2 Objetivos específicos",2)
for b in ["Unificar las tres modalidades por paciente y depurar las etiquetas.",
          "Resolver el carácter multietiqueta con etiquetas parciales (masking) y la falta de negativos (No Finding).",
          "Entrenar modelos por modalidad en tres niveles de complejidad (ligero/óptimo/pesado).",
          "Fusionar las modalidades sin fuga (OOF) y comparar con líneas base.",
          "Diseñar una herramienta de apoyo a la decisión explicable y evaluarla con ejemplos."]:
    bullet(b)

# ---- 5. DATOS ----
Hh("5. Datos y comprensión del problema",1)
Hh("5.1 Fuente de datos y unificación",2)
P("Symile-MIMIC-IV enlaza por ingreso (hadm_id) radiografía, ECG y analíticas del mismo paciente. La unificación realiza "
  "el **join por hadm_id** entre tablas, imágenes (.npy) y señales (.npy), produciendo CSV limpios de train/val/test "
  "(≈10.000 / 750 / 464 pacientes etiquetados). El test etiquetado es un subconjunto intencional del estudio de "
  "recuperación original.")
Hh("5.2 Modalidades y variable dependiente",2)
bullet("**Etiquetas (dependiente)**: seis hallazgos (CheXpert) extraídos del informe de la radiografía.")
bullet("**Radiografía**: imágenes en .npy (gris).")
bullet("**ECG**: 12 derivaciones, señal larga (5000 muestras).")
bullet("**Analíticas**: valores brutos y percentiles, con datos faltantes estructurales.")
Hh("5.3 Análisis exploratorio (EDA) y calidad",2)
bullet("**Desequilibrio** marcado por etiqueta.")
bullet("**Valores perdidos en la dependiente**: muchas etiquetas no anotadas (faltante no aleatorio, MNAR) → clave para el diseño.")
bullet("**Valores perdidos en features tabulares**: el propio patrón de ausencia es informativo (flags de missingness).")

# ---- 6. METODOLOGÍA ----
Hh("6. Metodología",1)
Hh("6.1 Decisiones sobre la variable dependiente (críticas)",2)
Hh("6.1.1 Enmascarado (masking) y problema multietiqueta",3)
P("Cada paciente puede presentar varios hallazgos → seis salidas independientes con sigmoide. La pérdida es **BCE "
  "enmascarada**: los **NaN** (no anotados) no entran en la pérdida ni en la métrica de esa etiqueta.")
Hh("6.1.2 Incertidumbre (−1)",3)
P("Los valores −1 (incertidumbre de CheXpert) se tratan según una política configurable; por defecto se mapean a 0.")
Hh("6.1.3 No Finding y negativos derivados por exclusividad (innovación clave)",3)
P("Atelectasia, Opacidad y No Finding apenas tenían negativos anotados → no entrenables. Se derivan negativos por "
  "exclusividad clínica rellenando **solo NaN**: si No Finding=1, las patologías pasan a 0 observado; si hay alguna "
  "patología, No Finding pasa a 0 observado. Nunca se pisa un valor explícito ni un −1. Esta decisión hizo entrenable "
  "el problema completo.", color=None)
Hh("6.2 Preprocesado y arquitecturas por modalidad",2)
Hh("6.2.1 Imagen — DenseNet121 congelado (corrección del AUC≈0.5)",3)
P("Las primeras versiones daban AUC≈0.5 por un preprocesado mal alineado. La corrección aplica **min-max por imagen a "
  "[−1024, 1024]**, 1 canal, 224×224 (lo que espera torchxrayvision) sobre **DenseNet121** (CheXNet) congelado; se "
  "cachean los embeddings y se entrena una cabeza encima. El AUC pasó de 0.5 a 0.81. El fine-tuning agresivo empeoraba.")
Hh("6.2.2 Señal — ResNet1D (ECG)",3)
P("ResNet1D entrenada desde cero, 12 derivaciones como canales, z-score por derivación y aumento de señal (ruido, "
  "escalado, desplazamiento, lead-masking).")
Hh("6.2.3 Tabular — ensemble multiarquitectura (analíticas)",3)
P("Árboles (XGBoost/LightGBM) con NaN nativo sobre el valor bruto y modelos densos (LogReg/MLP) sobre el percentil "
  "imputado por fold; ratios clínicos (BUN/Creatinina, Neutrófilos/Linfocitos, RDW×Edad) y flags de missingness; sin cxr_view.")
Hh("6.3 Tres niveles por modalidad, calibración y OOF",2)
bullet("**Tres versiones** por modalidad: v1 ligera, v2 óptima, v3 pesada (todas con masking, negativos derivados y export de OOF).")
bullet("**Calibración isotónica** ajustada en validación (probabilidades legibles).")
bullet("**OOF sin fuga**: cada muestra de train la predice un modelo que no la vio → entradas válidas para la fusión.")
Hh("6.4 Fusión multimodal (stacking)",2)
P("Fusión tardía sobre las probabilidades OOF de los modelos óptimos (v2). Stacking **base** (meta Regresión Logística/"
  "XGBoost + interpretación de pesos por modalidad) y **avanzado** (blending óptimo, Mixture-of-Experts, ensemble de Caruana).")

# ---- 7. RESULTADOS ----
Hh("7. Resultados",1)
Hh("7.1 Modelos por modalidad (AUC-ROC macro, test)",2)
table(["Modalidad","v1 ligera","v2 óptima","v3 pesada"],
      [["Imagen (CXR)","0.731","0.809","(pendiente)"],
       ["ECG","(pendiente)","0.637","0.639"],
       ["Analíticas","0.673","0.692","(pendiente)"]],best=None)
P("La imagen es la modalidad dominante; el salto de complejidad en ECG/LABS apenas mejora (techo de señal).", it=True, color=GRAY)
Hh("7.2 Fusión multimodal (test)",2)
table(["Enfoque","macro AUC (patologías)"],
      [["CXR en solitario","0.7825"],["Blending (media)","0.7738"],["Stacking (LR/XGB)","0.7875"],
       ["Blending óptimo ponderado","0.7914"],["Ensemble de Caruana (mejor)","0.7946"]],best=4)
P("La fusión mejora poco (las etiquetas vienen del CXR), pero la mejora es real y rigurosa; aporta sobre todo en Edema/"
  "Derrame (analíticas) y Cardiomegalia (ECG).")

# ---- 8. HERRAMIENTA ----
Hh("8. Herramienta de apoyo a la decisión clínica",1)
P("Para cada paciente y hallazgo, la herramienta combina **probabilidad calibrada** (qué dice cada prueba), **confianza** "
  "(|p−0.5|, cuán segura está) y **habilidad** (AUC de validación −0.5, en qué prueba fiarse para ese hallazgo). El "
  "**peso = habilidad × confianza**; la prueba con más peso es la que mirar, y se genera una recomendación en lenguaje "
  "natural (p. ej., «la radiografía duda → fíate de las analíticas»).")
img(FG/"exp_habilidad.png",13,"Habilidad por prueba y hallazgo (AUC−0.5, validación).")
img(FG/"exp_lead_distribucion.png",13,"Qué prueba lidera por hallazgo (test): analíticas en Edema, ECG en Cardiomegalia.")
P("Ejemplo (paciente real): la radiografía está dudosa sobre Atelectasia (54%) y las analíticas, más seguras, la "
  "confirman (68%) → la herramienta recomienda fiarse de las analíticas; la verdad clínica era positiva.")
img(FG/"ej_defiere_labs_acierta_hadm27652514.png",12)
P("Se acompaña de una aplicación gráfica (Streamlit) y de una API; consume las salidas de los modelos óptimos.")

# ---- 9. DISCUSIÓN ----
Hh("9. Discusión y honestidad metodológica",1)
bullet("Las seis etiquetas se extraen del informe del CXR → la imagen es casi su propio oráculo; por eso la fusión sube poco.")
bullet("El sistema es **apoyo, no diagnóstico**; suele recomendar la radiografía y su valor está en los casos de CXR dudoso.")
bullet("Las estimaciones de fiabilidad/habilidad se calculan en validación (n limitado) y deben tomarse como orientación.")

# ---- 10. CONCLUSIONES ----
Hh("10. Conclusiones y trabajo futuro",1)
P("Se ha construido un sistema multimodal completo, reproducible en CPU y honesto, cuya aportación metodológica es la "
  "fusión rigurosa y cuya aportación clínica es el triaje explicable. Como trabajo futuro (con GPU): atención cruzada "
  "sobre embeddings, atribución por SHAP y contrafactuales para señalar la región/derivación/analítica concreta que "
  "dispara cada recomendación.")
Hh("11. Referencias",1)
P("[Pendiente de completar: Symile (Saporta et al., 2025), MIMIC-IV, CheXNet/DenseNet121, torchxrayvision, XGBoost, etc.]",it=True,color=GRAY)

try: d.save(str(DOC)); print("Guardado:",DOC.name)
except PermissionError:
    alt=DOC.with_name("TFM_Memoria_v2 (nuevo).docx"); d.save(str(alt)); print("BLOQUEADO ->",alt.name)
print("apartados/headings y tablas:",len(d.tables),"· imágenes:",len(d.inline_shapes))
