# -*- coding: utf-8 -*-
# Borrador-resumen para el tutor: TODO el trabajo hecho, con decisiones críticas, modelos, fusión y herramienta.
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

NBO=Path(r"C:\TFM\1.Opción - Symile Mimic\tfm_multimodal_clinico\notebooks")
FG=NBO/"outputs_herramienta_decision"/"figuras"
DOC=Path(r"C:\TFM\1.Opción - Symile Mimic\DOCUMENTACIÓN\Borrador TFM - Resumen para tutor.docx")
BLUE=RGBColor(0x1F,0x4E,0x79); LBLUE=RGBColor(0x2E,0x75,0xB6); GREEN=RGBColor(0x2E,0x5A,0x1E); GRAY=RGBColor(0x59,0x59,0x59); REDC=RGBColor(0xB0,0x2A,0x2A)
doc=Document(); doc.styles["Normal"].font.name="Calibri"; doc.styles["Normal"].font.size=Pt(10.5)
def shade(c,h): sh=OxmlElement("w:shd"); sh.set(qn("w:val"),"clear"); sh.set(qn("w:fill"),h); c._tc.get_or_add_tcPr().append(sh)
def H1(t):
    p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(16); p.paragraph_format.space_after=Pt(6); r=p.add_run(t); r.bold=True; r.font.size=Pt(15); r.font.color.rgb=BLUE; return p
def H2(t):
    p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(10); p.paragraph_format.space_after=Pt(3); r=p.add_run(t); r.bold=True; r.font.size=Pt(12); r.font.color.rgb=LBLUE; return p
def P(t,sp=4,it=False,color=None,bold=False):
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(sp); r=p.add_run(t); r.italic=it; r.bold=bold
    if color: r.font.color.rgb=color
    return p
def bullet(t,sub=False):
    p=doc.add_paragraph(style="List Bullet 2" if sub else "List Bullet"); p.paragraph_format.space_after=Pt(2)
    parts=t.split("**")
    for i,seg in enumerate(parts):
        r=p.add_run(seg); r.bold=(i%2==1)
    return p
def img(path,w=15.5,cap=None):
    if not Path(path).exists(): P(f"[falta {Path(path).name}]",it=True,color=REDC); return
    doc.add_picture(str(path),width=Cm(w)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    if cap:
        c=doc.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=c.add_run(cap); r.italic=True; r.font.size=Pt(9); r.font.color.rgb=GRAY
def table(headers,rows,best=None,small=False):
    t=doc.add_table(rows=1,cols=len(headers)); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER; fs=Pt(8.5) if small else Pt(9.5)
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i]; shade(c,"1F4E79"); pr=c.paragraphs[0]; pr.alignment=WD_ALIGN_PARAGRAPH.CENTER; run=pr.add_run(h); run.bold=True; run.font.color.rgb=RGBColor(255,255,255); run.font.size=fs
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,v in enumerate(row):
            pr=cells[i].paragraphs[0]; pr.alignment=WD_ALIGN_PARAGRAPH.CENTER if i>0 else WD_ALIGN_PARAGRAPH.LEFT; run=pr.add_run(str(v)); run.font.size=fs
            if best is not None and ri==best: run.bold=True; shade(cells[i],"E2EFDA")
    doc.add_paragraph().paragraph_format.space_after=Pt(2); return t

# ===================== PORTADA =====================
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run("TFM — Sistema Multimodal de Apoyo a la Decisión Clínica"); r.bold=True; r.font.size=Pt(20); r.font.color.rgb=BLUE
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run("Radiografía de tórax + ECG + Analíticas de sangre (Symile-MIMIC-IV)"); r.font.size=Pt(12); r.font.color.rgb=LBLUE
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run("Borrador-resumen de todo el trabajo realizado · Universidad de Salamanca"); r.font.size=Pt(10); r.font.color.rgb=GRAY
P("")

# ===================== 0. RESUMEN EJECUTIVO =====================
H1("0. Resumen ejecutivo (1 minuto)")
bullet("**Objetivo**: predecir 6 hallazgos cardiopulmonares a partir de tres pruebas (imagen, ECG, sangre) y, sobre todo, construir una **herramienta de apoyo** que diga al médico, paciente a paciente, **en qué prueba fiarse** y por qué.")
bullet("**Restricción dura**: todo en **CPU** (portátil i5, 4 núcleos, sin GPU) → decisiones de diseño orientadas a viabilidad (backbones congelados, embeddings cacheados, modelos compactos).")
bullet("**Mejor resultado por modalidad** (AUC-ROC macro, patologías): **CXR 0.809** · LABS 0.692 · ECG 0.637. **Fusión multimodal**: hasta **0.795** (Caruana).")
bullet("**Hallazgo clave**: como las etiquetas se extraen del informe de la radiografía, el CXR es casi su propio oráculo y la fusión sube poco el AUC → el **valor real es la herramienta de triaje explicable**, no un número mayor.")

# ===================== 1. PUNTO DE PARTIDA =====================
H1("1. Punto de partida y unificación de los datos")
P("Se parte de Symile-MIMIC-IV, que enlaza por ingreso hospitalario (hadm_id) tres modalidades del mismo paciente: "
  "radiografía de tórax, electrocardiograma de 12 derivaciones y analíticas de sangre. El estudio original es de "
  "recuperación (retrieval); aquí se reformula como clasificación clínica multietiqueta.")
H2("Unificación de las bases de datos")
bullet("**Join por hadm_id** entre las tablas tabulares, las imágenes (.npy) y las señales de ECG (.npy); cada fila del CSV limpio se enlaza con su imagen y su ECG por índice.")
bullet("**CSV limpios** (train/val/test) con etiquetas, analíticas (valor bruto y percentil), demografía y contexto de ingreso. Separador ';'.")
bullet("**Conjuntos**: train ≈ 10.000 · val = 750 · test = 464 pacientes etiquetados (el test 'limpio' es un subconjunto intencional del retrieval original).")
bullet("**EDA**: 6 etiquetas (Atelectasia, Cardiomegalia, Edema, Opacidad pulmonar, No Finding, Derrame pleural); fuerte **desequilibrio** y **datos faltantes estructurales** (no todas las etiquetas están anotadas en todos los pacientes → faltante no aleatorio, MNAR).")

# ===================== 2. DECISIONES CRÍTICAS =====================
H1("2. Decisiones críticas (lo que hizo que todo tuviera sentido)")
P("Estas decisiones son el núcleo metodológico del TFM; sin ellas, varias etiquetas eran imposibles de entrenar.", color=GRAY, it=True)

H2("2.1 Problema multietiqueta con enmascarado (masking)")
P("Cada paciente puede tener varios hallazgos a la vez → no es clasificación de una sola clase, sino **multietiqueta** "
  "(6 salidas independientes con sigmoide). El reto es que las etiquetas tienen valores especiales:")
bullet("**NaN** (no anotado) → se **enmascara**: no entra en la pérdida ni en la métrica de esa etiqueta (no inventamos un 0).")
bullet("**−1** (incertidumbre de CheXpert) → se trata según una política (`uncertainty_policy`): por defecto se mapea a 0.")
bullet("La **pérdida es BCE enmascarada**: solo se penaliza donde hay etiqueta observada (`loss = Σ(bce·mask)/Σmask`).")

H2("2.2 'No Finding' y los negativos derivados por exclusividad (la innovación clave)")
P("Problema: 'No Finding' (sin hallazgos) y patologías como Atelectasia o Lung Opacity **casi no tenían negativos "
  "anotados** → AUC indefinida / F1 espurio, imposibles de entrenar.")
P("Solución — **negativos derivados por exclusividad clínica** (cómo se computa ahora):", bold=True)
bullet("Si **No Finding = 1** (paciente sano) → todas las patologías con NaN se rellenan a **0 observado** (si no hay hallazgos, no hay atelectasia, etc.).")
bullet("Si **alguna patología = 1** → 'No Finding' con NaN se rellena a **0 observado** (si hay un hallazgo, no está 'sin hallazgos').")
bullet("**Regla de oro**: solo se rellenan **NaN**; nunca se pisa un valor explícito ni un −1. Esto crea negativos legítimos sin inventar etiquetas.")
P("Efecto: Atelectasia, Lung Opacity y No Finding pasan de 'no entrenables' a tener miles de negativos válidos. Es la "
  "decisión que hizo viable el problema completo.", color=GREEN)

H2("2.3 El fallo del CXR (AUC≈0.5) y su corrección")
P("Las primeras versiones de imagen daban **AUC≈0.5** (azar). Causa real: el preprocesado no coincidía con lo que "
  "esperaba el backbone (z-score estilo ImageNet, 3→1 canal, 160×160), dándole al modelo entradas que nunca vio.")
bullet("**Corrección**: preprocesado alineado con torchxrayvision → 1 canal, **min-max por imagen a rango [−1024, 1024]**, 224×224.")
bullet("**Backbone**: **DenseNet121** (`densenet121-res224-all`, CheXNet), preentrenado justo en patologías torácicas. El AUC pasó de 0.5 a **0.81**.")

H2("2.4 Otras decisiones transversales")
bullet("**Desequilibrio**: `pos_weight` dinámico por etiqueta (CNN) y `scale_pos_weight` (árboles); umbral por F1 en validación.")
bullet("**Calibración isotónica** (ajustada en validación) → las probabilidades se leen como tales y son mejores features de fusión.")
bullet("**Predicciones out-of-fold (OOF) sin fuga**: cada muestra de train la predice un modelo que no la vio → imprescindible para entrenar la fusión sin trampa.")
bullet("**Sin `cxr_view`** en el modelo tabular y **imputación por fold** (sin leakage).")

# ===================== 3. MODELOS POR MODALIDAD =====================
H1("3. Modelos por modalidad: estructura y comparación de versiones")
P("Cada modalidad se organiza en **3 versiones por nivel** (v1 ligero, v2 óptimo, v3 pesado), todas con las bases "
  "anteriores y exportando OOF para la fusión.")

H2("3.1 Imagen — CXR (DenseNet121 congelado + embeddings cacheados)")
P("Estrategia que mejor funciona en CPU: pasar las imágenes una sola vez por el backbone congelado, cachear los "
  "embeddings (1024 dims) y entrenar encima una cabeza ligera.")
table(["Versión","Estructura","Test macro AUC"],
      [["v1 ligero","LogReg one-vs-rest sobre embeddings","0.731"],
       ["v2 ÓPTIMO","Cabeza MLP + metadatos + correlación de etiquetas + K-fold","0.809"],
       ["v3 pesado","DenseNet+TTA(flip)+ResNet50@512 + bagging + ensemble LogReg","(pendiente de ejecutar)"]],best=1)
P("Conclusión: el fine-tuning agresivo (probado en versiones previas) **empeoraba**; el backbone congelado ya marca el "
  "techo para estas etiquetas radiográficas.", it=True, color=GRAY)

H2("3.2 Señal — ECG (ResNet1D desde cero, 12 derivaciones)")
table(["Versión","Estructura","Test macro AUC"],
      [["v1 ligero","ResNet1D base32, 1 bloque/etapa, señal 1000","(pendiente)"],
       ["v2 ÓPTIMO","ResNet1D base48, 2 bloques/etapa, señal 1250, Optuna, K-fold ensemble","0.637"],
       ["v3 pesado","base64, señal 2500, K=5, aumento rico","0.639"]],best=1)
P("El ECG aporta señal indirecta (techo ~0.64). El salto de complejidad apenas mejora → su valor es como modalidad "
  "complementaria, no en solitario.", it=True, color=GRAY)

H2("3.3 Tabular — Analíticas de sangre")
table(["Versión","Estructura","Test macro AUC"],
      [["v1 ligero","Un XGBoost one-vs-rest (NaN nativo) + ratios clínicos + flags","0.673"],
       ["v2 ÓPTIMO","Ensemble XGBoost+LightGBM+LogReg+MLP, tuning por modelo","0.692"],
       ["v3 pesado","Ensemble + TabPFN (nube) + stacking de 2º nivel","(pendiente)"]],best=1)
bullet("**Ingeniería clínica**: ratios con sentido fisiológico (BUN/Creatinina, Neutrófilos/Linfocitos, RDW×Edad).")
bullet("**Flags de missingness** (el hecho de que falte una analítica es informativo, MNAR).")

# ===================== 4. FUSIÓN MULTIMODAL =====================
H1("4. Fusión multimodal (stacking)")
P("Se combinan las predicciones OOF de los tres modelos óptimos (v2) con un meta-modelo. Dos notebooks:")
bullet("**Stacking v1 (base)**: meta-modelos Regresión Logística y XGBoost; se comparan con blending (promedio) y con el mejor individual (CXR). Incluye la **interpretación** de qué modalidad pesa en cada enfermedad.")
bullet("**Stacking v2 (avanzado)**: técnicas para exprimir el AUC — **blending óptimo ponderado** (Nelder-Mead por etiqueta), **Mixture-of-Experts** (gating por paciente), **stacking enriquecido** y **ensemble de Caruana**.")
table(["Enfoque de fusión","Test macro AUC (patologías)"],
      [["CXR en solitario","0.7825"],["Blending (media simple)","0.7738"],["Stacking (LogReg/XGB)","0.7875"],
       ["Blending óptimo ponderado","0.7914"],["Ensemble Caruana (mejor)","0.7946"]],best=4)
bullet("**Por qué la fusión sube poco**: las 6 etiquetas se extraen del informe del CXR → la imagen es casi su propio oráculo; ECG y analíticas solo añaden señal indirecta.")
bullet("**Dónde sí aporta**: la mejora se concentra en **Edema y Derrame** (analíticas) y algo en **Cardiomegalia** (ECG); el promedio simple incluso empeora al CXR (lo diluye).")

# ===================== 5. HERRAMIENTA DE DECISIÓN =====================
H1("5. Herramienta de apoyo a la decisión clínica (el producto del TFM)")
P("La conclusión metodológica (la fusión sube poco) se convierte en el **valor clínico**: una herramienta que, por "
  "paciente y hallazgo, dice **en qué prueba fijarse** y lo justifica. Combina tres señales que ya tenemos:")
table(["Ingrediente","Qué mide","Cómo se calcula"],
      [["1. Probabilidad calibrada","qué dice cada prueba","p (isotónica) → un 0.80 ≈ 80% real de positivos"],
       ["2. Confianza","cuán segura está en ESE paciente","|p − 0.5|·2 (0 = duda, 1 = segura)"],
       ["3. Habilidad","en qué prueba fiarse para ESE hallazgo","AUC(validación) − 0.5, fijo por hallazgo"]])
P("Regla de combinación: **peso = habilidad × confianza** (normalizado). La prueba con más peso es 'la que mirar'; la "
  "decisión combinada es la media ponderada de las probabilidades.", bold=True)
img(FG/"exp_habilidad.png",13,"Habilidad por prueba y hallazgo (AUC−0.5, validación): CXR domina; analíticas complementan en Edema/Derrame; ECG en Cardiomegalia.")
img(FG/"exp_acierto_vs_confianza.png",11,"La confianza SIRVE: a más confianza de una prueba, más acierto (validación).")

H2("5.1 Ejemplos de pacientes (utilidad real)")
P("Caso A — la radiografía es clara → guíate por ella:", bold=True)
img(FG/"ej_cxr_seguro_acierta_hadm21846096.png",12)
P("Caso B — la radiografía duda y las analíticas aciertan (mensaje: 'la radiografía está dudosa → ten en cuenta sobre todo las analíticas'):", bold=True)
img(FG/"ej_defiere_labs_acierta_hadm27652514.png",12)
P("Caso C — las pruebas discrepan: se sigue a la más fiable (y se avisa de la discrepancia):", bold=True)
img(FG/"ej_discrepancia_hadm28401574.png",12)

H2("5.2 ¿Qué prueba 'manda' en cada hallazgo? (test completo)")
table(["Hallazgo","% Radiografía","% ECG","% Analíticas"],
      [["Edema","80%","3%","16%"],["Cardiomegalia","74%","15%","12%"],["Atelectasia","90%","2%","8%"],
       ["Opacidad pulmonar","92%","0%","8%"],["Derrame pleural","89%","2%","9%"]])
img(FG/"exp_lead_distribucion.png",13,"Reparto de la prueba que lidera por hallazgo. Clínicamente coherente: analíticas en Edema, ECG en Cardiomegalia.")

# ===================== 6. CONCLUSIONES =====================
H1("6. Conclusiones y honestidad metodológica")
bullet("Se ha construido un sistema multimodal completo y reproducible: **9 notebooks de modelos** (3 niveles × 3 modalidades), **2 de fusión** y **1 de la herramienta**, todos con masking, negativos derivados, calibración y OOF.")
bullet("La **fusión rigurosa** (OOF sin fuga, test limpio, comparación con baselines) es la aportación metodológica; el **triaje explicable** es la aportación clínica.")
bullet("**Honestidad**: es apoyo, no diagnóstico; como las etiquetas vienen del CXR, la herramienta suele recomendar la radiografía; su valor está en los casos de CXR dudoso. La fiabilidad se estima en validación (n limitado).")
bullet("**Siguiente nivel** (con GPU): atención cruzada sobre embeddings, SHAP y contrafactuales para señalar la región/derivación/analítica concreta que dispara la recomendación.")

P("")
P("Documento generado automáticamente a partir de los resultados reales del proyecto. Pendiente: actualización extensa de "
  "las documentaciones individuales (EDA→05) y app en modo API.", it=True, color=GRAY)

DOC.parent.mkdir(parents=True,exist_ok=True)
try:
    doc.save(str(DOC)); print("Guardado:",DOC.name)
except PermissionError:
    alt=DOC.with_name("Borrador TFM - Resumen para tutor (nuevo).docx"); doc.save(str(alt)); print("BLOQUEADO, guardado:",alt.name)
print("tablas/figuras embebidas OK")
