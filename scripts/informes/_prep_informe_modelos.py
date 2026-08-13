"""Prepara tablas curadas y figuras para el Informe de Modelos (01_Informe_MODELOS.docx).
Versión MÁXIMA: explota todo el material (kit clínico de las 3 modalidades, importancia,
MNAR, recalibración de fusión, consistencia, comparación de arquitecturas, fusión por etiqueta).
Salida: salidas/_informe_modelos/{tablas,figuras}. Reejecutable; no toca las salidas originales.
CXR v3: se deja su hueco en la tabla resumen (fila en blanco), sin narrativa.
Uso: venv/Scripts/python.exe _prep_informe_modelos.py
"""
import os, json, shutil, csv
import pandas as pd

S = "salidas"
OUT = os.path.join(S, "_informe_modelos")
TAB = os.path.join(OUT, "tablas"); FIG = os.path.join(OUT, "figuras")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

ES = {"Atelectasis": "Atelectasia", "Cardiomegaly": "Cardiomegalia", "Edema": "Edema",
      "Lung Opacity": "Opacidad pulmonar", "No Finding": "Sin hallazgo",
      "Pleural Effusion": "Derrame pleural"}

def f(x, n=3):
    if x is None or (isinstance(x, float) and pd.isna(x)): return ""
    try: return f"{float(x):.{n}f}".replace(".", ",")
    except (TypeError, ValueError): return str(x)

def write(name, rows, header):
    with open(os.path.join(TAB, name), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    print("  tabla:", name, f"({len(rows)} filas)")

def jload(p):
    with open(p, encoding="utf-8") as fh: return json.load(fh)

def ic_es(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): return ""
    s = str(s).strip()
    if not s or s.lower() == "nan": return ""
    try:
        lo, hi = s.strip("[]").split(","); return f"[{f(lo)} – {f(hi)}]"
    except ValueError:
        return s.replace(".", ",")

# ═══════════════ 1 · RESUMEN DE LOS 11 MODELOS (hueco CXR v3 en blanco) ═══════════════
NOM = {"01_cxr": "CXR (imagen)", "02_ecg": "ECG", "03_labs": "LABS (tabular)",
       "04_stacking": "FUSIÓN (stacking)"}
res = pd.read_csv(os.path.join(S, "RESULTADOS_todos_los_modelos.csv"))
rows = []
for _, r in res.iterrows():
    rows.append([NOM.get(r["modalidad"], r["modalidad"]), r["version"].upper(),
                 f(r["macro_AUC_PR"]), ic_es(r.get("IC95_AUC_PR")),
                 f(r["macro_AUC_ROC"]) if pd.notna(r.get("macro_AUC_ROC")) else ""])
    if r["modalidad"] == "01_cxr" and r["version"].lower() == "v2":
        rows.append(["CXR (imagen)", "V3", "pendiente", "", ""])   # pendiente de recálculo
# modelo FINAL: fusión v3 (media ponderada por patología)
try:
    d3 = jload(f"{S}/04_stacking/v3/veredicto_v3.json"); ic3 = d3.get("ic_v3")
    rows.append(["FUSIÓN v3 (MODELO FINAL)", "pesos/patología", f(d3["macro_ap_v3"]),
                 (f"[{f(ic3[0])} – {f(ic3[1])}]" if ic3 else ""), ""])
except Exception: pass
write("01_resumen_modelos.csv", rows,
      ["Modalidad", "Versión", "AUC-PR macro", "IC 95 % (AUC-PR)", "AUC-ROC macro"])

# ═══════════════ 2 · COMPARATIVA DE VERSIONES POR MODALIDAD ═══════════════
def versiones(mod, etiqueta, incluir_hueco_v3=False):
    d = res[res["modalidad"] == mod]
    rows = []
    for _, r in d.iterrows():
        rows.append([r["version"].upper(), f(r["macro_AUC_PR"]), ic_es(r.get("IC95_AUC_PR")),
                     f(r["macro_AUC_ROC"]) if pd.notna(r.get("macro_AUC_ROC")) else ""])
    if incluir_hueco_v3: rows.append(["V3", "pendiente", "", ""])
    write(etiqueta, rows, ["Versión", "AUC-PR macro", "IC 95 %", "AUC-ROC macro"])

versiones("01_cxr", "10_cxr_versiones.csv", incluir_hueco_v3=True)
versiones("02_ecg", "11_ecg_versiones.csv")
versiones("03_labs", "12_labs_versiones.csv")

# ═══════════════ 3 · MÉTRICAS POR ETIQUETA (referencia de cada modalidad) ═══════════════
def per_label(path, name):
    if not os.path.exists(path): print("  (falta)", path); return
    d = pd.read_csv(path); rows = []
    for _, r in d.iterrows():
        rows.append([ES.get(r["label"], r["label"]), f(r["prevalencia"]), f(r["AP"]),
                     f"[{f(r['AP_ci_lo'])} – {f(r['AP_ci_hi'])}]", f(r["AUC"]),
                     "Sí" if r.get("AP_supera_prevalencia", True) else "No"])
    write(name, rows, ["Hallazgo", "Prevalencia", "AUC-PR", "IC 95 %", "AUC-ROC", "¿Supera prev.?"])

per_label(f"{S}/01_cxr/v2/metrics_per_label_v2.csv", "02_cxr_v2_per_label.csv")
per_label(f"{S}/02_ecg/v2/metrics_per_label_v2.csv", "03_ecg_v2_per_label.csv")
per_label(f"{S}/03_labs/v1/metrics_per_label_v1.csv", "04_labs_v1_per_label.csv")

# ═══════════════ 4 · PUNTOS DE OPERACIÓN (las 3 modalidades) ═══════════════
def puntos(path, name):
    if not os.path.exists(path): print("  (falta)", path); return
    d = pd.read_csv(path); rows = []
    for _, r in d.iterrows():
        rows.append([r["punto"], ES.get(r["etiqueta"], r["etiqueta"]), f(r["umbral"], 2),
                     f(r["Se"]), f(r["Sp"]), f(r["VPP"]), f(r["VPN"]), int(r["FN"]), int(r["FP"])])
    write(name, rows, ["Punto", "Hallazgo", "Umbral", "Sensib.", "Especif.", "VPP", "VPN", "FN", "FP"])

puntos(f"{S}/01_cxr/v2/puntos_operacion_test.csv", "07_puntos_cxr_v2.csv")
puntos(f"{S}/02_ecg/v2/puntos_operacion_test.csv", "13_puntos_ecg_v2.csv")
puntos(f"{S}/03_labs/v1/puntos_operacion_test.csv", "14_puntos_labs_v1.csv")

# ═══════════════ 5 · CALIBRACIÓN (Brier antes/después isotónica) ═══════════════
rows = []
for mod, ver, p in [("CXR", "v2", f"{S}/01_cxr/v2/calibracion_brier.csv"),
                    ("ECG", "v2", f"{S}/02_ecg/v2/calibracion_brier.csv"),
                    ("LABS", "v1", f"{S}/03_labs/v1/calibracion_brier.csv")]:
    if not os.path.exists(p): continue
    d = pd.read_csv(p)
    rows.append([f"{mod} {ver}", f(d["Brier_sin_calibrar"].mean()),
                 f(d["Brier_calibrado"].mean()), f(d["mejora"].mean())])
write("08_calibracion.csv", rows, ["Modelo", "Brier sin calibrar", "Brier calibrado", "Mejora media"])

# ═══════════════ 6 · IMPORTANCIA DE VARIABLES (LABS, Gini vs MI/ANOVA) ═══════════════
imp = pd.read_csv(f"{S}/03_labs/v1/importancia_mi_anova.csv").head(10)
rows = [[r["variable"], f(r["importancia_MI"]), f(r["importancia_ANOVA"]), f(r["media"])]
        for _, r in imp.iterrows()]
write("15_labs_importancia.csv", rows, ["Variable", "Info. mutua", "ANOVA F", "Media"])

# ═══════════════ 7 · FUSIÓN: enfoques, veredicto y por etiqueta ═══════════════
cf = pd.read_csv(f"{S}/04_stacking/v2/comparativa_fusion_v4.csv")
LAB = ["Atelectasis", "Cardiomegaly", "Edema", "Lung Opacity", "No Finding", "Pleural Effusion"]
rows = []
for _, r in cf.iterrows():
    rows.append([r["enfoque"], f(r["macroP_val"]), f(r["macroP_test"])] + [f(r[c]) for c in LAB])
write("05_fusion_enfoques.csv", rows,
      ["Enfoque", "AP val", "AP test", "Atelect.", "Cardiom.", "Edema", "Opacidad", "Sin hall.", "Derrame"])

v1 = jload(f"{S}/04_stacking/v1/summary_stacking_v1.json")
v2 = jload(f"{S}/04_stacking/v2/veredicto_fusion_vs_mono.json")
rows = []
for et, d in [("Stacking v1 (base)", v1), ("Stacking v2 (avanzado)", v2)]:
    rows.append([et, d.get("mejor_fusor") or d.get("mejor_fusion", ""), f(d["macro_ap_fusion"]),
                 f"[{f(d['ic_fusion'][0])} – {f(d['ic_fusion'][1])}]",
                 f(d["macro_ap_mono"]), f"[{f(d['ic_mono'][0])} – {f(d['ic_mono'][1])}]",
                 f"+{f(d['diferencia'])}", "Sí" if d["ic_se_solapan"] else "No", d["veredicto"]])
write("06_fusion_veredicto.csv", rows,
      ["Versión", "Mejor fusor", "AP fusión", "IC 95 % fusión", "AP mejor mono",
       "IC 95 % mono", "Diferencia", "¿IC solapan?", "Veredicto"])

# fusión vs mono por etiqueta (CXR mono vs sel_por_etiqueta)
mono = cf[cf["enfoque"] == "CXR"].iloc[0]
fus = cf[cf["enfoque"] == "sel_por_etiqueta"].iloc[0]
rows = []
for c in LAB:
    d = fus[c] - mono[c]
    rows.append([ES[c], f(mono[c]), f(fus[c]), ("+" if d >= 0 else "") + f(d)])
write("16_fusion_por_etiqueta.csv", rows,
      ["Hallazgo", "AP CXR (mono)", "AP fusión", "Δ (fusión − mono)"])

# recalibración tras la fusión (stack v1)
rc = pd.read_csv(f"{S}/04_stacking/v1/recalibracion_brier.csv")
rows = [[ES.get(r["etiqueta"], r["etiqueta"]), f(r["Brier_sin_recalibrar"]),
         f(r["Brier_recalibrado"]),
         ("≈0" if abs(r["mejora"]) < 5e-4 else ("+" if r["mejora"] > 0 else "") + f(r["mejora"]))]
        for _, r in rc.iterrows()]
write("17_recalibracion_fusion.csv", rows,
      ["Hallazgo", "Brier antes", "Brier después", "Mejora"])

# ═══════════════ 7b · SÍNTESIS POR PATOLOGÍA (orientada a la herramienta) ═══════════════
def ap_por_label(path, col="AP"):
    d = pd.read_csv(path); return {r["label"]: r[col] for _, r in d.iterrows()}
ap_cxr = ap_por_label(f"{S}/01_cxr/v2/metrics_per_label_v2.csv")
ap_ecg = ap_por_label(f"{S}/02_ecg/v2/metrics_per_label_v2.csv")
ap_labs = ap_por_label(f"{S}/03_labs/v1/metrics_per_label_v1.csv")
ap_fus = {c: cf[cf["enfoque"] == "sel_por_etiqueta"].iloc[0][c] for c in LAB}
def fiabilidad(lab, apf):
    if lab == "No Finding": return "Chequeo de consistencia"
    if apf >= 0.65: return "Alta"
    if apf >= 0.55: return "Media-alta"
    if apf >= 0.50: return "Media"
    return "Baja (apoyar en fusión)"
rows = []
for c in LAB:
    segundo = "ECG" if ap_ecg[c] >= ap_labs[c] else "LABS"
    seg_ap = max(ap_ecg[c], ap_labs[c])
    rows.append([ES[c], f(ap_cxr[c]), f"{segundo} ({f(seg_ap)})", f(ap_fus[c]), fiabilidad(c, ap_fus[c])])
write("18_sintesis_por_patologia.csv", rows,
      ["Hallazgo", "AP imagen (líder)", "2ª modalidad", "AP fusión", "Fiabilidad en la herramienta"])

# ═══════════════ 7c · FUSIÓN v3 (pesos interpretables por patología) ═══════════════
p3 = f"{S}/04_stacking/v3/veredicto_v3.json"
if os.path.exists(p3):
    d3 = jload(p3)
    rows = [[ES.get(lab, lab), f(w[0], 1), f(w[1], 1), f(w[2], 1)] for lab, w in d3["pesos_opt"].items()]
    write("19_fusion_v3_pesos.csv", rows, ["Hallazgo", "Peso CXR", "Peso ECG", "Peso LABS"])
    write("20_fusion_v3_resumen.csv", [[
        "Fusión v3 (pesos óptimos/etiqueta)", f(d3["macro_ap_v3"]),
        f"[{f(d3['ic_v3'][0])} – {f(d3['ic_v3'][1])}]", "+" + f(d3["mejora_vs_mono"]),
        f(d3["mejora_vs_v2"]), "Sí" if d3["ic_solapan_con_mono"] else "No"]],
        ["Fusor", "AP test", "IC 95 %", "Δ vs CXR mono", "Δ vs v2 (0,594)", "¿IC solapan con mono?"])

# ═══════════════ 8 · ESTRATIFICACIÓN (CXR v2) ═══════════════
es = pd.read_csv(f"{S}/01_cxr/v2/estratificacion_subgrupos.csv")
VARMAP = {"gender": "Sexo", "race": "Etnia", "admission_type": "Tipo de ingreso", "cxr_view": "Proyección"}
GRP = {"ASIAN": "Asiática", "BLACK": "Negra", "HISPANIC_LATINO": "Hispana/Latina", "UNKNOWN": "Desconocida",
       "WHITE": "Blanca", "EMERGENCY": "Urgencias", "OBSERVATION": "Observación", "SCHEDULED": "Programado",
       "URGENT": "Urgente", "AP": "AP (portátil)", "PA": "PA (de pie)"}
rows = []
for _, r in es.iterrows():
    if pd.isna(r.get("macro_AP")): continue
    g = str(r["grupo"])
    if r["variable"] == "gender": g = "Femenino" if g == "0" else "Masculino"
    else: g = GRP.get(g, g)
    rows.append([VARMAP.get(r["variable"], r["variable"]), g, int(r["n"]), f(r["macro_AP"]), f(r["macro_AUC"])])
write("09_estratificacion_cxr_v2.csv", rows,
      ["Variable", "Subgrupo", "n", "AUC-PR macro", "AUC-ROC macro"])

# ═══════════════ 10b · TODOS LOS FUSORES (nombres descriptivos, no v1/v2/v3) ═══════════════
NOMFUS = {"CXR": "CXR en solitario (referencia)", "blend_opt(v3)": "Mezcla ponderada (probabilidad)",
    "blend_logit": "Mezcla en logit", "blend_raw": "Mezcla sin calibrar", "stack_LR": "Meta-regresión logística",
    "stack_enriquecido": "Meta-modelo enriquecido", "MoE": "Mezcla de expertos (MoE)",
    "ensemble_fusores": "Ensemble de fusores", "sel_por_etiqueta": "Selección por patología ★",
    "caruana": "Selección voraz (Caruana)"}
INTERP = {"CXR en solitario (referencia)": "—", "Mezcla ponderada (probabilidad)": "Alta",
    "Mezcla en logit": "Alta", "Mezcla sin calibrar": "Alta", "Meta-regresión logística": "Alta",
    "Meta-modelo enriquecido": "Media", "Mezcla de expertos (MoE)": "Baja", "Ensemble de fusores": "Baja",
    "Selección por patología ★": "Media", "Selección voraz (Caruana)": "Media",
    "Pesos fijos interpretables": "Máxima", "Ponderación por confianza": "Alta", "Fusión temprana": "Muy baja"}
filas = []
for _, r in cf.iterrows():
    nom = NOMFUS.get(r["enfoque"], r["enfoque"])
    filas.append([nom, float(r["macroP_test"])])
# añadir v3, A, B desde sus json
try:
    d3 = jload(f"{S}/04_stacking/v3/veredicto_v3.json"); filas.append(["Pesos fijos interpretables", d3["macro_ap_v3"]])
except Exception: pass
try:
    d45 = jload(f"{S}/04_stacking/veredicto_v4v5.json")
    filas.append(["Ponderación por confianza", d45["A_confianza"]["macro_ap"]])
    filas.append(["Fusión temprana", d45["B_temprana"]["macro_ap"]])
except Exception: pass
mono = next(v for n, v in filas if n.startswith("CXR"))
filas = sorted(filas, key=lambda x: -x[1])
rows = [[n, f(v), ("—" if n.startswith("CXR") else ("+" if v - mono >= 0 else "") + f(v - mono)), INTERP.get(n, "—")]
        for n, v in filas]
write("23_fusores_todos.csv", rows, ["Fusor", "AUC-PR (test)", "Δ vs CXR", "Interpretabilidad"])

# ═══════════════ 9 · UNA-LÍNEA: consistencia / calidad / MNAR (para prosa) ═══════════════
extra = {
    "cxr_v2_calidad": jload(f"{S}/01_cxr/v2/control_calidad_imagen.json"),
    "labs_v1_mnar": jload(f"{S}/03_labs/v1/dependencia_flags_mnar.json"),
    "labs_v3_consistencia": jload(f"{S}/03_labs/v3/consistencia_no_finding.json"),
    "stack_v1_consistencia": jload(f"{S}/04_stacking/v1/summary_stacking_v1.json").get("consistencia", {}),
}
with open(os.path.join(TAB, "_valores_prosa.json"), "w", encoding="utf-8") as fh:
    json.dump(extra, fh, ensure_ascii=False, indent=1)
print("  valores para prosa:", {k: (list(v.keys()) if isinstance(v, dict) else v) for k, v in extra.items()})

# ═══════════════ 10 · FIGURAS ═══════════════
COPIAS = [
    ("01_cxr/v1/figuras/roc_pr_v1.png",              "01_cxr_v1_roc_pr.png"),
    ("01_cxr/v1/figuras/auc_por_etiqueta_v1.png",    "02_cxr_v1_auc_etiqueta.png"),
    ("01_cxr/v1/figuras/confusion_v1.png",           "03_cxr_v1_confusion.png"),
    ("01_cxr/v2/figuras/curva_fiabilidad.png",       "04_cxr_v2_fiabilidad.png"),
    ("02_ecg/v2/figuras/auc_por_etiqueta_v3_1.png",  "05_ecg_v2_auc_etiqueta.png"),
    ("02_ecg/v2/figuras/curvas_v3_1.png",            "06_ecg_v2_curvas.png"),
    ("02_ecg/v2/figuras/confusion_matrices_v3_1.png","07_ecg_v2_confusion.png"),
    ("02_ecg/v2/figuras/curva_fiabilidad.png",       "08_ecg_v2_fiabilidad.png"),
    ("03_labs/v1/figuras/ap_vs_prevalencia_v1.png",  "09_labs_v1_ap_prevalencia.png"),
    ("03_labs/v1/figuras/importancia_v1.png",        "10_labs_v1_importancia.png"),
    ("03_labs/v1/figuras/importancia_auc_v1.png",    "11_labs_v1_importancia_auc.png"),
    ("03_labs/v2/figuras/boxplot_ap_modelos.png",    "12_labs_v2_boxplot_arqs.png"),
    ("03_labs/v2/figuras/importancia_variables.png", "13_labs_v2_importancia.png"),
    ("03_labs/v1/figuras/curva_fiabilidad_v1.png",   "14_labs_v1_fiabilidad.png"),
    ("04_stacking/v1/figuras/comparativa_enfoques.png",       "15_stack_v1_comparativa.png"),
    ("04_stacking/v1/figuras/pesos_modalidad_enfermedad.png", "16_stack_v1_pesos.png"),
    ("04_stacking/v1/figuras/roc_pr_mejor_enfoque.png",       "17_stack_v1_roc_pr.png"),
    ("04_stacking/v1/figuras/confusion_mejor_enfoque.png",    "18_stack_v1_confusion.png"),
    ("04_stacking/v2/figuras/comparativa_v4.png",             "19_stack_v2_comparativa.png"),
    ("04_stacking/v2/figuras/auc_etiqueta_v4.png",            "20_stack_v2_auc_etiqueta.png"),
]
n = 0
for src, dst in COPIAS:
    p = os.path.join(S, src)
    if os.path.exists(p):
        shutil.copyfile(p, os.path.join(FIG, dst)); n += 1
    else:
        print("  (figura ausente)", src)
print(f"\nOK · {n} figuras · tablas en {TAB}")
