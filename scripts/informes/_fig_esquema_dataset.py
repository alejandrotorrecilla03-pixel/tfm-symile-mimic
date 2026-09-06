"""Genera una figura de esquema del conjunto de trabajo para el capitulo 4 (Materiales).
No usa ningun dato real de MIMIC: la parte "esquema" solo nombra columnas y tipos, y la
parte "muestra" usa valores fabricados a mano para ilustrar el formato, no un registro real.
Salida: salidas/00_eda/figuras/00_esquema_dataset.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "salidas/00_eda/figuras"
os.makedirs(FIG, exist_ok=True)
BLUE = "#1F3864"; BLUE2 = "#2E5496"; GREY = "#595959"
XLGREEN = "#217346"      # verde de cabecera estilo Excel
XLZEBRA = "#E9F1EC"      # fila alterna verdosa muy suave
FILL = XLGREEN
plt.rcParams.update({"font.family": "Cambria", "font.size": 10, "axes.edgecolor": "#888",
                      "axes.titlecolor": BLUE, "figure.dpi": 150})

BLOQUES = [
    ("Identificadores", "2", "subject_id, hadm_id"),
    ("Rutas a tensores", "2", "cxr_path, ecg_path"),
    ("Demografia", "3", "age, gender, race"),
    ("Contexto del ingreso", "4", "admission_type, admission_location, cxr_view, hours_adm_to_cxr"),
    ("Etiquetas", "6", "Atelectasis, Cardiomegaly, Edema, Lung Opacity, No Finding, Pleural Effusion"),
    ("Analiticas", "41", "hematocrit_51221, creatinine_50912, albumin_50862, ..."),
]

MUESTRA_COLS = ["hadm_id", "age", "gender", "admission_type", "cxr_view",
                "hematocrit_51221", "creatinine_50912", "Cardiomegaly"]
MUESTRA_FILAS = [
    ["29XXXXX", "71", "F", "URGENTE", "AP", "31.4", "1.1", "1"],
    ["21XXXXX", "58", "M", "PROGRAMADO", "PA", "38.9", "0.8", "0"],
    ["25XXXXX", "84", "F", "URGENTE", "AP", "27.6", "1.9", "1"],
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 5.6), gridspec_kw={"height_ratios": [1.5, 1]})

for ax in (ax1, ax2):
    ax.axis("off")

ax1.set_title("Esquema del conjunto de trabajo · 58 columnas en 6 bloques", fontsize=11, fontweight="bold", loc="left")
t1 = ax1.table(cellText=BLOQUES, colLabels=["Bloque", "N.º de columnas", "Ejemplos de columnas"],
                cellLoc="left", colLoc="left", loc="center",
                colWidths=[0.22, 0.14, 0.64])
t1.auto_set_font_size(False); t1.set_fontsize(9); t1.scale(1, 1.9)
for (r, c), cell in t1.get_celld().items():
    cell.set_edgecolor("#C6D9CC")
    if r == 0:
        cell.set_facecolor(XLGREEN); cell.set_text_props(fontweight="bold", color="white")
    else:
        cell.set_facecolor("white" if r % 2 else XLZEBRA)

ax2.set_title("Muestra ilustrativa del formato (valores sintéticos, no proceden de ningún paciente real)",
              fontsize=10.5, fontweight="bold", loc="left", color=GREY)
t2 = ax2.table(cellText=MUESTRA_FILAS, colLabels=MUESTRA_COLS, cellLoc="center", colLoc="center", loc="center")
t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1, 1.8)
for (r, c), cell in t2.get_celld().items():
    cell.set_edgecolor("#C6D9CC")
    if r == 0:
        cell.set_facecolor(XLGREEN); cell.set_text_props(fontweight="bold", color="white")
    else:
        cell.set_facecolor("white" if r % 2 else XLZEBRA)

fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(f"{FIG}/00_esquema_dataset.png", bbox_inches="tight")
plt.close(fig)
print("OK: 00_esquema_dataset.png")
