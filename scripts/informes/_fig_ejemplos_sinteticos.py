"""Genera una figura con ejemplos SINTETICOS de las modalidades de senal (ECG y radiografia).
Nada procede de un paciente real: la senal ECG es una simulacion PQRST y la radiografia es un
esquema dibujado. Complementa la figura del esquema del conjunto (tabular + demografico).
Salida: salidas/00_eda/figuras/00b_ejemplos_sinteticos.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

FIG = "salidas/00_eda/figuras"
os.makedirs(FIG, exist_ok=True)
BLUE = "#1F3864"; ECGC = "#1F7A6D"; GREY = "#595959"
plt.rcParams.update({"font.family": "Cambria", "font.size": 10, "axes.edgecolor": "#888",
                      "axes.titlecolor": BLUE, "figure.dpi": 150})


def pqrst(n=1000, hr=75, fs=500):
    """Una tira de ECG sintetica (varios latidos) a partir de gaussianas P-Q-R-S-T.
    Anchuras en desviacion tipica (s): P y T anchas, QRS estrecho y alto."""
    t = np.arange(n) / fs
    beat = 60.0 / hr
    sig = np.zeros(n)
    # (centro rel s, amplitud, sigma s)
    waves = [(0.16, 0.12, 0.018), (0.215, -0.10, 0.007), (0.24, 1.25, 0.0045),
             (0.265, -0.28, 0.007), (0.40, 0.32, 0.028)]
    k = 0
    while k * beat < t[-1] + beat:
        c0 = k * beat
        for cr, amp, wd in waves:
            sig += amp * np.exp(-((t - (c0 + cr)) ** 2) / (2 * wd * wd))
        k += 1
    sig += np.random.default_rng(3).normal(0, 0.006, n)  # ruido leve
    return t, sig


fig = plt.figure(figsize=(9.6, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])

# ---- ECG sintetico (3 derivaciones ilustrativas) ----
axe = fig.add_subplot(gs[0, 0])
for i, (nm, sc) in enumerate([("I", 1.0), ("II", 1.15), ("V2", 0.8)]):
    t, s = pqrst()
    axe.plot(t, s * sc - i * 1.8, color=ECGC, lw=1.1)
    axe.text(-0.02, -i * 1.8 + 0.3, nm, ha="right", va="center", fontsize=9, color=BLUE, fontweight="bold")
axe.set_title("Electrocardiograma (señal sintética, 3 de 12 derivaciones)", fontsize=10.5, fontweight="bold")
axe.set_xlabel("Tiempo (s)"); axe.set_yticks([]); axe.set_xlim(-0.05, t[-1]); axe.grid(axis="x", alpha=.2, color="#C88")
for sp in ["top", "right", "left"]:
    axe.spines[sp].set_visible(False)

# ---- Radiografia esquematica ----
axr = fig.add_subplot(gs[0, 1])
axr.set_xlim(0, 10); axr.set_ylim(0, 12); axr.axis("off")
axr.set_title("Radiografía de tórax (esquema, no es real)", fontsize=10.5, fontweight="bold")
axr.add_patch(FancyBboxPatch((0.4, 0.4), 9.2, 11.2, boxstyle="round,pad=0.1", fc="#0F1420", ec="#333", lw=1))
# pulmones
for cx in (3.4, 6.6):
    axr.add_patch(Ellipse((cx, 6.6), 2.7, 6.2, fc="#20304a", ec="#43597e", lw=1.2))
# silueta cardiaca
axr.add_patch(Ellipse((5.4, 4.6), 3.0, 3.4, fc="#33465f", ec="#5b7ba6", lw=1.2))
# costillas (arcos)
th = np.linspace(0.2, np.pi - 0.2, 60)
for r in np.linspace(2.2, 5.4, 5):
    axr.plot(5 + r * np.cos(th) * 0.95, 6.6 + r * np.sin(th) * 0.7, color="#54658a", lw=0.8, alpha=.6)
axr.plot([5, 5], [1.2, 11.4], color="#54658a", lw=1.0, alpha=.5)  # columna
axr.text(5, 0.05, "aire = oscuro · hueso/corazón = claro", ha="center", va="bottom", fontsize=8, color=GREY)

fig.suptitle("Ejemplos ilustrativos de las señales (datos sintéticos, ningún paciente real)",
             color=BLUE, fontweight="bold", fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/00b_ejemplos_sinteticos.png", bbox_inches="tight")
plt.close(fig)
print("OK: 00b_ejemplos_sinteticos.png")
