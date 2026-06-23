# -*- coding: utf-8 -*-
"""
API de apoyo a la decisión clínica multimodal (FastAPI).

Sirve, por paciente y hallazgo, qué dice cada prueba (CXR/ECG/analíticas), cuánto fiarse de cada una
(peso = habilidad × confianza), la decisión combinada y una recomendación en lenguaje natural.

Usa las SALIDAS DE LOS MODELOS ÓPTIMOS (v2) ya calculadas (probabilidades calibradas OOF/val/test).

Cómo ejecutar (en local; los datos derivan de MIMIC -> atado a 127.0.0.1 por la DUA):
    uvicorn api_decision:app --host 127.0.0.1 --port 8000
Documentación interactiva (Swagger):  http://127.0.0.1:8000/docs
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from fastapi import FastAPI, HTTPException, Query

# ----------------------------- carga de datos y motor -----------------------------
ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"
BASE = Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV = BASE / "data_csv" / "clean"
DIRS = {  # MODELOS ÓPTIMOS (v2) de cada modalidad
    "CXR":  (NB / "outputs_cxr_densenet121_v2", "cxr"),
    "ECG":  (NB / "outputs_ecg_resnet1d_v2",    "ecg"),
    "LABS": (NB / "outputs_labs_tabular_v2",    "labs"),
}
LABELS = ["Atelectasis", "Cardiomegaly", "Edema", "Lung Opacity", "No Finding", "Pleural Effusion"]
N = len(LABELS); NO_FINDING = "No Finding"
ES = {"Atelectasis": "Atelectasia", "Cardiomegaly": "Cardiomegalia", "Edema": "Edema",
      "Lung Opacity": "Opacidad pulmonar", "No Finding": "Sin hallazgos", "Pleural Effusion": "Derrame pleural"}
MODS = ["CXR", "ECG", "LABS"]
MOD_ES = {"CXR": "la radiografía", "ECG": "el ECG", "LABS": "las analíticas"}

def _targets(df):
    raw = df[LABELS].to_numpy(float); n = raw.shape[0]
    y = np.zeros((n, N), np.float32); m = np.zeros((n, N), np.float32)
    m[~np.isnan(raw)] = 1; y[raw == 1] = 1; y[raw == -1] = 0
    nf = LABELS.index(NO_FINDING); pc = [j for j in range(N) if j != nf]; nfp = (raw[:, nf] == 1)
    for j in pc:
        f = nfp & np.isnan(raw[:, j]); y[f, j] = 0; m[f, j] = 1
    ap = (raw[:, pc] == 1).any(1); fn = ap & np.isnan(raw[:, nf]); y[fn, nf] = 0; m[fn, nf] = 1
    return y, m

def _load(split, df):
    out = {}
    for mod, (d, pref) in DIRS.items():
        c = pd.read_csv(d / f"{pref}_pred_{split}.csv").set_index("hadm_id")
        cols = [f"{pref}_{l.replace(' ', '_')}_cal" for l in LABELS]
        out[mod] = c.reindex(df["hadm_id"].to_numpy())[cols].fillna(0.5).to_numpy(np.float32)
    return out

class Engine:
    """Carga val (para habilidad/fiabilidad) y test (pacientes consultables) una sola vez."""
    def __init__(self):
        self.df_vl = pd.read_csv(CSV / "val_clean.csv", sep=";")
        self.df_te = pd.read_csv(CSV / "test_clean.csv", sep=";")
        self.y_vl, self.m_vl = _targets(self.df_vl); self.y_te, self.m_te = _targets(self.df_te)
        self.Bvl = _load("val", self.df_vl); self.Bte = _load("test", self.df_te)
        self.skill = np.zeros((3, N))
        for mi, mod in enumerate(MODS):
            for j in range(N):
                s = self.m_vl[:, j] == 1; yt = self.y_vl[s, j]
                if yt.sum() >= 2 and (1 - yt).sum() >= 2:
                    self.skill[mi, j] = max(roc_auc_score(yt, self.Bvl[mod][s, j]) - 0.5, 0.0)
        self.idx_by_hadm = {int(h): i for i, h in enumerate(self.df_te["hadm_id"].to_numpy())}

    def reliability(self, mod, j, conf):
        s = self.m_vl[:, j] == 1; p = self.Bvl[mod][s, j]; yt = self.y_vl[s, j]; c = np.abs(p - 0.5) * 2
        sel = (c >= max(conf - 0.15, 0)) & (c <= min(conf + 0.15, 1.0))
        if sel.sum() < 10: sel = np.ones_like(c, bool)
        return float(((p[sel] >= 0.5).astype(int) == yt[sel]).mean())

    def explain(self, hadm_id, topk=5):
        if int(hadm_id) not in self.idx_by_hadm:
            raise HTTPException(404, f"hadm_id {hadm_id} no está en el conjunto de test")
        idx = self.idx_by_hadm[int(hadm_id)]; B, y, m = self.Bte, self.y_te, self.m_te
        cand = [j for j in range(N) if LABELS[j] != NO_FINDING]
        order = sorted(cand, key=lambda j: -np.mean([B[mo][idx, j] for mo in MODS]))[:topk]
        findings = []
        for j in order:
            probs = {mo: float(B[mo][idx, j]) for mo in MODS}
            conf = {mo: abs(probs[mo] - 0.5) * 2 for mo in MODS}
            w = {mo: self.skill[mi, j] * max(conf[mo], 0.05) for mi, mo in enumerate(MODS)}
            Z = sum(w.values()) + 1e-9; wn = {mo: w[mo] / Z for mo in MODS}
            comb = sum(wn[mo] * probs[mo] for mo in MODS); lead = max(MODS, key=lambda mo: w[mo])
            disagree = [mo for mi, mo in enumerate(MODS) if self.skill[mi, j] > 0.05 and (probs[mo] >= 0.5) != (probs[lead] >= 0.5)]
            f = {"hallazgo": ES[LABELS[j]], "label": LABELS[j],
                 "probabilidades": {MOD_ES[mo]: round(probs[mo], 3) for mo in MODS},
                 "confianza": {MOD_ES[mo]: round(conf[mo], 2) for mo in MODS},
                 "habilidad": {MOD_ES[mo]: round(float(self.skill[mi, j]), 2) for mi, mo in enumerate(MODS)},
                 "peso": {MOD_ES[mo]: round(wn[mo], 2) for mo in MODS},
                 "decision_combinada": round(comb, 3), "prueba_lider": MOD_ES[lead],
                 "verdad": (None if m[idx, j] != 1 else int(y[idx, j]))}
            f["recomendacion"] = self._narrate(f, conf, probs, lead, comb, j)
            findings.append(f)
        r = self.df_te.iloc[idx]
        demo = {"edad": int(r.get("age", 0)), "sexo": str(r.get("gender", "")),
                "tipo_ingreso": str(r.get("admission_type", "")), "raza": str(r.get("race", ""))}
        return {"hadm_id": int(hadm_id), "datos_paciente": demo, "modelos": "óptimos v2 (CXR/ECG/LABS)", "hallazgos": findings}

    def _narrate(self, f, conf, probs, lead, comb, j):
        l = f["hallazgo"]; lead_es = MOD_ES[lead]; cxr = probs["CXR"]
        if lead == "CXR" and conf["CXR"] >= 0.4:
            s = f"La radiografía es clara y fiable aquí ({cxr:.0%}); guíate sobre todo por ella."
        elif lead != "CXR":
            rel = self.reliability(lead, j, conf[lead])
            s = (f"La radiografía está dudosa para {l} ({cxr:.0%}, baja confianza); {lead_es} aportan más "
                 f"({probs[lead]:.0%}) y aciertan ~{rel:.0%} así de seguras → ten en cuenta sobre todo {lead_es}.")
        else:
            s = f"La radiografía es la referencia ({cxr:.0%}) con confianza media; contrasta si la clínica no encaja."
        if f["verdad"] is not None:
            s += f"  [verdad: {'SÍ' if f['verdad'] == 1 else 'no'} {l}]"
        return f"Decisión combinada {comb:.0%}. {s}"

ENG = Engine()

# ----------------------------------- API -----------------------------------
app = FastAPI(title="Apoyo a la Decisión Clínica Multimodal (TFM)",
              description="Por paciente y hallazgo: qué dice cada prueba (CXR/ECG/analíticas), cuánto fiarse de cada una "
                          "(peso = habilidad × confianza), la decisión combinada y la recomendación. Modelos óptimos v2.",
              version="1.0")

@app.get("/salud", tags=["estado"], summary="Comprobación de salud")
def salud():
    return {"estado": "ok", "pacientes_test": len(ENG.df_te), "modelos": "óptimos v2 (CXR/ECG/LABS)"}

@app.get("/pacientes", tags=["consulta"], summary="Lista de pacientes consultables (test)")
def pacientes(limite: int = Query(50, ge=1, le=500, description="máximo de hadm_id a devolver")):
    hs = ENG.df_te["hadm_id"].to_numpy()[:limite]
    return {"n_total": len(ENG.df_te), "hadm_ids": [int(h) for h in hs]}

@app.get("/habilidad", tags=["consulta"], summary="Habilidad (AUC−0.5) por prueba y hallazgo")
def habilidad():
    return {ES[LABELS[j]]: {MOD_ES[mo]: round(float(ENG.skill[mi, j]), 3) for mi, mo in enumerate(MODS)} for j in range(N) if LABELS[j] != NO_FINDING}

@app.get("/paciente/{hadm_id}", tags=["consulta"],
         summary="Informe de decisión de un paciente",
         description="Devuelve, por hallazgo, las probabilidades de cada prueba, su confianza y habilidad, el peso, "
                     "la decisión combinada, la prueba en la que fiarse y la recomendación en lenguaje natural.")
def paciente(hadm_id: int):
    return ENG.explain(hadm_id)

@app.get("/", include_in_schema=False)
def raiz():
    return {"mensaje": "API de apoyo a la decisión clínica multimodal. Ve a /docs para la interfaz interactiva.",
            "endpoints": ["/salud", "/pacientes", "/habilidad", "/paciente/{hadm_id}", "/docs"]}
