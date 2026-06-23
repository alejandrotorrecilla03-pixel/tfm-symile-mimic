# -*- coding: utf-8 -*-
"""
Herramienta de Apoyo a la Decisión Clínica Multimodal (CXR + ECG + Analíticas).
Interfaz gráfica: se elige un paciente y la app muestra, por hallazgo, qué dice cada
prueba, cuánto fiarse de cada una (peso = habilidad × confianza), la decisión combinada
y una recomendación en lenguaje natural.

Ejecutar:  streamlit run app_decision.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

# ───────────────────────── CONFIG / RUTAS ─────────────────────────
ROOT = Path(__file__).resolve().parent
NB   = ROOT / "notebooks"
BASE = Path(r"C:\TFM\1.Opción - Symile Mimic\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0")
CSV  = BASE / "data_csv" / "clean"
DIRS = {"CXR": (NB/"outputs_cxr_densenet121_v2", "cxr"),
        "ECG": (NB/"outputs_ecg_resnet1d_v2", "ecg"),
        "LABS": (NB/"outputs_labs_tabular_v2", "labs")}
LABELS = ["Atelectasis","Cardiomegaly","Edema","Lung Opacity","No Finding","Pleural Effusion"]
PATHO  = ["Atelectasis","Cardiomegaly","Edema","Lung Opacity","Pleural Effusion"]
N = len(LABELS)
ES = {"Atelectasis":"Atelectasia","Cardiomegaly":"Cardiomegalia","Edema":"Edema",
      "Lung Opacity":"Opacidad pulmonar","No Finding":"Sin hallazgos","Pleural Effusion":"Derrame pleural"}
MODS = ["CXR","ECG","LABS"]
MOD_ES = {"CXR":"la radiografía","ECG":"el ECG","LABS":"las analíticas"}
MOD_LBL = {"CXR":"Radiografía","ECG":"ECG","LABS":"Analíticas"}
COL = {"CXR":"#2980b9","ECG":"#8e44ad","LABS":"#16a085"}
GENDER = {0:0,1:1,"0":0,"1":1,"M":1,"F":0}
PRESETS = {"Radiografía clara y segura": 21846096,
           "La radiografía duda → mandan las analíticas": 27652514,
           "Las pruebas se contradicen": 28401574,
           "ECG seguro pero poco hábil": 23530010}

st.set_page_config(page_title="Apoyo a la decisión clínica multimodal", layout="wide", page_icon="🩺")

# ───────────────────────── CARGA DE DATOS ─────────────────────────
@st.cache_data(show_spinner=False)
def build_targets(df):
    raw = df[LABELS].to_numpy(float); n = raw.shape[0]
    y = np.zeros((n,N),np.float32); m = np.zeros((n,N),np.float32)
    m[~np.isnan(raw)] = 1; y[raw==1] = 1; y[(raw==-1)] = 0
    nf = LABELS.index("No Finding"); pc = [j for j in range(N) if j!=nf]; nfp = (raw[:,nf]==1)
    for j in pc:
        f = nfp & np.isnan(raw[:,j]); y[f,j]=0; m[f,j]=1
    ap = (raw[:,pc]==1).any(1); fn = ap & np.isnan(raw[:,nf]); y[fn,nf]=0; m[fn,nf]=1
    return y, m

@st.cache_data(show_spinner=False)
def load_split(split, hadm):
    df = pd.read_csv(CSV/f"{split}_clean.csv", sep=";")
    P = {}
    for mod,(d,pref) in DIRS.items():
        c = pd.read_csv(d/f"{pref}_pred_{split}.csv").set_index("hadm_id")
        cols = [f"{pref}_{l.replace(' ','_')}_cal" for l in LABELS]
        P[mod] = c.reindex(df["hadm_id"].to_numpy())[cols].fillna(0.5).to_numpy(np.float32)
    return df, P

@st.cache_data(show_spinner=False)
def compute_skill():
    dfv, Pv = load_split("val", True); yv, mv = build_targets(dfv)
    sk = np.zeros((3,N))
    for mi,mod in enumerate(MODS):
        for j in range(N):
            s = mv[:,j]==1; yt = yv[s,j]
            if yt.sum()>=2 and (1-yt).sum()>=2: sk[mi,j] = max(roc_auc_score(yt,Pv[mod][s,j])-0.5,0.0)
    return sk, dfv, Pv, yv, mv

def reliability(mod, j, conf, Pv, yv, mv):
    s = mv[:,j]==1; p = Pv[mod][s,j]; yt = yv[s,j]; c = np.abs(p-0.5)*2
    sel = (c>=max(conf-0.15,0)) & (c<=min(conf+0.15,1.0))
    if sel.sum()<10: sel = np.ones_like(c, bool)
    return float(((p[sel]>=0.5).astype(int)==yt[sel]).mean())

# ───────────────────────── LÓGICA DE DECISIÓN ─────────────────────────
def assess_finding(idx, j, Bte, yte, mte, skill, Pv, yv, mv):
    probs = {mod: float(Bte[mod][idx,j]) for mod in MODS}
    conf  = {mod: abs(probs[mod]-0.5)*2 for mod in MODS}
    w     = {mod: skill[mi,j]*max(conf[mod],0.05) for mi,mod in enumerate(MODS)}
    Z = sum(w.values())+1e-9; wn = {mod: w[mod]/Z for mod in MODS}
    comb = sum(wn[mod]*probs[mod] for mod in MODS)
    lead = max(MODS, key=lambda mod: w[mod])
    disagree = [mod for mi,mod in enumerate(MODS) if skill[mi,j]>0.05 and (probs[mod]>=0.5)!=(probs[lead]>=0.5)]
    rel = reliability(lead, j, conf[lead], Pv, yv, mv)
    truth = int(yte[idx,j]) if mte[idx,j]==1 else None
    aport = {mod: wn[mod]*probs[mod] for mod in MODS}            # peso × prob → suman la decisión
    cf = {}                                                       # contrafactual: decisión SIN cada prueba
    for mod in MODS:
        otros = [o for o in MODS if o!=mod]; zz = sum(w[o] for o in otros)+1e-9
        cf[mod] = sum((w[o]/zz)*probs[o] for o in otros)
    influ = {mod: comb-cf[mod] for mod in MODS}                   # cuánto cambia la decisión esa prueba
    abst = max(conf.values()) < 0.25                              # ninguna prueba se moja → baja certeza
    return dict(label=LABELS[j], probs=probs, conf=conf, skill={mod:float(skill[mi,j]) for mi,mod in enumerate(MODS)},
                w=wn, comb=comb, lead=lead, rel=rel, disagree=disagree, truth=truth,
                aport=aport, cf=cf, influ=influ, abst=abst)

def narrate(f):
    l = ES[f["label"]]; lead = f["lead"]; cxr = f["probs"]["CXR"]; lead_es = MOD_ES[lead]
    if lead=="CXR" and f["conf"]["CXR"]>=0.4:
        s = f"La radiografía es clara y fiable aquí ({cxr:.0%}); **guíate sobre todo por la radiografía**."
    elif lead!="CXR":
        s = (f"La radiografía está **dudosa** ({cxr:.0%}, baja confianza); {lead_es} aportan más en este hallazgo "
             f"({f['probs'][lead]:.0%}) y aciertan ~{f['rel']:.0%} cuando están así de seguras → "
             f"**ten en cuenta sobre todo {lead_es}**.")
    else:
        s = f"La radiografía es la referencia ({cxr:.0%}) con confianza media; contrasta con las demás si la clínica no encaja."
    if f["disagree"]:
        s += f" ⚠️ Discrepancia: {' y '.join(MOD_ES[d] for d in f['disagree'])} no coinciden con {lead_es}."
    return s

def bar_figure(f):
    order = ["LABS","ECG","CXR"]  # plotly dibuja de abajo a arriba -> CXR queda arriba
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[MOD_LBL[m] for m in order], x=[f["probs"][m] for m in order], orientation="h",
        marker=dict(color=[COL[m] for m in order],
                    line=dict(color=["#111" if m==f["lead"] else "rgba(0,0,0,0)" for m in order], width=2)),
        text=[f"{f['probs'][m]:.0%} · peso {f['w'][m]:.0%}" for m in order],
        textposition="outside", textfont=dict(size=13), hoverinfo="skip", showlegend=False))
    fig.add_vline(x=0.5, line=dict(color="gray", width=1, dash="dash"))
    fig.add_vline(x=f["comb"], line=dict(color="#E24B4A", width=3),
                  annotation_text=f"decisión {f['comb']:.0%}", annotation_position="top",
                  annotation_font=dict(color="#E24B4A", size=13))
    fig.update_xaxes(range=[0,1.18], tickformat=".0%", showgrid=False, title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(height=190, margin=dict(l=10,r=10,t=26,b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ───────────────────────── INTERFAZ ─────────────────────────
skill, dfv, Pv, yv, mv = compute_skill()
df_te, Bte = load_split("test", True)
y_te, m_te = build_targets(df_te)
hadms = df_te["hadm_id"].tolist()

st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1150px}
div[data-testid="stMetricValue"]{font-size:1.5rem}
</style>""", unsafe_allow_html=True)

st.title("🩺 Apoyo a la decisión clínica multimodal")
st.caption("¿En qué prueba fiarse para cada paciente y hallazgo? · Radiografía + ECG + Analíticas · Herramienta de APOYO, no diagnóstica.")

with st.sidebar:
    st.header("Paciente")
    preset = st.selectbox("Casos de ejemplo", ["—"]+list(PRESETS), index=1)
    default_hadm = PRESETS.get(preset, hadms[0])
    only_no_cxr = st.checkbox("Solo casos donde NO manda la radiografía", value=False)
    pool = hadms
    if only_no_cxr:
        pool = [h for h in hadms
                if any(assess_finding(df_te.index[df_te.hadm_id==h][0], LABELS.index(l), Bte,y_te,m_te,skill,Pv,yv,mv)["lead"]!="CXR" for l in PATHO)]
        if default_hadm not in pool and pool: default_hadm = pool[0]
    hadm = st.selectbox("hadm_id del paciente", pool,
                        index=pool.index(default_hadm) if default_hadm in pool else 0)
    st.markdown("---")
    with st.expander("¿Qué significan los números?"):
        st.markdown("""
**Barra = probabilidad** que da cada prueba a ese hallazgo (0% = seguro que no, 100% = seguro que sí).

**Confianza** `|p−0.5|`: cuánto se moja esa prueba en este paciente.

**Habilidad** `AUC(validación)−0.5`: el *track record* de la prueba en ese hallazgo.

**Peso** = habilidad × confianza → cuánto le hacemos caso.

**Decisión** = media de las barras ponderada por el peso.

**Aportación** = peso × prob (las 3 suman la decisión).

**Decisión sin ella** (contrafactual) = qué saldría quitando esa prueba.

**Influencia** = cuánto cambia la decisión esa prueba.

**⚠️ Baja certeza** = ninguna prueba se moja → poco fiable.
""")
    st.markdown("---")
    st.caption("Modelos óptimos solo-CPU (v2): CXR DenseNet121 · ECG ResNet1D · Analíticas (ensemble).")

idx = df_te.index[df_te.hadm_id==hadm][0]
row = df_te.loc[idx]

# Cabecera del paciente
sexo = "Hombre" if GENDER.get(row.get("gender"),0)==1 else "Mujer"
c = st.columns(4)
c[0].metric("Paciente (hadm_id)", int(hadm))
c[1].metric("Edad", f"{int(row['age'])} años" if not pd.isna(row.get('age')) else "—")
c[2].metric("Sexo", sexo)
c[3].metric("Ingreso", str(row.get("admission_type","—")).title())

# Resumen
findings = [assess_finding(idx, LABELS.index(l), Bte,y_te,m_te,skill,Pv,yv,mv) for l in PATHO]
prob_pos = [f for f in findings if f["comb"]>=0.5]
st.markdown("### Resumen")
if prob_pos:
    txt = ", ".join(f"**{ES[f['label']]}** ({f['comb']:.0%})" for f in sorted(prob_pos,key=lambda f:-f['comb']))
    st.info(f"Hallazgos probables según la fusión: {txt}.")
else:
    st.success("La fusión no señala ningún hallazgo claramente probable (todos por debajo del 50%).")

st.markdown("### Detalle por hallazgo")
for f in sorted(findings, key=lambda f:-f["comb"]):
    with st.container(border=True):
        a, b = st.columns([2.6, 1])
        with a:
            lead_color = COL[f["lead"]]
            st.markdown(f"#### {ES[f['label']]} &nbsp; "
                        f"<span style='background:{lead_color};color:white;padding:2px 10px;border-radius:10px;font-size:0.8rem'>"
                        f"fíate de: {MOD_LBL[f['lead']]}</span>", unsafe_allow_html=True)
            st.plotly_chart(bar_figure(f), width="stretch", key=f"fig_{f['label']}")
        with b:
            st.metric("Decisión combinada", f"{f['comb']:.0%}")
            if f["truth"] is not None:
                ok = (f["comb"]>=0.5)==(f["truth"]==1)
                st.markdown(("✅ " if ok else "❌ ") + f"Verdad clínica: **{'SÍ' if f['truth']==1 else 'no'}**"
                            + ("" if ok else " (el sistema falló aquí)"))
            else:
                st.caption("Sin etiqueta de referencia para este hallazgo.")
        st.markdown("🧠 " + narrate(f))
        if f["abst"]:
            st.warning("⚠️ **Baja certeza**: ninguna prueba se moja en este hallazgo; la decisión es poco fiable, apóyate en la clínica.")
        with st.expander("🔬 Cómo se ha decidido (desglose y contrafactual)"):
            dd = pd.DataFrame({
                "prob (p)":          [f"{f['probs'][m]:.0%}" for m in MODS],
                "confianza":         [f"{f['conf'][m]:.2f}"  for m in MODS],
                "habilidad":         [f"{f['skill'][m]:.2f}" for m in MODS],
                "peso":              [f"{f['w'][m]:.0%}"     for m in MODS],
                "aportación":        [f"{f['aport'][m]:.2f}" for m in MODS],
                "decisión sin ella": [f"{f['cf'][m]:.0%}"    for m in MODS],
                "influencia":        [("+" if f['influ'][m]>=0 else "")+f"{round(100*f['influ'][m])} pp" for m in MODS],
            }, index=[MOD_LBL[m] for m in MODS])
            st.table(dd)
            st.caption("**Aportación** = peso × prob (las tres suman la decisión). **Decisión sin ella** = qué saldría "
                       "quitando esa prueba (contrafactual). **Influencia** = cuánto cambia la decisión esa prueba, en "
                       "puntos porcentuales. La líder es la de mayor peso.")

with st.expander("📊 Habilidad de cada prueba por hallazgo (validación) — el 'track record'"):
    sk_df = pd.DataFrame({MOD_LBL[m]: [round(float(skill[mi,LABELS.index(l)]),2) for l in PATHO] for mi,m in enumerate(MODS)},
                         index=[ES[l] for l in PATHO])
    st.dataframe(sk_df.style.background_gradient(cmap="YlGnBu", axis=None), width="stretch")
    st.caption("Cuanto más alto, más fiable es esa prueba para ese hallazgo. La radiografía lidera; las analíticas "
               "complementan en Edema/Derrame; el ECG aporta sobre todo en Cardiomegalia.")

st.markdown("---")
st.caption("⚠️ Herramienta de **apoyo**, no diagnóstica: surfacea y justifica información; el médico decide. "
           "Como las etiquetas provienen del informe de la radiografía, suele recomendarla; su valor está en los "
           "casos de radiografía dudosa. Fiabilidades estimadas en validación (n limitado).")
