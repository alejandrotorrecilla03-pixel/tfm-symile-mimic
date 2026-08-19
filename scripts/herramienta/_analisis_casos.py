"""Revisión EXHAUSTIVA de los 464 ingresos para elegir los casos que mejor y más
EXCLUSIVAMENTE ilustran cada concepto didáctico, indicando el MODO en que ocurre.
Salida: salidas/_herramienta/CASOS_RECOMENDADOS.md  (+ resumen por consola).
"""
import json
S="salidas"; OUT=f"{S}/_herramienta"
d=json.load(open(f"{OUT}/data.json",encoding="utf-8"))
P=d["pacientes"]
PAT=["Cardiomegalia","Edema","Derrame pleural","Atelectasia","Opacidad pulmonar"]
def H(p,n): return p["hallazgos"][n]
def dec(p,n,m): return H(p,n)["decision"][m]
def real(p,n): return H(p,n)["real"]
def fus(p,n): return H(p,n)["fusion"]
def reales(p): return [n for n in PAT if real(p,n)==1]
def pos(p,m): return [n for n in PAT if dec(p,n,m)=="positivo"]

lines=["# Casos recomendados — revisión de los 464 ingresos\n",
       "Para cada concepto: el/los ingreso(s) que lo ilustran de forma más **limpia y exclusiva**, con el **modo**.\n"]
def bloque(titulo, explicacion, filas):
    lines.append(f"\n## {titulo}\n\n{explicacion}\n")
    if not filas: lines.append("_(sin candidatos claros)_\n"); return
    lines.append("| Ingreso | Detalle | Modo |\n|---|---|---|")
    for hid,det,modo in filas: lines.append(f"| {hid} | {det} | {modo} |")
    lines.append("")

# 1 · POSITIVO CLARO Y EXCLUSIVO: 1 solo hallazgo real, detectado en AMBOS modos, sin otros positivos
c=[]
for p in P:
    rs=reales(p)
    if len(rs)!=1: continue
    n=rs[0]
    if dec(p,n,"cribado")=="positivo" and dec(p,n,"confirmacion")=="positivo":
        otros=[x for x in PAT if x!=n and (dec(p,x,"cribado")=="positivo" or dec(p,x,"confirmacion")=="positivo")]
        if not otros:
            c.append((p["hadm_id"], n, fus(p,n)))
c.sort(key=lambda x:-x[2])
bloque("1 · Positivo claro y EXCLUSIVO (acierto rotundo)",
       "Un único hallazgo real, detectado como POSITIVO en **cribado y confirmación**, y ningún otro hallazgo positivo. Ideal para enseñar un acierto sin ruido.",
       [(h,f"{n} — fusión {v:.2f}; único real y único positivo",f"cribado+confirmación") for h,n,v in c[:5]])

# 2 · FALSO NEGATIVO HONESTO: real=1 pero NEGATIVO en ambos modos (se escapa)
c=[]
for p in P:
    rs=reales(p)
    if len(rs)!=1: continue
    n=rs[0]
    if dec(p,n,"cribado")=="negativo" and dec(p,n,"confirmacion")=="negativo":
        fp=len(pos(p,"cribado"))
        c.append((p["hadm_id"], n, fus(p,n), H(p,n)["fiabilidad"], fp))
c.sort(key=lambda x:(x[4], x[2]))  # menos falsos positivos y menor fusión = más limpio
bloque("2 · Falso negativo honesto (exclusivo)",
       "Un hallazgo real que el sistema NO detecta en **ningún modo** (ni cribado). Mejor si además tiene fiabilidad alta (engaña) y pocos falsos positivos alrededor.",
       [(h,f"{n} real pero NEGATIVO; fusión {v:.2f}; fiab. {fb}; {fp} falsos+ en cribado",f"ambos (negativo)") for h,n,v,fb,fp in c[:5]])

# 3 · MODO-GAP: positivo en CRIBADO y negativo en CONFIRMACIÓN (sospecha, no diagnóstico)
c=[]
for p in P:
    for n in PAT:
        if dec(p,n,"cribado")=="positivo" and dec(p,n,"confirmacion")=="negativo":
            otros=[x for x in PAT if x!=n and dec(p,x,"cribado")=="positivo"]
            c.append((p["hadm_id"], n, fus(p,n), real(p,n), len(otros)))
c.sort(key=lambda x:x[4])  # menos ruido de otros positivos
bloque("3 · Cribado POSITIVO / Confirmación NEGATIVO (el modo importa)",
       "El hallazgo cruza el umbral de **cribado** pero no el de **confirmación**: es una *sospecha a revisar*, no un diagnóstico. Se pide el más aislado (pocos otros positivos).",
       [(h,f"{n} — fusión {v:.2f}; real={'sí' if r==1 else 'no'}; {o} otros positivos en cribado","cribado sí / confirmación no") for h,n,v,r,o in c[:6]])

# 4 · OPACIDAD que ilustra baja fiabilidad: opacidad positiva en cribado, negativa en confirmación
c=[]
for p in P:
    n="Opacidad pulmonar"
    if dec(p,n,"cribado")=="positivo" and dec(p,n,"confirmacion")=="negativo":
        otros=[x for x in PAT if x!=n and (dec(p,x,"cribado")=="positivo")]
        c.append((p["hadm_id"], fus(p,n), real(p,n), len(otros)))
c.sort(key=lambda x:x[3])
bloque("4 · Opacidad de baja fiabilidad (cribado sí, confirmación no)",
       "Sustituto del ejemplo actual (26572612 salía POSITIVO/POSITIVO y no ilustraba el punto). Aquí la Opacidad cruza cribado pero **no** confirmación, mostrando su inestabilidad.",
       [(h,f"Opacidad fusión {v:.2f}; real={'sí' if r==1 else 'no'}; {o} otros positivos","cribado sí / confirmación no") for h,v,r,o in c[:6]])

# 5 · DISCREPANCIA entre modalidades: una prueba alta y otra baja, con fusión intermedia
c=[]
for p in P:
    for n in PAT:
        im,ec,an=H(p,n)["imagen"],H(p,n)["ecg"],H(p,n)["analiticas"]
        spread=max(im,ec,an)-min(im,ec,an)
        if max(im,ec,an)>=0.6 and spread>=0.45:
            c.append((p["hadm_id"], n, im,ec,an, fus(p,n), real(p,n), spread))
c.sort(key=lambda x:-x[7])
bloque("5 · Discrepancia entre modalidades",
       "Una modalidad ve el hallazgo con fuerza mientras otra apenas: la fusión (media ponderada) queda a medio camino. Enseña que no es una votación por mayoría.",
       [(h,f"{n} — img {im:.2f}/ecg {ec:.2f}/an {an:.2f} → fusión {fu:.2f} (real={'sí' if r==1 else 'no'})","según predomine la prueba") for h,n,im,ec,an,fu,r,sp in c[:6]])

# 6 · SANO COHERENTE: No Finding real=1 y ningún hallazgo positivo en ningún modo
c=[]
for p in P:
    if real(p,"Sin hallazgo")==1 and not pos(p,"cribado") and not pos(p,"confirmacion"):
        c.append((p["hadm_id"], fus(p,"Sin hallazgo")))
c.sort(key=lambda x:-x[1])
bloque("6 · Sano coherente y EXCLUSIVO",
       "«Sin hallazgo» real y **ningún** hallazgo positivo en ninguno de los dos modos: el chequeo de coherencia en su forma más limpia.",
       [(h,f"«Sin hallazgo» fusión {v:.2f}; 0 positivos","ambos (todo negativo)") for h,v in c[:5]])

# 7 · CARDIOMEGALIA en AP (aviso de magnificación)
c=[]
for p in P:
    if p["proyeccion"]=="AP" and real(p,"Cardiomegalia")==1 and dec(p,"Cardiomegalia","cribado")=="positivo":
        otros=[x for x in PAT if x!="Cardiomegalia" and dec(p,x,"cribado")=="positivo"]
        c.append((p["hadm_id"], fus(p,"Cardiomegalia"), len(otros)))
c.sort(key=lambda x:(x[2],-x[1]))
bloque("7 · Cardiomegalia en proyección AP (aviso de magnificación)",
       "Positiva en una placa AP (portátil), donde la silueta cardíaca puede magnificarse: dispara el aviso. Mejor si es la única positiva.",
       [(h,f"Cardiomegalia fusión {v:.2f} en AP; {o} otros positivos","cribado (con aviso AP)") for h,v,o in c[:5]])

open(f"{OUT}/CASOS_RECOMENDADOS.md","w",encoding="utf-8").write("\n".join(lines))
print("OK · CASOS_RECOMENDADOS.md generado")
# resumen por consola
for L in lines:
    if L.startswith("## ") or L.startswith("| 2") or L.startswith("| 1") or L.startswith("| 3") or L.startswith("| 4"):
        print(L[:120])
