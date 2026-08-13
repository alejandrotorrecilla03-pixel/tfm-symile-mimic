"""Ejecuta un notebook como SCRIPT (evita el kernel de Jupyter, que muere en este equipo).
Uso: python run_nb.py <ruta.ipynb>   ·  Guarda log y resultados; NO incrusta outputs en el .ipynb."""
import json,io,sys,os,psutil,traceback
import matplotlib; matplotlib.use('Agg')
p=sys.argv[1]
nb=json.load(io.open(p,encoding='utf-8'))
cells=[c for c in nb['cells'] if c['cell_type']=='code']
proc=psutil.Process(os.getpid()); g={'__name__':'__main__'}
for i,c in enumerate(cells):
    src=''.join(c['source'])
    if 'pip install' in src or 'subprocess.run' in src:
        print(f'[{i}] saltada (deps)',flush=True); continue
    vm=psutil.virtual_memory()
    print(f'[{i}] rss={proc.memory_info().rss/1e6:,.0f}MB libre={vm.available/1e9:.2f}GB | {src.splitlines()[0][:60]}',flush=True)
    try: exec(compile(src,f'<c{i}>','exec'),g)
    except Exception as e:
        print(f'[{i}] EXCEPCION {type(e).__name__}: {str(e)[:300]}',flush=True); traceback.print_exc(); sys.exit(1)
print('FIN OK',flush=True)
