"""Genera un QR (PNG) apuntando a la URL de la herramienta.
Uso:  py -3.14 _gen_qr.py "https://mi-enlace-privado.example/"  [salida.png]
Si no se pasa URL, usa la del servidor local en la red Wi-Fi (solo LAN).
"""
import sys, socket, qrcode

def lan_url(port=5178):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    finally:
        s.close()
    return f"http://{ip}:{port}/"

url = sys.argv[1] if len(sys.argv) > 1 else lan_url()
out = sys.argv[2] if len(sys.argv) > 2 else "salidas/_herramienta/qr_herramienta.png"
qr = qrcode.QRCode(box_size=10, border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data(url); qr.make(fit=True)
img = qr.make_image(fill_color="#1C5E82", back_color="white")
img.save(out)
print(f"OK · QR -> {out}  ({url})")
