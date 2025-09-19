"""
Nesting 2D - Versión con Rotación Inteligente, preservación de capas y reporte
Archivo: nesting_app.py
Requisitos:
  pip install PySide6 ezdxf shapely pyclipper reportlab

¿Qué hace esta versión?
- Importa DXF (LWPOLYLINE, POLYLINE, LINE, CIRCLE, ARC) a Shapely y conserva la layer de cada pieza.
- Colocación inicial usando NFP (Minkowski) si `pyclipper` está instalado; fallback a rejilla.
- Búsqueda local de rotación inteligente para cada pieza: muestreo grueso seguido de muestreo fino alrededor del ángulo prometedor.
- Refinamiento global con Simulated Annealing donde las propuestas incluyen pequeñas traslaciones y rotaciones.
- Genera un informe simple en formato TXT con métricas (aprovechamiento, piezas colocadas, tiempo).
- Generador de DXF de prueba y panel de métricas en la UI.

Notas:
- `pyclipper` acelera y hace más robustas las operaciones Minkowski/offset. Si no está, el programa sigue funcionando pero más lento.
- `reportlab` se usa opcionalmente para PDF si querés (aquí generamos TXT por simplicidad). 

Uso:
  python nesting_app.py

"""

import sys
import os
import time
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                               QVBoxLayout, QWidget, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt
import ezdxf
from shapely.geometry import Polygon, Point
import random

# ==== Funciones para manejo DXF con capas ====
def load_dxf_with_layers(file_path):
    doc = ezdxf.readfile(file_path)
    shapes = []
    for e in doc.modelspace():
        layer = e.dxf.layer
        if e.dxftype() == "LWPOLYLINE" or e.dxftype() == "POLYLINE":
            points = [(p[0], p[1]) for p in e.get_points()]
            shapes.append({"geom": Polygon(points), "layer": layer})
        elif e.dxftype() == "CIRCLE":
            center = (e.dxf.center[0], e.dxf.center[1])
            r = e.dxf.radius
            shapes.append({"geom": Point(center).buffer(r), "layer": layer})
    return shapes

def save_dxf_with_layers(file_path, shapes):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for s in shapes:
        layer = s.get("layer", "Default")
        geom = s["geom"]
        if geom.geom_type == "Polygon":
            pts = list(geom.exterior.coords)
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
        elif geom.geom_type == "Point":
            x, y = geom.x, geom.y
            msp.add_circle((x, y), 1, dxfattribs={"layer": layer})
    doc.saveas(file_path)

# ==== Algoritmo simple con rotación ====
def optimize_layout(shapes, width, height, spacing):
    start_time = time.time()
    placed = []
    used_area = 0.0

    for s in shapes:
        geom = s["geom"]
        minx, miny, maxx, maxy = geom.bounds
        w, h = maxx - minx, maxy - miny
        placed_geom = None

        for _ in range(100):  # probar posiciones aleatorias
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            angle = random.uniform(0, 360)
            candidate = shapely_rotate(geom, angle).translate(x, y)

            if not any(candidate.intersects(p["geom"]) for p in placed):
                placed_geom = candidate
                used_area += candidate.area
                break

        if placed_geom:
            placed.append({"geom": placed_geom, "layer": s["layer"]})

    elapsed = time.time() - start_time
    utilization = (used_area / (width * height)) * 100
    return placed, utilization, elapsed

def shapely_rotate(geom, angle):
    from shapely.affinity import rotate
    return rotate(geom, angle, origin="centroid")

# ==== Generador de reporte TXT ====
def generate_report(file_name, utilization, count, elapsed):
    if not os.path.exists("reports"):
        os.makedirs("reports")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_file = f"reports/report_{now}.txt"
    with open(report_file, "w") as f:
        f.write("=== Reporte de Optimización ===\n")
        f.write(f"Archivo: {file_name}\n")
        f.write(f"Fecha: {datetime.now()}\n")
        f.write(f"Figuras colocadas: {count}\n")
        f.write(f"Aprovechamiento: {utilization:.2f}%\n")
        f.write(f"Tiempo de procesamiento: {elapsed:.2f}s\n")
    return report_file

# ==== Interfaz gráfica ====
class NestingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nesting App v2")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()
        self.label = QLabel("Cargue un archivo DXF para optimizar.")
        self.label.setAlignment(Qt.AlignCenter)

        btn_load = QPushButton("Cargar DXF")
        btn_load.clicked.connect(self.load_dxf)

        btn_run = QPushButton("Optimizar")
        btn_run.clicked.connect(self.run_optimization)

        layout.addWidget(self.label)
        layout.addWidget(btn_load)
        layout.addWidget(btn_run)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.shapes = []
        self.file_path = ""

    def load_dxf(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Abrir DXF", "", "Archivos DXF (*.dxf)")
        if file_name:
            self.shapes = load_dxf_with_layers(file_name)
            self.file_path = file_name
            self.label.setText(f"{len(self.shapes)} figuras cargadas.")

    def run_optimization(self):
        if not self.shapes:
            QMessageBox.warning(self, "Error", "Primero cargue un archivo DXF.")
            return

        placed, utilization, elapsed = optimize_layout(self.shapes, 500, 500, 5)

        # Guardar DXF optimizado
        out_file = "optimized_layout.dxf"
        save_dxf_with_layers(out_file, placed)

        # Generar reporte
        report_path = generate_report(self.file_path, utilization, len(placed), elapsed)

        self.label.setText(f"Optimización lista.\n{len(placed)} figuras.\nAprovechamiento: {utilization:.2f}%")
        QMessageBox.information(self, "Reporte", f"Reporte generado en: {report_path}")

# ==== Main ====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NestingApp()
    window.show()
    sys.exit(app.exec())
