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
import math
import random
import time
import traceback
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QLabel,
    QPushButton, QLineEdit, QMessageBox, QSpinBox, QDoubleSpinBox, QHBoxLayout
)
from PySide6.QtCore import Qt

import ezdxf
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

# optional libs
try:
    import pyclipper
    PYCLIPPER_AVAILABLE = True
except Exception:
    PYCLIPPER_AVAILABLE = False

MM_TO_PX = 2.0
PYCLIPPER_SCALE = 1000.0

# ---------------- geometry helpers ----------------

def shapely_from_dxf_entity(ent):
    etype = ent.dxftype()
    try:
        if etype == 'LINE':
            p1 = ent.dxf.start; p2 = ent.dxf.end
            return LineString([(p1[0], p1[1]), (p2[0], p2[1])]).buffer(0)
        if etype in ('LWPOLYLINE', 'POLYLINE'):
            pts = []
            if etype == 'LWPOLYLINE':
                pts = [(p[0], p[1]) for p in ent]
            else:
                try:
                    pts = [(v[0], v[1]) for v in ent.points()]
                except Exception:
                    pts = []
            if len(pts) >= 3:
                return Polygon(pts)
            else:
                return LineString(pts).buffer(0.1)
        if etype == 'CIRCLE':
            c = ent.dxf.center; r = ent.dxf.radius
            return Point(c[0], c[1]).buffer(r, resolution=64)
        if etype == 'ARC':
            center = ent.dxf.center; r = ent.dxf.radius
            start = math.radians(ent.dxf.start_angle); end = math.radians(ent.dxf.end_angle)
            if end < start: end += 2*math.pi
            steps = max(6, int((end-start)/(2*math.pi)*64))
            pts = []
            for i in range(steps+1):
                a = start + (end-start)*i/steps
                pts.append((center[0]+r*math.cos(a), center[1]+r*math.sin(a)))
            return Polygon(pts).buffer(0.01)
    except Exception:
        traceback.print_exc()
    return None


def shapely_to_int_paths(poly, scale=PYCLIPPER_SCALE):
    paths = []
    try:
        exterior = [(int(round(x*scale)), int(round(y*scale))) for (x,y) in poly.exterior.coords]
        paths.append(exterior)
        for interior in poly.interiors:
            ip = [(int(round(x*scale)), int(round(y*scale))) for (x,y) in interior.coords]
            paths.append(ip)
    except Exception:
        pass
    return paths


def int_paths_to_shapely(paths, scale=PYCLIPPER_SCALE):
    polys = []
    for p in paths:
        coords = [(x/scale, y/scale) for (x,y) in p]
        if len(coords) >= 3:
            try:
                polys.append(Polygon(coords))
            except Exception:
                pass
    if not polys:
        return Polygon()
    if len(polys) == 1:
        return polys[0]
    return unary_union(polys)


def offset_poly(poly, offset, scale=PYCLIPPER_SCALE):
    if not PYCLIPPER_AVAILABLE:
        return poly.buffer(offset)
    pco = pyclipper.PyclipperOffset()
    delta = int(round(offset*scale))
    paths = shapely_to_int_paths(poly, scale)
    for path in paths:
        pco.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        sol = pco.Execute(delta)
        return int_paths_to_shapely(sol, scale)
    except Exception:
        traceback.print_exc()
        return poly.buffer(offset)


def minkowski_forbidden(rotated_poly, placed_offsets, scale=PYCLIPPER_SCALE):
    if not PYCLIPPER_AVAILABLE:
        return None
    try:
        pattern = [(int(round(-x*scale)), int(round(-y*scale))) for (x,y) in rotated_poly.exterior.coords]
    except Exception:
        return None
    forbidden = []
    for placed in placed_offsets:
        subj_paths = shapely_to_int_paths(placed, scale)
        for subj in subj_paths:
            try:
                sol = pyclipper.MinkowskiSum(subj, pattern, True)
                fp = int_paths_to_shapely(sol, scale)
                if not fp.is_empty:
                    forbidden.append(fp)
            except Exception:
                pass
    if not forbidden:
        return Polygon()
    return unary_union(forbidden)

# ---------------- core classes ----------------
class Piece:
    def __init__(self, geom: Polygon, layer: str = '0', name: str = 'piece'):
        self.orig_geom = geom
        self.layer = layer
        self.name = name
        self.placement = None  # x,y,angle
        # normalize geometry so minx,miny = 0
        minx, miny, _, _ = self.orig_geom.bounds
        if abs(minx)>1e-9 or abs(miny)>1e-9:
            self.orig_geom = translate(self.orig_geom, xoff=-minx, yoff=-miny)
            self.anchor = (minx, miny)
        else:
            self.anchor = (0.0, 0.0)

    def placed_geom(self):
        if self.placement is None: return None
        x,y,a = self.placement
        g = rotate(self.orig_geom, a, origin=(0,0), use_radians=False)
        g = translate(g, xoff=x, yoff=y)
        return g

class Workspace:
    def __init__(self, w=1000, h=600, margin=15, clearance=2):
        self.w = w; self.h = h; self.margin = margin; self.clearance = clearance
    def usable_rect(self):
        return (self.margin, self.margin, self.w-2*self.margin, self.h-2*self.margin)

# ---------------- GUI and logic ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Nesting 2D - Rotación Inteligente')
        self.resize(1100,700)
        self.workspace = Workspace()
        self.pieces = []
        self.placed_offsets = []
        self.placed_bboxes = []
        self._build_ui()

    def _build_ui(self):
        load_btn = QPushButton('Abrir DXF'); load_btn.clicked.connect(self.open_dxf)
        export_btn = QPushButton('Exportar DXF'); export_btn.clicked.connect(self.export_dxf)
        gen_btn = QPushButton('Generar DXF de Prueba'); gen_btn.clicked.connect(self.generate_test_dxf)
        run_btn = QPushButton('Run placement (NFP/Rejilla)'); run_btn.clicked.connect(self.run_placement)
        refine_btn = QPushButton('Refine (SA)'); refine_btn.clicked.connect(self.run_refine_sa)
        rot_refine_btn = QPushButton('Rotación Inteligente (local)'); rot_refine_btn.clicked.connect(self.run_rotation_refinement)
        report_btn = QPushButton('Generar Reporte'); report_btn.clicked.connect(self.generate_report)

        # controls
        self.w_spin = QSpinBox(); self.w_spin.setRange(100,5000); self.w_spin.setValue(self.workspace.w)
        self.h_spin = QSpinBox(); self.h_spin.setRange(100,5000); self.h_spin.setValue(self.workspace.h)
        self.margin_spin = QSpinBox(); self.margin_spin.setRange(0,200); self.margin_spin.setValue(self.workspace.margin)
        self.clear_spin = QDoubleSpinBox(); self.clear_spin.setRange(0,50); self.clear_spin.setDecimals(1); self.clear_spin.setValue(self.workspace.clearance)
        self.angle_spin = QSpinBox(); self.angle_spin.setRange(1,90); self.angle_spin.setValue(10)
        self.sa_iters = QSpinBox(); self.sa_iters.setRange(0,50000); self.sa_iters.setValue(2000)
        self.sa_temp = QDoubleSpinBox(); self.sa_temp.setRange(0.01,1000); self.sa_temp.setDecimals(2); self.sa_temp.setValue(5.0)
        self.sa_time = QSpinBox(); self.sa_time.setRange(0,36000); self.sa_time.setValue(0)

        self.metrics_label = QLabel('Métricas: --')
        self.info_label = QLabel('Piezas: 0')

        # layout
        left = QWidget(); left_layout = QVBoxLayout(left)
        left_layout.addWidget(load_btn); left_layout.addWidget(export_btn); left_layout.addWidget(gen_btn)
        left_layout.addWidget(run_btn); left_layout.addWidget(rot_refine_btn); left_layout.addWidget(refine_btn); left_layout.addWidget(report_btn)
        left_layout.addWidget(QLabel('Workspace (mm)')); hl = QHBoxLayout(); hl.addWidget(QLabel('W:')); hl.addWidget(self.w_spin); hl.addWidget(QLabel('H:')); hl.addWidget(self.h_spin); left_layout.addLayout(hl)
        left_layout.addWidget(QLabel('Margin (mm)')); left_layout.addWidget(self.margin_spin)
        left_layout.addWidget(QLabel('Clearance (mm)')); left_layout.addWidget(self.clear_spin)
        left_layout.addWidget(QLabel('Angle sampling (deg) - coarse)')); left_layout.addWidget(self.angle_spin)
        left_layout.addWidget(QLabel('SA iterations')); left_layout.addWidget(self.sa_iters); left_layout.addWidget(QLabel('SA temp')); left_layout.addWidget(self.sa_temp); left_layout.addWidget(QLabel('SA time limit (s)')); left_layout.addWidget(self.sa_time)
        left_layout.addStretch(); left_layout.addWidget(self.info_label); left_layout.addWidget(self.metrics_label)

        from PySide6.QtWidgets import QHBoxLayout, QSplitter, QFrame
        center = QFrame()
        from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPolygonItem
        self.scene = QGraphicsScene(); self.view = QGraphicsView(self.scene)

        splitter = QSplitter(); splitter.addWidget(left); splitter.addWidget(self.view)
        main_layout = QHBoxLayout(); main_layout.addWidget(splitter)
        container = QWidget(); container.setLayout(main_layout); self.setCentralWidget(container)
        self.draw_workspace()

    def draw_workspace(self):
        self.scene.clear()
        w = self.w_spin.value(); h = self.h_spin.value(); self.workspace.w=w; self.workspace.h=h; self.workspace.margin=self.margin_spin.value(); self.workspace.clearance=self.clear_spin.value()
        margin = self.workspace.margin; rect_w=self.workspace.w-2*margin; rect_h=self.workspace.h-2*margin
        outer = (0,0,w*MM_TO_PX,h*MM_TO_PX)
        self.scene.addRect(outer[0],outer[1],outer[2],outer[3])
        self.scene.addRect(margin*MM_TO_PX,margin*MM_TO_PX,rect_w*MM_TO_PX,rect_h*MM_TO_PX)
        for p in self.pieces:
            g = p.placed_geom()
            if g is None:
                self._draw_shapely(p.orig_geom, offset=(10,10))
            else:
                self._draw_shapely(g)
        self.info_label.setText(f'Piezas: {len(self.pieces)}')

    def _draw_shapely(self, geom, layer='0', offset=(0,0)):
        try:
            if geom.is_empty: return
            if geom.geom_type == 'Polygon' or geom.geom_type == 'MultiPolygon':
                polys = [geom] if geom.geom_type=='Polygon' else list(geom.geoms)
                from PySide6.QtGui import QPolygonF, QColor, QPen
                from PySide6.QtCore import QPointF
                for poly in polys:
                    ext = poly.exterior.coords[:]
                    qpoly = QPolygonF([QPointF((x+offset[0])*MM_TO_PX,(y+offset[1])*MM_TO_PX) for (x,y) in ext])
                    item = QGraphicsPolygonItem(qpoly); pen = QPen(QColor('blue')); pen.setWidth(1); item.setPen(pen); item.setBrush(QColor(0,0,255,20)); self.scene.addItem(item)
            else:
                b = geom.bounds; from PySide6.QtCore import QRectF; self.scene.addRect(b[0]*MM_TO_PX,b[1]*MM_TO_PX,(b[2]-b[0])*MM_TO_PX,(b[3]-b[1])*MM_TO_PX)
        except Exception as e:
            print('draw error', e)

    def open_dxf(self):
        path,_ = QFileDialog.getOpenFileName(self,'Abrir DXF', filter='DXF Files (*.dxf)')
        if not path: return
        doc = ezdxf.readfile(path); msp = doc.modelspace(); self.pieces=[]
        for ent in msp:
            try:
                geom = shapely_from_dxf_entity(ent)
                if geom is None or geom.is_empty: continue
                layer = ent.dxf.layer if hasattr(ent.dxf,'layer') else '0'
                self.pieces.append(Piece(geom, layer, ent.dxftype()))
            except Exception as e:
                print('skip', getattr(ent,'dxftype', lambda:'?')(), e)
        self.draw_workspace()

    def export_dxf(self):
        if not self.pieces: return
        path,_ = QFileDialog.getSaveFileName(self,'Exportar DXF', filter='DXF Files (*.dxf)')
        if not path: return
        doc = ezdxf.new('R2010'); msp = doc.modelspace()
        for p in self.pieces:
            g = p.placed_geom() if p.placement else p.orig_geom
            layer = p.layer
            if g.geom_type=='Polygon': msp.add_lwpolyline(list(g.exterior.coords), dxfattribs={'layer':layer})
            else:
                if hasattr(g,'geoms'):
                    for sub in g.geoms:
                        msp.add_lwpolyline(list(sub.exterior.coords), dxfattribs={'layer':layer})
        doc.write(path); QMessageBox.information(self,'Exportar DXF', f'Exportado: {path}')

    def generate_test_dxf(self):
        doc = ezdxf.new('R2010'); msp = doc.modelspace()
        for i in range(6):
            x = i*60; y = 0; msp.add_lwpolyline([(x,y),(x+50,y),(x+50,y+35),(x,y+35)], close=True, dxfattribs={'layer':'TEST'})
        for i in range(4):
            msp.add_circle((i*70+25,90), 22, dxfattribs={'layer':'TEST'})
        p = Path.cwd() / 'test_shapes.dxf'; doc.saveas(str(p)); QMessageBox.information(self,'Test DXF', f'Generado: {p}')

    # ---------------- placement (NFP or grid) ----------------
    def run_placement(self):
        ux,uy,uw,uh = self.workspace.usable_rect(); usable_poly = Polygon([(ux,uy),(ux+uw,uy),(ux+uw,uy+uh),(ux,uy+uh)])
        for p in self.pieces: p.placement=None
        self.placed_offsets=[]; self.placed_bboxes=[]
        ordered = sorted(self.pieces, key=lambda q: -q.orig_geom.area)
        angle_step = max(1, int(self.angle_spin.value())); angles = list(range(0,360,angle_step))
        use_nfp = PYCLIPPER_AVAILABLE
        for piece in ordered:
            placed=False
            for a in angles:
                rotated = rotate(piece.orig_geom, a, origin=(0,0), use_radians=False)
                if use_nfp:
                    forbidden = minkowski_forbidden(rotated, self.placed_offsets)
                    if forbidden is None: use_nfp=False; break
                    feasible = usable_poly.difference(forbidden)
                    if feasible.is_empty: continue
                    candidates = [feasible] if feasible.geom_type=='Polygon' else list(feasible.geoms)
                    best_pt = None; best_y=None
                    for poly in candidates:
                        minx,miny,_,_ = poly.bounds
                        if best_pt is None or (miny<best_y) or (abs(miny-best_y)<1e-9 and minx<best_pt[0]): best_pt=(minx,miny); best_y=miny
                    if best_pt is None: continue
                    piece.placement=(best_pt[0], best_pt[1], a)
                    off = offset_poly(piece.placed_geom(), self.workspace.clearance) if PYCLIPPER_AVAILABLE else piece.placed_geom().buffer(self.workspace.clearance)
                    self.placed_offsets.append(off); self.placed_bboxes.append(off.bounds); placed=True; break
                else:
                    minx_r,miny_r,maxx_r,maxy_r = rotated.bounds
                    step = max(1.0, min(5.0, min(uw,uh)/200.0))
                    y = uy
                    while y <= uy+uh:
                        x = ux
                        while x <= ux+uw:
                            dx_min = x+minx_r; dy_min=y+miny_r; dx_max = x+maxx_r; dy_max = y+maxy_r
                            if dx_min<ux-1e-6 or dy_min<uy-1e-6 or dx_max>ux+uw+1e-6 or dy_max>uy+uh+1e-6: x+=step; continue
                            g = translate(rotated, xoff=x, yoff=y)
                            g_offset = offset_poly(g, self.workspace.clearance) if PYCLIPPER_AVAILABLE else g.buffer(self.workspace.clearance)
                            collision = False
                            for idx, ob in enumerate(self.placed_bboxes):
                                if not (g_offset.bounds[2]<ob[0] or g_offset.bounds[0]>ob[2] or g_offset.bounds[3]<ob[1] or g_offset.bounds[1]>ob[3]):
                                    if g_offset.intersects(self.placed_offsets[idx]): collision=True; break
                            if not collision:
                                piece.placement=(x,y,a); self.placed_offsets.append(g_offset); self.placed_bboxes.append(g_offset.bounds); placed=True; break
                            x+=step
                        if placed: break
                        y+=step
                    if placed: break
            if not placed:
                print('No se pudo colocar pieza', piece.name)
        self.draw_workspace()

    # ---------------- rotation refinement (local) ----------------
    def run_rotation_refinement(self):
        """Para cada pieza colocada: muestreo fino alrededor de su ángulo actual para mejorar densidad local."""
        ux,uy,uw,uh = self.workspace.usable_rect(); usable_area = uw*uh
        if not self.pieces or any(p.placement is None for p in self.pieces):
            QMessageBox.information(self,'Rot. Refine','Primero ejecuta placement')
            return
        start = time.time()
        # compute current union area
        best_union = unary_union([p.placed_geom() for p in self.pieces if p.placed_geom() is not None]).area
        angle_step_coarse = max(1, int(self.angle_spin.value()))
        for p in self.pieces:
            x0,y0,a0 = p.placement
            # sample around current angle +/- angle_step_coarse with finer resolution
            best_local = a0
            local_best_area = best_union
            for da in range(-angle_step_coarse, angle_step_coarse+1, max(1, angle_step_coarse//4)):
                a_candidate = (a0 + da) % 360
                g = translate(rotate(p.orig_geom, a_candidate, origin=(0,0), use_radians=False), xoff=x0, yoff=y0)
                # quick bounds check
                minx,miny,maxx,maxy = g.bounds
                if minx<ux-1e-6 or miny<uy-1e-6 or maxx>ux+uw+1e-6 or maxy>uy+uh+1e-6:
                    continue
                # check collisions with offsets
                candidate_ok = True
                candidate_off = offset_poly(g, self.workspace.clearance) if PYCLIPPER_AVAILABLE else g.buffer(self.workspace.clearance)
                for other in self.pieces:
                    if other is p: continue
                    og = other.placed_geom();
                    if og is None: continue
                    og_off = offset_poly(og, self.workspace.clearance) if PYCLIPPER_AVAILABLE else og.buffer(self.workspace.clearance)
                    if candidate_off.intersects(og_off): candidate_ok=False; break
                if not candidate_ok: continue
                # compute union area if placed
                # temporarily set angle
                old_angle = p.placement[2]
                p.placement = (x0,y0,a_candidate)
                try:
                    new_union = unary_union([q.placed_geom() for q in self.pieces if q.placed_geom() is not None]).area
                except Exception:
                    new_union = best_union
                if new_union > local_best_area + 1e-6:
                    local_best_area = new_union; best_local = a_candidate
                # revert
                p.placement = (x0,y0,old_angle)
            # set best local
            p.placement = (x0,y0,best_local)
            best_union = local_best_area
        elapsed = time.time()-start
        utilization = best_union / (uw*uh) * 100 if uw*uh>0 else 0
        self.metrics_label.setText(f'Aprovechamiento: {utilization:.2f}% | Tiempo rot.refine: {elapsed:.2f}s')
        self.draw_workspace()

    # ---------------- Simulated Annealing (global) ----------------
    def run_refine_sa(self):
        iters = int(self.sa_iters.value()); T0 = float(self.sa_temp.value()); time_limit = int(self.sa_time.value())
        if iters <= 0: return
        if any(p.placement is None for p in self.pieces): QMessageBox.information(self,'SA','Ejecuta placement primero'); return
        ux,uy,uw,uh = self.workspace.usable_rect(); usable_area = uw*uh
        # build offsets cache
        offsets_cache = []
        for p in self.pieces:
            g = p.placed_geom(); off = offset_poly(g, self.workspace.clearance) if PYCLIPPER_AVAILABLE else g.buffer(self.workspace.clearance); offsets_cache.append(off)
        best_union = unary_union([p.placed_geom() for p in self.pieces if p.placed_geom() is not None]).area
        best_positions = [(p.placement[0],p.placement[1],p.placement[2]) for p in self.pieces]
        cur_union = best_union; cur_density = cur_union/usable_area if usable_area>0 else 0
        start=time.time()
        for k in range(iters):
            if time_limit>0 and (time.time()-start)>time_limit: break
            frac = k/max(1,iters); T = T0*(1-frac)
            i = random.randrange(len(self.pieces)); p = self.pieces[i]
            old = p.placement
            step = max(1.0, min(5.0, min(uw,uh)/200.0)); angle_step = max(1, int(self.angle_spin.value()))
            dx = random.uniform(-step*2, step*2); dy = random.uniform(-step*2, step*2); da = random.uniform(-angle_step, angle_step)
            nx = old[0]+dx; ny = old[1]+dy; na = (old[2]+da)%360
            rotated = rotate(p.orig_geom, na, origin=(0,0), use_radians=False); minx_r,miny_r,maxx_r,maxy_r=rotated.bounds
            if nx+minx_r<ux-1e-6 or ny+miny_r<uy-1e-6 or nx+maxx_r>ux+uw+1e-6 or ny+maxy_r>uy+uh+1e-6: continue
            newg = translate(rotated, xoff=nx, yoff=ny)
            newoff = offset_poly(newg, self.workspace.clearance) if PYCLIPPER_AVAILABLE else newg.buffer(self.workspace.clearance)
            collision=False
            for j,og in enumerate(offsets_cache):
                if j==i: continue
                bbox=og.bounds; nb = newoff.bounds
                if not (nb[2]<bbox[0] or nb[0]>bbox[2] or nb[3]<bbox[1] or nb[1]>bbox[3]):
                    if newoff.intersects(og): collision=True; break
            if collision: continue
            # accept prob
            p.placement = (nx,ny,na)
            try: new_union = unary_union([q.placed_geom() for q in self.pieces if q.placed_geom() is not None]).area
            except Exception: new_union = cur_union
            new_density = new_union/usable_area if usable_area>0 else 0; delta = new_density - cur_density
            accept=False
            if delta>1e-9: accept=True
            else:
                prob = math.exp(delta/(T+1e-9)) if T>1e-9 else 0
                if random.random()<prob: accept=True
            if accept:
                cur_union = new_union; cur_density = new_density; offsets_cache[i]=newoff
                if cur_density > best_union/usable_area:
                    best_union = cur_union; best_positions = [(q.placement[0],q.placement[1],q.placement[2]) for q in self.pieces]
            else:
                p.placement = old
            if k%200==0:
                self.metrics_label.setText(f'SA iter {k}/{iters} dens {cur_density*100:.2f}%')
        # set best
        for idx,p in enumerate(self.pieces): p.placement = best_positions[idx]
        self.draw_workspace()
        total_time = time.time()-start
        self.metrics_label.setText(f'SA done best_density={(best_union/(uw*uh) if uw*uh>0 else 0)*100:.2f}% | time {total_time:.1f}s')

    # ---------------- report ----------------
    def generate_report(self):
        if not self.pieces:
            QMessageBox.information(self,'Reporte','No hay datos')
            return
        ux,uy,uw,uh = self.workspace.usable_rect(); usable_area = uw*uh
        placed = sum(1 for p in self.pieces if p.placement is not None)
        union_area = unary_union([p.placed_geom() for p in self.pieces if p.placed_geom() is not None]).area if placed>0 else 0
        utilization = (union_area/usable_area)*100 if usable_area>0 else 0
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = Path.cwd() / f'nesting_report_{now}.txt'
        with open(p,'w',encoding='utf-8') as f:
            f.write('Nesting Report
')
            f.write(f'Date: {datetime.now()}
')
            f.write(f'Workspace: {self.workspace.w} x {self.workspace.h} mm
')
            f.write(f'Margin: {self.workspace.margin} mm, Clearance: {self.workspace.clearance} mm
')
            f.write(f'Total pieces: {len(self.pieces)}
')
            f.write(f'Placed pieces: {placed}
')
            f.write(f'Union area: {union_area:.2f} mm2
')
            f.write(f'Utilization: {utilization:.2f}%
')
        QMessageBox.information(self,'Reporte', f'Reporte guardado: {p}')

if __name__=='__main__':
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())
