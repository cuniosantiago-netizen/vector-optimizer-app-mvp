"""
Nesting 2D - Versión con Rotación y Optimización (Simulated Annealing)
Archivo: mvp_nesting.py
Requisitos:
  pip install PySide6 ezdxf shapely pyclipper

Descripción:
Esta versión incluye:
- Importación básica de DXF (LWPOLYLINE, LINE, CIRCLE, ARC) a Shapely.
- Rotación muestreada (ángulo configurable) para las piezas.
- Colocación inicial: NFP (Minkowski) con pyclipper si está disponible; fallback a rejilla.
- Refinamiento de la solución por Simulated Annealing (optimiza la densidad en el rectángulo usable).
- Exportación DXF básica preservando capas y bloques simples.

Instrucciones de uso:
  python mvp_nesting.py

"""

import sys
import math
import random
import time
import traceback
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsView,
    QGraphicsPolygonItem, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox
)
from PySide6.QtGui import QPolygonF, QColor, QPen, QAction
from PySide6.QtCore import QPointF, QRectF

import ezdxf
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

# Optional libraries
try:
    import pyclipper
    PYCLIPPER_AVAILABLE = True
except Exception:
    PYCLIPPER_AVAILABLE = False

MM_TO_PX = 2.0
PYCLIPPER_SCALE = 1000.0


# ---------------- Geometry helpers ----------------

def shapely_from_dxf_entity(ent):
    etype = ent.dxftype()
    try:
        if etype == 'LINE':
            p1 = ent.dxf.start
            p2 = ent.dxf.end
            return LineString([(p1[0], p1[1]), (p2[0], p2[1])]).buffer(0)
        if etype in ('LWPOLYLINE', 'POLYLINE'):
            pts = []
            if etype == 'LWPOLYLINE':
                pts = [(p[0], p[1]) for p in ent.get_points()]
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
            c = ent.dxf.center
            r = ent.dxf.radius
            return Point(c[0], c[1]).buffer(r, resolution=64)
        if etype == 'ARC':
            center = ent.dxf.center
            r = ent.dxf.radius
            start = math.radians(ent.dxf.start_angle)
            end = math.radians(ent.dxf.end_angle)
            if end < start:
                end += 2 * math.pi
            steps = max(6, int((end - start) / (2 * math.pi) * 64))
            pts = []
            for i in range(steps + 1):
                a = start + (end - start) * i / steps
                pts.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a)))
            return Polygon(pts).buffer(0.01)
    except Exception:
        traceback.print_exc()
    return None


def shapely_to_int_paths(poly, scale=PYCLIPPER_SCALE):
    paths = []
    try:
        exterior = [(int(round(x * scale)), int(round(y * scale))) for (x, y) in poly.exterior.coords]
        paths.append(exterior)
        for interior in poly.interiors:
            ip = [(int(round(x * scale)), int(round(y * scale))) for (x, y) in interior.coords]
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
    delta = int(round(offset * scale))
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


# ---------------- Classes ----------------
class Piece:
    def __init__(self, geom, layer='0', name='piece'):
        self.orig_geom = geom
        self.layer = layer
        self.name = name
        self.placement = None
        # normalize so minx,miny = 0
        minx, miny, _, _ = self.orig_geom.bounds
        if abs(minx) > 1e-9 or abs(miny) > 1e-9:
            self.orig_geom = translate(self.orig_geom, xoff=-minx, yoff=-miny)
            self.anchor = (minx, miny)
        else:
            self.anchor = (0.0, 0.0)
    def placed_geom(self):
        if self.placement is None:
            return None
        x,y,a = self.placement
        g = rotate(self.orig_geom, a, origin=(0,0), use_radians=False)
        g = translate(g, xoff=x, yoff=y)
        return g

class Workspace:
    def __init__(self, w=1000,h=600,margin=15,clearance=2):
        self.w = w; self.h = h; self.margin = margin; self.clearance = clearance
    def usable_rect(self):
        return (self.margin, self.margin, self.w - 2*self.margin, self.h - 2*self.margin)


# ---------------- GUI & Logic ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Nesting 2D - Rotación + SA')
        self.resize(1200,800)
        self.workspace = Workspace()
        self.pieces = []
        self.placed_offsets = []
        self.placed_bboxes = []
        self._build_ui()
    def _build_ui(self):
        load_action = QAction('Abrir DXF', self); load_action.triggered.connect(self.open_dxf)
        save_action = QAction('Exportar DXF', self); save_action.triggered.connect(self.export_dxf)
        menubar = self.menuBar(); file_menu = menubar.addMenu('Archivo'); file_menu.addAction(load_action); file_menu.addAction(save_action)

        central = QWidget(); self.setCentralWidget(central); layout = QHBoxLayout(central)
        ctrl = QVBoxLayout(); layout.addLayout(ctrl,0)
        ctrl.addWidget(QLabel('Workspace (mm)'))
        hbox = QHBoxLayout(); hbox.addWidget(QLabel('W:')); self.w_spin=QSpinBox(); self.w_spin.setRange(100,5000); self.w_spin.setValue(self.workspace.w); hbox.addWidget(self.w_spin)
        hbox.addWidget(QLabel('H:')); self.h_spin=QSpinBox(); self.h_spin.setRange(100,5000); self.h_spin.setValue(self.workspace.h); hbox.addWidget(self.h_spin); ctrl.addLayout(hbox)
        ctrl.addWidget(QLabel('Margin (mm)')); self.margin_spin=QSpinBox(); self.margin_spin.setRange(0,200); self.margin_spin.setValue(self.workspace.margin); ctrl.addWidget(self.margin_spin)
        ctrl.addWidget(QLabel('Clearance (mm)')); self.clear_spin=QDoubleSpinBox(); self.clear_spin.setRange(0,50); self.clear_spin.setDecimals(1); self.clear_spin.setValue(self.workspace.clearance); ctrl.addWidget(self.clear_spin)
        ctrl.addWidget(QLabel('Angle sampling (deg)')); self.angle_spin=QSpinBox(); self.angle_spin.setRange(1,90); self.angle_spin.setValue(5); ctrl.addWidget(self.angle_spin)
        if not PYCLIPPER_AVAILABLE: ctrl.addWidget(QLabel('pyclipper NO instalado: NFP desactivado (fallback rejilla)'))
        else: ctrl.addWidget(QLabel('pyclipper instalado: NFP activado'))
        self.run_btn = QPushButton('Run placement'); self.run_btn.clicked.connect(self.run_placement); ctrl.addWidget(self.run_btn)
        ctrl.addWidget(QLabel('Refine (Simulated Annealing)'))
        self.sa_iters = QSpinBox(); self.sa_iters.setRange(0,200000); self.sa_iters.setValue(2000); ctrl.addWidget(QLabel('Iterations')); ctrl.addWidget(self.sa_iters)
        self.sa_temp = QDoubleSpinBox(); self.sa_temp.setRange(0.01,1000); self.sa_temp.setDecimals(2); self.sa_temp.setValue(5.0); ctrl.addWidget(QLabel('Initial Temperature')); ctrl.addWidget(self.sa_temp)
        self.sa_time = QSpinBox(); self.sa_time.setRange(0,36000); self.sa_time.setValue(0); ctrl.addWidget(QLabel('Time limit (s)')); ctrl.addWidget(self.sa_time)
        self.refine_btn = QPushButton('Run Refine (SA)'); self.refine_btn.clicked.connect(self.run_refine); ctrl.addWidget(self.refine_btn)
        self.info_label = QLabel('Piezas: 0'); ctrl.addWidget(self.info_label); ctrl.addStretch()

        self.scene = QGraphicsScene(); self.view = QGraphicsView(self.scene); layout.addWidget(self.view,1)
        self.draw_workspace()

    def draw_workspace(self):
        self.scene.clear(); w=self.w_spin.value(); h=self.h_spin.value(); self.workspace.w=w; self.workspace.h=h; self.workspace.margin=self.margin_spin.value(); self.workspace.clearance=self.clear_spin.value()
        margin=self.workspace.margin; rect_w=self.workspace.w-2*margin; rect_h=self.workspace.h-2*margin
        outer = QRectF(0,0,w*MM_TO_PX,h*MM_TO_PX); self.scene.addRect(outer,QPen(QColor('black')))
        usable = QRectF(margin*MM_TO_PX,margin*MM_TO_PX,rect_w*MM_TO_PX,rect_h*MM_TO_PX); self.scene.addRect(usable,QPen(QColor('green')))
        for p in self.pieces:
            g = p.placed_geom()
            if g is None: self._draw_shapely(p.orig_geom,offset=(10,10))
            else: self._draw_shapely(g)
        self.info_label.setText(f'Piezas: {len(self.pieces)}')

    def _draw_shapely(self,geom,layer='0',offset=(0,0)):
        try:
            if geom.is_empty: return
            if geom.geom_type=='Polygon' or geom.geom_type=='MultiPolygon':
                polys = [geom] if geom.geom_type=='Polygon' else list(geom.geoms)
                for poly in polys:
                    ext = poly.exterior.coords[:]
                    qpoly = QPolygonF([QPointF((x+offset[0])*MM_TO_PX,(y+offset[1])*MM_TO_PX) for (x,y) in ext])
                    item = QGraphicsPolygonItem(qpoly); item.setPen(QPen(QColor('blue'))); item.setBrush(QColor(0,0,255,20)); self.scene.addItem(item)
            else:
                b=geom.bounds; rect=QRectF(b[0]*MM_TO_PX,b[1]*MM_TO_PX,(b[2]-b[0])*MM_TO_PX,(b[3]-b[1])*MM_TO_PX); self.scene.addRect(rect)
        except Exception as e:
            print('draw error',e)

    def open_dxf(self):
        path,_=QFileDialog.getOpenFileName(self,'Abrir DXF',filter='DXF Files (*.dxf)')
        if not path: return
        doc=ezdxf.readfile(path); msp=doc.modelspace(); self.pieces=[]
        for ent in msp:
            try:
                geom = shapely_from_dxf_entity(ent)
                if geom is None or geom.is_empty: continue
                layer = ent.dxf.layer if hasattr(ent.dxf,'layer') else '0'
                self.pieces.append(Piece(geom,layer,ent.dxftype()))
            except Exception as e:
                print('skip',getattr(ent,'dxftype',lambda:'?')(),e)
        self.draw_workspace()

    def export_dxf(self):
        if not self.pieces: return
        path,_=QFileDialog.getSaveFileName(self,'Exportar DXF',filter='DXF Files (*.dxf)')
        if not path: return
        doc=ezdxf.new('R2010'); msp=doc.modelspace()
        for p in self.pieces:
            g = p.placed_geom() if p.placement else p.orig_geom
            layer = p.layer
            if g.geom_type=='Polygon': msp.add_lwpolyline(list(g.exterior.coords),dxfattribs={'layer':layer})
            else:
                if hasattr(g,'geoms'):
                    for sub in g.geoms:
                        msp.add_lwpolyline(list(sub.exterior.coords),dxfattribs={'layer':layer})
        doc.write(path); print('exported',path)

    def run_placement(self):
        ux,uy,uw,uh = self.workspace.usable_rect(); usable_poly = Polygon([(ux,uy),(ux+uw,uy),(ux+uw,uy+uh),(ux,uy+uh)])
        for p in self.pieces: p.placement=None
        self.placed_offsets=[]; self.placed_bboxes=[]
        ordered = sorted(self.pieces,key=lambda q:-q.orig_geom.area)
        angle_step = max(1,int(self.angle_spin.value())); angles=list(range(0,360,angle_step))
        use_nfp = PYCLIPPER_AVAILABLE
        for piece in ordered:
            placed=False
            for a in angles:
                rotated = rotate(piece.orig_geom,a,origin=(0,0),use_radians=False)
                if use_nfp:
                    forbidden = minkowski_forbidden(rotated,self.placed_offsets)
                    if forbidden is None: use_nfp=False; break
                    feasible = usable_poly.difference(forbidden)
                    if feasible.is_empty: continue
                    candidates = [feasible] if feasible.geom_type=='Polygon' else list(feasible.geoms)
                    best=None; by=None
                    for c in candidates:
                        minx,miny,_,_ = c.bounds
                        if best is None or (miny<by) or (abs(miny-by)<1e-9 and minx<best[0]): best=(minx,miny); by=miny
                    if best is None: continue
                    piece.placement=(best[0],best[1],a)
                    try: off=offset_poly(piece.placed_geom(),self.workspace.clearance)
                    except Exception: off=piece.placed_geom().buffer(self.workspace.clearance)
                    self.placed_offsets.append(off); self.placed_bboxes.append(off.bounds); placed=True; break
                else:
                    minx_r,miny_r,maxx_r,maxy_r = rotated.bounds
                    step = max(1.0,min(5.0,min(uw,uh)/200.0))
                    y=uy
                    while y<=uy+uh:
                        x=ux
                        while x<=ux+uw:
                            dx_min = x+minx_r; dy_min=y+miny_r; dx_max=x+maxx_r; dy_max=y+maxy_r
                            if dx_min<ux-1e-6 or dy_min<uy-1e-6 or dx_max>ux+uw+1e-6 or dy_max>uy+uh+1e-6:
                                x+=step; continue
                            g=translate(rotated,xoff=x,yoff=y)
                            g_offset = offset_poly(g,self.workspace.clearance) if PYCLIPPER_AVAILABLE else g.buffer(self.workspace.clearance)
                            collision=False
                            for idx,ob in enumerate(self.placed_bboxes):
                                if not (g_offset.bounds[2]<ob[0] or g_offset.bounds[0]>ob[2] or g_offset.bounds[3]<ob[1] or g_offset.bounds[1]>ob[3]):
                                    if g_offset.intersects(self.placed_offsets[idx]): collision=True; break
                            if not collision:
                                piece.placement=(x,y,a); self.placed_offsets.append(g_offset); self.placed_bboxes.append(g_offset.bounds); placed=True; break
                            x+=step
                        if placed: break
                        y+=step
                    if placed: break
            if not placed: print('no place',piece.name)
        self.draw_workspace()

    def current_union_area(self):
        geoms=[]
        for p in self.pieces:
            g=p.placed_geom()
            if g is not None: geoms.append(g)
        if not geoms: return 0.0
        try: return unary_union(geoms).area
        except Exception: return sum([g.area for g in geoms])

    def run_refine(self):
        iters = int(self.sa_iters.value()); T0=float(self.sa_temp.value()); time_limit=int(self.sa_time.value())
        if iters<=0: return
        if any(p.placement is None for p in self.pieces): print('Run initial placement first'); return
        ux,uy,uw,uh = self.workspace.usable_rect(); usable_area = uw*uh
        offsets_cache=[]
        for p in self.pieces:
            g=p.placed_geom(); off = offset_poly(g,self.workspace.clearance) if PYCLIPPER_AVAILABLE else g.buffer(self.workspace.clearance); offsets_cache.append(off)
        best_area = self.current_union_area(); best_density = best_area/usable_area if usable_area>0 else 0; best_positions=[(p.placement[0],p.placement[1],p.placement[2]) for p in self.pieces]
        cur_area=best_area; cur_density=best_density; start=time.time()
        for k in range(iters):
            if time_limit>0 and (time.time()-start)>time_limit: print('SA time limit'); break
            frac=k/max(1,iters); T=T0*(1-frac)
            i=random.randrange(len(self.pieces)); piece=self.pieces[i]; old=piece.placement
            step = max(1.0,min(5.0,min(uw,uh)/200.0)); angle_step=max(1,int(self.angle_spin.value())); dx=random.uniform(-step*2,step*2); dy=random.uniform(-step*2,step*2); da=random.uniform(-angle_step,angle_step)
            nx=old[0]+dx; ny=old[1]+dy; na=(old[2]+da)%360
            rotated=rotate(piece.orig_geom,na,origin=(0,0),use_radians=False); minx_r,miny_r,maxx_r,maxy_r=rotated.bounds
            if nx+minx_r<ux-1e-6 or ny+miny_r<uy-1e-6 or nx+maxx_r>ux+uw+1e-6 or ny+maxy_r>uy+uh+1e-6: continue
            newg=translate(rotated,xoff=nx,yoff=ny); newoff = offset_poly(newg,self.workspace.clearance) if PYCLIPPER_AVAILABLE else newg.buffer(self.workspace.clearance)
            collision=False
            for j,og in enumerate(offsets_cache):
                if j==i: continue
                bbox=og.bounds; nb=newoff.bounds
                if not (nb[2]<bbox[0] or nb[0]>bbox[2] or nb[3]<bbox[1] or nb[1]>bbox[3]):
                    if newoff.intersects(og): collision=True; break
            if collision: continue
            piece.placement=(nx,ny,na)
            try: new_area=self.current_union_area()
            except Exception: new_area=cur_area
            new_density=new_area/usable_area if usable_area>0 else 0; delta=new_density-cur_density
            accept=False
            if delta>1e-9: accept=True
            else:
                prob=math.exp(delta/(T+1e-9)) if T>1e-9 else 0
                if random.random()<prob: accept=True
            if accept:
                cur_area=new_area; cur_density=new_density; offsets_cache[i]=newoff
                if cur_density>best_density: best_density=cur_density; best_area=cur_area; best_positions=[(p.placement[0],p.placement[1],p.placement[2]) for p in self.pieces]
            else:
                piece.placement=old
            if k%200==0:
                print(f'SA {k}/{iters} density {cur_density*100:.2f}%'); self.draw_workspace()
        for idx,p in enumerate(self.pieces): p.placement=best_positions[idx]
        self.draw_workspace(); print(f'SA done best_density={best_density*100:.2f}% best_area={best_area:.2f} time={time.time()-start:.1f}s')


if __name__=='__main__':
    app=QApplication(sys.argv); w=MainWindow(); w.show(); sys.exit(app.exec())
