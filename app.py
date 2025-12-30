import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import math
from collections import Counter

# 幾何学計算ライブラリ
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, GeometryCollection
from shapely.affinity import translate
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid

import ezdxf
import ezdxf.path

# Vカービング用
from scipy.spatial import Voronoi

# --- 0. ポストプロセッサ定義 ---
POST_PROCESSORS = {
    "Generic (汎用)": {
        "desc": "一般的なCNCルーター向け",
        "start": "G21 ; Metric\nG90 ; Absolute\nG00 Z10.0 ; Safe Z\nM3 S10000 ; Spindle On",
        "end": "M5 ; Spindle Off\nG00 Z10.0\nM30 ; End",
        "format": "G00/G01"
    },
    "GRBL / Candle": {
        "desc": "Arduinoベース (3018等)",
        "start": "G21 G90 G17\nG0 Z10.0\nM3 S10000",
        "end": "M5\nG0 Z10.0\nM30",
        "format": "G0/G1"
    },
    "Mach3 / Mach4": {
        "desc": "Mach3/4 コントローラ",
        "start": "G21 G90 G40 G80\nG00 Z10.0\nM03 S10000",
        "end": "M05\nG00 Z10.0\nM30",
        "format": "G00/G01"
    }
}

# --- 1. 幾何学ユーティリティ ---

def ensure_list_of_polys(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        return list(geometry.geoms)
    elif geometry.geom_type == 'GeometryCollection':
        polys = []
        for g in geometry.geoms:
            if g.geom_type == 'Polygon':
                polys.append(g)
            elif g.geom_type == 'MultiPolygon':
                polys.extend(g.geoms)
        return polys
    return []

def dist_lseg(l1, l2, p):
    x0, y0 = l1[0], l1[1]
    xa, ya = l2[0], l2[1]
    xi, yi = p[0], p[1]
    dx, dy = xa-x0, ya-y0
    d2 = dx*dx + dy*dy
    if d2 == 0: return 0
    t = max(0, min(1, ((xi-x0)*dx + (yi-y0)*dy) / d2))
    return math.sqrt((xi - (x0 + t*dx))**2 + (yi - (y0 + t*dy))**2)

def douglas_peucker(points, tolerance):
    if len(points) < 3: return points
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = dist_lseg(points[0], points[end], points[i])
        if d > dmax:
            index = i
            dmax = d
    if dmax > tolerance:
        return douglas_peucker(points[:index+1], tolerance)[:-1] + douglas_peucker(points[index:], tolerance)
    else:
        return [points[0], points[end]]

# --- 2. データ読み込み ---

def dxf_to_shapely_list(dxf_bytes):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(dxf_bytes)
        tmp_path = tmp.name
    try:
        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()
        polys = []
        for e in msp:
            if e.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'SPLINE', 'CIRCLE'):
                try:
                    if e.dxftype() == 'CIRCLE':
                        poly = Point(e.dxf.center[:2]).buffer(e.dxf.radius, resolution=64)
                    else:
                        p_obj = ezdxf.path.make_path(e)
                        # Ver 3.7の値に戻す
                        pts = list(p_obj.flattening(0.01))
                        poly = Polygon([(v.x, v.y) for v in pts]) if len(pts) > 2 else None
                    if poly and poly.is_valid and poly.area > 0.0001:
                        polys.append(poly)
                    elif poly:
                        clean = make_valid(poly)
                        if clean.area > 0.0001:
                            if clean.geom_type == 'Polygon':
                                polys.append(clean)
                            else:
                                polys.extend(clean.geoms)
                except:
                    pass
        polys.sort(key=lambda x: x.area, reverse=True)
        return polys
    except Exception as e:
        st.error(f"DXF Read Error: {e}")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def merge_polygons_xor(polys):
    if not polys: return None
    combined = Polygon()
    for p in polys:
        combined = combined.symmetric_difference(p) if not combined.is_empty else p
    return combined

def analyze_holes(geometry):
    polys = ensure_list_of_polys(geometry)
    sizes = []
    def check_p(p):
        minx, miny, maxx, maxy = p.bounds
        w, h = maxx - minx, maxy - miny
        if abs(w - h) > 0.1: return None
        expected_area = math.pi * ((w/2)**2)
        if abs(p.area - expected_area) / expected_area > 0.2: return None
        return w
    for p in polys:
        d = check_p(p)
        if d:
            sizes.append(round(d, 2))
        for interior in p.interiors:
            d_h = check_p(Polygon(interior))
            if d_h:
                sizes.append(round(d_h, 2))
    return Counter(sizes)

# --- 3. 加工パス生成 ---

def find_drill_points(geometry, target_dia, tolerance=0.1):
    polys = ensure_list_of_polys(geometry)
    drill_points = []
    def check_p(p):
        minx, miny, maxx, maxy = p.bounds
        w, h = maxx - minx, maxy - miny
        if abs(w - h) > tolerance: return None
        if not (target_dia - tolerance <= w <= target_dia + tolerance): return None
        return p.centroid
    for p in polys:
        pt = check_p(p)
        if pt:
            drill_points.append(pt)
        for interior in p.interiors:
            pt_h = check_p(Polygon(interior))
            if pt_h:
                drill_points.append(pt_h)
    return drill_points

def generate_drill_gcode(points, z_start, z_final, peck_depth, feed, tool_name, header, footer, fmt):
    if not points: return None
    gc = [header.strip(), f"; Tool: {tool_name} (Drill)", "T1 M06", f"F{int(feed)}", ""]
    G0 = "G0" if "G0/" in fmt else "G00"
    G1 = "G1" if "G0/" in fmt else "G01"
    safe = 5.0
    for pt in points:
        gc.append(f"; Drill X{pt.x:.3f} Y{pt.y:.3f}\n{G0} X{pt.x:.3f} Y{pt.y:.3f}\n{G0} Z{z_start + 1.0}")
        current_z = z_start
        while current_z > z_final:
            target_z = max(current_z - peck_depth, z_final)
            gc.append(f"{G1} Z{target_z:.3f}")
            if target_z > z_final:
                gc.append(f"{G0} Z{z_start + 0.5}\n{G0} Z{target_z + 0.5}")
            current_z = target_z
        gc.append(f"{G0} Z{safe}")
    gc.append(footer.strip())
    return "\n".join(gc)

def generate_pocket(geometry, tool_d, clearance, stepover):
    paths_rough = []
    paths_finish = []
    r = tool_d / 2.0
    step = tool_d * stepover
    offset_rough = -(r + clearance)
    try:
        current_rough = geometry.buffer(offset_rough, join_style=2)
    except:
        current_rough = Polygon()
    while not current_rough.is_empty and current_rough.area > 0.01:
        current_polys = ensure_list_of_polys(current_rough)
        for p in current_polys:
            paths_rough.append(p.exterior)
            paths_rough.extend(p.interiors)
        try:
            current_rough = current_rough.buffer(-step, join_style=2)
        except:
            break
    if clearance > 0:
        try:
            finish_pass = geometry.buffer(-r, join_style=2)
            finish_polys = ensure_list_of_polys(finish_pass)
            for p in finish_polys:
                paths_finish.append(p.exterior)
                paths_finish.extend(p.interiors)
        except:
            pass
    return ([LineString(p.coords) for p in paths_rough if p.length > 0.1], 
            [LineString(p.coords) for p in paths_finish if p.length > 0.1])

def generate_chamfer_separated(geometry, tip_offset, finish_allowance=0.0):
    rough_paths = []
    finish_paths = []
    if finish_allowance > 0:
        try:
            p_rough = geometry.buffer(tip_offset + finish_allowance, join_style=1)
            p_list = ensure_list_of_polys(p_rough)
            for poly in p_list:
                rough_paths.append(poly.exterior)
                rough_paths.extend(poly.interiors)
        except:
            pass
    try:
        p_finish = geometry.buffer(tip_offset, join_style=1)
        p_list_f = ensure_list_of_polys(p_finish)
        for poly in p_list_f:
            finish_paths.append(poly.exterior)
            finish_paths.extend(poly.interiors)
    except:
        pass
    return ([LineString(ls.coords) for ls in rough_paths], 
            [LineString(ls.coords) for ls in finish_paths])

# --- Vカーブ ロジック (Ver 3.7仕様へロールバック) ---

class PathGraph:
    def __init__(self):
        self.adj = {}
    def add_edge(self, p1, p2):
        p1, p2 = (round(p1[0], 3), round(p1[1], 3)), (round(p2[0], 3), round(p2[1], 3))
        if p1 == p2: return
        if p1 not in self.adj: self.adj[p1] = set()
        if p2 not in self.adj: self.adj[p2] = set()
        self.adj[p1].add(p2); self.adj[p2].add(p1)
    def prune(self, min_len=4.0): # Ver 3.7の値に戻す
        for _ in range(10):
            leaves = [n for n, neigh in self.adj.items() if len(neigh) == 1]
            if not leaves: break
            for leaf in leaves:
                if leaf not in self.adj: continue
                neighbor = list(self.adj[leaf])[0]
                if math.sqrt((leaf[0]-neighbor[0])**2 + (leaf[1]-neighbor[1])**2) < min_len and len(self.adj[neighbor]) > 2:
                    self.adj[neighbor].remove(leaf); del self.adj[leaf]
    def get_chains(self):
        chains = []; visited = set(); nodes = sorted(self.adj.keys(), key=lambda n: (len(self.adj[n])%2!=1, -len(self.adj[n])))
        for start in nodes:
            if start not in self.adj: continue
            for neighbor in list(self.adj[start]):
                edge = tuple(sorted((start, neighbor)))
                if edge in visited: continue
                chain = [start, neighbor]; visited.add(edge); curr = neighbor
                while True:
                    cands = [n for n in self.adj[curr] if tuple(sorted((curr, n))) not in visited]
                    if len(cands) == 1:
                        nxt = cands[0]; visited.add(tuple(sorted((curr, nxt)))); chain.append(nxt); curr = nxt
                        if len(self.adj[curr]) > 2: break
                    else: break
                if len(chain) > 1: chains.append(chain)
        return chains

def generate_vcarve(geometry, angle_deg, use_limit, max_d, step_len=0.1, z_offset=0.0):
    polys = ensure_list_of_polys(geometry)
    if not polys: return []
    tan_a = np.tan(np.radians(angle_deg/2)); graph = PathGraph()
    combined_geom = unary_union(polys); boundary = combined_geom.boundary 
    for poly in polys:
        # Ver 3.7の値に戻す
        smooth = poly.simplify(0.02, preserve_topology=True)
        line = smooth.exterior
        # Ver 3.7の値に戻す
        sample_res = 0.2
        num = max(50, min(8000, int(line.length / sample_res)))
        pts = [line.interpolate(i * line.length / (num-1)) for i in range(num)]
        coords = np.array([(p.x, p.y) for p in pts])
        try: Voronoi(coords)
        except: continue
        for (p1_idx, p2_idx), (v1_idx, v2_idx) in zip(vor.ridge_points, vor.ridge_vertices):
            if v1_idx < 0 or v2_idx < 0: continue
            v1, v2 = vor.vertices[v1_idx], vor.vertices[v2_idx]
            if combined_geom.contains(Point(v1)) and combined_geom.contains(Point(v2)):
                if np.linalg.norm(vor.points[p1_idx] - vor.points[p2_idx]) > sample_res * 5.0:
                    graph.add_edge(v1, v2)
    
    graph.prune(min_len=4.0) # Ver 3.7の値に戻す
    chains = graph.get_chains()
    all_paths = []
    for chain in chains:
        path_3d = []
        ls = LineString(chain)
        pts_count = max(2, int(ls.length / step_len) + 1)
        for i in range(pts_count):
            pt = ls.interpolate(i * ls.length / (pts_count - 1))
            d = pt.distance(boundary)
            z = -(d / tan_a) + z_offset
            if z > 0: z = 0
            if use_limit and z < max_d: z = max_d
            path_3d.append((pt.x, pt.y, z))
        if len(path_3d) > 1: all_paths.append(douglas_peucker(path_3d, 0.02))
    return all_paths

# --- 4. Gコードエンジン ---

def make_gcode_phases_advanced(phases, tool_name, header, footer, fmt="G00/G01", is_3d=False):
    gc = [header.strip(), f"; Tool: {tool_name}", "T1 M06"]
    G0 = "G0" if "G0/" in fmt else "G00"
    G1 = "G1" if "G0/" in fmt else "G01"
    safe = 5.0
    gc.append(f"{G0} Z{safe}")
    for i, phase in enumerate(phases):
        paths = phase.get('paths', [])
        if not paths: continue
        feed = int(phase.get('feed', 300))
        z_start = phase.get('z_start', 0); z_final = phase.get('z_final', 0)
        z_step = max(0.01, phase.get('z_step', abs(z_final - z_start)))
        gc.append(f"; Phase {i+1}: {phase.get('name','')} (F{feed})"); gc.append(f"F{feed}")
        if is_3d:
            for p_pts in paths:
                gc.append(f"{G0} X{p_pts[0][0]:.3f} Y{p_pts[0][1]:.3f}\n{G1} Z{p_pts[0][2]:.3f}")
                for p in p_pts[1:]: gc.append(f"{G1} X{p[0]:.3f} Y{p[1]:.3f} Z{p[2]:.3f}")
                gc.append(f"{G0} Z{safe}")
        else:
            cur_z = z_start
            while cur_z > z_final:
                tgt_z = max(cur_z - z_step, z_final)
                for path in paths:
                    coords = np.array(path.coords)
                    gc.append(f"{G0} X{coords[0,0]:.3f} Y{coords[0,1]:.3f}\n{G0} Z{cur_z + 1.0}\n{G1} Z{tgt_z:.3f}")
                    for xy in coords[1:]: gc.append(f"{G1} X{xy[0]:.3f} Y{xy[1]:.3f}")
                    gc.append(f"{G0} Z{safe}")
                cur_z = tgt_z
    gc.append(footer.strip()); return "\n".join(gc)

# --- 5. UI ---

st.set_page_config(page_title="yosikeiCAM", layout="wide")
st.title("yosikeiCAM 3.9")
st.caption("Ver 3.9: Vカーブ計算ロジック ロールバック版")

with st.sidebar:
    st.header("原点設定")
    origin = st.radio("原点基準", ["Bottom-Left (左下)", "Center (中心)", "Original (CAD座標)"], index=0)
    st.divider(); st.header("加工設定")
    tab1, tab2, tab3, tab4 = st.tabs(["ポケット", "面取り", "Vカーブ", "ドリル"])
    
    with tab1:
        enable_p = st.checkbox("有効", value=True, key="cp")
        dia = st.number_input("工具径 (mm)", value=3.0, step=0.1)
        ucp = st.checkbox("2回加工 (粗+仕上げ)", value=False, key="uccp")
        clear = 0.0; fed_p_rough = st.number_input("送り速度 (粗)", value=300, min_value=1, key="fp_r")
        fed_p_finish = fed_p_rough; finish_mode = "Step-down"
        if ucp:
            clear = st.number_input("仕上げ代 (mm)", value=0.2, step=0.1)
            c3, c4 = st.columns(2)
            with c3: fed_p_finish = st.number_input("仕上げ送り", value=300, min_value=1, key="fp_f")
            with c4:
                f_opt = st.radio("仕上げ深さ", ["ピッチ刻み", "最終一括"], index=0, key="f_m_p")
                finish_mode = "Full-Depth" if f_opt == "最終一括" else "Step-down"
        c1, c2 = st.columns(2)
        with c1: dep_p = st.number_input("最終深さ Z", value=-1.0, step=0.1)
        with c2: stp_p = st.number_input("切込ピッチ", value=1.0, min_value=0.01)
    
    with tab2:
        enable_c = st.checkbox("有効", value=True, key="cc")
        cw = st.number_input("面取り幅 (mm)", value=0.5, step=0.1)
        to = st.number_input("刃先オフセット", value=1.0, step=0.1)
        z_c = -(cw + to)
        st.caption(f"切込深さ: {z_c:.2f}mm")
        fed_c_r = st.number_input("送り速度 (通常/粗)", value=300, min_value=1, key="fc_r")
        ucf = st.checkbox("2回加工 (粗+仕上げ)", value=False, key="uccf")
        cfa = 0.0; fed_c_f = fed_c_r
        if ucf:
            c1, c2 = st.columns(2)
            with c1: cfa = st.number_input("仕上げ代 (mm)", value=0.2, step=0.1)
            with c2: fed_c_f = st.number_input("仕上げ送り", value=300, min_value=1, key="fc_f")
        
    with tab3:
        enable_v = st.checkbox("有効", value=False, key="cv")
        va = st.number_input("角度 (度)", value=60.0, step=10.0)
        ucv = st.checkbox("2回加工 (粗+仕上げ)", value=False, key="uccv")
        v_cl = 0.0; fed_v_f = 300; fed_v = st.number_input("送り速度 (通常/粗)", value=300, min_value=1, key="fv")
        if ucv:
            v_cl = st.number_input("仕上げ代 (mm)", value=0.2, step=0.1)
            fed_v_f = st.number_input("仕上げ送り", value=300, min_value=1, key="fv_f")
        uvl = st.checkbox("深さ制限", value=False); vl = st.number_input("最大深さ (mm)", value=-3.0) if uvl else -100.0
        vr = st.slider("精度", value=0.05, min_value=0.02, max_value=0.2)
        
    with tab4:
        enable_d = st.checkbox("有効", value=False, key="cd")
        ddt = st.number_input("対象径 (mm)", value=3.0); ddz = st.number_input("深さ Z", value=-5.0)
        pck = st.number_input("ペック量", value=2.0, min_value=0.1); fd = st.number_input("送り", value=200, key="fd")
        
    st.divider(); ppn = st.selectbox("ポストプロセッサ", list(POST_PROCESSORS.keys())); pp = POST_PROCESSORS[ppn]
    h_c = st.text_area("Header", pp["start"]); f_c = st.text_area("Footer", pp["end"])

st.header("DXFアップロード")
f = st.file_uploader("", type=["dxf"])

if f:
    bn = os.path.splitext(f.name)[0]; polys_raw = dxf_to_shapely_list(f.getvalue())
    if polys_raw:
        tu = unary_union(polys_raw); minx, miny, maxx, maxy = tu.bounds; ox, oy = 0, 0
        w_size, h_size = maxx-minx, maxy-miny
        if origin.startswith("Bottom-Left"): ox, oy = -minx, -miny
        elif origin.startswith("Center"): ox, oy = -(minx+w_size/2), -(miny+h_size/2)
        polys_moved = [translate(p, ox, oy) for p in polys_raw]
        
        st.sidebar.divider(); st.sidebar.subheader("パス選択")
        cont = st.sidebar.container(); all_c = cont.checkbox("すべて選択", value=True, key="sa")
        selected = []
        for i, p in enumerate(polys_moved):
            if cont.checkbox(f"Path #{i+1} (Area:{p.area:.1f})", value=all_c, key=f"p{i}"):
                selected.append(i)
                
        target_polys = [polys_moved[i] for i in selected]; gfc = merge_polygons_xor(target_polys)
        if gfc:
            hs = analyze_holes(gfc)
            if hs: st.sidebar.info("検出された円: " + ", ".join([f"φ{d}({c}個)" for d, c in hs.items()]))
        
        # 加工サイズ表示
        st.success(f"加工サイズ: {w_size:.1f} x {h_size:.1f} mm")
        st.header("パス生成")
        
        c1, c2 = st.columns(2)
        with c1:
            # オリジナルプレビュー (グリッドあり)
            fig, ax = plt.subplots(figsize=(5,5))
            ax.plot(0, 0, 'r+', markersize=20)
            ax.axhline(0, color='red', alpha=0.3); ax.axvline(0, color='red', alpha=0.3)
            for i, p in enumerate(polys_moved):
                ax.plot(*p.exterior.xy, 'k-' if i in selected else 'k:', alpha=1.0 if i in selected else 0.1)
                for interior in p.interiors: ax.plot(*interior.xy, 'k-' if i in selected else 'k:', alpha=1.0 if i in selected else 0.1)
            ax.axis('equal'); ax.grid(True, linestyle=':'); st.pyplot(fig)
            
        with c2:
            # パス計算
            gc_v, gc_p, gc_c, gc_d = None, None, None, None; v_d, p_r_d, p_f_d, c_d = [], [], [], []
            if gfc and not gfc.is_empty:
                if enable_v:
                    with st.spinner("VCarve..."):
                        v_r_ps = generate_vcarve(gfc, va, uvl, vl, vr, z_offset=v_cl) if ucv else []
                        v_f_ps = generate_vcarve(gfc, va, uvl, vl, vr, z_offset=0.0); v_d = v_r_ps + v_f_ps; phs_v = []
                        if v_r_ps: phs_v.append({'name':'V-Rough','paths':v_r_ps,'z_start':0,'z_final':0,'feed':fed_v})
                        if v_f_ps: phs_v.append({'name':'V-Finish','paths':v_f_ps,'z_start':0,'z_final':0,'feed':fed_v_f if ucv else fed_v})
                        if phs_v: gc_v = make_gcode_phases_advanced(phs_v, "VBit", h_c, f_c, pp["format"], True)
                if enable_p:
                    p_r, p_f = generate_pocket(gfc, dia, clear if ucp else 0.0, 0.5); p_r_d, p_f_d = p_r, p_f; phs = []
                    if p_r: phs.append({'name':'Rough','paths':p_r,'z_start':0,'z_final':dep_p,'feed':fed_p_rough,'z_step':stp_p})
                    if p_f and ucp: phs.append({'name':'Finish','paths':p_f,'z_start':0,'z_final':dep_p,'feed':fed_p_finish,'z_step':(abs(dep_p) if finish_mode=="Full-Depth" else stp_p)})
                    if phs: gc_p = make_gcode_phases_advanced(phs, "EndMill", h_c, f_c, pp["format"])
                if enable_c:
                    rp_c, fp_c = generate_chamfer_separated(gfc, to, cfa if ucf else 0.0); c_d = rp_c + fp_c; phs_c = []
                    if rp_c and ucf: phs_c.append({'name':'Rough','paths':rp_c,'z_start':0,'z_final':z_c,'feed':fed_c_r})
                    if fp_c: phs_c.append({'name':'Finish','paths':fp_c,'z_start':0,'z_final':z_c,'feed':fed_c_f if ucf else fed_c_r})
                    if phs_c: gc_c = make_gcode_phases_advanced(phs_c, "Chamfer", h_c, f_c, pp["format"])
                if enable_d:
                    dpts = find_drill_points(gfc, ddt); gc_d = generate_drill_gcode(dpts, 0, ddz, pck, fd, f"Drill {ddt}mm", h_c, f_c, pp["format"])

            # 生成プレビュー (左側とグリッドを完全に同期)
            fig2, ax2 = plt.subplots(figsize=(5,5))
            ax2.plot(0, 0, 'r+', markersize=20)
            ax2.axhline(0, color='red', alpha=0.3); ax2.axvline(0, color='red', alpha=0.3)
            for p in polys_moved: ax2.plot(*p.exterior.xy, 'k--', alpha=0.05)
            for ls in p_r_d: ax2.plot(*ls.xy, color='tab:blue', alpha=0.3)
            for ls in p_f_d: ax2.plot(*ls.xy, color='tab:cyan', alpha=0.8)
            for ls in c_d: ax2.plot(*ls.xy, color='tab:green', alpha=0.8)
            for pts in v_d: ax2.plot([p[0] for p in pts], [p[1] for p in pts], 'r-', linewidth=0.8)
            ax2.axis('equal'); ax2.grid(True, linestyle=':'); st.pyplot(fig2)
            
            col1, col2 = st.columns(2)
            if gc_v: col1.download_button("VCARVE 保存", gc_v, f"{bn}_vcarve.nc")
            if gc_p: col2.download_button("POCKET 保存", gc_p, f"{bn}_pocket.nc")
            if gc_c: col1.download_button("CHAMFER 保存", gc_c, f"{bn}_chamfer.nc")
            if gc_d: col2.download_button("DRILL 保存", gc_d, f"{bn}_drill.nc")
    else: st.error("図形が見つかりません。")
