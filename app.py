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

# --- 0. 多言語定義 (辞書) ---
LANG_DICT = {
    "Japanese": {
        "title": "yosikeiCAM 4.1",
        "origin_setting": "原点設定",
        "origin_option": ["左下 (Bottom-Left)", "中心 (Center)", "CAD座標 (Original)"],
        "process_setting": "加工設定",
        "tabs": ["ポケット", "面取り", "Vカーブ", "ドリル"],
        "enable": "有効",
        "tool_dia": "工具径 (mm)",
        "double_pass": "2回加工 (粗+仕上げ)",
        "finish_allowance": "仕上げ代 (mm)",
        "rough_feed": "送り速度 (粗)",
        "finish_feed": "仕上げ送り",
        "finish_depth_mode": "仕上げ深さ",
        "depth_options": ["ピッチ刻み", "最終一括"],
        "final_depth": "最終深さ Z",
        "pitch": "切込ピッチ",
        "chamfer_width": "幅 (mm)",
        "tip_offset": "刃先オフセット",
        "v_angle": "角度 (度)",
        "depth_limit": "深さ制限",
        "max_depth": "最大深さ (mm)",
        "precision": "精度",
        "drill_dia": "対象径 (mm)",
        "drill_depth": "深さ Z",
        "peck": "ペック量",
        "feed": "送り速度",
        "post_processor": "ポストプロセッサ",
        "dxf_upload": "DXFアップロード",
        "path_select": "パス選択",
        "select_all": "すべて選択",
        "size_display": "加工サイズ",
        "gen_header": "パス生成",
        "download": ["VCARVE 保存", "POCKET 保存", "CHAMFER 保存", "DRILL 保存"],
        "error_no_geom": "図形が見つかりません。",
        "hole_detected": "検出された円"
    },
    "English": {
        "title": "yosikeiCAM 4.1",
        "origin_setting": "Origin Setup",
        "origin_option": ["Bottom-Left", "Center", "Original (CAD)"],
        "process_setting": "Machining Setup",
        "tabs": ["Pocket", "Chamfer", "V-Carve", "Drill"],
        "enable": "Enable",
        "tool_dia": "Tool Dia (mm)",
        "double_pass": "Double Pass (Rough+Finish)",
        "finish_allowance": "Finish Allowance (mm)",
        "rough_feed": "Rough Feed Rate",
        "finish_feed": "Finish Feed Rate",
        "finish_depth_mode": "Finish Depth Mode",
        "depth_options": ["Step-down", "Full-Depth"],
        "final_depth": "Final Depth Z",
        "pitch": "Z Step Pitch",
        "chamfer_width": "Width (mm)",
        "tip_offset": "Tip Offset",
        "v_angle": "V-Angle (deg)",
        "depth_limit": "Depth Limit",
        "max_depth": "Max Depth (mm)",
        "precision": "Precision",
        "drill_dia": "Hole Dia (mm)",
        "drill_depth": "Depth Z",
        "peck": "Peck Amount",
        "feed": "Feed Rate",
        "post_processor": "Post Processor",
        "dxf_upload": "DXF Upload",
        "path_select": "Path Selection",
        "select_all": "Select All",
        "size_display": "Machining Size",
        "gen_header": "Toolpath Generation",
        "download": ["Save VCARVE", "Save POCKET", "Save CHAMFER", "Save DRILL"],
        "error_no_geom": "No valid geometry found.",
        "hole_detected": "Circles detected"
    }
}

# --- 1. ポストプロセッサ定義 ---
POST_PROCESSORS = {
    "Generic (汎用)": {
        "start": "G21 ; Metric\nG90 ; Absolute\nG00 Z10.0 ; Safe Z\nM3 S10000 ; Spindle On",
        "end": "M5 ; Spindle Off\nG00 Z10.0\nM30 ; End",
        "format": "G00/G01"
    },
    "GRBL / Candle": {
        "start": "G21 G90 G17\nG0 Z10.0\nM3 S10000",
        "end": "M5 G0 Z10.0\nM30",
        "format": "G0/G1"
    },
    "Mach3 / Mach4": {
        "start": "G21 G90 G40 G80\nG00 Z10.0\nM03 S10000",
        "end": "M05\nG00 Z10.0\nM30",
        "format": "G00/G01"
    }
}

# --- 2. 幾何学ユーティリティ (中略：V 4.0と同様) ---

def ensure_list_of_polys(geometry):
    if geometry is None or geometry.is_empty: return []
    if geometry.geom_type == 'Polygon': return [geometry]
    elif geometry.geom_type == 'MultiPolygon': return list(geometry.geoms)
    elif geometry.geom_type == 'GeometryCollection':
        polys = []
        for g in geometry.geoms:
            if g.geom_type == 'Polygon': polys.append(g)
            elif g.geom_type == 'MultiPolygon': polys.extend(g.geoms)
        return polys
    return []

def dist_lseg(l1, l2, p):
    x0, y0 = l1[0], l1[1]; xa, ya = l2[0], l2[1]; xi, yi = p[0], p[1]
    dx, dy = xa-x0, ya-y0; d2 = dx*dx + dy*dy
    if d2 == 0: return 0
    t = max(0, min(1, ((xi-x0)*dx + (yi-y0)*dy) / d2))
    return math.sqrt((xi - (x0 + t*dx))**2 + (yi - (y0 + t*dy))**2)

def douglas_peucker(points, tolerance):
    if len(points) < 3: return points
    dmax, index, end = 0, 0, len(points) - 1
    for i in range(1, end):
        d = dist_lseg(points[0], points[end], points[i])
        if d > dmax: index, dmax = i, d
    if dmax > tolerance:
        return douglas_peucker(points[:index+1], tolerance)[:-1] + douglas_peucker(points[index:], tolerance)
    else: return [points[0], points[end]]

def dxf_to_shapely_list(dxf_bytes):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(dxf_bytes); tmp_path = tmp.name
    try:
        doc = ezdxf.readfile(tmp_path); msp = doc.modelspace(); polys = []
        for e in msp:
            if e.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'SPLINE', 'CIRCLE'):
                try:
                    if e.dxftype() == 'CIRCLE':
                        poly = Point(e.dxf.center[:2]).buffer(e.dxf.radius, resolution=64)
                    else:
                        p_obj = ezdxf.path.make_path(e); pts = list(p_obj.flattening(0.01))
                        poly = Polygon([(v.x, v.y) for v in pts]) if len(pts) > 2 else None
                    if poly and poly.is_valid and poly.area > 0.0001: polys.append(poly)
                    elif poly:
                        clean = make_valid(poly)
                        if clean.area > 0.0001:
                            if clean.geom_type == 'Polygon': polys.append(clean)
                            else: polys.extend(clean.geoms)
                except: pass
        polys.sort(key=lambda x: x.area, reverse=True); return polys
    except Exception as e: st.error(f"DXF Read Error: {e}"); return []
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

def merge_polygons_xor(polys):
    if not polys: return None
    combined = Polygon()
    for p in polys: combined = combined.symmetric_difference(p) if not combined.is_empty else p
    return combined

def find_drill_points(geometry, target_dia, tolerance=0.1):
    polys = ensure_list_of_polys(geometry); drill_points = []
    def check_p(p):
        minx, miny, maxx, maxy = p.bounds; w, h = maxx - minx, maxy - miny
        if abs(w - h) > tolerance: return None
        if not (target_dia - tolerance <= w <= target_dia + tolerance): return None
        return p.centroid
    for p in polys:
        pt = check_p(p); if pt: drill_points.append(pt)
        for interior in p.interiors:
            pt_h = check_p(Polygon(interior)); if pt_h: drill_points.append(pt_h)
    return drill_points

def generate_drill_gcode(points, z_start, z_final, peck_depth, feed, tool_name, header, footer, fmt):
    if not points: return None
    gc = [header.strip(), f"; Tool: {tool_name}", "T1 M06", f"F{int(feed)}", ""]
    G0, G1 = ("G0", "G1") if "G0/" in fmt else ("G00", "G01"); safe = 5.0
    for pt in points:
        gc.append(f"; Drill X{pt.x:.3f} Y{pt.y:.3f}\n{G0} X{pt.x:.3f} Y{pt.y:.3f}\n{G0} Z{z_start + 1.0}")
        current_z = z_start
        while current_z > z_final:
            target_z = max(current_z - peck_depth, z_final); gc.append(f"{G1} Z{target_z:.3f}")
            if target_z > z_final: gc.append(f"{G0} Z{z_start + 0.5}\n{G0} Z{target_z + 0.5}")
            current_z = target_z
        gc.append(f"{G0} Z{safe}")
    gc.append(footer.strip()); return "\n".join(gc)

def generate_pocket(geometry, tool_d, clearance, stepover):
    paths_rough, paths_finish = [], []; r, step = tool_d / 2.0, tool_d * stepover
    offset_rough = -(r + clearance)
    try: current_rough = geometry.buffer(offset_rough, join_style=2)
    except: current_rough = Polygon()
    while not current_rough.is_empty and current_rough.area > 0.01:
        for p in ensure_list_of_polys(current_rough): paths_rough.append(p.exterior); paths_rough.extend(p.interiors)
        try: current_rough = current_rough.buffer(-step, join_style=2)
        except: break
    if clearance > 0:
        try:
            for p in ensure_list_of_polys(geometry.buffer(-r, join_style=2)): paths_finish.append(p.exterior); paths_finish.extend(p.interiors)
        except: pass
    return ([LineString(p.coords) for p in paths_rough if p.length > 0.1], [LineString(p.coords) for p in paths_finish if p.length > 0.1])

def generate_chamfer_separated(geometry, tip_offset, finish_allowance=0.0):
    rough_paths, finish_paths = [], []
    if finish_allowance > 0:
        try:
            for poly in ensure_list_of_polys(geometry.buffer(tip_offset + finish_allowance, join_style=1)): rough_paths.append(poly.exterior); rough_paths.extend(poly.interiors)
        except: pass
    try:
        for poly in ensure_list_of_polys(geometry.buffer(tip_offset, join_style=1)): finish_paths.append(poly.exterior); finish_paths.extend(poly.interiors)
    except: pass
    return ([LineString(ls.coords) for ls in rough_paths], [LineString(ls.coords) for ls in finish_paths])

class PathGraph:
    def __init__(self): self.adj = {}
    def add_edge(self, p1, p2):
        p1, p2 = (round(p1[0], 3), round(p1[1], 3)), (round(p2[0], 3), round(p2[1], 3))
        if p1 == p2: return
        if p1 not in self.adj: self.adj[p1] = set()
        if p2 not in self.adj: self.adj[p2] = set()
        self.adj[p1].add(p2); self.adj[p2].add(p1)
    def prune(self, min_len=4.0):
        for _ in range(10):
            leaves = [n for n, neigh in self.adj.items() if len(neigh) == 1]
            if not leaves: break
            for leaf in leaves:
                if leaf not in self.adj: continue
                neighbor = list(self.adj[leaf])[0]
                if math.sqrt((leaf[0]-neighbor[0])**2 + (leaf[1]-neighbor[1])**2) < min_len and len(self.adj[neighbor]) > 2:
                    self.adj[neighbor].remove(leaf); del self.adj[leaf]
    def get_chains(self):
        chains, visited = [], set(); nodes = sorted(self.adj.keys(), key=lambda n: (len(self.adj[n])%2!=1, -len(self.adj[n])))
        for start in nodes:
            if start not in self.adj: continue
            for neighbor in list(self.adj[start]):
                edge = tuple(sorted((start, neighbor)))
                if edge in visited: continue
                chain, curr = [start, neighbor], neighbor; visited.add(edge)
                while True:
                    cands = [n for n in self.adj[curr] if tuple(sorted((curr, n))) not in visited]
                    if len(cands) == 1:
                        nxt = cands[0]; visited.add(tuple(sorted((curr, nxt)))); chain.append(nxt); curr = nxt
                        if len(self.adj[curr]) > 2: break
                    else: break
                if len(chain) > 1: chains.append(chain)
        return chains

def generate_vcarve(geometry, angle_deg, use_limit, max_d, step_len=0.1, z_offset=0.0):
    polys = ensure_list_of_polys(geometry); if not polys: return []
    tan_a, graph, boundary = np.tan(np.radians(angle_deg/2)), PathGraph(), unary_union(polys).boundary
    for poly in polys:
        smooth = poly.simplify(0.02, preserve_topology=True); line = smooth.exterior; sample_res = 0.2
        num = max(50, min(8000, int(line.length / sample_res)))
        coords = np.array([(p.x, p.y) for p in [line.interpolate(i * line.length / (num-1)) for i in range(num)]])
        try:
            vor_obj = Voronoi(coords)
            for (p1_idx, p2_idx), (v1_idx, v2_idx) in zip(vor_obj.ridge_points, vor_obj.ridge_vertices):
                if v1_idx < 0 or v2_idx < 0: continue
                v1, v2 = vor_obj.vertices[v1_idx], vor_obj.vertices[v2_idx]
                if unary_union(polys).contains(Point(v1)) and unary_union(polys).contains(Point(v2)):
                    if np.linalg.norm(vor_obj.points[p1_idx] - vor_obj.points[p2_idx]) > sample_res * 5.0: graph.add_edge(v1, v2)
        except: continue
    graph.prune(min_len=4.0); chains, all_paths = graph.get_chains(), []
    for chain in chains:
        path_3d, ls = [], LineString(chain); pts_count = max(2, int(ls.length / step_len) + 1)
        for i in range(pts_count):
            pt = ls.interpolate(i * ls.length / (pts_count - 1)); z = -(pt.distance(boundary) / tan_a) + z_offset
            if z > 0: z = 0
            if use_limit and z < max_d: z = max_d
            path_3d.append((pt.x, pt.y, z))
        if len(path_3d) > 1: all_paths.append(douglas_peucker(path_3d, 0.02))
    return all_paths

def make_gcode_phases_advanced(phases, tool_name, header, footer, fmt="G00/G01", is_3d=False):
    gc = [header.strip(), f"; Tool: {tool_name}", "T1 M06"]; G0, G1 = ("G0", "G1") if "G0/" in fmt else ("G00", "G01"); safe = 5.0; gc.append(f"{G0} Z{safe}")
    for i, phase in enumerate(phases):
        if not phase.get('paths', []): continue
        feed, z_start, z_final = int(phase.get('feed', 300)), phase.get('z_start', 0), phase.get('z_final', 0)
        z_step = max(0.01, phase.get('z_step', abs(z_final - z_start)))
        gc.append(f"; Phase {i+1}: {phase.get('name','')} (F{feed})\nF{feed}")
        if is_3d:
            for p_pts in phase['paths']:
                gc.append(f"{G0} X{p_pts[0][0]:.3f} Y{p_pts[0][1]:.3f}\n{G1} Z{p_pts[0][2]:.3f}")
                for p in p_pts[1:]: gc.append(f"{G1} X{p[0]:.3f} Y{p[1]:.3f} Z{p[2]:.3f}")
                gc.append(f"{G0} Z{safe}")
        else:
            cur_z = z_start
            while cur_z > z_final:
                tgt_z = max(cur_z - z_step, z_final)
                for path in phase['paths']:
                    coords = np.array(path.coords)
                    gc.append(f"{G0} X{coords[0,0]:.3f} Y{coords[0,1]:.3f}\n{G0} Z{cur_z + 1.0}\n{G1} Z{tgt_z:.3f}")
                    for xy in coords[1:]: gc.append(f"{G1} X{xy[0]:.3f} Y{xy[1]:.3f}")
                    gc.append(f"{G0} Z{safe}")
                cur_z = tgt_z
    gc.append(footer.strip()); return "\n".join(gc)

# --- 3. UIロジック ---

st.set_page_config(page_title="yosikeiCAM", layout="wide")

# 言語選択
with st.sidebar:
    lang = st.selectbox("Language / 言語", list(LANG_DICT.keys()))
    T = LANG_DICT[lang] # 選択された言語辞書

st.title(T["title"])
st.caption(f"Ver 4.1: Multi-language Support")

with st.sidebar:
    st.header(T["origin_setting"])
    origin = st.radio("", T["origin_option"], index=0)
    st.divider(); st.header(T["process_setting"])
    tab1, tab2, tab3, tab4 = st.tabs(T["tabs"])
    
    with tab1:
        enable_p = st.checkbox(T["enable"], value=True, key="cp")
        dia = st.number_input(T["tool_dia"], value=3.0, step=0.1)
        ucp = st.checkbox(T["double_pass"], value=False, key="uccp")
        clear = 0.0; f_p_r = st.number_input(T["rough_feed"], value=300, min_value=1, key="fp_r")
        f_p_f, f_mode = f_p_r, "Step-down"
        if ucp:
            clear = st.number_input(T["finish_allowance"], value=0.2, step=0.1)
            c3, c4 = st.columns(2)
            with c3: f_p_f = st.number_input(T["finish_feed"], value=300, min_value=1, key="fp_f")
            with c4: f_o = st.radio(T["finish_depth_mode"], T["depth_options"], index=0, key="fmp"); f_mode = "Full-Depth" if f_o == T["depth_options"][1] else "Step-down"
        c1, c2 = st.columns(2)
        with c1: dep_p = st.number_input(T["final_depth"], value=-1.0, step=0.1)
        with c2: stp_p = st.number_input(T["pitch"], value=1.0, min_value=0.01)
    
    with tab2:
        enable_c = st.checkbox(T["enable"], value=True, key="cc")
        cw = st.number_input(T["chamfer_width"], value=0.5, step=0.1); to = st.number_input(T["tip_offset"], value=1.0, step=0.1); z_c = -(cw + to)
        fc_r = st.number_input(T["rough_feed"], value=300, key="fc_r")
        ucf = st.checkbox(T["double_pass"], value=False, key="uccf"); cfa = 0.0; fc_f = fc_r
        if ucf:
            c1, c2 = st.columns(2)
            with c1: cfa = st.number_input(T["finish_allowance"], value=0.2, step=0.1)
            with c2: fc_f = st.number_input(T["finish_feed"], value=300, key="fc_f")
        
    with tab3:
        enable_v = st.checkbox(T["enable"], value=False, key="cv")
        va = st.number_input(T["v_angle"], value=60.0, step=10.0); ucv = st.checkbox(T["double_pass"], value=False, key="uccv")
        v_cl, fvf, fv = 0.0, 300, st.number_input(T["rough_feed"], value=300, key="fv")
        if ucv:
            v_cl = st.number_input(T["finish_allowance"], value=0.2, step=0.1); fvf = st.number_input(T["finish_feed"], value=300, key="fvf")
        uvl = st.checkbox(T["depth_limit"], value=False); vl = st.number_input(T["max_depth"], value=-3.0) if uvl else -100.0; vr = st.slider(T["precision"], value=0.05, min_value=0.02, max_value=0.2)
        
    with tab4:
        enable_d = st.checkbox(T["enable"], value=False, key="cd")
        ddt = st.number_input(T["drill_dia"], value=3.0); ddz = st.number_input(T["drill_depth"], value=-5.0); pck = st.number_input(T["peck"], value=2.0, min_value=0.1); fd = st.number_input(T["feed"], value=200, key="fd")
        
    st.divider(); ppn = st.selectbox(T["post_processor"], list(POST_PROCESSORS.keys())); pp = POST_PROCESSORS[ppn]
    h_c = st.text_area("Header", pp["start"]); f_c = st.text_area("Footer", pp["end"])

st.header(T["dxf_upload"])
f = st.file_uploader("", type=["dxf"])

if f:
    bn = os.path.splitext(f.name)[0]; polys_raw = dxf_to_shapely_list(f.getvalue())
    if polys_raw:
        tu = unary_union(polys_raw); minx, miny, maxx, maxy = tu.bounds; ox, oy = 0, 0
        w_size, h_size = maxx-minx, maxy-miny
        if origin.startswith(T["origin_option"][0].split(" ")[0]): ox, oy = -minx, -miny
        elif origin.startswith(T["origin_option"][1].split(" ")[0]): ox, oy = -(minx+w_size/2), -(miny+h_size/2)
        polys_moved = [translate(p, ox, oy) for p in polys_raw]
        
        st.sidebar.divider(); st.sidebar.subheader(T["path_select"])
        all_c = st.sidebar.checkbox(T["select_all"], value=True, key="sa")
        selected = [i for i, p in enumerate(polys_moved) if st.sidebar.checkbox(f"Path #{i+1} (Area:{p.area:.1f})", value=all_c, key=f"p{i}")]
                
        target_polys = [polys_moved[i] for i in selected]; gfc = merge_polygons_xor(target_polys)
        if gfc:
            hs = analyze_holes(gfc); 
            if hs: st.sidebar.info(f"{T['hole_detected']}: " + ", ".join([f"φ{d}({c})" for d, c in hs.items()]))
        
        st.success(f"{T['size_display']}: {w_size:.1f} x {h_size:.1f} mm")
        st.header(T["gen_header"])
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5,5)); ax.plot(0, 0, 'r+', markersize=20); ax.axhline(0, color='red', alpha=0.3); ax.axvline(0, color='red', alpha=0.3)
            for i, p in enumerate(polys_moved):
                ax.plot(*p.exterior.xy, 'k-' if i in selected else 'k:', alpha=1.0 if i in selected else 0.1)
                for interior in p.interiors: ax.plot(*interior.xy, 'k-' if i in selected else 'k:', alpha=1.0 if i in selected else 0.1)
            ax.axis('equal'); ax.grid(True, linestyle=':'); st.pyplot(fig)
            
        with c2:
            gc_v, gc_p, gc_c, gc_d = None, None, None, None; v_d, p_r_d, p_f_d, c_d = [], [], [], []
            if gfc and not gfc.is_empty:
                if enable_v:
                    with st.spinner("VCarve..."):
                        v_r_ps = generate_vcarve(gfc, va, uvl, vl, vr, z_offset=v_cl) if ucv else []
                        v_f_ps = generate_vcarve(gfc, va, uvl, vl, vr, z_offset=0.0); v_d = v_r_ps + v_f_ps; phs_v = []
                        if v_r_ps: phs_v.append({'name':'V-Rough','paths':v_r_ps,'z_start':0,'z_final':0,'feed':fv})
                        if v_f_ps: phs_v.append({'name':'V-Finish','paths':v_f_ps,'z_start':0,'z_final':0,'feed':fvf if ucv else fv})
                        if phs_v: gc_v = make_gcode_phases_advanced(phs_v, "VBit", h_c, f_c, pp["format"], True)
                if enable_p:
                    p_r, p_f = generate_pocket(gfc, dia, clear if ucp else 0.0, 0.5); p_r_d, p_f_d = p_r, p_f; phs = []
                    if p_r: phs.append({'name':'Rough','paths':p_r,'z_start':0,'z_final':dep_p,'feed':f_p_r,'z_step':stp_p})
                    if p_f and ucp: phs.append({'name':'Finish','paths':p_f,'z_start':0,'z_final':dep_p,'feed':f_p_f,'z_step':(abs(dep_p) if f_mode=="Full-Depth" else stp_p)})
                    if phs: gc_p = make_gcode_phases_advanced(phs, "EndMill", h_c, f_c, pp["format"])
                if enable_c:
                    rp_c, fp_c = generate_chamfer_separated(gfc, to, cfa if ucf else 0.0); c_d = rp_c + fp_c; phs_c = []
                    if rp_c and ucf: phs_c.append({'name':'Rough','paths':rp_c,'z_start':0,'z_final':z_c,'feed':fed_c_r})
                    if fp_c: phs_c.append({'name':'Finish','paths':fp_c,'z_start':0,'z_final':z_c,'feed':fc_f if ucf else fed_c_r})
                    if phs_c: gc_c = make_gcode_phases_advanced(phs_c, "Chamfer", h_c, f_c, pp["format"])
                if enable_d:
                    dpts = find_drill_points(gfc, ddt); gc_d = generate_drill_gcode(dpts, 0, ddz, pck, fd, f"Drill {ddt}mm", h_c, f_c, pp["format"])

            fig2, ax2 = plt.subplots(figsize=(5,5)); ax2.plot(0, 0, 'r+', markersize=20); ax2.axhline(0, color='red', alpha=0.3); ax2.axvline(0, color='red', alpha=0.3)
            for p in polys_moved: ax2.plot(*p.exterior.xy, 'k--', alpha=0.05)
            for ls in p_r_d: ax2.plot(*ls.xy, color='tab:blue', alpha=0.3)
            for ls in p_f_d: ax2.plot(*ls.xy, color='tab:cyan', alpha=0.8)
            for ls in c_d: ax2.plot(*ls.xy, color='tab:green', alpha=0.8)
            for pts in v_d: ax2.plot([p[0] for p in pts], [p[1] for p in pts], 'r-', linewidth=0.8)
            ax2.axis('equal'); ax2.grid(True, linestyle=':'); st.pyplot(fig2)
            
            col1, col2 = st.columns(2)
            if gc_v: col1.download_button(T["download"][0], gc_v, f"{bn}_vcarve.nc")
            if gc_p: col2.download_button(T["download"][1], gc_p, f"{bn}_pocket.nc")
            if gc_c: col1.download_button(T["download"][2], gc_c, f"{bn}_chamfer.nc")
            if gc_d: col2.download_button(T["download"][3], gc_d, f"{bn}_drill.nc")
    else: st.error(T["error_no_geom"])
