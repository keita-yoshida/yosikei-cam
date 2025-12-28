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
    t = ((xi-x0)*dx + (yi-y0)*dy) / d2
    t = max(0, min(1, t))
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

def apply_dogbone_single(polygon: Polygon, tool_dia: float) -> Polygon:
    if polygon.is_empty: return polygon
    poly = polygon.simplify(0.001)
    if poly.geom_type != 'Polygon': return polygon
    rings = [poly.exterior] + list(poly.interiors)
    dogbone_circles = []
    r = tool_dia / 2.0
    for ring in rings:
        coords = list(ring.coords)
        if coords[0] == coords[-1]: coords.pop()
        num_pts = len(coords)
        for i in range(num_pts):
            p_curr = np.array(coords[i])
            p_prev = np.array(coords[(i - 1) % num_pts])
            p_next = np.array(coords[(i + 1) % num_pts])
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6: continue
            v1 /= n1; v2 /= n2
            cross = np.cross(v1, v2)
            dot = np.dot(v1, v2)
            angle_deg = math.degrees(math.atan2(cross, dot))
            if 5 < abs(angle_deg) < 175:
                bisector = v2 - v1
                bn = np.linalg.norm(bisector)
                if bn > 1e-6:
                    bisector /= bn
                    test_pt = p_curr + bisector * 0.01
                    if not poly.contains(Point(test_pt)): bisector = -bisector
                    half_angle_rad = math.radians((180 - abs(angle_deg)) / 2)
                    if half_angle_rad < 0.1: half_angle_rad = 0.1
                    dist_theoretical = r / math.sin(half_angle_rad)
                    offset = (dist_theoretical - r) + (r * 0.05)
                    center = p_curr + bisector * offset
                    circle = Point(center).buffer(r, resolution=16)
                    dogbone_circles.append(circle)
    if not dogbone_circles: return polygon
    try: return unary_union([polygon] + dogbone_circles).simplify(0.001)
    except: return polygon

def apply_dogbone(geometry, tool_dia):
    polys = ensure_list_of_polys(geometry)
    if not polys: return geometry
    new_parts = [apply_dogbone_single(p, tool_dia) for p in polys]
    return unary_union(new_parts)

# --- 2. データ読み込み ---

def dxf_to_shapely_list(dxf_bytes):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(dxf_bytes); tmp_path = tmp.name
    try:
        doc = ezdxf.readfile(tmp_path); msp = doc.modelspace()
        polys = []
        for e in msp:
            if e.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'SPLINE', 'CIRCLE'):
                try:
                    if e.dxftype() == 'CIRCLE':
                        center = e.dxf.center; radius = e.dxf.radius
                        poly = Point(center[:2]).buffer(radius, resolution=64)
                    else:
                        path_obj = ezdxf.path.make_path(e)
                        pts = list(path_obj.flattening(0.01))
                        if len(pts) > 2: poly = Polygon([(v.x, v.y) for v in pts])
                    if 'poly' in locals():
                        if poly.is_valid and poly.area > 0.0001: polys.append(poly)
                        elif not poly.is_valid:
                            clean = make_valid(poly)
                            if clean.area > 0.0001:
                                if clean.geom_type == 'Polygon': polys.append(clean)
                                elif clean.geom_type == 'MultiPolygon': polys.extend(clean.geoms)
                except: pass
        polys.sort(key=lambda x: x.area, reverse=True)
        return polys
    except Exception as e:
        st.error(f"DXF Read Error: {e}"); return []
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

def merge_polygons_xor(polys):
    if not polys: return None
    combined = Polygon()
    for p in polys:
        if combined.is_empty: combined = p
        else: combined = combined.symmetric_difference(p)
    return combined

# --- 3. ドリル解析 ---

def analyze_holes(geometry):
    polys = ensure_list_of_polys(geometry); sizes = []
    def check_poly(p):
        minx, miny, maxx, maxy = p.bounds; w, h = maxx - minx, maxy - miny
        if abs(w - h) > 0.1: return None
        expected_area = math.pi * ((w/2)**2)
        if abs(p.area - expected_area) / expected_area > 0.2: return None
        return w
    for p in polys:
        d = check_poly(p)
        if d: sizes.append(round(d, 2))
        for interior in p.interiors:
            d = check_poly(Polygon(interior))
            if d: sizes.append(round(d, 2))
    return Counter(sizes)

def find_drill_points(geometry, target_dia, tolerance=0.1):
    polys = ensure_list_of_polys(geometry); drill_points = []
    def check_poly(p):
        minx, miny, maxx, maxy = p.bounds; w, h = maxx - minx, maxy - miny
        if abs(w - h) > tolerance: return None
        if not (target_dia - tolerance <= w <= target_dia + tolerance): return None
        expected_area = math.pi * ((w/2)**2)
        if abs(p.area - expected_area) / expected_area > 0.2: return None
        return p.centroid
    for p in polys:
        pt = check_poly(p)
        if pt: drill_points.append(pt)
        for interior in p.interiors:
            pt = check_poly(Polygon(interior))
            if pt: drill_points.append(pt)
    return drill_points

def generate_drill_gcode(points, z_start, z_final, peck_depth, feed, tool_name, header, footer, fmt):
    if not points: return None
    gc = [header.strip(), f"; Tool: {tool_name} (Drill)", "T1 M06", f"F{int(feed)}", ""]
    G0 = "G0" if "G0/" in fmt else "G00"; G1 = "G1" if "G0/" in fmt else "G01"; safe = 5.0
    for pt in points:
        x, y = pt.x, pt.y
        gc.append(f"; Drill Hole at X{x:.3f} Y{y:.3f}")
        gc.append(f"{G0} X{x:.3f} Y{y:.3f}")
        gc.append(f"{G0} Z{z_start + 1.0}")
        current_z = z_start
        while current_z > z_final:
            target_z = current_z - peck_depth
            if target_z < z_final: target_z = z_final
            gc.append(f"{G1} Z{target_z:.3f}")
            if target_z > z_final:
                gc.append(f"{G0} Z{z_start + 0.5}"); gc.append(f"{G0} Z{target_z + 0.5}")
            current_z = target_z
        gc.append(f"{G0} Z{safe}")
    gc.append(footer.strip())
    return "\n".join(gc)

# --- 4. パス生成 ---

def generate_pocket(geometry, tool_d, clearance, stepover, dogbone):
    paths_rough = []; paths_finish = []
    r = tool_d / 2.0; step = tool_d * stepover
    offset_rough = -(r + clearance)
    try: current_rough = geometry.buffer(offset_rough, join_style=2)
    except: current_rough = Polygon()
    while not current_rough.is_empty and current_rough.area > 0.01:
        current_polys = ensure_list_of_polys(current_rough)
        if not current_polys: break
        for p in current_polys:
            paths_rough.append(p.exterior); paths_rough.extend(p.interiors)
        try: current_rough = current_rough.buffer(-step, join_style=2)
        except: break
    if clearance > 0:
        work_geom_finish = geometry
        if dogbone: work_geom_finish = apply_dogbone(geometry, tool_d)
        offset_finish = -r
        try:
            finish_pass = work_geom_finish.buffer(offset_finish, join_style=2)
            finish_polys = ensure_list_of_polys(finish_pass)
            for p in finish_polys:
                paths_finish.append(p.exterior); paths_finish.extend(p.interiors)
        except: pass
    return ([LineString(p.coords) for p in paths_rough if p.length > 0.1], [LineString(p.coords) for p in paths_finish if p.length > 0.1])

def generate_chamfer_separated(geometry, width, tip_offset, finish_allowance=0.0):
    rough_paths = []; finish_paths = []
    if finish_allowance > 0:
        offset_rough = tip_offset + finish_allowance
        try:
            p_rough = geometry.buffer(offset_rough, join_style=1); p_list_rough = ensure_list_of_polys(p_rough)
            for poly in p_list_rough: rough_paths.append(poly.exterior); rough_paths.extend(poly.interiors)
        except: pass
    try:
        p_finish = geometry.buffer(tip_offset, join_style=1); p_list_finish = ensure_list_of_polys(p_finish)
        for poly in p_list_finish: finish_paths.append(poly.exterior); finish_paths.extend(poly.interiors)
    except: pass
    return ([LineString(ls.coords) for ls in rough_paths], [LineString(ls.coords) for ls in finish_paths])

# --- Vカーブ グラフ理論ロジック ---

class PathGraph:
    def __init__(self): self.adj = {}
    def add_edge(self, p1, p2):
        p1 = (round(p1[0], 3), round(p1[1], 3)); p2 = (round(p2[0], 3), round(p2[1], 3))
        if p1 == p2: return
        if p1 not in self.adj: self.adj[p1] = set()
        if p2 not in self.adj: self.adj[p2] = set()
        self.adj[p1].add(p2); self.adj[p2].add(p1)
    def prune_short_leaves(self, min_len=0.5):
        for _ in range(5):
            pruned_count = 0; leaves = [node for node, neighbors in self.adj.items() if len(neighbors) == 1]
            for leaf in leaves:
                if leaf not in self.adj: continue
                neighbor = list(self.adj[leaf])[0]
                dist = math.sqrt((leaf[0]-neighbor[0])**2 + (leaf[1]-neighbor[1])**2)
                if dist < min_len and len(self.adj[neighbor]) > 2:
                    self.adj[neighbor].remove(leaf); del self.adj[leaf]; pruned_count += 1
            if pruned_count == 0: break
    def get_chains(self):
        chains = []; visited_edges = set(); sorted_nodes = sorted(self.adj.keys(), key=lambda n: (len(self.adj[n]) % 2 != 1, -len(self.adj[n])))
        for start_node in sorted_nodes:
            if start_node not in self.adj: continue
            neighbors = list(self.adj[start_node])
            for next_node in neighbors:
                edge_key = tuple(sorted((start_node, next_node)))
                if edge_key in visited_edges: continue
                current_chain = [start_node, next_node]; visited_edges.add(edge_key); curr = next_node
                while True:
                    n_neighbors = list(self.adj[curr]); candidates = [n for n in n_neighbors if tuple(sorted((curr, n))) not in visited_edges]
                    if len(candidates) == 1:
                        nxt = candidates[0]; visited_edges.add(tuple(sorted((curr, nxt)))); current_chain.append(nxt); curr = nxt
                        if len(self.adj[curr]) > 2: break
                    else: break
                if len(current_chain) > 1: chains.append(current_chain)
        return chains

def generate_vcarve(geometry, angle_deg, use_limit, max_d, step_len=0.1):
    polys = ensure_list_of_polys(geometry); all_paths = []; tan_a = np.tan(np.radians(angle_deg/2)); graph = PathGraph()
    for poly in polys:
        simple = poly.simplify(0.01); line = simple.exterior; length = line.length; sample_res = 0.3
        num = int(length / sample_res)
        if num < 50: num = 50
        if num > 4000: num = 4000
        pts = [line.interpolate(i * length / num) for i in range(num)]
        coords = np.array([(p.x, p.y) for p in pts])
        try: vor = Voronoi(coords)
        except: continue
        for (p1_idx, p2_idx), (v1_idx, v2_idx) in zip(vor.ridge_points, vor.ridge_vertices):
            if v1_idx < 0 or v2_idx < 0: continue
            v1, v2 = vor.vertices[v1_idx], vor.vertices[v2_idx]
            if not simple.contains(Point(v1)) or not simple.contains(Point(v2)): continue
            g1, g2 = vor.points[p1_idx], vor.points[p2_idx]
            if np.linalg.norm(g1 - g2) < sample_res * 2.8: continue
            graph.add_edge(v1, v2)
    graph.prune_short_leaves(min_len=1.0); raw_chains = graph.get_chains()
    for chain in raw_chains:
        path_3d = []; ls = LineString(chain); pts_count = int(ls.length / step_len) + 1
        if pts_count < 2: pts_count = 2
        for i in range(pts_count):
            pt = ls.interpolate(i * ls.length / (pts_count - 1)); d = poly.distance(pt); z = -(d / tan_a)
            if use_limit and z < max_d: z = max_d
            path_3d.append((pt.x, pt.y, z))
        if len(path_3d) > 1: all_paths.append(douglas_peucker(path_3d, 0.02))
    return all_paths

# --- 5. Gコードエンジン ---

def make_gcode_phases_advanced(phases, tool_name, header, footer, fmt="G00/G01", is_3d=False):
    gc = [header.strip(), f"; Tool: {tool_name}", "T1 M06"]; G0 = "G0" if "G0/" in fmt else "G00"; G1 = "G1" if "G0/" in fmt else "G01"; safe = 5.0; gc.append(f"{G0} Z{safe}")
    for i, phase in enumerate(phases):
        paths = phase.get('paths', [])
        if not paths: continue
        feed = int(phase.get('feed', 300)); z_start = phase.get('z_start', 0); z_final = phase.get('z_final', 0); z_step = phase.get('z_step', abs(z_final - z_start)); use_ramp = phase.get('use_ramp', False)
        if z_step <= 0: z_step = abs(z_final - z_start)
        gc.append(f"; --- Phase {i+1}: {phase.get('name','')} (F{feed}) ---"); gc.append(f"F{feed}")
        if is_3d:
            for path_pts in paths:
                if not path_pts: continue
                gc.append(f"{G0} X{path_pts[0][0]:.3f} Y{path_pts[0][1]:.3f}"); gc.append(f"{G1} Z{path_pts[0][2]:.3f}")
                for p in path_pts[1:]: gc.append(f"{G1} X{p[0]:.3f} Y{p[1]:.3f} Z{p[2]:.3f}")
                gc.append(f"{G0} Z{safe}")
        else:
            current_z = z_start
            while current_z > z_final:
                target_z = max(current_z - z_step, z_final)
                for path in paths:
                    coords = np.array(path.coords)
                    if len(coords) < 2: continue
                    gc.append(f"{G0} X{coords[0,0]:.3f} Y{coords[0,1]:.3f}"); gc.append(f"{G0} Z{current_z + 1.0}")
                    if use_ramp:
                        dist_accum = 0.0; path_len = path.length; z_diff = current_z - target_z; reached_target = False
                        for j in range(1, len(coords)):
                            dist_accum += np.linalg.norm(coords[j] - coords[j-1]); ratio = min(dist_accum / min(path_len, 50.0), 1.0)
                            gc.append(f"{G1} X{coords[j,0]:.3f} Y{coords[j,1]:.3f} Z{current_z - (z_diff * ratio):.3f}")
                            if ratio >= 1.0:
                                reached_target = True
                                for k in range(j+1, len(coords)): gc.append(f"{G1} X{coords[k,0]:.3f} Y{coords[k,1]:.3f} Z{target_z:.3f}")
                                break
                        if not reached_target: gc.append(f"{G1} Z{target_z:.3f}")
                    else:
                        gc.append(f"{G1} Z{target_z:.3f}")
                        for xy in coords[1:]: gc.append(f"{G1} X{xy[0]:.3f} Y{xy[1]:.3f}")
                    gc.append(f"{G0} Z{safe}")
                current_z = target_z
    gc.append(footer.strip()); return "\n".join(gc)

# --- 6. UI ---

st.set_page_config(page_title="yosikeiCAM", layout="wide")
st.title("⚡ yosikeiCAM 1.7")
st.caption("Ver 1.7: 構文エラー修正・Vカーブノイズ除去強化")

with st.sidebar:
    st.header("📍 原点設定")
    origin = st.radio("原点基準", ["Bottom-Left (左下)", "Center (中心)", "Original (CAD座標)"], index=0)
    st.divider(); st.header("⚙️ 加工設定")
    tab1, tab2, tab3, tab4 = st.tabs(["ポケット", "面取り", "Vカーブ", "ドリル"])
    with tab1:
        enable_pocket = st.checkbox("有効", True); st.divider()
        dia = st.number_input("工具径 (mm)", value=3.0, min_value=0.01, step=0.1, format="%.3f")
        clear = st.number_input("仕上げ代 (mm)", value=0.0, step=0.1)
        c1, c2 = st.columns(2)
        with c1: depth = st.number_input("最終深さ Z", value=-1.0, step=0.1)
        with c2: z_step_p = st.number_input("Zピッチ", value=1.0, min_value=0.01, step=0.1)
        step = st.slider("ステップオーバー (%)", 10, 90, 50) / 100.0; use_dogbone = st.checkbox("ドッグボーン", True); use_ramp = st.checkbox("ランピング進入", False)
        st.caption("▼ 送り速度")
        feed_p_rough = st.number_input("粗送り速度", value=300, min_value=1, step=50, key="fp_r")
        feed_p_finish = feed_p_rough; finish_mode = "Step-down"
        if clear > 0:
            st.markdown("---"); st.caption("▼ 仕上げ設定")
            c3, c4 = st.columns(2)
            with c3: feed_p_finish = st.number_input("仕上げ速度", value=300, min_value=1, step=50, key="fp_f")
            with c4: finish_mode_opt = st.radio("仕上げ深さ", ["刻み", "一括"], index=0); finish_mode = "Full-Depth" if finish_mode_opt == "一括" else "Step-down"
    with tab2:
        enable_chamfer = st.checkbox("有効", True); st.divider()
        chamfer_w = st.number_input("幅 (mm)", 0.5, step=0.1); tip_off = st.number_input("刃先オフセット", value=1.0, min_value=0.0, step=0.1); z_c = -(chamfer_w + tip_off)
        feed_c_rough = st.number_input("速度", value=300, min_value=1, step=50, key="fc_r")
        use_chamfer_finish = st.checkbox("2回加工", False); chamfer_finish_allowance = 0.0; feed_c_finish = 300
        if use_chamfer_finish:
            c1, c2 = st.columns(2)
            with c1: chamfer_finish_allowance = st.number_input("代 (mm)", value=0.2, min_value=0.01, step=0.1)
            with c2: feed_c_finish = st.number_input("仕上速度", value=300, min_value=1, step=50)
    with tab3:
        enable_vcarve = st.checkbox("有効", False); st.divider(); v_ang = st.number_input("角度", 60.0, step=10.0); use_v_limit = st.checkbox("深さ制限", False)
        v_lim = st.number_input("制限 (mm)", value=-3.0) if use_v_limit else -100.0; feed_v = st.number_input("速度", value=300, min_value=1, step=50); v_res = st.slider("精度", 0.2, 0.02, 0.05)
    with tab4:
        enable_drill = st.checkbox("有効", False); st.divider(); drill_dia_target = st.number_input("穴径 (mm)", value=3.0, min_value=0.01, step=0.1); drill_depth = st.number_input("深さ Z", value=-5.0); peck_depth = st.number_input("ペック", value=2.0, min_value=0.1); feed_d = st.number_input("速度", value=200, min_value=1, step=50)
    st.divider(); pp_name = st.selectbox("ポストプロセッサ", list(POST_PROCESSORS.keys())); pp = POST_PROCESSORS[pp_name]
    with st.expander("Gコード詳細"): h_code = st.text_area("Header", pp["start"]); f_code = st.text_area("Footer", pp["end"])

st.header("1. DXFアップロード"); f = st.file_uploader("", type=["dxf"])
if f:
    base_name = os.path.splitext(f.name)[0]; polys_raw = dxf_to_shapely_list(f.getvalue())
    if polys_raw:
        temp_union = unary_union(polys_raw); minx, miny, maxx, maxy = temp_union.bounds; w, h = maxx-minx, maxy-miny
        offset_x, offset_y = 0, 0
        if origin.startswith("Bottom-Left"): offset_x, offset_y = -minx, -miny
        elif origin.startswith("Center"): offset_x, offset_y = -(minx+w/2), -(miny+h/2)
        polys_moved = [translate(p, offset_x, offset_y) for p in polys_raw]
        st.sidebar.divider(); st.sidebar.subheader("📐 パス選択"); selected_indices = []
        container = st.sidebar.container(); all_checked = container.checkbox("すべて選択", value=True)
        for i, p in enumerate(polys_moved):
            is_checked = container.checkbox(f"Path #{i+1} (Area:{p.area:.1f})", value=all_checked, key=f"p_{i}")
            if is_checked: selected_indices.append(i)
        target_polys = [polys_moved[i] for i in selected_indices]; geom_for_calc = merge_polygons_xor(target_polys)
        if geom_for_calc:
            drill_sizes = analyze_holes(geom_for_calc)
            if drill_sizes: st.sidebar.info("💡 穴: " + ", ".join([f"φ{d}({c}個)" for d, c in drill_sizes.items()]))
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"加工サイズ: {w:.1f} x {h:.1f} mm")
            fig, ax = plt.subplots(figsize=(5,5)); ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=2, zorder=10); ax.axhline(0, color='red', alpha=0.5); ax.axvline(0, color='red', alpha=0.5)
            for i, p in enumerate(polys_moved):
                style = 'k-' if i in selected_indices else 'k:'; alpha = 1.0 if i in selected_indices else 0.1
                ax.plot(*p.exterior.xy, style, alpha=alpha, linewidth=1)
                for interior in p.interiors: ax.plot(*interior.xy, style, alpha=alpha, linewidth=1)
            if enable_drill and geom_for_calc:
                drill_preview = find_drill_points(geom_for_calc, drill_dia_target)
                for pt in drill_preview: ax.plot(pt.x, pt.y, 'x', color='tab:purple')
            ax.axis('equal'); ax.grid(True, linestyle=':'); st.pyplot(fig)
        with c2:
            st.header("2. パス生成"); gc_p, gc_c, gc_v, gc_d = None, None, None, None; p_disp_r, p_disp_f, c_disp, v_disp = [], [], [], []
            if geom_for_calc and not geom_for_calc.is_empty:
                if enable_pocket:
                    p_rough, p_finish = generate_pocket(geom_for_calc, dia, clear, step, use_dogbone); p_disp_r, p_disp_f = p_rough, p_finish; phases = []
                    if p_rough: phases.append({'name': 'Roughing', 'paths': p_rough, 'z_start': 0, 'z_final': depth, 'feed': feed_p_rough, 'z_step': z_step_p, 'use_ramp': use_ramp})
                    if p_finish: phases.append({'name': 'Finishing', 'paths': p_finish, 'z_start': 0, 'z_final': depth, 'feed': feed_p_finish, 'z_step': (abs(depth) if finish_mode == "Full-Depth" else z_step_p), 'use_ramp': use_ramp})
                    if phases: gc_p = make_gcode_phases_advanced(phases, "EndMill", h_code, f_code, pp["format"])
                if enable_chamfer:
                    r_p, f_p = generate_chamfer_separated(geom_for_calc, chamfer_w, tip_off, chamfer_finish_allowance if use_chamfer_finish else 0.0); c_disp = r_p + f_p; phases = []
                    if r_p: phases.append({'name':'Rough', 'paths': r_p, 'z_start': 0, 'z_final': z_c, 'feed': feed_c_rough})
                    if f_p: phases.append({'name':'Finish', 'paths': f_p, 'z_start': 0, 'z_final': z_c, 'feed': (feed_c_finish if use_chamfer_finish else feed_c_rough)})
                    if phases: gc_c = make_gcode_phases_advanced(phases, "Chamfer", h_code, f_code, pp["format"])
                if enable_vcarve:
                    with st.spinner("VCarve..."): v_paths = generate_vcarve(geom_for_calc, v_ang, use_v_limit, v_lim, v_res); v_disp = v_paths
                    if v_paths: gc_v = make_gcode_phases_advanced([{'name': 'V-Carve', 'paths': v_paths, 'z_start': 0, 'z_final': 0, 'feed': feed_v}], "VBit", h_code, f_code, pp["format"], True)
                if enable_drill:
                    d_pts = find_drill_points(geom_for_calc, drill_dia_target)
                    if d_pts: gc_d = generate_drill_gcode(d_pts, 0, drill_depth, peck_depth, feed_d, f"Drill {drill_dia_target}mm", h_code, f_code, pp["format"])
            fig2, ax2 = plt.subplots(figsize=(5,5)); ax2.plot(0, 0, 'r+', markersize=15); ax2.axhline(0, color='red', alpha=0.5); ax2.axvline(0, color='red', alpha=0.5)
            for p in polys_moved: ax2.plot(*p.exterior.xy, 'k--', alpha=0.1)
            for ls in p_disp_r: ax2.plot(*ls.xy, color='tab:blue', alpha=0.5); 
            for ls in p_disp_f: ax2.plot(*ls.xy, color='tab:cyan', alpha=1.0)
            for ls in c_disp: ax2.plot(*ls.xy, color='tab:green', alpha=0.9)
            for pts in v_disp: ax2.plot([p[0] for p in pts], [p[1] for p in pts], color='tab:red', linewidth=0.8)
            ax2.axis('equal'); st.pyplot(fig2); b1, b2, b3, b4 = st.columns(4)
            if gc_p: b1.download_button("📥 POCKET", gc_p, f"{base_name}_pocket.nc")
            if gc_c: b2.download_button("📥 CHAMFER", gc_c, f"{base_name}_chamfer.nc")
            if gc_v: b3.download_button("📥 VCARVE", gc_v, f"{base_name}_vcarve.nc")
            if gc_d: b4.download_button("📥 DRILL", gc_d, f"{base_name}_drill.nc")
    else: st.error("有効な閉じた図形が見つかりません。")
