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
            p_prev = np.array(coords[(i-1)%num_pts])
            p_next = np.array(coords[(i+1)%num_pts])
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6: continue
            v1 /= n1
            v2 /= n2
            cross = np.cross(v1, v2)
            dot = np.dot(v1, v2)
            angle_deg = math.degrees(math.atan2(cross, dot))
            
            if 5 < abs(angle_deg) < 175:
                bisector = v2 - v1
                bn = np.linalg.norm(bisector)
                if bn > 1e-6:
                    bisector /= bn
                    test_pt = p_curr + bisector * 0.01
                    if not poly.contains(Point(test_pt)):
                        bisector = -bisector
                    half_angle_rad = math.radians((180 - abs(angle_deg)) / 2)
                    dist_theoretical = r / math.sin(max(0.1, half_angle_rad))
                    center = p_curr + bisector * ((dist_theoretical - r) + (r * 0.05))
                    dogbone_circles.append(Point(center).buffer(r, resolution=16))
                    
    try:
        return unary_union([polygon] + dogbone_circles).simplify(0.001)
    except:
        return polygon

def apply_dogbone(geometry, tool_dia):
    polys = ensure_list_of_polys(geometry)
    if not polys: return geometry
    return unary_union([apply_dogbone_single(p, tool_dia) for p in polys])

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
                        center = e.dxf.center
                        radius = e.dxf.radius
                        poly = Point(center[:2]).buffer(radius, resolution=64)
                    else:
                        path_obj = ezdxf.path.make_path(e)
                        pts = list(path_obj.flattening(0.01))
                        if len(pts) > 2:
                            poly = Polygon([(v.x, v.y) for v in pts])
                    
                    if 'poly' in locals():
                        if poly.is_valid and poly.area > 0.0001:
                            polys.append(poly)
                        elif not poly.is_valid:
                            clean = make_valid(poly)
                            if clean.area > 0.0001:
                                if clean.geom_type == 'Polygon':
                                    polys.append(clean)
                                elif clean.geom_type == 'MultiPolygon':
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
        if combined.is_empty:
            combined = p
        else:
            combined = combined.symmetric_difference(p)
    return combined

# --- 3. ドリル解析 ---

def analyze_holes(geometry):
    polys = ensure_list_of_polys(geometry)
    sizes = []
    def check_poly(p):
        minx, miny, maxx, maxy = p.bounds
        w = maxx - minx
        h = maxy - miny
        if abs(w - h) > 0.1: return None
        expected_area = math.pi * ((w/2)**2)
        if abs(p.area - expected_area) / expected_area > 0.2: return None
        return w

    for p in polys:
        d = check_poly(p)
        if d:
            sizes.append(round(d, 2))
        for interior in p.interiors:
            d = check_poly(Polygon(interior))
            if d:
                sizes.append(round(d, 2))
    return Counter(sizes)

def find_drill_points(geometry, target_dia, tolerance=0.1):
    polys = ensure_list_of_polys(geometry)
    drill_points = []
    def check_poly(p):
        minx, miny, maxx, maxy = p.bounds
        w = maxx - minx
        h = maxy - miny
        if abs(w - h) > tolerance: return None
        if not (target_dia - tolerance <= w <= target_dia + tolerance): return None
        expected_area = math.pi * ((w/2)**2)
        if abs(p.area - expected_area) / expected_area > 0.2: return None
        return p.centroid

    for p in polys:
        pt = check_poly(p)
        if pt:
            drill_points.append(pt)
        for interior in p.interiors:
            pt = check_poly(Polygon(interior))
            if pt:
                drill_points.append(pt)
    return drill_points

def generate_drill_gcode(points, z_start, z_final, peck_depth, feed, tool_name, header, footer, fmt):
    if not points: return None
    gc = [header.strip(), f"; Tool: {tool_name} (Drill)", "T1 M06", f"F{int(feed)}", ""]
    G0 = "G0" if "G0/" in fmt else "G00"
    G1 = "G1" if "G0/" in fmt else "G01"
    safe = 5.0
    for pt in points:
        gc.append(f"; Drill X{pt.x:.3f} Y{pt.y:.3f}")
        gc.append(f"{G0} X{pt.x:.3f} Y{pt.y:.3f}")
        gc.append(f"{G0} Z{z_start + 1.0}")
        current_z = z_start
        while current_z > z_final:
            target_z = max(current_z - peck_depth, z_final)
            gc.append(f"{G1} Z{target_z:.3f}")
            if target_z > z_final:
                gc.append(f"{G0} Z{z_start + 0.5}")
                gc.append(f"{G0} Z{target_z + 0.5}")
            current_z = target_z
        gc.append(f"{G0} Z{safe}")
    gc.append(footer.strip())
    return "\n".join(gc)

# --- 4. パス生成 ---

def generate_pocket(geometry, tool_d, clearance, stepover, dogbone):
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
        work_geom_finish = geometry
        if dogbone:
            work_geom_finish = apply_dogbone(geometry, tool_d)
        try:
            finish_pass = work_geom_finish.buffer(-r, join_style=2)
            finish_polys = ensure_list_of_polys(finish_pass)
            for p in finish_polys:
                paths_finish.append(p.exterior)
                paths_finish.extend(p.interiors)
        except:
            pass
            
    return ([LineString(p.coords) for p in paths_rough if p.length > 0.1], 
            [LineString(p.coords) for p in paths_finish if p.length > 0.1])

def generate_chamfer_separated(geometry, width, tip_offset, finish_allowance=0.0):
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
        p_list_finish = ensure_list_of_polys(p_finish)
        for poly in p_list_finish:
            finish_paths.append(poly.exterior)
            finish_paths.extend(poly.interiors)
    except:
        pass
    return ([LineString(ls.coords) for ls in rough_paths], 
            [LineString(ls.coords) for ls in finish_paths])

# --- Vカーブ グラフ理論ロジック (Ver 2.1) ---

class PathGraph:
    def __init__(self):
        self.adj = {}
        
    def add_edge(self, p1, p2):
        p1 = (round(p1[0], 3), round(p1[1], 3))
        p2 = (round(p2[0], 3), round(p2[1], 3))
        if p1 == p2: return
        if p1 not in self.adj: self.adj[p1] = set()
        if p2 not in self.adj: self.adj[p2] = set()
        self.adj[p1].add(p2)
        self.adj[p2].add(p1)
    
    def prune_logic(self, min_len):
        pruned = 0
        leaves = [n for n, neigh in self.adj.items() if len(neigh) == 1]
        for leaf in leaves:
            if leaf not in self.adj: continue
            neighbor = list(self.adj[leaf])[0]
            dist = math.sqrt((leaf[0]-neighbor[0])**2 + (leaf[1]-neighbor[1])**2)
            # 分岐点から伸びる短い枝を消す
            if dist < min_len and len(self.adj[neighbor]) > 2:
                self.adj[neighbor].remove(leaf)
                del self.adj[leaf]
                pruned += 1
        return pruned

    def prune_short_leaves(self, min_len=4.0):
        for _ in range(15):
            if self.prune_logic(min_len) == 0:
                break

    def get_chains(self):
        chains = []
        visited = set()
        nodes = sorted(self.adj.keys(), key=lambda n: (len(self.adj[n])%2!=1, -len(self.adj[n])))
        for start in nodes:
            if start not in self.adj: continue
            for neighbor in list(self.adj[start]):
                edge = tuple(sorted((start, neighbor)))
                if edge in visited: continue
                chain = [start, neighbor]
                visited.add(edge)
                curr = neighbor
                while True:
                    cands = [n for n in self.adj[curr] if tuple(sorted((curr, n))) not in visited]
                    if len(cands) == 1:
                        nxt = cands[0]
                        visited.add(tuple(sorted((curr, nxt))))
                        chain.append(nxt)
                        curr = nxt
                        if len(self.adj[curr]) > 2: break
                    else:
                        break
                if len(chain) > 1:
                    chains.append(chain)
        return chains

def generate_vcarve(geometry, angle_deg, use_limit, max_d, step_len=0.1):
    polys = ensure_list_of_polys(geometry)
    all_paths = []
    tan_a = np.tan(np.radians(angle_deg/2))
    graph = PathGraph()
    
    for poly in polys:
        # 入力図形を滑らかにする（ノイズ対策）
        smooth = poly.simplify(0.02, preserve_topology=True)
        line = smooth.exterior
        length = line.length
        sample_res = 0.2
        num = max(50, min(8000, int(length / sample_res)))
        pts = [line.interpolate(i * length / (num-1)) for i in range(num)]
        coords = np.array([(p.x, p.y) for p in pts])
        try:
            vor = Voronoi(coords)
        except:
            continue
            
        for (p1_idx, p2_idx), (v1_idx, v2_idx) in zip(vor.ridge_points, vor.ridge_vertices):
            if v1_idx < 0 or v2_idx < 0: continue
            v1, v2 = vor.vertices[v1_idx], vor.vertices[v2_idx]
            
            # ボロノイ頂点が図形の内側にあるか厳密にチェック
            pv1, pv2 = Point(v1), Point(v2)
            if not smooth.contains(pv1) or not smooth.contains(pv2): continue
            
            g1, g2 = vor.points[p1_idx], vor.points[p2_idx]
            # 隣接点からのノイズを強力にカット
            if np.linalg.norm(g1 - g2) < sample_res * 5.0: continue
            
            graph.add_edge(v1, v2)
            
    graph.prune_short_leaves(min_len=4.0)
    raw_chains = graph.get_chains()
    
    for chain in raw_chains:
        path_3d = []
        ls = LineString(chain)
        pts_count = max(2, int(ls.length / step_len) + 1)
        for i in range(pts_count):
            pt = ls.interpolate(i * ls.length / (pts_count - 1))
            
            # ★ 重要: 図形の「境界線」までの距離を正しく測る ★
            # 点が内側にあっても0にならないように exterior と interiors を直接使う
            d = pt.distance(smooth.exterior)
            for interior in smooth.interiors:
                d = min(d, pt.distance(interior))
            
            z = -(d / tan_a)
            if use_limit and z < max_d: z = max_d
            path_3d.append((pt.x, pt.y, z))
            
        if len(path_3d) > 1:
            all_paths.append(douglas_peucker(path_3d, 0.02))
    return all_paths

# --- 5. Gコードエンジン ---

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
        z_start = phase.get('z_start', 0)
        z_final = phase.get('z_final', 0)
        z_step = max(0.01, phase.get('z_step', abs(z_final - z_start)))
        use_ramp = phase.get('use_ramp', False)
        
        gc.append(f"; Phase {i+1}: {phase.get('name','')} (F{feed})")
        gc.append(f"F{feed}")
        
        if is_3d:
            for p_pts in paths:
                if not p_pts: continue
                gc.append(f"{G0} X{p_pts[0][0]:.3f} Y{p_pts[0][1]:.3f}")
                gc.append(f"{G1} Z{p_pts[0][2]:.3f}")
                for p in p_pts[1:]:
                    gc.append(f"{G1} X{p[0]:.3f} Y{p[1]:.3f} Z{p[2]:.3f}")
                gc.append(f"{G0} Z{safe}")
        else:
            cur_z = z_start
            while cur_z > z_final:
                tgt_z = max(cur_z - z_step, z_final)
                for path in paths:
                    coords = np.array(path.coords)
                    if len(coords) < 2: continue
                    gc.append(f"{G0} X{coords[0,0]:.3f} Y{coords[0,1]:.3f}")
                    gc.append(f"{G0} Z{cur_z + 1.0}")
                    if use_ramp:
                        dist = 0.0
                        p_len = path.length
                        z_d = cur_z - tgt_z
                        reached = False
                        for j in range(1, len(coords)):
                            dist += np.linalg.norm(coords[j] - coords[j-1])
                            ratio = min(dist / min(p_len, 50.0), 1.0)
                            gc.append(f"{G1} X{coords[j,0]:.3f} Y{coords[j,1]:.3f} Z{cur_z - (z_d * ratio):.3f}")
                            if ratio >= 1.0:
                                reached = True
                                for k in range(j+1, len(coords)):
                                    gc.append(f"{G1} X{coords[k,0]:.3f} Y{coords[k,1]:.3f} Z{tgt_z:.3f}")
                                break
                        if not reached:
                            gc.append(f"{G1} Z{tgt_z:.3f}")
                    else:
                        gc.append(f"{G1} Z{tgt_z:.3f}")
                        for xy in coords[1:]:
                            gc.append(f"{G1} X{xy[0]:.3f} Y{xy[1]:.3f}")
                    gc.append(f"{G0} Z{safe}")
                cur_z = tgt_z
    gc.append(footer.strip())
    return "\n".join(gc)

# --- 6. UI ---

st.set_page_config(page_title="yosikeiCAM", layout="wide")
st.title("⚡ yosikeiCAM 2.1")
st.caption("Ver 2.1: Vカーブ深さ計算修正・パス安定版")

with st.sidebar:
    st.header("📍 原点設定")
    origin = st.radio("原点基準", ["Bottom-Left (左下)", "Center (中心)", "Original (CAD座標)"], index=0)
    st.divider()
    st.header("⚙️ 加工設定")
    tab1, tab2, tab3, tab4 = st.tabs(["ポケット", "面取り", "Vカーブ", "ドリル"])
    
    with tab1:
        enable_pocket = st.checkbox("有効", True, key="cp")
        st.divider()
        dia = st.number_input("工具径 (mm)", value=3.0, min_value=0.01, step=0.1, format="%.3f")
        clear = st.number_input("仕上げ代 (mm)", value=0.0, step=0.1)
        c1, c2 = st.columns(2)
        with c1: depth = st.number_input("最終深さ Z", value=-1.0, step=0.1)
        with c2: z_step_p = st.number_input("Zピッチ", value=1.0, min_value=0.01, step=0.1)
        step = st.slider("ステップオーバー (%)", 10, 90, 50)/100.0
        use_dogbone = st.checkbox("ドッグボーン", True)
        use_ramp = st.checkbox("ランピング進入", False)
        st.caption("▼ 送り速度")
        f_r = st.number_input("粗送り速度", value=300, min_value=1, step=50, key="fr")
        f_f = f_r
        f_m = "Step-down"
        if clear > 0:
            st.markdown("---")
            st.caption("▼ 仕上げ設定")
            c3, c4 = st.columns(2)
            with c3: f_f = st.number_input("仕上げ速度", value=300, min_value=1, step=50, key="ff")
            with c4:
                f_o = st.radio("仕上げ深さ", ["刻み", "一括"], index=0, key="fo")
                f_m = "Full-Depth" if f_o == "一括" else "Step-down"
                
    with tab2:
        enable_chamfer = st.checkbox("有効", True, key="cc")
        st.divider()
        cw = st.number_input("幅 (mm)", 0.5, step=0.1)
        to = st.number_input("刃先オフセット", value=1.0, min_value=0.0, step=0.1)
        z_c = -(cw + to)
        fc_r = st.number_input("速度", value=300, min_value=1, step=50, key="fcr")
        ucf = st.checkbox("2回加工", False, key="uccf")
        cfa = 0.0
        fc_f = 300
        if ucf:
            c1, c2 = st.columns(2)
            with c1: cfa = st.number_input("代 (mm)", value=0.2, min_value=0.01, step=0.1)
            with c2: fc_f = st.number_input("仕上速度", value=300, min_value=1, step=50, key="fcf")
            
    with tab3:
        enable_vcarve = st.checkbox("有効", False, key="cv")
        st.divider()
        va = st.number_input("角度", 60.0, step=10.0)
        uvl = st.checkbox("深さ制限", False)
        vl = st.number_input("制限 (mm)", value=-3.0) if uvl else -100.0
        fv = st.number_input("速度", value=300, min_value=1, step=50, key="fv")
        vr = st.slider("精度", 0.2, 0.02, 0.05)
        
    with tab4:
        enable_drill = st.checkbox("有効", False, key="cd")
        st.divider()
        ddt = st.number_input("穴径 (mm)", value=3.0, min_value=0.01, step=0.1)
        dd = st.number_input("深さ Z", value=-5.0)
        pd = st.number_input("ペック", value=2.0, min_value=0.1)
        fd = st.number_input("速度", value=200, min_value=1, step=50, key="fd")
        
    st.divider()
    ppn = st.selectbox("ポストプロセッサ", list(POST_PROCESSORS.keys()))
    pp = POST_PROCESSORS[ppn]
    with st.expander("Gコード詳細"):
        h_c = st.text_area("Header", pp["start"])
        f_c = st.text_area("Footer", pp["end"])

st.header("1. DXFアップロード")
f = st.file_uploader("", type=["dxf"])

if f:
    bn = os.path.splitext(f.name)[0]
    polys_raw = dxf_to_shapely_list(f.getvalue())
    if polys_raw:
        tu = unary_union(polys_raw)
        minx, miny, maxx, maxy = tu.bounds
        w, h = maxx-minx, maxy-miny
        ox, oy = 0, 0
        if origin.startswith("Bottom-Left"): ox, oy = -minx, -miny
        elif origin.startswith("Center"): ox, oy = -(minx+w/2), -(miny+h/2)
        
        polys_moved = [translate(p, ox, oy) for p in polys_raw]
        st.sidebar.divider()
        st.sidebar.subheader("📐 パス選択")
        cont = st.sidebar.container()
        all_c = cont.checkbox("すべて選択", value=True, key="sa")
        selected = []
        for i, p in enumerate(polys_moved):
            if cont.checkbox(f"Path #{i+1} (Area:{p.area:.1f})", value=all_c, key=f"p{i}"):
                selected.append(i)
                
        target_polys = [polys_moved[i] for i in selected]
        gfc = merge_polygons_xor(target_polys)
        
        if gfc:
            ds = analyze_holes(gfc)
            if ds:
                st.sidebar.info("💡 穴: " + ", ".join([f"φ{d}({c}個)" for d, c in ds.items()]))
                
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"加工サイズ: {w:.1f} x {h:.1f} mm")
            fig, ax = plt.subplots(figsize=(5,5))
            ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=2, zorder=10)
            ax.axhline(0, color='red', alpha=0.5)
            ax.axvline(0, color='red', alpha=0.5)
            for i, p in enumerate(polys_moved):
                st_l = 'k-' if i in selected else 'k:'
                al = 1.0 if i in selected else 0.1
                ax.plot(*p.exterior.xy, st_l, alpha=al, linewidth=1)
                for interior in p.interiors:
                    ax.plot(*interior.xy, st_l, alpha=al, linewidth=1)
            if enable_drill and gfc:
                dp = find_drill_points(gfc, ddt)
                for pt in dp:
                    ax.plot(pt.x, pt.y, 'x', color='tab:purple')
            ax.axis('equal')
            ax.grid(True, linestyle=':')
            st.pyplot(fig)
            
        with c2:
            st.header("2. パス生成")
            gc_p, gc_c, gc_v, gc_d = None, None, None, None
            dpr, dpf, dpc, dpv = [], [], [], []
            if gfc and not gfc.is_empty:
                if enable_pocket:
                    p_r, p_f = generate_pocket(gfc, dia, clear, step, use_dogbone)
                    dpr, dpf = p_r, p_f
                    phs = []
                    if p_r:
                        phs.append({'name': 'Rough', 'paths': p_r, 'z_start': 0, 'z_final': depth, 'feed': f_r, 'z_step': z_step_p, 'use_ramp': use_ramp})
                    if p_f:
                        phs.append({'name': 'Finish', 'paths': p_f, 'z_start': 0, 'z_final': depth, 'feed': f_f, 'z_step': (abs(depth) if f_m == "Full-Depth" else z_step_p), 'use_ramp': use_ramp})
                    if phs:
                        gc_p = make_gcode_phases_advanced(phs, "EndMill", h_c, f_c, pp["format"])
                if enable_chamfer:
                    rp, fp = generate_chamfer_separated(gfc, cw, to, cfa if ucf else 0.0)
                    dpc = rp + fp
                    phs = []
                    if rp: phs.append({'name':'Rough', 'paths': rp, 'z_start': 0, 'z_final': z_c, 'feed': fc_r})
                    if fp: phs.append({'name':'Finish', 'paths': fp, 'z_start': 0, 'z_final': z_c, 'feed': (fc_f if ucf else fc_r)})
                    if phs: gc_c = make_gcode_phases_advanced(phs, "Chamfer", h_c, f_c, pp["format"])
                if enable_vcarve:
                    with st.spinner("VCarve..."):
                        vps = generate_vcarve(gfc, va, uvl, vl, vr)
                        dpv = vps
                    if vps:
                        gc_v = make_gcode_phases_advanced([{'name': 'V-Carve', 'paths': vps, 'z_start': 0, 'z_final': 0, 'feed': fv}], "VBit", h_c, f_c, pp["format"], True)
                if enable_drill:
                    dpts = find_drill_points(gfc, ddt)
                    if dpts:
                        gc_d = generate_drill_gcode(dpts, 0, dd, pd, fd, f"Drill {ddt}mm", h_c, f_c, pp["format"])
            
            fig2, ax2 = plt.subplots(figsize=(5,5))
            ax2.plot(0, 0, 'r+', markersize=15)
            ax2.axhline(0, color='red', alpha=0.5)
            ax2.axvline(0, color='red', alpha=0.5)
            for p in polys_moved:
                ax2.plot(*p.exterior.xy, 'k--', alpha=0.1)
            for ls in dpr: ax2.plot(*ls.xy, color='tab:blue', alpha=0.5)
            for ls in dpf: ax2.plot(*ls.xy, color='tab:cyan', alpha=1.0)
            for ls in dpc: ax2.plot(*ls.xy, color='tab:green', alpha=0.9)
            for pts in dpv:
                ax2.plot([p[0] for p in pts], [p[1] for p in pts], color='tab:red', linewidth=0.8)
            ax2.axis('equal')
            st.pyplot(fig2)
            
            b1, b2, b3, b4 = st.columns(4)
            if gc_p: b1.download_button("📥 POCKET", gc_p, f"{bn}_pocket.nc")
            if gc_c: b2.download_button("📥 CHAMFER", gc_c, f"{bn}_chamfer.nc")
            if gc_v: b3.download_button("📥 VCARVE", gc_v, f"{bn}_vcarve.nc")
            if gc_d: b4.download_button("📥 DRILL", gc_d, f"{bn}_drill.nc")
    else:
        st.error("有効な閉じた図形が見つかりません。")
