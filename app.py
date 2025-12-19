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
    overcut_ratio = 1.05 
    
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
                    is_inside = poly.contains(Point(test_pt))
                    if not is_inside: bisector = -bisector
                    
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
                        p = ezdxf.path.make_path(e)
                        pts = list(p.flattening(0.01))
                        if len(pts) > 2:
                            poly = Polygon([(v.x, v.y) for v in pts])
                    
                    if 'poly' in locals():
                        if poly.is_valid:
                            if poly.area > 0.0001: 
                                polys.append(poly)
                        elif not poly.is_valid:
                            clean = make_valid(poly)
                            if clean.area > 0.0001: 
                                if clean.geom_type == 'Polygon': polys.append(clean)
                                elif clean.geom_type == 'MultiPolygon': polys.extend(clean.geoms)
                except: pass
        polys.sort(key=lambda x: x.area, reverse=True)
        return polys
    except Exception as e:
        st.error(f"DXF Read Error: {e}")
        return []
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
        if d: sizes.append(round(d, 2))
        for interior in p.interiors:
            d = check_poly(Polygon(interior))
            if d: sizes.append(round(d, 2))
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
        if pt: drill_points.append(pt)
        for interior in p.interiors:
            pt = check_poly(Polygon(interior))
            if pt: drill_points.append(pt)
    return drill_points

def generate_drill_gcode(points, z_start, z_final, peck_depth, feed, tool_name, header, footer, fmt):
    if not points: return None
    gc = [header.strip(), f"; Tool: {tool_name} (Drill)", "T1 M06", f"F{int(feed)}", ""]
    G0 = "G0" if "G0/" in fmt else "G00"
    G1 = "G1" if "G0/" in fmt else "G01"
    safe = 5.0
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
                gc.append(f"{G0} Z{z_start + 0.5}")
                gc.append(f"{G0} Z{target_z + 0.5}")
            current_z = target_z
        gc.append(f"{G0} Z{safe}")
    gc.append(footer.strip())
    return "\n".join(gc)

# --- 4. パス生成 ---

def generate_pocket(geometry, tool_d, clearance, stepover, dogbone):
    paths = []
    r = tool_d / 2.0
    step = tool_d * stepover
    offset_rough = -(r - clearance + step)
    try: current_rough = geometry.buffer(offset_rough, join_style=2)
    except: current_rough = Polygon()

    while not current_rough.is_empty and current_rough.area > 0.01:
        current_polys = ensure_list_of_polys(current_rough)
        if not current_polys: break
        for p in current_polys:
            paths.append(p.exterior)
            paths.extend(p.interiors)
        try: current_rough = current_rough.buffer(-step, join_style=2)
        except: break

    work_geom_finish = geometry
    if dogbone: work_geom_finish = apply_dogbone(geometry, tool_d)
    offset_finish = -(r - clearance)
    try:
        finish_pass = work_geom_finish.buffer(offset_finish, join_style=2)
        finish_polys = ensure_list_of_polys(finish_pass)
        for p in finish_polys:
            paths.insert(0, p.exterior)
            paths.extend(p.interiors)
    except: pass
    return [LineString(p.coords) for p in paths if p.length > 0.1]

# ★ 面取り (2回加工対応: パスを分離して返す)
def generate_chamfer_separated(geometry, width, tip_offset, finish_allowance=0.0):
    rough_paths = []
    finish_paths = []
    
    # 1. 粗加工 (仕上げ代がある場合のみ)
    if finish_allowance > 0:
        offset_rough = tip_offset + finish_allowance
        try:
            p_rough = geometry.buffer(offset_rough, join_style=1)
            p_list_rough = ensure_list_of_polys(p_rough)
            for poly in p_list_rough:
                rough_paths.append(poly.exterior)
                rough_paths.extend(poly.interiors)
        except: pass

    # 2. 仕上げ加工
    try:
        p_finish = geometry.buffer(tip_offset, join_style=1)
        p_list_finish = ensure_list_of_polys(p_finish)
        for poly in p_list_finish:
            finish_paths.append(poly.exterior)
            finish_paths.extend(poly.interiors)
    except: pass
    
    return (
        [LineString(ls.coords) for ls in rough_paths],
        [LineString(ls.coords) for ls in finish_paths]
    )

def generate_vcarve(geometry, angle_deg, use_limit, max_d, step_len=0.1):
    polys = ensure_list_of_polys(geometry)
    all_paths = []
    tan_a = np.tan(np.radians(angle_deg/2))
    for poly in polys:
        simple = poly.simplify(0.05)
        line = simple.exterior
        length = line.length
        num = int(length / step_len)
        if num > 800: num = 800 
        if num < 20: num = 20
        pts = [line.interpolate(i * length / num) for i in range(num)]
        coords = np.array([(p.x, p.y) for p in pts])
        try: vor = Voronoi(coords)
        except: continue
        segments = []
        for p1i, p2i in vor.ridge_vertices:
            if p1i < 0 or p2i < 0: continue
            p1 = vor.vertices[p1i]
            p2 = vor.vertices[p2i]
            if simple.contains(Point(p1)) and simple.contains(Point(p2)):
                segments.append(LineString([p1, p2]))
        if not segments: continue
        merged = linemerge(segments)
        lines = []
        if merged.geom_type == 'LineString': lines = [merged]
        elif merged.geom_type == 'MultiLineString': lines = list(merged.geoms)
        else: lines = list(merged)
        for l in lines:
            l_pts = []
            dist_pts = int(l.length / step_len) + 1
            if dist_pts < 2: dist_pts = 2
            for i in range(dist_pts):
                pt = l.interpolate(i * step_len)
                d = line.distance(pt)
                z = -(d / tan_a)
                if use_limit:
                    if z < max_d: z = max_d
                l_pts.append((pt.x, pt.y, z))
            if len(l_pts) > 1:
                l_pts = douglas_peucker(l_pts, 0.05)
                all_paths.append(l_pts)
    return all_paths

# ★ Gコード生成エンジン (フェーズ対応版)
def make_gcode_phases(phases, tool_name, header, footer, fmt="G00/G01", is_3d=False):
    """
    phases: list of dict {'paths': [], 'z_start': float, 'z_final': float, 'feed': int}
    """
    gc = [header.strip(), f"; Tool: {tool_name}", "T1 M06"]
    
    G0 = "G0" if "G0/" in fmt else "G00"
    G1 = "G1" if "G0/" in fmt else "G01"
    safe = 5.0
    
    # 最初のフェーズの前にG0 Zsafeを入れる
    gc.append(f"{G0} Z{safe}")

    for i, phase in enumerate(phases):
        paths = phase['paths']
        if not paths: continue
        
        feed = int(phase['feed'])
        z_start = phase['z_start']
        z_final = phase['z_final']
        
        gc.append(f"; --- Phase {i+1}: F{feed} ---")
        gc.append(f"F{feed}")
        
        if is_3d:
            # 3Dパス (V-Carve)
            for path_pts in paths:
                if not path_pts: continue
                p0 = path_pts[0]
                gc.append(f"{G0} X{p0[0]:.3f} Y{p0[1]:.3f}")
                gc.append(f"{G1} Z{p0[2]:.3f}")
                for p in path_pts[1:]:
                    gc.append(f"{G1} X{p[0]:.3f} Y{p[1]:.3f} Z{p[2]:.3f}")
                gc.append(f"{G0} Z{safe}")
        else:
            # 2Dパス
            for path in paths:
                coords = np.array(path.coords)
                if len(coords) < 1: continue
                gc.append(f"{G0} X{coords[0,0]:.3f} Y{coords[0,1]:.3f}")
                gc.append(f"{G1} Z{z_start:.3f}")
                gc.append(f"{G1} Z{z_final:.3f}")
                for xy in coords[1:]:
                    gc.append(f"{G1} X{xy[0]:.3f} Y{xy[1]:.3f}")
                gc.append(f"{G0} Z{safe}")

    gc.append(footer.strip())
    return "\n".join(gc)

# --- 5. UI ---

st.set_page_config(page_title="Multi-Path CAM", layout="wide")
st.title("⚡ Multi-Path CAM")
st.caption("Ver 4.9: 面取り速度分離・原点自動補正")

with st.sidebar:
    st.header("📍 原点設定")
    origin = st.radio("原点基準", ["Bottom-Left (全図形の左下)", "Center (全図形の中心)", "Original (CAD座標)"], index=0)
    
    st.divider()
    st.header("⚙️ 加工設定")
    tab1, tab2, tab3, tab4 = st.tabs(["ポケット", "面取り", "Vカーブ", "ドリル"])
    
    with tab1:
        enable_pocket = st.checkbox("ポケット有効", True)
        st.divider()
        dia = st.number_input("工具径 (mm)", value=3.0, min_value=0.01, max_value=None, step=0.1, format="%.3f")
        clear = st.number_input("クリアランス (mm)", 0.0, step=0.1, help="仕上げ代")
        depth = st.number_input("深さ Z (mm)", -1.0, max_value=0.0, step=0.1)
        step = st.slider("ステップオーバー (%)", 10, 90, 50) / 100.0
        use_dogbone = st.checkbox("ドッグボーン (角逃げ)", True)
        feed_p = st.number_input("送り速度 (mm/min)", value=300, min_value=1, max_value=None, step=50, key="fp")
        
    with tab2:
        enable_chamfer = st.checkbox("面取り有効", True)
        st.divider()
        chamfer_w = st.number_input("面取り幅 (mm)", 0.5, step=0.1)
        tip_off = st.number_input("刃先オフセット (mm)", value=1.0, min_value=0.0, max_value=None, step=0.1, format="%.3f")
        
        # ★ 面取りの2回加工オプション
        use_chamfer_finish = st.checkbox("2回加工 (粗+仕上げ)", False)
        chamfer_finish_allowance = 0.0
        feed_c_rough = 300
        feed_c_finish = 300
        
        if use_chamfer_finish:
            c1, c2 = st.columns(2)
            with c1:
                chamfer_finish_allowance = st.number_input("仕上げ代 (mm)", value=0.2, min_value=0.01, step=0.1, format="%.2f")
            with c2:
                feed_c_finish = st.number_input("仕上げ送り速度", value=300, min_value=1, step=50)
            feed_c_rough = st.number_input("粗送り速度", value=300, min_value=1, step=50)
        else:
            feed_c = st.number_input("送り速度", value=300, min_value=1, step=50, key="fc_single")
            feed_c_rough = feed_c
            feed_c_finish = feed_c # 使わないが変数確保
            
        z_c = -(chamfer_w + tip_off)
        st.caption(f"切込深さ: {z_c:.2f}mm")
        
    with tab3:
        enable_vcarve = st.checkbox("Vカーブ有効", False)
        st.divider()
        v_ang = st.number_input("Vビット角度 (度)", 60.0, step=10.0)
        use_v_limit = st.checkbox("深さ制限", False)
        if use_v_limit: v_lim = st.number_input("最大深さ (mm)", value=-3.0, max_value=0.0, step=0.1)
        else: v_lim = -100.0
        feed_v = st.number_input("送り速度 (mm/min)", 300, step=50, key="fv")
        v_res = st.slider("計算精度 (粗---細)", 0.2, 0.02, 0.05, format="%.2f")

    with tab4:
        enable_drill = st.checkbox("ドリル有効", False)
        st.divider()
        drill_dia_target = st.number_input("対象円の直径 (mm)", value=3.0, min_value=0.01, max_value=None, step=0.1, format="%.3f")
        drill_depth = st.number_input("穴深さ Z (mm)", value=-5.0, max_value=0.0, step=0.5)
        peck_depth = st.number_input("ペッキング深さ (mm)", value=2.0, min_value=0.1, step=0.5)
        feed_d = st.number_input("送り速度 (mm/min)", 200, step=50, key="fd")

    st.divider()
    pp_name = st.selectbox("ポストプロセッサ", list(POST_PROCESSORS.keys()))
    pp = POST_PROCESSORS[pp_name]
    with st.expander("Gコード詳細設定"):
        h_code = st.text_area("Header", pp["start"])
        f_code = st.text_area("Footer", pp["end"])

st.header("1. DXFアップロード")
f = st.file_uploader("", type=["dxf"])

if f:
    base_name = os.path.splitext(f.name)[0]
    polys_raw = dxf_to_shapely_list(f.getvalue())
    
    if polys_raw:
        temp_union = unary_union(polys_raw)
        minx, miny, maxx, maxy = temp_union.bounds
        w, h = maxx-minx, maxy-miny
        
        offset_x, offset_y = 0, 0
        if origin.startswith("Bottom-Left"):
            offset_x, offset_y = -minx, -miny
        elif origin.startswith("Center"):
            offset_x, offset_y = -(minx+w/2), -(miny+h/2)
            
        polys_moved = [translate(p, offset_x, offset_y) for p in polys_raw]
        
        st.sidebar.divider()
        st.sidebar.subheader("📐 パス選択")
        selected_indices = []
        container = st.sidebar.container()
        all_checked = container.checkbox("すべて選択", value=True)
        for i, p in enumerate(polys_moved):
            label = f"Path #{i+1} (Area:{p.area:.1f})"
            is_checked = container.checkbox(label, value=all_checked, key=f"p_{i}")
            if is_checked: selected_indices.append(i)
        
        target_polys = [polys_moved[i] for i in selected_indices]
        geom_for_calc = merge_polygons_xor(target_polys)
        
        if geom_for_calc:
            drill_sizes = analyze_holes(geom_for_calc)
            if drill_sizes:
                msg = "💡 検出された円: " + ", ".join([f"φ{d}mm({c}個)" for d, c in drill_sizes.items()])
                st.sidebar.info(msg)
        
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"加工サイズ: {w:.1f} x {h:.1f} mm")
            fig, ax = plt.subplots(figsize=(5,5))
            ax.plot(0, 0, 'r+', markersize=20, markeredgewidth=2, zorder=10, label="原点 (0,0)")
            ax.axhline(0, color='red', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='red', linewidth=0.5, alpha=0.5)
            for i, p in enumerate(polys_moved):
                style = 'k-' if i in selected_indices else 'k:'
                alpha = 1.0 if i in selected_indices else 0.1
                ax.plot(*p.exterior.xy, style, alpha=alpha, linewidth=1)
                for interior in p.interiors: 
                    ax.plot(*interior.xy, style, alpha=alpha, linewidth=1)
            
            if enable_drill and geom_for_calc:
                 drill_preview = find_drill_points(geom_for_calc, drill_dia_target)
                 for pt in drill_preview: ax.plot(pt.x, pt.y, 'x', color='tab:purple')
            ax.axis('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='lower right')
            st.pyplot(fig)
            
        with c2:
            st.header("2. パス生成")
            
            # --- 計算 & Gコード生成 (フェーズ対応) ---
            gc_p, gc_c, gc_v, gc_d = None, None, None, None
            p_disp, c_disp, v_disp = [], [], [] # 描画用リスト

            if geom_for_calc and not geom_for_calc.is_empty:
                # 1. Pocket
                if enable_pocket:
                    # Pocketは現状1フェーズだが、構造統一のためリスト化
                    p_paths = generate_pocket(geom_for_calc, dia, clear, step, use_dogbone)
                    p_disp = p_paths
                    if p_paths:
                        phases = [{'paths': p_paths, 'z_start': 0, 'z_final': depth, 'feed': feed_p}]
                        gc_p = make_gcode_phases(phases, "EndMill", h_code, f_code, pp["format"])
                
                # 2. Chamfer (2フェーズ対応)
                if enable_chamfer:
                    # 戻り値: (rough_paths, finish_paths)
                    rough_paths, finish_paths = generate_chamfer_separated(
                        geom_for_calc, chamfer_w, tip_off, 
                        chamfer_finish_allowance if use_chamfer_finish else 0.0
                    )
                    c_disp = rough_paths + finish_paths
                    
                    if rough_paths or finish_paths:
                        phases = []
                        if rough_paths:
                            phases.append({'paths': rough_paths, 'z_start': 0, 'z_final': z_c, 'feed': feed_c_rough})
                        if finish_paths:
                            phases.append({'paths': finish_paths, 'z_start': 0, 'z_final': z_c, 'feed': feed_c_finish})
                        
                        gc_c = make_gcode_phases(phases, "Chamfer", h_code, f_code, pp["format"])
                
                # 3. VCarve
                if enable_vcarve:
                    if tab3: 
                        with st.spinner("Vカービングパス計算中..."):
                            v_paths = generate_vcarve(geom_for_calc, v_ang, use_v_limit, v_lim, v_res)
                    v_disp = v_paths
                    if v_paths:
                        phases = [{'paths': v_paths, 'z_start': 0, 'z_final': 0, 'feed': feed_v}] # Zはパス内に含む
                        gc_v = make_gcode_phases(phases, "VBit", h_code, f_code, pp["format"], is_3d=True)
                
                # 4. Drill
                if enable_drill:
                    drill_pts = find_drill_points(geom_for_calc, drill_dia_target)
                    gc_d = generate_drill_gcode(drill_pts, 0, drill_depth, peck_depth, feed_d, f"Drill {drill_dia_target}mm", h_code, f_code, pp["format"]) if drill_pts else None

            # --- プレビュー ---
            fig2, ax2 = plt.subplots(figsize=(5,5))
            ax2.plot(0, 0, 'r+', markersize=15, markeredgewidth=2, zorder=10)
            ax2.axhline(0, color='red', linewidth=0.5, alpha=0.5)
            ax2.axvline(0, color='red', linewidth=0.5, alpha=0.5)
            for p in polys_moved:
                ax2.plot(*p.exterior.xy, 'k--', alpha=0.1)
                for interior in p.interiors: ax2.plot(*interior.xy, 'k--', alpha=0.1)
            
            if enable_pocket: ax2.plot([], [], color='tab:blue', linewidth=1.5, label='Pocket')
            if enable_chamfer: ax2.plot([], [], color='tab:green', linewidth=1.5, label='Chamfer')
            if enable_vcarve: ax2.plot([], [], color='tab:red', linewidth=1.0, label='V-Carve')
            if enable_drill: ax2.plot([], [], color='tab:purple', marker='x', linestyle='None', label='Drill')

            for ls in p_disp: ax2.plot(*ls.xy, color='tab:blue', alpha=0.9, linewidth=1.0)
            for ls in c_disp: ax2.plot(*ls.xy, color='tab:green', alpha=0.9, linewidth=1.0)
            for pts in v_disp: ax2.plot([p[0] for p in pts], [p[1] for p in pts], color='tab:red', linewidth=0.8)
            if enable_drill and 'drill_pts' in locals() and drill_pts:
                for pt in drill_pts: ax2.plot(pt.x, pt.y, 'x', color='tab:purple', markersize=8, markeredgewidth=2)
            
            ax2.legend(loc='upper right', framealpha=0.9)
            ax2.axis('equal')
            st.pyplot(fig2)
            
            b1, b2, b3, b4 = st.columns(4)
            if gc_p: b1.download_button("📥 POCKET", gc_p, f"{base_name}_pocket.nc")
            if gc_c: b2.download_button("📥 CHAMFER", gc_c, f"{base_name}_chamfer.nc")
            if gc_v: b3.download_button("📥 VCARVE", gc_v, f"{base_name}_vcarve.nc")
            if gc_d: b4.download_button("📥 DRILL", gc_d, f"{base_name}_drill.nc")
            
            if enable_drill and 'drill_pts' in locals() and drill_pts:
                st.success(f"ドリル穴: {len(drill_pts)}箇所")
            
    else:
        st.error("有効な閉じた図形が見つかりません。")
