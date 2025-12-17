import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import math

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
    """GeometryをPolygonのリストに変換"""
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
    """ドッグボーン適用"""
    if polygon.is_empty: return polygon
    poly = polygon.simplify(0.001)
    if poly.geom_type != 'Polygon': return polygon
    
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]: coords.pop()
    
    num_pts = len(coords)
    dogbone_circles = []
    r = tool_dia / 2.0
    overcut_ratio = 1.05 
    
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
        angle = math.degrees(math.atan2(cross, dot))
        
        if abs(angle) > 5:
            bisector = v2 - v1
            bn = np.linalg.norm(bisector)
            if bn > 1e-6:
                bisector /= bn
                test_pt = p_curr + bisector * 0.01
                direction_is_outward = not poly.contains(Point(test_pt))
                if not direction_is_outward:
                    bisector = -bisector
                
                dist = r * overcut_ratio
                center = p_curr + bisector * dist
                circle = Point(center).buffer(r, resolution=16)
                dogbone_circles.append(circle)

    if not dogbone_circles: return polygon
    try:
        return unary_union([polygon] + dogbone_circles).simplify(0.001)
    except:
        return polygon

def apply_dogbone(geometry, tool_dia):
    polys = ensure_list_of_polys(geometry)
    if not polys: return geometry
    new_parts = [apply_dogbone_single(p, tool_dia) for p in polys]
    return unary_union(new_parts)

# --- 2. データ読み込み ---

def dxf_to_shapely(dxf_bytes):
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
                        # 円は多角形近似
                        center = e.dxf.center
                        radius = e.dxf.radius
                        poly = Point(center[:2]).buffer(radius, resolution=64)
                    else:
                        p = ezdxf.path.make_path(e)
                        pts = list(p.flattening(0.01))
                        if len(pts) > 2:
                            poly = Polygon([(v.x, v.y) for v in pts])
                    
                    if 'poly' in locals():
                        if poly.is_valid and poly.area > 0.01:
                            polys.append(poly)
                        elif not poly.is_valid:
                            clean = make_valid(poly)
                            if clean.area > 0.01: polys.append(clean)
                except: pass
        if not polys: return None
        return unary_union(polys)
    except Exception as e:
        st.error(f"DXF Read Error: {e}")
        return None
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

# --- 3. ドリル穴検出ロジック ---

def find_drill_points(geometry, target_dia, tolerance=0.1):
    """
    指定された直径に近い円形ポリゴンを探し、中心座標を返す
    """
    polys = ensure_list_of_polys(geometry)
    drill_points = []
    
    target_r = target_dia / 2.0
    tol_r = tolerance / 2.0
    
    for p in polys:
        minx, miny, maxx, maxy = p.bounds
        w = maxx - minx
        h = maxy - miny
        
        # 1. バウンディングボックスのアスペクト比チェック (正方形に近いか)
        if abs(w - h) > tolerance: continue
        
        # 2. サイズチェック
        if not (target_dia - tolerance <= w <= target_dia + tolerance): continue
        
        # 3. 真円度チェック (面積 / (π*r^2))
        # ポリゴン近似されているので、少し甘めに判定
        expected_area = math.pi * ((w/2)**2)
        if abs(p.area - expected_area) / expected_area > 0.15: continue # 15%以上の誤差なら円じゃないかも
        
        # 円と認定
        drill_points.append(p.centroid)
        
    return drill_points

def generate_drill_gcode(points, z_start, z_final, peck_depth, feed, tool_name, header, footer, fmt):
    if not points: return None
    
    gc = [header.strip(), f"; Tool: {tool_name} (Drill)", "T1 M06", f"F{int(feed)}", ""]
    G0 = "G0" if "G0/" in fmt else "G00"
    G1 = "G1" if "G0/" in fmt else "G01"
    safe = 5.0
    retract = 1.0 # ペッキング時の戻り高さ (穴底から少し上)
    
    for pt in points:
        x, y = pt.x, pt.y
        gc.append(f"; Drill Hole at X{x:.2f} Y{y:.2f}")
        gc.append(f"{G0} X{x:.3f} Y{y:.3f}")
        gc.append(f"{G0} Z{z_start + 1.0}") # 早送りで素材直上へ
        
        current_z = z_start
        
        # ペッキングループ
        while current_z > z_final:
            target_z = current_z - peck_depth
            if target_z < z_final: target_z = z_final
            
            gc.append(f"{G1} Z{target_z:.3f}") # 切削
            
            if target_z > z_final:
                # 切り粉排出（リトラクト）
                gc.append(f"{G0} Z{z_start + 0.5}") # 穴の外へ一旦出る
                gc.append(f"{G0} Z{target_z + 0.5}") # 戻る（少し手前まで）
                
            current_z = target_z
            
        gc.append(f"{G0} Z{safe}") # 安全高さへ
        
    gc.append(footer.strip())
    return "\n".join(gc)

# --- 4. 既存パス生成ロジック ---

def generate_pocket(geometry, tool_d, clearance, stepover, dogbone):
    work_geom = geometry
    if dogbone: work_geom = apply_dogbone(work_geom, tool_d)
    paths = []
    r = tool_d / 2.0
    offset_dist = -(r - clearance)
    try: current = work_geom.buffer(offset_dist, join_style=2)
    except: return []
    step = tool_d * stepover
    while not current.is_empty and current.area > 0.01:
        current_polys = ensure_list_of_polys(current)
        if not current_polys: break
        for p in current_polys:
            paths.append(p.exterior)
            paths.extend(p.interiors)
        try: current = current.buffer(-step, join_style=2)
        except: break
    return [LineString(p.coords) for p in paths if p.length > 0.1]

def generate_chamfer(geometry, width, tip_offset):
    offset = tip_offset
    polys = ensure_list_of_polys(geometry)
    if offset <= 0:
        paths = []
        for p in polys:
            paths.append(p.exterior)
            paths.extend(p.interiors)
        return [LineString(ls.coords) for ls in paths]
    try:
        p = geometry.buffer(offset, join_style=1)
        p_list = ensure_list_of_polys(p)
        paths = []
        for poly in p_list:
            paths.append(poly.exterior)
            paths.extend(poly.interiors)
        return [LineString(ls.coords) for ls in paths]
    except: return []

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

def make_gcode(paths, z_start, z_final, feed, tool_name, header, footer, fmt="G00/G01", is_3d=False):
    gc = [header.strip(), f"; Tool: {tool_name}", "T1 M06", f"F{int(feed)}", ""]
    G0 = "G0" if "G0/" in fmt else "G00"
    G1 = "G1" if "G0/" in fmt else "G01"
    safe = 5.0
    if is_3d:
        for path in paths:
            if not path: continue
            x, y, z = path[0]
            gc.append(f"{G0} X{x:.3f} Y{y:.3f}")
            gc.append(f"{G1} Z{z:.3f}")
            for p in path[1:]:
                gc.append(f"{G1} X{p[0]:.3f} Y{p[1]:.3f} Z{p[2]:.3f}")
            gc.append(f"{G0} Z{safe}")
    else:
        for path in paths:
            coords = list(path.coords)
            if not coords: continue
            x, y = coords[0]
            gc.append(f"{G0} X{x:.3f} Y{y:.3f}")
            gc.append(f"{G1} Z{z_start:.3f}")
            gc.append(f"{G1} Z{z_final:.3f}")
            for p in coords[1:]:
                gc.append(f"{G1} X{p[0]:.3f} Y{p[1]:.3f}")
            gc.append(f"{G0} Z{safe}")
    gc.append(footer.strip())
    return "\n".join(gc)

# --- 5. UI ---

st.set_page_config(page_title="Multi-Path CAM", layout="wide")
st.title("⚡ Multi-Path CAM")
st.caption("Ver 3.0: ドリル加工(穴あけ) 対応版")

with st.sidebar:
    st.header("📍 原点設定")
    origin = st.radio("加工原点 (0,0)", ["Bottom-Left (左下)", "Center (中心)", "Original (DXF座標)"], index=0)
    st.divider()
    
    # ★ タブに「ドリル」を追加
    tab1, tab2, tab3, tab4 = st.tabs(["ポケット", "面取り", "Vカーブ", "ドリル"])
    
    with tab1:
        st.subheader("エンドミル (ポケット)")
        dia = st.number_input("工具径 (mm)", 3.0, step=0.1)
        clear = st.number_input("クリアランス (mm)", 0.0, step=0.1, help="仕上げ代")
        depth = st.number_input("深さ Z (mm)", -1.0, max_value=0.0, step=0.1)
        step = st.slider("ステップオーバー (%)", 10, 90, 50) / 100.0
        use_dogbone = st.checkbox("ドッグボーン (角逃げ)", True, help="鋭角も鈍角もしっかりえぐります")
        feed_p = st.number_input("送り速度 (mm/min)", 300, step=50, key="fp")
        
    with tab2:
        st.subheader("Vビット (面取り)")
        chamfer_w = st.number_input("面取り幅 (mm)", 0.5, step=0.1)
        tip_off = st.number_input("刃先オフセット (mm)", 1.0, step=0.1)
        feed_c = st.number_input("送り速度 (mm/min)", 300, step=50, key="fc")
        z_c = -(chamfer_w + tip_off)
        st.caption(f"切込深さ: {z_c:.2f}mm")
        
    with tab3:
        st.subheader("Vカービング (彫刻)")
        v_ang = st.number_input("Vビット角度 (度)", 60.0, step=10.0)
        use_v_limit = st.checkbox("深さ制限を有効にする", value=False)
        if use_v_limit:
            v_lim = st.number_input("最大深さ制限 (mm)", value=-3.0, max_value=0.0, step=0.1)
        else:
            v_lim = -100.0
        feed_v = st.number_input("送り速度 (mm/min)", 300, step=50, key="fv")
        v_res = st.slider("計算精度 (粗---細)", 0.2, 0.02, 0.05, format="%.2f")

    # ★ ドリル設定UI
    with tab4:
        st.subheader("ドリル加工 (穴あけ)")
        drill_dia_target = st.number_input("対象円の直径 (mm)", value=3.0, step=0.1, help="DXF内でこの直径を持つ円だけを穴あけします")
        drill_depth = st.number_input("穴深さ Z (mm)", value=-5.0, max_value=0.0, step=0.5)
        peck_depth = st.number_input("ペッキング深さ (mm)", value=2.0, min_value=0.1, step=0.5, help="1回に掘り進む深さ。深く掘る場合に切り粉を排出します")
        feed_d = st.number_input("送り速度 (mm/min)", 200, step=50, key="fd")
        st.info("※DXF内で「円」として描かれた図形、または正方形に近いポリゴンを検出します。")

    st.divider()
    pp_name = st.selectbox("ポストプロセッサ", list(POST_PROCESSORS.keys()))
    pp = POST_PROCESSORS[pp_name]
    with st.expander("Gコード詳細設定"):
        h_code = st.text_area("Header", pp["start"])
        f_code = st.text_area("Footer", pp["end"])

st.header("1. DXFアップロード")
f = st.file_uploader("", type=["dxf"])

if f:
    geom = dxf_to_shapely(f.getvalue())
    
    if geom and not geom.is_empty:
        minx, miny, maxx, maxy = geom.bounds
        w, h = maxx-minx, maxy-miny
        
        if origin == "Bottom-Left":
            geom = translate(geom, -minx, -miny)
        elif origin == "Center":
            geom = translate(geom, -(minx+w/2), -(miny+h/2))
            
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"読み込み成功: {w:.1f} x {h:.1f} mm")
            fig, ax = plt.subplots(figsize=(5,5))
            
            polys = ensure_list_of_polys(geom)
            for i, p in enumerate(polys):
                label = "Original" if i == 0 else ""
                ax.plot(*p.exterior.xy, 'k', linewidth=1.5, label=label)
                for interior in p.interiors: ax.plot(*interior.xy, 'k', linewidth=1.5)
                    
            ax.axis('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='upper right')
            st.pyplot(fig)
            
        with c2:
            st.header("2. パス生成")
            
            # --- 計算処理 ---
            # 1. ポケット
            p_paths = generate_pocket(geom, dia, clear, step, use_dogbone)
            gc_p = make_gcode(p_paths, 0, depth, feed_p, "EndMill", h_code, f_code, pp["format"]) if p_paths else None
            
            # 2. 面取り
            c_paths = generate_chamfer(geom, chamfer_w, tip_off)
            gc_c = make_gcode(c_paths, 0, z_c, feed_c, "Chamfer", h_code, f_code, pp["format"]) if c_paths else None
            
            # 3. Vカーブ
            v_paths = []
            if tab3: 
                with st.spinner("Vカービングパス計算中..."):
                    v_paths = generate_vcarve(geom, v_ang, use_v_limit, v_lim, v_res)
            gc_v = make_gcode(v_paths, 0, 0, feed_v, "VBit", h_code, f_code, pp["format"], True) if v_paths else None
            
            # 4. ドリル (新規)
            drill_pts = find_drill_points(geom, drill_dia_target)
            gc_d = generate_drill_gcode(drill_pts, 0, drill_depth, peck_depth, feed_d, f"Drill {drill_dia_target}mm", h_code, f_code, pp["format"]) if drill_pts else None

            # --- プレビュー ---
            fig2, ax2 = plt.subplots(figsize=(5,5))
            
            # 元図形(薄く)
            for p in polys:
                ax2.plot(*p.exterior.xy, 'k--', alpha=0.15)
                for interior in p.interiors: ax2.plot(*interior.xy, 'k--', alpha=0.15)
            
            # 凡例用ダミー
            ax2.plot([], [], color='tab:blue', linewidth=1.5, label='Pocket')
            ax2.plot([], [], color='tab:green', linewidth=1.5, label='Chamfer')
            ax2.plot([], [], color='tab:red', linewidth=1.0, label='V-Carve')
            ax2.plot([], [], color='tab:purple', marker='x', linestyle='None', label='Drill')

            # 実プロット
            if p_paths:
                for ls in p_paths: ax2.plot(*ls.xy, color='tab:blue', alpha=0.9, linewidth=1.0)
            if c_paths:
                for ls in c_paths: ax2.plot(*ls.xy, color='tab:green', alpha=0.9, linewidth=1.0)
            if v_paths:
                for pts in v_paths:
                    ax2.plot([p[0] for p in pts], [p[1] for p in pts], color='tab:red', linewidth=0.8)
            if drill_pts:
                for pt in drill_pts:
                    ax2.plot(pt.x, pt.y, 'x', color='tab:purple', markersize=8, markeredgewidth=2)
            
            ax2.legend(loc='upper right', framealpha=0.9)
            ax2.axis('equal')
            st.pyplot(fig2)
            
            # ダウンロードボタン
            b1, b2, b3, b4 = st.columns(4)
            if gc_p: b1.download_button("📥 POCKET", gc_p, "pocket.nc")
            if gc_c: b2.download_button("📥 CHAMFER", gc_c, "chamfer.nc")
            if gc_v: b3.download_button("📥 VCARVE", gc_v, "vcarve.nc")
            if gc_d: b4.download_button("📥 DRILL", gc_d, "drill.nc")
            
            if drill_pts:
                st.success(f"ドリル穴を {len(drill_pts)} 箇所検出しました (φ{drill_dia_target}mm)")
            
    else:
        st.error("有効な閉じた図形が見つかりません。")
