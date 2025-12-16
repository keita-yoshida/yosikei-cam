import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import math

# 幾何学計算ライブラリ (Shapely)
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
    """どんなGeometryが来ても必ずPolygonのリストにして返す(エラー回避の要)"""
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
    """点pと線分l1-l2の距離"""
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
    """パスの間引き (データ削減)"""
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
        rec_results1 = douglas_peucker(points[:index+1], tolerance)
        rec_results2 = douglas_peucker(points[index:], tolerance)
        return rec_results1[:-1] + rec_results2
    else:
        return [points[0], points[end]]

def apply_dogbone_single(polygon: Polygon, tool_dia: float) -> Polygon:
    """
    ドッグボーン適用処理:
    角の頂点に円（ドリル穴）を配置して合成する
    """
    if polygon.is_empty: return polygon
    
    # 処理前に軽く正規化（ノイズ除去）
    poly = polygon.simplify(0.001) 
    
    # 単純化の結果、MultiPolygonになった場合は再帰的に処理
    if poly.geom_type == 'MultiPolygon':
        parts = [apply_dogbone_single(p, tool_dia) for p in poly.geoms]
        return unary_union(parts)
    if poly.geom_type != 'Polygon': return polygon
    
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]: coords.pop()
    
    num_pts = len(coords)
    dogbone_circles = []
    r = tool_dia / 2.0
    
    for i in range(num_pts):
        p_curr = np.array(coords[i])
        p_prev = np.array(coords[(i - 1) % num_pts])
        p_next = np.array(coords[(i + 1) % num_pts])
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6: continue
        v1 /= norm1
        v2 /= norm2
        
        # 角度計算
        cross_prod = np.cross(v1, v2)
        dot_prod = np.dot(v1, v2)
        angle = math.degrees(math.atan2(cross_prod, dot_prod))
        
        # ターゲット：約90度の角 (45度〜135度, -135度〜-45度)
        # ポケット加工の場合、すべての鋭い角に逃げが必要
        if 45 < abs(angle) < 135:
            # 二等分線方向
            bisector = -v1 + v2
            norm_b = np.linalg.norm(bisector)
            if norm_b > 1e-6:
                bisector /= norm_b
                
                # 頂点から内側へ少し入った位置に円の中心を置く
                # これにより、円周がちょうど頂点を通る（T-Bone/Dogbone）
                offset_dist = r * (math.sqrt(2) - 1)
                
                # 向きの補正（ポリゴンの内側に向ける）
                test_pt = p_curr + bisector * 0.1
                if not poly.contains(Point(test_pt)):
                    bisector = -bisector
                
                center_pos = p_curr + bisector * offset_dist
                
                # 解像度高めの円を作成 (resolution=32)
                circle = Point(center_pos).buffer(r, resolution=32)
                dogbone_circles.append(circle)

    if not dogbone_circles:
        return polygon
        
    # 元のポリゴンとドッグボーン穴（円）を結合
    combined = unary_union([polygon] + dogbone_circles)
    
    # ★修正: 結合後のsimplifyは行わないか、極めて小さくして円形を保つ
    return combined

def apply_dogbone(geometry, tool_dia):
    """全てのパーツにドッグボーンを適用"""
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
            if e.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'SPLINE'):
                try:
                    p = ezdxf.path.make_path(e)
                    pts = list(p.flattening(0.01)) # 精度高め
                    if len(pts) > 2:
                        poly = Polygon([(v.x, v.y) for v in pts])
                        if poly.is_valid and poly.area > 0.1:
                            polys.append(poly)
                        elif not poly.is_valid:
                            clean = make_valid(poly)
                            if clean.area > 0.1: polys.append(clean)
                except: pass
        
        if not polys: return None
        # 全図形を結合
        combined = unary_union(polys)
        return combined
        
    except Exception as e:
        st.error(f"DXF Error: {e}")
        return None
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

# --- 3. パス生成ロジック ---

def generate_pocket(geometry, tool_d, clearance, stepover, dogbone):
    # ドッグボーン処理
    work_geom = geometry
    if dogbone:
        work_geom = apply_dogbone(work_geom, tool_d)
    
    paths = []
    r = tool_d / 2.0
    offset_dist = -(r - clearance) # 内側へオフセット
    
    try:
        current = work_geom.buffer(offset_dist, join_style=2) # 2=Miter
    except:
        return []

    step = tool_d * stepover
    
    # ループ処理 (MultiPolygon対応強化)
    while not current.is_empty and current.area > 0.01:
        # ★重要: ここで必ずリスト化してループする
        current_polys = ensure_list_of_polys(current)
        
        for p in current_polys:
            paths.append(p.exterior)
            paths.extend(p.interiors)
        
        # 次のステップへオフセット
        current = current.buffer(-step, join_style=2)
    
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
    
    p = geometry.buffer(offset, join_style=1) # 1=Round
    p_list = ensure_list_of_polys(p)
    
    paths = []
    for poly in p_list:
        paths.append(poly.exterior)
        paths.extend(poly.interiors)
            
    return [LineString(ls.coords) for ls in paths]

def generate_vcarve_single(polygon, angle_deg, use_limit, max_d, step_len=0.1):
    """1つのPolygonに対するVカーブ計算"""
    # ★重要: 単純化の結果、分裂する可能性を考慮
    simple_geom = polygon.simplify(0.02)
    poly_list = ensure_list_of_polys(simple_geom)
    
    all_final_paths = []
    
    for simple_poly in poly_list:
        line = simple_poly.exterior
        length = line.length
        num = int(length / step_len)
        if num > 1000: num = 1000
        if num < 20: num = 20
        
        pts = [line.interpolate(i * length / num) for i in range(num)]
        coords = np.array([(p.x, p.y) for p in pts])
        
        try:
            vor = Voronoi(coords)
        except:
            continue
            
        segments = []
        for p1i, p2i in vor.ridge_vertices:
            if p1i < 0 or p2i < 0: continue
            p1 = vor.vertices[p1i]
            p2 = vor.vertices[p2i]
            if simple_poly.contains(Point(p1)) and simple_poly.contains(Point(p2)):
                segments.append(LineString([p1, p2]))
                
        if not segments: continue
        
        merged = linemerge(segments)
        lines = []
        if merged.geom_type == 'LineString': lines = [merged]
        elif merged.geom_type == 'MultiLineString': lines = list(merged.geoms)
        else: lines = list(merged)
        
        tan_a = np.tan(np.radians(angle_deg/2))
        
        for l in lines:
            l_pts = []
            dist_pts = int(l.length / step_len) + 1
            if dist_pts < 2: dist_pts = 2
            
            for i in range(dist_pts):
                pt = l.interpolate(i * step_len)
                d = line.distance(pt)
                z = -(d / tan_a)
                
                # 深さ制限
                if use_limit:
                    if z < max_d: z = max_d
                    
                l_pts.append((pt.x, pt.y, z))
                
            if len(l_pts) > 1:
                l_pts = douglas_peucker(l_pts, 0.05)
                all_final_paths.append(l_pts)
            
    return all_final_paths

def generate_vcarve(geometry, angle_deg, use_limit, max_d, step_len=0.1):
    """MultiPolygon対応 Vカーブ"""
    all_paths = []
    polys = ensure_list_of_polys(geometry)
    
    for p in polys:
        paths = generate_vcarve_single(p, angle_deg, use_limit, max_d, step_len)
        all_paths.extend(paths)
        
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

# --- 3. UI ---

st.set_page_config(page_title="Multi-Path CAM", layout="wide")
st.title("⚡ Multi-Path CAM")
st.caption("複数図形完全対応 / 高品質ドッグボーン / Vカーブ")

with st.sidebar:
    st.header("📍 原点")
    origin = st.radio("原点位置", ["Bottom-Left", "Center", "Original"], horizontal=True)
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["ポケット", "面取り", "Vカーブ"])
    
    with tab1:
        st.subheader("エンドミル (ポケット)")
        dia = st.number_input("工具径", 3.0, step=0.1)
        clear = st.number_input("クリアランス", 0.0, step=0.1, help="仕上げ代")
        depth = st.number_input("深さ (Z)", -1.0, max_value=0.0, step=0.1)
        step = st.slider("ステップオーバー%", 10, 90, 50) / 100.0
        use_dogbone = st.checkbox("ドッグボーン (角逃げ)", True)
        feed_p = st.number_input("送り速度 (P)", 300, step=50)
        
    with tab2:
        st.subheader("Vビット (面取り)")
        chamfer_w = st.number_input("面取り幅", 0.5, step=0.1)
        tip_off = st.number_input("刃先オフセット", 1.0, step=0.1)
        feed_c = st.number_input("送り速度 (C)", 300, step=50)
        z_c = -(chamfer_w + tip_off)
        if z_c < 0: st.info(f"切込深さ: {z_c:.2f}mm")
        
    with tab3:
        st.subheader("Vカービング")
        v_ang = st.number_input("Vビット角度", 60.0, step=10.0)
        
        # 深さ制限UI
        use_v_limit = st.checkbox("深さ制限を有効にする", value=False)
        v_lim = -10.0 
        if use_v_limit:
            v_lim = st.number_input("最大深さ制限 (mm)", value=-3.0, max_value=0.0, step=0.1)
            
        feed_v = st.number_input("送り速度 (V)", 300, step=50)
        v_res = st.slider("計算精度 (粗---細)", 0.2, 0.02, 0.05, format="%.2f")

    st.divider()
    pp_name = st.selectbox("ポストプロセッサ", list(POST_PROCESSORS.keys()))
    pp = POST_PROCESSORS[pp_name]
    with st.expander("Gコード詳細"):
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
            for p in polys:
                ax.plot(*p.exterior.xy, 'k', linewidth=1)
                for interior in p.interiors: ax.plot(*interior.xy, 'k', linewidth=1)
                    
            ax.axis('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig)
            
        with c2:
            st.header("2. パス生成")
            
            p_paths = generate_pocket(geom, dia, clear, step, use_dogbone)
            gc_p = make_gcode(p_paths, 0, depth, feed_p, "EndMill", h_code, f_code, pp["format"]) if p_paths else None
            
            c_paths = generate_chamfer(geom, chamfer_w, tip_off)
            gc_c = make_gcode(c_paths, 0, z_c, feed_c, "Chamfer", h_code, f_code, pp["format"]) if c_paths else None
            
            v_paths = []
            if tab3: 
                with st.spinner("Vカービングパス計算中..."):
                    v_paths = generate_vcarve(geom, v_ang, use_v_limit, v_lim, v_res)
            gc_v = make_gcode(v_paths, 0, 0, feed_v, "VBit", h_code, f_code, pp["format"], True) if v_paths else None
            
            # プレビュー
            fig2, ax2 = plt.subplots(figsize=(5,5))
            # 元図形
            for p in polys:
                ax2.plot(*p.exterior.xy, 'k--', alpha=0.2)
                for interior in p.interiors: ax2.plot(*interior.xy, 'k--', alpha=0.2)
            
            # 凡例用のダミープロット
            ax2.plot([], [], color='orange', linewidth=1.5, label='Pocket')
            ax2.plot([], [], color='green', linewidth=1.5, label='Chamfer')
            ax2.plot([], [], color='red', linewidth=1.0, label='V-Carve')

            if p_paths:
                for ls in p_paths: ax2.plot(*ls.xy, color='orange', alpha=0.8, linewidth=1.5)
            if c_paths:
                for ls in c_paths: ax2.plot(*ls.xy, color='green', alpha=0.8, linewidth=1.5)
            if v_paths:
                for pts in v_paths:
                    ax2.plot([p[0] for p in pts], [p[1] for p in pts], color='red', linewidth=1.0)
            
            ax2.legend(loc='upper right')
            ax2.axis('equal')
            st.pyplot(fig2)
            
            # ダウンロード
            b1, b2, b3 = st.columns(3)
            if gc_p: b1.download_button("📥 POCKET.nc", gc_p, "pocket.nc")
            if gc_c: b2.download_button("📥 CHAMFER.nc", gc_c, "chamfer.nc")
            if gc_v: b3.download_button("📥 VCARVE.nc", gc_v, "vcarve.nc")
            
    else:
        st.error("有効な閉じた図形が見つかりません。")
