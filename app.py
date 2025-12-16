import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import math

# 幾何学計算ライブラリ (Shapely)
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.affinity import translate
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid

import ezdxf
import ezdxf.path

# Vカービング用 (Voronoi)
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

# --- 1. 幾何学アルゴリズム ---

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
    """パスの間引き (Douglas-Peucker)"""
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
    単一ポリゴンに対するきれいなドッグボーン処理
    内角に円を配置してUnion結合する
    """
    if polygon.is_empty: return polygon
    
    # 単純化してノイズ除去
    poly = polygon.simplify(0.01)
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
        
        # 内角判定 (外積を使用)
        # ポケット(穴)の内側を加工する場合、多角形の内角が対象
        cross_prod = np.cross(v1, v2)
        angle = math.degrees(math.atan2(cross_prod, np.dot(v1, v2)))
        
        # Shapelyは通常反時計回り。左折(正)が凸、右折(負)が凹ではない
        # 通常のポケット加工では「内側の角」=「90度などの凸角」に対して逃げを作る
        # ここでは簡易的に 45度〜135度の角を対象とする
        if 45 < abs(angle) < 135:
            # 角の二等分線方向 (内側へ向かうベクトル)
            bisector = -v1 + v2
            norm_b = np.linalg.norm(bisector)
            if norm_b > 1e-6:
                bisector /= norm_b
                
                # 円の中心位置の計算
                # 90度の場合、頂点から r/sqrt(2) だけ内側に入った位置に中心を置くと、
                # 円周がちょうど頂点を通るドッグボーンになる
                dist_from_corner = r / math.sqrt(2)
                
                # 向きの調整 (ポリゴンの内側へ)
                # 単純に頂点から少しずらしてテスト配置
                test_pt = p_curr + bisector * 0.1
                if not poly.contains(Point(test_pt)):
                    bisector = -bisector # 向き反転
                
                center_pos = p_curr + bisector * dist_from_corner
                dogbone_circles.append(Point(center_pos).buffer(r + 0.01)) # 少し大きめに

    if not dogbone_circles:
        return polygon
        
    # 元のポリゴンとドッグボーン穴（円）を結合
    combined = unary_union([polygon] + dogbone_circles)
    return combined.simplify(0.01)

def apply_dogbone(geometry, tool_dia):
    """MultiPolygon対応のドッグボーンラッパー"""
    if geometry.geom_type == 'Polygon':
        return apply_dogbone_single(geometry, tool_dia)
    elif geometry.geom_type == 'MultiPolygon':
        # 全ての島に対して適用
        new_parts = [apply_dogbone_single(p, tool_dia) for p in geometry.geoms]
        return unary_union(new_parts)
    return geometry

# --- 2. パス生成ロジック ---

def dxf_to_shapely(dxf_bytes):
    """DXFを読み込み、すべての閉じたパスを結合して返す"""
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
                    # 高速化のため精度を少し落とす (0.01 -> 0.05)
                    pts = list(p.flattening(0.05))
                    if len(pts) > 2:
                        poly = Polygon([(v.x, v.y) for v in pts])
                        if poly.is_valid and poly.area > 0.5: # 小さすぎるゴミは無視
                            polys.append(poly)
                        elif not poly.is_valid:
                            clean = make_valid(poly)
                            if clean.area > 0.5: polys.append(clean)
                except: pass
        
        if not polys: return None
        
        # ★ 全てのパスを結合して1つのGeometryにする
        combined = unary_union(polys)
        return combined.simplify(0.02, preserve_topology=True)
        
    except Exception as e:
        st.error(f"DXF Error: {e}")
        return None
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

def generate_pocket(geometry, tool_d, clearance, stepover, dogbone):
    # 1. ドッグボーン適用
    work_geom = geometry
    if dogbone:
        work_geom = apply_dogbone(work_geom, tool_d)
    
    paths = []
    # 壁際オフセット (クリアランス分内側へ)
    r = tool_d / 2.0
    offset_dist = -(r - clearance)
    
    try:
        current = work_geom.buffer(offset_dist, join_style=2) # 2=Miter
    except:
        return []

    step = tool_d * stepover
    
    # ポケット切削ループ (MultiPolygon対応)
    while not current.is_empty and current.area > 0.1:
        if current.geom_type == 'Polygon':
            paths.append(current.exterior)
            paths.extend(current.interiors)
        elif current.geom_type == 'MultiPolygon':
            for p in current.geoms:
                paths.append(p.exterior)
                paths.extend(p.interiors)
        elif current.geom_type == 'GeometryCollection':
             for g in current.geoms:
                 if g.geom_type == 'Polygon':
                     paths.append(g.exterior)
        
        # 次のオフセット
        current = current.buffer(-step, join_style=2)
    
    return [LineString(p.coords) for p in paths if p.length > 0.1]

def generate_chamfer(geometry, width, tip_offset):
    offset = tip_offset
    if offset <= 0:
        # オフセットなし
        if geometry.geom_type == 'Polygon': return [geometry.exterior]
        elif geometry.geom_type == 'MultiPolygon': return [g.exterior for g in geometry.geoms]
        return []
    
    # オフセット実行 (MultiPolygon対応)
    p = geometry.buffer(offset, join_style=1) # 1=Round
    paths = []
    
    if p.geom_type == 'Polygon':
        paths.append(p.exterior)
        paths.extend(p.interiors)
    elif p.geom_type == 'MultiPolygon':
        for g in p.geoms:
            paths.append(g.exterior)
            paths.extend(g.interiors)
            
    return [LineString(ls.coords) for ls in paths]

def generate_vcarve_single(polygon, angle_deg, max_d, step_len=0.1):
    """単一ポリゴンに対するVカーブ計算"""
    simple_poly = polygon.simplify(0.05)
    line = simple_poly.exterior
    
    length = line.length
    num = int(length / step_len)
    if num > 1500: num = 1500 # 高速化のための点数制限
    if num < 10: num = 10
    
    pts = [line.interpolate(i * length / num) for i in range(num)]
    coords = np.array([(p.x, p.y) for p in pts])
    
    try:
        vor = Voronoi(coords)
    except:
        return []
        
    segments = []
    for p1i, p2i in vor.ridge_vertices:
        if p1i < 0 or p2i < 0: continue
        p1 = vor.vertices[p1i]
        p2 = vor.vertices[p2_idx] # Typo fix: p2_idx -> p2i
        p2 = vor.vertices[p2i]
        
        # ポリゴン内部の線分のみ抽出
        if simple_poly.contains(Point(p1)) and simple_poly.contains(Point(p2)):
            segments.append(LineString([p1, p2]))
            
    if not segments: return []
    
    # パス結合 (LineMerge)
    merged = linemerge(segments)
    lines = []
    if merged.geom_type == 'LineString': lines = [merged]
    elif merged.geom_type == 'MultiLineString': lines = list(merged.geoms)
    else: lines = list(merged)
    
    final_paths = []
    tan_a = np.tan(np.radians(angle_deg/2))
    
    for l in lines:
        l_pts = []
        # 再サンプリングしてZ計算
        dist_pts = int(l.length / step_len) + 1
        if dist_pts < 2: dist_pts = 2
        
        for i in range(dist_pts):
            pt = l.interpolate(i * step_len)
            d = line.distance(pt)
            z = -(d / tan_a)
            if z < max_d: z = max_d
            l_pts.append((pt.x, pt.y, z))
            
        # Douglas-Peucker で間引き (データ量削減)
        if len(l_pts) > 1:
            l_pts = douglas_peucker(l_pts, 0.02)
            final_paths.append(l_pts)
            
    return final_paths

def generate_vcarve(geometry, angle_deg, max_d, step_len=0.1):
    """MultiPolygon対応 Vカーブラッパー"""
    all_paths = []
    
    # 入力形状をリスト化
    polys = []
    if geometry.geom_type == 'Polygon':
        polys = [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        polys = list(geometry.geoms)
    elif geometry.geom_type == 'GeometryCollection':
        for g in geometry.geoms:
            if g.geom_type == 'Polygon': polys.append(g)
            
    # 各ポリゴンごとに計算して結合
    for p in polys:
        paths = generate_vcarve_single(p, angle_deg, max_d, step_len)
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
st.caption("複数図形対応 / 高品質ドッグボーン / Vカーブ")

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
        v_lim = st.number_input("最大深さ制限", -3.0, max_value=0.0)
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
    # 読み込み処理
    geom = dxf_to_shapely(f.getvalue())
    
    if geom and not geom.is_empty:
        # 原点移動
        minx, miny, maxx, maxy = geom.bounds
        w, h = maxx-minx, maxy-miny
        
        if origin == "Bottom-Left":
            geom = translate(geom, -minx, -miny)
        elif origin == "Center":
            geom = translate(geom, -(minx+w/2), -(miny+h/2))
            
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"読み込み成功: {w:.1f} x {h:.1f} mm")
            fig, ax = plt.subplots(figsize=(4,4))
            
            # 元図形の描画
            if geom.geom_type == 'Polygon':
                ax.plot(*geom.exterior.xy, 'b')
                for interior in geom.interiors: ax.plot(*interior.xy, 'b')
            elif geom.geom_type == 'MultiPolygon':
                for g in geom.geoms:
                    ax.plot(*g.exterior.xy, 'b')
                    for interior in g.interiors: ax.plot(*interior.xy, 'b')
                    
            ax.axis('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig)
            
        with c2:
            st.header("2. パス生成")
            
            # ポケット
            p_paths = generate_pocket(geom, dia, clear, step, use_dogbone)
            gc_p = make_gcode(p_paths, 0, depth, feed_p, "EndMill", h_code, f_code, pp["format"]) if p_paths else None
            
            # 面取り
            c_paths = generate_chamfer(geom, chamfer_w, tip_off)
            gc_c = make_gcode(c_paths, 0, z_c, feed_c, "Chamfer", h_code, f_code, pp["format"]) if c_paths else None
            
            # Vカーブ
            v_paths = []
            if tab3: 
                # 計算時間短縮のためスピナーを表示
                with st.spinner("Vカービングパス計算中..."):
                    v_paths = generate_vcarve(geom, v_ang, v_lim, v_res)
            gc_v = make_gcode(v_paths, 0, 0, feed_v, "VBit", h_code, f_code, pp["format"], True) if v_paths else None
            
            # プレビュー
            fig2, ax2 = plt.subplots(figsize=(4,4))
            # 元図形(薄く)
            if geom.geom_type == 'Polygon':
                ax2.plot(*geom.exterior.xy, 'k--', alpha=0.3)
            elif geom.geom_type == 'MultiPolygon':
                for g in geom.geoms: ax2.plot(*g.exterior.xy, 'k--', alpha=0.3)
            
            if p_paths:
                for ls in p_paths: ax2.plot(*ls.xy, 'orange', alpha=0.8, linewidth=1, label='Pocket')
            if c_paths:
                for ls in c_paths: ax2.plot(*ls.xy, 'g', alpha=0.8, linewidth=1, label='Chamfer')
            if v_paths:
                for pts in v_paths:
                    ax2.plot([p[0] for p in pts], [p[1] for p in pts], 'r', linewidth=0.6, label='V-Carve')
                    
            ax2.axis('equal')
            st.pyplot(fig2)
            
            # ダウンロード
            b1, b2, b3 = st.columns(3)
            if gc_p: b1.download_button("📥 POCKET.nc", gc_p, "pocket.nc")
            if gc_c: b2.download_button("📥 CHAMFER.nc", gc_c, "chamfer.nc")
            if gc_v: b3.download_button("📥 VCARVE.nc", gc_v, "vcarve.nc")
            
    else:
        st.error("有効な閉じた図形が見つかりません。")
