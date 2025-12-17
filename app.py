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
    """どんなGeometryが来ても必ずPolygonのリストにして返す"""
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
        rec_results1 = douglas_peucker(points[:index+1], tolerance)
        rec_results2 = douglas_peucker(points[index:], tolerance)
        return rec_results1[:-1] + rec_results2
    else:
        return [points[0], points[end]]

# ★★★ 改良版ドッグボーン生成ロジック ★★★
def apply_dogbone_single(polygon: Polygon, tool_dia: float) -> Polygon:
    """
    角の逃げ（ドッグボーン）処理
    アクリル等の嵌め合いのために、内角の頂点から外側へドリル穴を追加する
    """
    if polygon.is_empty: return polygon
    
    # ノイズ除去は最小限にする（角を丸めないため）
    poly = polygon.simplify(0.0001)
    if poly.geom_type != 'Polygon': return polygon
    
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]: coords.pop()
    
    num_pts = len(coords)
    dogbone_circles = []
    r = tool_dia / 2.0
    
    # 確実に角を落とすためのオーバーカット係数
    # 1.0だとぴったり、少し大きくすると確実
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
        
        # 角度計算
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        angle = math.degrees(math.atan2(cross, dot))
        
        # 判定: 直線(0度)以外の「曲がっている角」はすべてチェック対象にする
        # 特にポケット加工の場合、あらゆる凸角（内側に突き出た角）が干渉の原因になる
        if abs(angle) > 5: # 5度以上曲がっていれば処理対象
            
            # 二等分線ベクトルを計算
            # (v2 - v1) は角の「外側」を向くベクトルになる
            bisector = v2 - v1
            bn = np.linalg.norm(bisector)
            
            if bn > 1e-6:
                bisector /= bn
                
                # 【重要】ベクトルの向き判定
                # ポケット加工(穴)の場合、「穴の外側へ向かう方向」に逃げを作りたい。
                # 頂点からわずかにbisector方向に進んだ点が、ポリゴンの「外」であれば正解。
                test_pt = p_curr + bisector * 0.01
                
                # Shapelyの contains は境界を含む場合があるため注意
                # 穴加工なので、「ポリゴンの外」＝「削るべき領域の外（残すべき母材）」
                # ドッグボーンは「削る領域（ポリゴン）」を「拡張」するもの。
                # なので、拡張する方向（円を置く方向）は、元のポリゴンの「外」でなければならない。
                
                direction_is_outward = not poly.contains(Point(test_pt))
                
                # もし内側を向いていたら反転させる
                if not direction_is_outward:
                    bisector = -bisector
                
                # 配置位置の計算
                # ユーザー提案:「中心から外側に1本飛び出る形」
                # これを実現するには、角の頂点から半径Rの距離に円の中心を置けば、
                # 円周がちょうど頂点を通る形になる。
                # 確実に頂点をクリアするために、ほんの少し(overcut_ratio)奥に配置する。
                dist = r * overcut_ratio
                center = p_curr + bisector * dist
                
                # 円を生成して追加
                # resolution=16 で十分滑らか
                circle = Point(center).buffer(r, resolution=16)
                dogbone_circles.append(circle)

    if not dogbone_circles:
        return polygon
        
    # 元の形状とドッグボーン穴を結合 (Union)
    try:
        combined = unary_union([polygon] + dogbone_circles)
        return combined.simplify(0.001)
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
            if e.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'SPLINE'):
                try:
                    p = ezdxf.path.make_path(e)
                    pts = list(p.flattening(0.01))
                    if len(pts) > 2:
                        poly = Polygon([(v.x, v.y) for v in pts])
                        if poly.is_valid and poly.area > 0.1:
                            polys.append(poly)
                        elif not poly.is_valid:
                            clean = make_valid(poly)
                            if clean.area > 0.1: polys.append(clean)
                except: pass
        
        if not polys: return None
        combined = unary_union(polys)
        return combined
        
    except Exception as e:
        st.error(f"DXF Read Error: {e}")
        return None
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

# --- 3. パス生成ロジック ---

def generate_pocket(geometry, tool_d, clearance, stepover, dogbone):
    work_geom = geometry
    # ★重要: まずドッグボーンで形状自体を拡張する
    if dogbone:
        work_geom = apply_dogbone(work_geom, tool_d)
    
    paths = []
    r = tool_d / 2.0
    offset_dist = -(r - clearance) # 内側へ
    
    try:
        # ドッグボーン適用後の形状に対してオフセットを開始
        current = work_geom.buffer(offset_dist, join_style=2) # 2=Miter
    except:
        return []

    step = tool_d * stepover
    
    while not current.is_empty and current.area > 0.01:
        current_polys = ensure_list_of_polys(current)
        if not current_polys: break

        for p in current_polys:
            paths.append(p.exterior)
            paths.extend(p.interiors)
        
        try:
            current = current.buffer(-step, join_style=2)
        except:
            break
    
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
        p = geometry.buffer(offset, join_style=1) # 1=Round
        p_list = ensure_list_of_polys(p)
        paths = []
        for poly in p_list:
            paths.append(poly.exterior)
            paths.extend(poly.interiors)
        return [LineString(ls.coords) for ls in paths]
    except:
        return []

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
        
        try:
            vor = Voronoi(coords)
        except:
            continue
            
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

# --- 4. UI ---

st.set_page_config(page_title="Multi-Path CAM", layout="wide")
st.title("⚡ Multi-Path CAM")
st.caption("Ver 2.3: 強力ドッグボーン搭載・アクリル嵌め合い対応")

with st.sidebar:
    st.header("📍 原点設定")
    origin = st.radio("加工原点 (0,0)", ["Bottom-Left (左下)", "Center (中心)", "Original (DXF座標)"], index=0)
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["ポケット", "面取り", "Vカーブ"])
    
    with tab1:
        st.subheader("エンドミル (ポケット)")
        dia = st.number_input("工具径 (mm)", 3.0, step=0.1)
        clear = st.number_input("クリアランス (mm)", 0.0, step=0.1, help="仕上げ代")
        depth = st.number_input("深さ Z (mm)", -1.0, max_value=0.0, step=0.1)
        step = st.slider("ステップオーバー (%)", 10, 90, 50) / 100.0
        # ★ ドッグボーンの説明を明確化
        use_dogbone = st.checkbox("ドッグボーン (角逃げ)", True, help="すべての角にドリル穴を追加して、アクリルの角が入るようにします")
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
        
        # 深さ制限チェックボックス
        use_v_limit = st.checkbox("深さ制限を有効にする", value=False)
        if use_v_limit:
            v_lim = st.number_input("最大深さ制限 (mm)", value=-3.0, max_value=0.0, step=0.1)
        else:
            v_lim = -100.0

        feed_v = st.number_input("送り速度 (mm/min)", 300, step=50, key="fv")
        v_res = st.slider("計算精度 (粗---細)", 0.2, 0.02, 0.05, format="%.2f")

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
        w
