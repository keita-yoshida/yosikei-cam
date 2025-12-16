import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import math

# 幾何学計算ライブラリ (Shapely)
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.affinity import translate, rotate
from shapely.ops import linemerge, unary_union, voronoi_diagram
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

# --- 1. 幾何学アルゴリズム ---

def dist_lseg(l1, l2, p):
    """点pと線分l1-l2の距離 (2D/3D)"""
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

def apply_dogbone(polygon: Polygon, tool_dia: float) -> Polygon:
    """
    きれいなドッグボーン生成:
    内角(90度付近)を検出し、そこにツール径の円を配置してUnion結合する
    """
    if polygon.is_empty: return polygon
    
    # 単純化してノイズ除去
    poly = polygon.simplify(0.01)
    if poly.geom_type != 'Polygon': return polygon # MultiPolygon等はスキップ
    
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]: coords.pop()
    
    num_pts = len(coords)
    dogbone_circles = []
    
    # ツール半径
    r = tool_dia / 2.0
    # 少しだけ食い込ませて確実に角を出すための係数
    overcut = r * 0.05 
    dist_from_corner = (r / math.sqrt(2)) - overcut

    for i in range(num_pts):
        p_curr = np.array(coords[i])
        p_prev = np.array(coords[(i - 1) % num_pts])
        p_next = np.array(coords[(i + 1) % num_pts])
        
        # ベクトル計算
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        # 正規化
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6: continue
        v1 /= norm1
        v2 /= norm2
        
        # 外積で凸凹判定 (左回り前提: 正なら凸, 負なら凹=内角)
        # ShapelyのPolygonは通常反時計回り(CCW)
        cross_prod = np.cross(v1, v2)
        
        # 内角(Concave)かつ、鋭角すぎない(一直線でない)場合
        # ドッグボーンが必要なのは90度前後の内角
        # cross_prod > 0: 左折(凸), < 0: 右折(凹) ※座標系によるがShapely標準では
        
        # 角度計算
        angle = math.degrees(math.atan2(cross_prod, np.dot(v1, v2)))
        
        # 内角判定 (右折 = 凹 = ドッグボーン対象)
        # 厳密にはポリゴンの巻き方向によるが、ここでは簡易的に判定
        # 90度コーナー(angle ~ -90) をターゲットにする
        if -135 < angle < -45: 
            # 角の二等分線方向へ円を配置
            # 入射ベクトルの逆 + 出射ベクトル の方向が角の「外」への二等分線
            bisector = -v1 + v2
            bisector /= np.linalg.norm(bisector)
            
            # 円の中心位置
            # 角の頂点から bisector方向に dist_from_corner だけ進んだ点
            circle_center = p_curr + bisector * (r / math.sin(math.radians(45)) - r) 
            
            # 中心を少し内側に寄せる（過剰切削防止）ための補正
            # シンプルに「角の頂点」に円を置くと削りすぎるので、
            # 「円周が角の頂点を通る」位置からさらにr*sqrt(2)だけ内側へ...
            # 実用的な「T-Bone」や「Dogbone」は、角の頂点から45度方向に r/sqrt(2) ずらした位置に中心を置く
            
            # 実践的ドッグボーン位置: 角の頂点から、二等分線方向に r*sqrt(2) - r 離す
            offset_dist = r * (math.sqrt(2) - 1)
            # 内側へ食い込ませる
            center_pos = p_curr + bisector * offset_dist
            
            dogbone_circles.append(Point(center_pos).buffer(r + 0.05)) # 少し大きめに結合

    if not dogbone_circles:
        return polygon
        
    # 元のポリゴンとドッグボーン穴（円）を結合
    combined = unary_union([polygon] + dogbone_circles)
    return combined.simplify(0.01) # 結合後も整理

# --- 2. パス生成ロジック ---

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
                    # ★ 高速化: flatening の距離を少し粗くする (0.01 -> 0.05)
                    pts = list(p.flattening(0.05))
                    if len(pts) > 2:
                        poly = Polygon([(v.x, v.y) for v in pts])
                        if poly.is_valid and poly.area > 0.1:
                            polys.append(poly)
                        else:
                            polys.append(poly.buffer(0)) # 修復
                except: pass
        if not polys: return None
        # 最大の図形を返す (簡易化)
        main = max(polys, key=lambda x: x.area)
        # ★ 高速化: 読み込み直後に単純化
        return main.simplify(0.02, preserve_topology=True)
    except Exception as e:
        st.error(f"DXF Error: {e}")
        return None
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

def generate_pocket(polygon, tool_d, clearance, stepover, dogbone):
    # 1. ドッグボーン適用 (形状自体を変える)
    work_poly = polygon
    if dogbone:
        work_poly = apply_dogbone(work_poly, tool_d)
    
    # 2. オフセットループ
    paths = []
    # 最初の壁際パス (クリアランス分内側)
    r = tool_d / 2.0
    offset_dist = -(r - clearance) # 内側へ
    
    # そもそも内側に入る隙間があるか？
    try:
        current = work_poly.buffer(offset_dist, join_style=2) # 2=Miter
    except:
        return []

    step = tool_d * stepover
    while not current.is_empty and current.area > 0.1:
        if current.geom_type == 'Polygon':
            paths.append(current.exterior)
            paths.extend(current.interiors)
        elif current.geom_type == 'MultiPolygon':
            for p in current.geoms:
                paths.append(p.exterior)
                paths.extend(p.interiors)
        
        # 次のオフセット
        current = current.buffer(-step, join_style=2)
    
    # LineString化
    return [LineString(p.coords) for p in paths]

def generate_chamfer(polygon, width, tip_offset):
    # 面取りは単純オフセット
    offset = tip_offset
    if offset <= 0: return [polygon.exterior]
    
    p = polygon.buffer(offset, join_style=1) # 1=Round
    if p.geom_type == 'Polygon':
        return [p.exterior]
    elif p.geom_type == 'MultiPolygon':
        return [g.exterior for g in p.geoms]
    return []

def generate_vcarve(polygon, angle_deg, max_d, step_len=0.1):
    # ★ 高速化: 計算前にさらに単純化
    simple_poly = polygon.simplify(0.05)
    line = simple_poly.exterior
    
    # サンプリング
    length = line.length
    num = int(length / step_len)
    if num > 2000: num = 2000 # ★ 高速化: 点数制限
    if num < 10: num = 10
    
    pts = [line.interpolate(i * length / num) for i in range(num)]
    coords = np.array([(p.x, p.y) for p in pts])
    
    # Voronoi
    try:
        vor = Voronoi(coords)
    except:
        return []
        
    segments = []
    for p1i, p2i in vor.ridge_vertices:
        if p1i < 0 or p2i < 0: continue
        p1 = vor.vertices[p1i]
        p2 = vor.vertices[p2i]
        if simple_poly.contains(Point(p1)) and simple_poly.contains(Point(p2)):
            segments.append(LineString([p1, p2]))
            
    if not segments: return []
    
    # 結合 (LineMerge)
    merged = linemerge(segments)
    lines = []
    if merged.geom_type == 'LineString': lines = [merged]
    else: lines = list(merged.geoms)
    
    final_paths = []
    tan_a = np.tan(np.radians(angle_deg/2))
    
    for l in lines:
        # Z計算
        l_pts = []
        # パス上の点を再サンプリング
        dist_pts = int(l.length / step_len) + 1
        for i in range(dist_pts):
            pt = l.interpolate(i * step_len)
            d = line.distance(pt)
            z = -(d / tan_a)
            if z < max_d: z = max_d
            l_pts.append((pt.x, pt.y, z))
            
        # Douglas-Peucker で間引き
        if len(l_pts) > 2:
            l_pts = douglas_peucker(l_pts, 0.02)
            
        final_paths.append(l_pts)
        
    return final_paths

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

st.set_page_config(page_title="Speedy CAM", layout="wide")
st.title("⚡ Speedy CAM (高速版)")
st.caption("DXFからポケット(ドッグボーン付)、面取り、VカーブのGコードを生成")

with st.sidebar:
    st.header("📍 原点")
    origin = st.radio("原点位置", ["Bottom-Left", "Center", "Original"], horizontal=True)
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["ポケット", "面取り", "Vカーブ"])
    
    with tab1:
        dia = st.number_input("工具径", 3.0, step=0.1)
        clear = st.number_input("クリアランス", 0.0, step=0.1, help="仕上げ代(マイナスで食い込み)")
        depth = st.number_input("深さ (Z)", -1.0, max_value=0.0, step=0.1)
        step = st.slider("ステップオーバー%", 10, 90, 50) / 100.0
        use_dogbone = st.checkbox("ドッグボーン (角逃げ)", True)
        feed_p = st.number_input("送り速度 (P)", 300, step=50)
        
    with tab2:
        chamfer_w = st.number_input("面取り幅", 0.5, step=0.1)
        tip_off = st.number_input("刃先オフセット", 1.0, step=0.1)
        feed_c = st.number_input("送り速度 (C)", 300, step=50)
        z_c = -(chamfer_w + tip_off)
        if z_c < 0: st.info(f"切込深さ: {z_c:.2f}mm")
        
    with tab3:
        v_ang = st.number_input("Vビット角度", 60.0, step=10.0)
        v_lim = st.number_input("最大深さ制限", -3.0, max_value=0.0)
        feed_v = st.number_input("送り速度 (V)", 300, step=50)
        v_res = st.slider("計算精度 (粗---細)", 0.1, 0.01, 0.05, format="%.2f")

    st.divider()
    pp_name = st.selectbox("ポストプロセッサ", list(POST_PROCESSORS.keys()))
    pp = POST_PROCESSORS[pp_name]
    with st.expander("Gコード詳細"):
        h_code = st.text_area("Header", pp["start"])
        f_code = st.text_area("Footer", pp["end"])

st.header("1. DXFアップロード")
f = st.file_uploader("", type=["dxf"])

if f:
    poly = dxf_to_shapely(f.getvalue())
    if poly:
        # 原点移動
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx-minx, maxy-miny
        if origin == "Bottom-Left":
            poly = translate(poly, -minx, -miny)
        elif origin == "Center":
            poly = translate(poly, -(minx+w/2), -(miny+h/2))
            
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"読み込み成功: {w:.1f} x {h:.1f} mm")
            fig, ax = plt.subplots(figsize=(4,4))
            x,y = poly.exterior.xy
            ax.plot(x, y, 'b')
            ax.axis('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig)
            
        with c2:
            st.header("2. パス生成")
            
            # ポケット
            p_paths = generate_pocket(poly, dia, clear, step, use_dogbone)
            gc_p = make_gcode(p_paths, 0, depth, feed_p, "EndMill", h_code, f_code, pp["format"]) if p_paths else None
            
            # 面取り
            c_paths = generate_chamfer(poly, chamfer_w, tip_off)
            gc_c = make_gcode(c_paths, 0, z_c, feed_c, "Chamfer", h_code, f_code, pp["format"]) if c_paths else None
            
            # Vカーブ
            v_paths = []
            if tab3: # タブが開かれてる時のみ計算したいが、Streamlitの仕様上難しいので計算フラグにするか、常に計算するか
                # ここではボタンにするか、常時計算（高速化したので）
                v_paths = generate_vcarve(poly, v_ang, v_lim, v_res)
            gc_v = make_gcode(v_paths, 0, 0, feed_v, "VBit", h_code, f_code, pp["format"], True) if v_paths else None
            
            # プレビュー
            fig2, ax2 = plt.subplots(figsize=(4,4))
            ax2.plot(x, y, 'k--', alpha=0.3, label="Original")
            
            if p_paths:
                for ls in p_paths:
                    ax2.plot(*ls.xy, 'orange', alpha=0.8)
            if c_paths:
                for ls in c_paths:
                    ax2.plot(*ls.xy, 'g', alpha=0.8)
            if v_paths:
                for pts in v_paths:
                    ax2.plot([p[0] for p in pts], [p[1] for p in pts], 'r', linewidth=0.5)
                    
            ax2.axis('equal')
            st.pyplot(fig2)
            
            # ダウンロード
            b1, b2, b3 = st.columns(3)
            if gc_p: b1.download_button("POCKET.nc", gc_p, "pocket.nc")
            if gc_c: b2.download_button("CHAMFER.nc", gc_c, "chamfer.nc")
            if gc_v: b3.download_button("VCARVE.nc", gc_v, "vcarve.nc")
            
    else:
        st.error("有効な閉じた図形が見つかりません")
