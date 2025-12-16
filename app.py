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

def ensure_valid_polygons(geometry):
    """
    どんな形状データが来ても、必ず「有効なPolygonのリスト」に変換して返す。
    エラー('NullPolygon'など)の原因をここで排除する。
    """
    if geometry is None or geometry.is_empty:
        return []
    
    # 無効な形状なら修復を試みる
    if not geometry.is_valid:
        geometry = make_valid(geometry)

    polys = []
    if geometry.geom_type == 'Polygon':
        polys.append(geometry)
    elif geometry.geom_type == 'MultiPolygon':
        polys.extend(geometry.geoms)
    elif geometry.geom_type == 'GeometryCollection':
        for g in geometry.geoms:
            if g.geom_type == 'Polygon':
                polys.append(g)
            elif g.geom_type == 'MultiPolygon':
                polys.extend(g.geoms)
    
    # 面積がほぼゼロのゴミを除去
    clean_polys = [p for p in polys if p.area > 0.001]
    return clean_polys

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
    """ドッグボーン適用（高感度版）"""
    if polygon.is_empty: return polygon
    
    # 形状を整える
    poly = polygon.simplify(0.001)
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
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-5 or n2 < 1e-5: continue
        v1 /= n1
        v2 /= n2
        
        # 外積で内角判定
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        angle = math.degrees(math.atan2(cross, dot))
        
        # ★修正: 判定範囲を大幅に拡大 (10度〜170度)
        # これにより菱形の鋭角(30度など)や六角形(120度)も全て対象にする
        # Shapelyの標準的な外周は反時計回り。左折が凸、右折が凹(内角)。
        # ここでは「曲がっている角」すべてに対して、内側への逃げを検討する
        
        # 内側への曲がり（凹角＝ポケットの隅）を検出
        # 時計回りのデータの場合もあるため、絶対値や符号に頼りすぎず、
        # 「角の二等分線方向に少し進んで、ポリゴン内部なら内角」と判定する
        
        if 5 < abs(angle) < 175: # ほぼ一直線以外はすべてチェック
            # 二等分線ベクトル (外向きか内向きかはまだ不明)
            bisector = -v1 + v2
            bn = np.linalg.norm(bisector)
            if bn > 1e-5:
                bisector /= bn
                
                # 判定: 頂点からごくわずかに二等分線方向に進んだ点が、ポリゴンの「内側」にあるか？
                # 内側にあるなら、そこは「ポケットの隅」なのでドッグボーンが必要
                test_pt = p_curr + bisector * 0.01
                is_inner_corner = poly.contains(Point(test_pt))
                
                # もし内側でなければ、逆方向（-bisector）が内側か確認（データの回り順対策）
                if not is_inner_corner:
                    test_pt_rev = p_curr - bisector * 0.01
                    if poly.contains(Point(test_pt_rev)):
                        is_inner_corner = True
                        bisector = -bisector
                
                if is_inner_corner:
                    # ここは削るべき隅である
                    # 配置位置: 頂点から内側へ少し入った位置
                    # 90度なら r*(sqrt(2)-1) だが、鋭角だと中心をもっと奥にしないといけない
                    # 簡易的に、rの50%程度内側に入れれば大抵の角は落ちる
                    offset = r * 0.5
                    center = p_curr + bisector * offset
                    
                    # 円を作成
                    circle = Point(center).buffer(r, resolution=16)
                    dogbone_circles.append(circle)

    if not dogbone_circles:
        return polygon
        
    # 合成
    try:
        combined = unary_union([polygon] + dogbone_circles)
        return combined
    except:
        return polygon

def apply_dogbone(geometry, tool_dia):
    polys = ensure_valid_polygons(geometry)
    if not polys: return geometry
    
    # 各パーツごとにドッグボーン処理
    processed = []
    for p in polys:
        processed.append(apply_dogbone_single(p, tool_dia))
        
    return unary_union(processed)

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
                    # 読み込み精度
                    pts = list(p.flattening(0.02))
                    if len(pts) > 2:
                        poly = Polygon([(v.x, v.y) for v in pts])
                        if poly.is_valid and poly.area > 0.5:
                            polys.append(poly)
                        elif not poly.is_valid:
                            clean = make_valid(poly)
                            if clean.area > 0.5: polys.append(clean)
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
    # ドッグボーン処理 (元形状自体を変更して膨らませる)
    if dogbone:
        work_geom = apply_dogbone(work_geom, tool_d)
    
    paths = []
    r = tool_d / 2.0
    offset_dist = -(r - clearance)
    
    try:
        current = work_geom.buffer(offset_dist, join_style=2)
    except:
        return []

    step = tool_d * stepover
    
    # ループ処理 (エラー回避強化)
    while not current.is_empty and current.area > 0.01:
        current_polys = ensure_valid_polygons(current)
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
    polys = ensure_valid_polygons(geometry)
    
    if offset <= 0:
        paths = []
        for p in polys:
            paths.append(p.exterior)
            paths.extend(p.interiors)
        return [LineString(ls.coords) for ls in paths]
    
    try:
        p = geometry.buffer(offset, join_style=1) # 1=Round
        p_list = ensure_valid_polygons(p)
        
        paths = []
        for poly in p_list:
            paths.append(poly.exterior)
            paths.extend(poly.interiors)
        return [LineString(ls.coords) for ls in paths]
    except:
        return []

def generate_vcarve(geometry, angle_deg, use_limit, max_d, step_len=0.1):
    polys = ensure_valid_polygons(geometry)
    all_paths = []
    
    tan_a = np.tan(np.radians(angle_deg/2))

    for poly in polys:
        # 高速化のため少し単純化
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
st.caption("Ver 2.2: 菱形/六角形ドッグボーン対応・凡例表示・深さ制限修正")

with st.sidebar:
    st.header("📍 原点設定")
    origin = st.radio("加工原点 (0,0)", ["Bottom-Left (左下)", "Center (中心)", "Original (DXF座標)"], index=0)
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["ポケット", "面取り", "Vカーブ"])
    
    with tab1:
        st.subheader("エンドミル (ポケット)")
        dia = st.number_input("工具径 (mm)", 0.1, step=0.1)
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
        
        # ★★★ 修正: 深さ制限のチェックボックス ★★★
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
        w, h = maxx-minx, maxy-miny
        
        if origin == "Bottom-Left":
            geom = translate(geom, -minx, -miny)
        elif origin == "Center":
            geom = translate(geom, -(minx+w/2), -(miny+h/2))
            
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"読み込み成功: {w:.1f} x {h:.1f} mm")
            fig, ax = plt.subplots(figsize=(5,5))
            
            # 元図形の描画
            polys = ensure_valid_polygons(geom)
            for i, p in enumerate(polys):
                label = "Original" if i == 0 else ""
                ax.plot(*p.exterior.xy, 'k', linewidth=1.5, label=label)
                for interior in p.interiors: ax.plot(*interior.xy, 'k', linewidth=1.5)
                    
            ax.axis('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            
        with c2:
            st.header("2. パス生成")
            
            # パス計算
            p_paths = generate_pocket(geom, dia, clear, step, use_dogbone)
            c_paths = generate_chamfer(geom, chamfer_w, tip_off)
            
            v_paths = []
            if tab3: 
                with st.spinner("Vカービングパス計算中..."):
                    v_paths = generate_vcarve(geom, v_ang, use_v_limit, v_lim, v_res)

            # Gコード生成
            gc_p = make_gcode(p_paths, 0, depth, feed_p, "EndMill", h_code, f_code, pp["format"]) if p_paths else None
            gc_c = make_gcode(c_paths, 0, z_c, feed_c, "Chamfer", h_code, f_code, pp["format"]) if c_paths else None
            gc_v = make_gcode(v_paths, 0, 0, feed_v, "VBit", h_code, f_code, pp["format"], True) if v_paths else None
            
            # --- プレビュー (色分け & 凡例付き) ---
            fig2, ax2 = plt.subplots(figsize=(5,5))
            
            # 元図形(薄く)
            for p in polys:
                ax2.plot(*p.exterior.xy, 'k--', alpha=0.15)
                for interior in p.interiors: ax2.plot(*interior.xy, 'k--', alpha=0.15)
            
            # 凡例用ダミープロット (各色1回だけ登録)
            ax2.plot([], [], color='tab:blue', linewidth=1.5, label='Pocket')
            ax2.plot([], [], color='tab:green', linewidth=1.5, label='Chamfer')
            ax2.plot([], [], color='tab:red', linewidth=1.0, label='V-Carve')

            # 実プロット
            if p_paths:
                for ls in p_paths: ax2.plot(*ls.xy, color='tab:blue', alpha=0.9, linewidth=1.0)
            if c_paths:
                for ls in c_paths: ax2.plot(*ls.xy, color='tab:green', alpha=0.9, linewidth=1.0)
            if v_paths:
                for pts in v_paths:
                    ax2.plot([p[0] for p in pts], [p[1] for p in pts], color='tab:red', linewidth=0.8)
            
            ax2.legend(loc='upper right', framealpha=0.9)
            ax2.axis('equal')
            st.pyplot(fig2)
            
            # ダウンロードボタン
            b1, b2, b3 = st.columns(3)
            if gc_p: b1.download_button("📥 POCKET.nc", gc_p, "pocket.nc")
            if gc_c: b2.download_button("📥 CHAMFER.nc", gc_c, "chamfer.nc")
            if gc_v: b3.download_button("📥 VCARVE.nc", gc_v, "vcarve.nc")
            
    else:
        st.error("有効な閉じた図形が見つかりません。")
