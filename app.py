import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO

# 幾何学計算ライブラリ
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate
import ezdxf
import ezdxf.path
# Vカービング用
from scipy.spatial import Voronoi
from shapely.geometry import Point

# --- 0. ポストプロセッサ定義 ---
POST_PROCESSORS = {
    "Generic (汎用)": {
        "desc": "一般的なCNCルーター向け",
        "start": "G21 ; Metric (mm)\nG90 ; Absolute positioning\nG00 Z10.0 ; Safe Z\nM3 S10000 ; Spindle On",
        "end": "M5 ; Spindle Off\nG00 Z10.0 ; Retract\nM30 ; Program End",
        "format": "G00/G01"
    },
    "GRBL / Candle": {
        "desc": "Arduinoベースの小型CNC (3018など) 向け",
        "start": "G21 G90 G17\nG0 Z10.0\nM3 S10000",
        "end": "M5\nG0 Z10.0\nM30",
        "format": "G0/G1"
    },
    "Mach3 / Mach4": {
        "desc": "Mach3/4 コントローラ向け",
        "start": "G21 G90 G40 G49 G80\nG00 Z10.0\nM03 S10000",
        "end": "M05\nG00 Z10.0\nM30",
        "format": "G00/G01"
    },
    "Fanuc (Industrial)": {
        "desc": "ファナック系 産業用MC向け",
        "start": "%\nO1001 (PROGRAM START)\nG21 G90 G40 G49 G80\nG00 G91 G28 Z0.\nG90\nM03 S10000",
        "end": "M05\nG91 G28 Z0.\nG28 Y0.\nM30\n%",
        "format": "G00/G01"
    }
}

# --- 1. コアロジック機能 ---

def dxf_to_shapely_polygon(dxf_content) -> Polygon | None:
    tmp_file_path = None
    polygons = []
    try:
        dxf_bytes = dxf_content.encode('utf-8') if isinstance(dxf_content, str) else dxf_content
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
            tmp_file.write(dxf_bytes)
            tmp_file_path = tmp_file.name
        
        doc = ezdxf.readfile(tmp_file_path)
        msp = doc.modelspace()
        
        for entity in msp:
            if entity.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'SPLINE'):
                try:
                    path = ezdxf.path.make_path(entity)
                    vertices = list(path.flattening(distance=0.01))
                    points = [(v.x, v.y) for v in vertices]
                    if len(points) < 3: continue

                    is_closed = False
                    if entity.dxftype() == 'SPLINE':
                        if hasattr(entity, 'closed') and entity.closed: is_closed = True
                    elif entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                        if entity.is_closed: is_closed = True
                    
                    if not is_closed:
                        if np.linalg.norm(np.array(points[0]) - np.array(points[-1])) < 1e-3:
                            is_closed = True
                    
                    if is_closed:
                        unique_points = [points[0]]
                        for p in points[1:]:
                            if p != unique_points[-1]: unique_points.append(p)
                        poly = Polygon(unique_points)
                        if poly.is_valid and poly.area > 1e-6:
                            polygons.append(poly)
                        elif not poly.is_valid:
                            fixed_poly = poly.buffer(0)
                            if fixed_poly.is_valid and fixed_poly.area > 1e-6:
                                if fixed_poly.geom_type == 'Polygon': polygons.append(fixed_poly)
                                elif fixed_poly.geom_type == 'MultiPolygon': polygons.append(max(fixed_poly.geoms, key=lambda g: g.area))
                except Exception: continue
        
        if not polygons: return None
        return max(polygons, key=lambda p: p.area)
    except Exception as e:
        st.error(f"DXF解析エラー: {e}")
        return None
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try: os.unlink(tmp_file_path)
            except Exception: pass

def align_polygon(polygon: Polygon, mode: str) -> Polygon:
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    if mode == "Bottom-Left (左下)": return translate(polygon, xoff=-minx, yoff=-miny)
    elif mode == "Center (中心)": return translate(polygon, xoff=-(minx + width/2), yoff=-(miny + height/2))
    return polygon

def add_dogbone_relief(polygon: Polygon, diameter: float) -> LineString:
    tool_r = diameter / 2.0
    relief_offset = tool_r * 0.4 
    coords = list(polygon.exterior.coords)
    new_coords = []
    num_points = len(coords) - 1
    for i in range(num_points):
        current = np.array(coords[i])
        prev = np.array(coords[(i - 1 + num_points) % num_points])
        next_point = np.array(coords[(i + 1) % num_points])
        new_coords.append(tuple(current))
        v_in = prev - current
        v_out = next_point - current
        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)
        if norm_in > 1e-6 and norm_out > 1e-6:
            v_in_n = v_in / norm_in
            v_out_n = v_out / norm_out
            relief_pt1 = current + v_in_n * relief_offset
            relief_pt2 = current + v_out_n * relief_offset
            new_coords.append(tuple(relief_pt1))
            new_coords.append(tuple(relief_pt2))
    new_coords.append(new_coords[0])
    return LineString(new_coords)

# ★★★ 修正: MultiPolygon対応版ポケット加工生成 ★★★
def generate_pocket_paths(polygon: Polygon, diameter: float, clearance: float, stepover_ratio: float, dogbone: bool = True) -> list[LineString]:
    """治具ポケット加工パス生成 (MultiPolygon対応)"""
    tool_r = diameter / 2.0
    boundary_offset = clearance - tool_r
    
    try:
        # 初期オフセット (ここでMultiPolygonになる可能性がある)
        pocket_boundary = polygon.buffer(boundary_offset, join_style=2)
    except Exception:
        return [] 
    
    stepover = diameter * stepover_ratio 
    current_geom = pocket_boundary
    tool_paths = []
    
    # 面積がある限りループ
    while current_geom and not current_geom.is_empty and current_geom.area > 1e-6:
        # 現在の形状からパス(外形線)を抽出
        if current_geom.geom_type == 'Polygon':
            tool_paths.append(current_geom.exterior)
            # 中抜き(島)がある場合はその境界もパスに追加
            for interior in current_geom.interiors:
                tool_paths.append(interior)
                
        elif current_geom.geom_type == 'MultiPolygon':
            # 分離した全ての島について処理
            for poly in current_geom.geoms:
                tool_paths.append(poly.exterior)
                for interior in poly.interiors:
                    tool_paths.append(interior)
        
        # 次のステップへ内側にオフセット
        try:
            current_geom = current_geom.buffer(-stepover, join_style=2)
        except Exception:
            break 
            
    # ドッグボーン追加 (最初のパスのみ適用してクラッシュ回避)
    if dogbone and tool_paths:
        try:
            first_path = tool_paths[0]
            # パスが閉じているか確認してPolygon化
            if first_path.is_closed:
                tool_paths[0] = add_dogbone_relief(Polygon(first_path), diameter)
        except Exception:
             pass 
             
    return [LineString(p.coords) for p in tool_paths if p.geom_type in ('LineString', 'LinearRing')]

def generate_chamfer_paths(polygon: Polygon, chamfer_width: float, tip_offset: float = 0.0) -> list[LineString]:
    if chamfer_width <= 0: return []
    total_offset = tip_offset
    try:
        if total_offset > 0: chamfer_path_poly = polygon.buffer(total_offset, join_style=1)
        else: chamfer_path_poly = polygon
    except Exception: return []
    paths = []
    if chamfer_path_poly.geom_type == 'Polygon': paths.append(chamfer_path_poly.exterior)
    elif chamfer_path_poly.geom_type == 'MultiPolygon':
         for g in chamfer_path_poly.geoms:
             if g.geom_type == 'Polygon': paths.append(g.exterior)
    return [LineString(p.coords) for p in paths]

def generate_vcarve_paths(polygon: Polygon, tool_angle_deg: float, max_depth: float, step_length: float = 0.1) -> list:
    boundary_line = polygon.exterior
    length = boundary_line.length
    num_points = int(length / step_length)
    if num_points < 10: num_points = 10
    
    sample_points = [boundary_line.interpolate(i * step_length) for i in range(num_points)]
    coords = np.array([(p.x, p.y) for p in sample_points])
    
    vor = Voronoi(coords)
    medial_axis_segments = []
    tool_angle_rad = np.radians(tool_angle_deg)
    
    for p1_idx, p2_idx in vor.ridge_vertices:
        if p1_idx == -1 or p2_idx == -1: continue
        p1 = vor.vertices[p1_idx]
        p2 = vor.vertices[p2_idx]
        pt1 = Point(p1)
        pt2 = Point(p2)
        
        if polygon.contains(pt1) and polygon.contains(pt2):
            dist1 = boundary_line.distance(pt1)
            dist2 = boundary_line.distance(pt2)
            tan_half_angle = np.tan(tool_angle_rad / 2.0)
            z1 = - (dist1 / tan_half_angle)
            z2 = - (dist2 / tan_half_angle)
            z1 = max(z1, max_depth)
            z2 = max(z2, max_depth)
            
            medial_axis_segments.append([(p1[0], p1[1], z1), (p2[0], p2[1], z2)])
    return medial_axis_segments

def generate_gcode(paths: list, z_start: float, z_final: float, feed_rate: float, tool_name: str, header_code: str, footer_code: str, format_type: str = "G00/G01", is_3d: bool = False) -> str:
    gcode = []
    CMD_G0 = "G0" if format_type == "G0/G1" else "G00"
    CMD_G1 = "G1" if format_type == "G0/G1" else "G01"

    gcode.append(header_code.strip())
    gcode.append(f"; --- Tool: {tool_name} ---")
    gcode.append(f"T1 M06")
    gcode.append(f"F{int(feed_rate)}")
    gcode.append("")
    
    if is_3d:
        safe_z = 5.0 
        for segment in paths:
            p1, p2 = segment
            gcode.append(f"{CMD_G0} X{p1[0]:.3f} Y{p1[1]:.3f}")
            gcode.append(f"{CMD_G1} Z{p1[2]:.3f}")
            gcode.append(f"{CMD_G1} X{p2[0]:.3f} Y{p2[1]:.3f} Z{p2[2]:.3f}")
            gcode.append(f"{CMD_G0} Z{safe_z}")
    else:
        for path in paths:
            coords = np.array(path.coords)
            if len(coords) < 1: continue
            gcode.append(f"{CMD_G0} X{coords[0, 0]:.3f} Y{coords[0, 1]:.3f}")
            gcode.append(f"{CMD_G1} Z{z_start:.3f}")
            gcode.append(f"{CMD_G1} Z{z_final:.3f}")
            for x, y in coords[1:]:
                gcode.append(f"{CMD_G1} X{x:.3f} Y{y:.3f}")
            gcode.append(f"{CMD_G0} Z10.0")

    gcode.append("")
    gcode.append(footer_code.strip())
    return "\n".join(gcode)


# --- 2. Streamlit アプリケーション UI ---

st.set_page_config(page_title="Simple CAM + V-Carve", layout="wide")
st.title("🛠️ 簡易 CNC Gコードジェネレーター")
st.caption("DXFから治具ポケットと面取り加工のGコードを生成します")

with st.sidebar:
    st.header("📍 原点設定")
    origin_mode = st.radio("加工原点 (0,0)", ("Bottom-Left (左下)", "Center (中心)", "Original (DXF座標)"), index=0)
    st.divider()

    st.header("⚙️ 加工設定")
    
    tab1, tab2, tab3 = st.tabs(["ポケット", "面取り", "Vカーブ"])
    
    with tab1:
        st.subheader("エンドミル (ポケット)")
        tool_diameter = st.number_input("工具径 (mm)", value=3.0, step=0.1, format="%.1f", key="pocket_tool")
        clearance = st.number_input("クリアランス (mm)", value=0.05, step=0.01, format="%.2f", help="治具と製品の隙間")
        pocket_depth = st.number_input("ポケット深さ (mm)", value=-1.0, max_value=0.0, step=0.1, format="%.1f", help="負の値")
        stepover_ratio = st.slider("ステップオーバー (%)", 10, 90, 70, 5) / 100.0
        add_dogbone = st.checkbox("ドッグボーン追加", value=True)
        feed_rate_pocket = st.number_input("送り速度 (mm/min)", value=300, step=10, key="feed_pocket")

    with tab2:
        st.subheader("Vビット (面取り)")
        chamfer_width = st.number_input("面取り幅 (mm)", value=0.5, step=0.1, format="%.1f")
        tip_offset = st.number_input("刃先オフセット (mm)", value=1.0, step=0.1, format="%.1f")
        z_chamfer_depth = -1.0 
        if tip_offset < 0: st.error("⚠️ 警告: 刃先オフセットがマイナスです")
        
        # 面取り深さ計算
        z_chamfer_final = - (chamfer_width + tip_offset)
        if z_chamfer_final < 0:
             if z_chamfer_final < -5.0: 
                 st.warning(f"面取り深さ Z{z_chamfer_final:.2f} が深すぎる可能性があります")
        feed_rate_chamfer = st.number_input("送り速度 (mm/min)", value=300, step=10, key="feed_chamfer")

    with tab3:
        st.subheader("Vカービング (彫刻)")
        v_tool_angle = st.number_input("Vビット角度 (度)", value=60.0, step=5.0, format="%.1f")
        v_max_depth = st.number_input("最大深さ制限 (mm)", value=-3.0, max_value=0.0, step=0.1, format="%.1f")
        v_feed_rate = st.number_input("送り速度 (mm/min)", value=300, step=10, key="feed_vcarve")

    st.divider()
    st.header("📝 Gコード設定")
    selected_machine = st.selectbox("ポストプロセッサ", list(POST_PROCESSORS.keys()), index=0)
    machine_config = POST_PROCESSORS[selected_machine]
    with st.expander("詳細設定"):
        start_code_input = st.text_area("スタートコード", value=machine_config["start"], height=100, key=f"s_{selected_machine}")
        end_code_input = st.text_area("エンドコード", value=machine_config["end"], height=100, key=f"e_{selected_machine}")


st.header("1. DXFファイル入力")
uploaded_file = st.file_uploader("DXFファイルをアップロード", type=["dxf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    main_polygon = dxf_to_shapely_polygon(file_bytes)

    if main_polygon:
        main_polygon = align_polygon(main_polygon, origin_mode)

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.success(f"解析成功: 面積 {main_polygon.area:.1f} mm²")
            fig, ax = plt.subplots(figsize=(5, 5))
            x, y = main_polygon.exterior.xy
            ax.plot(x, y, color='blue', label='Original')
            ax.axhline(y=0, color='k', linewidth=0.8, linestyle='-')
            ax.axvline(x=0, color='k', linewidth=0.8, linestyle='-')
            ax.plot(0, 0, 'ro')
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)

        with col2:
            st.header("2. 生成結果")
            
            # 1. ポケット
            pocket_paths = generate_pocket_paths(main_polygon, tool_diameter, clearance, stepover_ratio, add_dogbone)
            
            # 2. 面取り
            z_chamfer_final = - (chamfer_width + tip_offset)
            chamfer_paths = generate_chamfer_paths(main_polygon, chamfer_width, tip_offset)

            # 3. Vカービング
            vcarve_paths = []
            if tab3:
                vcarve_paths = generate_vcarve_paths(main_polygon, v_tool_angle, v_max_depth)

            # プロット
            fig_path, ax_path = plt.subplots(figsize=(5, 5))
            ax_path.plot(x, y, color='gray', linestyle='--', alpha=0.5)
            ax_path.axhline(y=0, color='k', linewidth=0.8)
            ax_path.axvline(x=0, color='k', linewidth=0.8)
            
            gcode_pocket = None
            gcode_chamfer = None
            gcode_vcarve = None

            if pocket_paths:
                for p in pocket_paths: ax_path.plot(p.xy[0], p.xy[1], color='orange', linewidth=1)
                gcode_pocket = generate_gcode(pocket_paths, 0.0, pocket_depth, feed_rate_pocket, "Pocket_EM", start_code_input, end_code_input, machine_config["format"])

            if chamfer_paths:
                for p in chamfer_paths: ax_path.plot(p.xy[0], p.xy[1], color='green', linewidth=1)
                gcode_chamfer = generate_gcode(chamfer_paths, 0.0, z_chamfer_final, feed_rate_chamfer, "Chamfer_Bit", start_code_input, end_code_input, machine_config["format"])

            if vcarve_paths:
                for seg in vcarve_paths:
                    p1, p2 = seg
                    ax_path.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=0.5)
                gcode_vcarve = generate_gcode(vcarve_paths, 0, 0, v_feed_rate, f"V-Carve_{v_tool_angle}deg", start_code_input, end_code_input, machine_config["format"], is_3d=True)

            ax_path.set_aspect('equal')
            st.pyplot(fig_path)
            
            # ダウンロード
            st.subheader("Gコード ダウンロード")
            c1, c2, c3 = st.columns(3)
            with c1:
                if gcode_pocket:
                    st.download_button("📥 ポケット (.nc)", gcode_pocket, "pocket.nc", key="dl_p")
                else: st.info("ポケットなし")
            with c2:
                if gcode_chamfer:
                    st.download_button("📥 面取り (.nc)", gcode_chamfer, "chamfer.nc", key="dl_c")
                else: st.info("面取りなし")
            with c3:
                if gcode_vcarve:
                    st.download_button("📥 Vカーブ (.nc)", gcode_vcarve, "vcarve.nc", key="dl_v")
                else: st.info("Vカーブなし")

    else:
        st.error("有効な図形が見つかりませんでした。")
else:
    st.info("DXFファイルをアップロードして開始してください。")
