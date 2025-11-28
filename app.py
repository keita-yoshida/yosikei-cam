import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO

# 幾何学計算ライブラリ
from shapely.geometry import Polygon, LineString
import ezdxf
import ezdxf.path

# --- 1. コアロジック機能 ---

def dxf_to_shapely_polygon(dxf_content) -> Polygon | None:
    """
    DXFデータから閉じた図形を抽出し、最大の面積を持つ Shapely Polygon を返します。
    """
    tmp_file_path = None
    polygons = []
    
    try:
        # 文字列ならバイト列に変換
        dxf_bytes = dxf_content.encode('utf-8') if isinstance(dxf_content, str) else dxf_content
            
        # 1. バイトデータを一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
            tmp_file.write(dxf_bytes)
            tmp_file_path = tmp_file.name
        
        # 2. ezdxf でファイルを読み込む
        doc = ezdxf.readfile(tmp_file_path)
        msp = doc.modelspace()
        
        # 3. エンティティの解析と変換
        for entity in msp:
            dxftype = entity.dxftype()
            
            # 対応するエンティティのみ処理
            if dxftype in ('LWPOLYLINE', 'POLYLINE', 'SPLINE'):
                try:
                    # ezdxf.path を使用して、あらゆる曲線を直線近似パスに変換
                    path = ezdxf.path.make_path(entity)
                    vertices = list(path.flattening(distance=0.01))
                    points = [(v.x, v.y) for v in vertices]
                    
                    # 頂点数が足りない場合は無効
                    if len(points) < 3:
                        continue

                    # 閉じた図形かどうかの判定
                    is_closed = False
                    if dxftype == 'SPLINE':
                        if hasattr(entity, 'closed') and entity.closed:
                            is_closed = True
                    elif dxftype in ('LWPOLYLINE', 'POLYLINE'):
                        if entity.is_closed:
                            is_closed = True
                    
                    # 幾何学的な一致確認 (誤差 0.001mm)
                    if not is_closed:
                        start_pt = np.array(points[0])
                        end_pt = np.array(points[-1])
                        if np.linalg.norm(start_pt - end_pt) < 1e-3:
                            is_closed = True
                    
                    if is_closed:
                        # 連続する重複点を削除
                        unique_points = [points[0]]
                        for p in points[1:]:
                            if p != unique_points[-1]:
                                unique_points.append(p)
                        
                        # Polygon 作成と検証
                        poly = Polygon(unique_points)
                        
                        if poly.is_valid and poly.area > 1e-6:
                            polygons.append(poly)
                        elif not poly.is_valid:
                            # 自己交差などの修復
                            fixed_poly = poly.buffer(0)
                            if fixed_poly.is_valid and fixed_poly.area > 1e-6:
                                if fixed_poly.geom_type == 'Polygon':
                                    polygons.append(fixed_poly)
                                elif fixed_poly.geom_type == 'MultiPolygon':
                                    largest = max(fixed_poly.geoms, key=lambda g: g.area)
                                    polygons.append(largest)

                except Exception:
                    continue # 個別のエンティティ変換エラーはスキップ
        
        if not polygons:
            return None
            
        # 候補の中から最大面積のポリゴンを返す
        return max(polygons, key=lambda p: p.area)
        
    except Exception as e:
        st.error(f"DXF解析エラー: {e}")
        return None
    finally:
        # 一時ファイルのクリーンアップ
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass


def add_dogbone_relief(polygon: Polygon, diameter: float) -> LineString:
    """ポケット内角にドッグボーン（直線延長）の逃げを追加"""
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


def generate_pocket_paths(polygon: Polygon, diameter: float, clearance: float, stepover_ratio: float = 0.7, dogbone: bool = True) -> list[LineString]:
    """治具ポケット加工パス生成"""
    tool_r = diameter / 2.0
    
    # 1. 境界オフセット (外側へ)
    try:
        pocket_boundary = polygon.buffer(tool_r + clearance, join_style=2)
    except Exception:
        return [] 
    
    # 2. 内部切削パスの生成
    stepover = diameter * stepover_ratio 
    current_poly = pocket_boundary
    tool_paths = []
    
    while current_poly and current_poly.area > 1e-6:
        if current_poly.exterior:
            tool_paths.append(current_poly.exterior)
        try:
            # 内側へステップオーバー分オフセット
            current_poly = current_poly.buffer(-stepover, join_style=2)
        except Exception:
            break 
            
        if current_poly.geom_type == 'MultiPolygon':
            if current_poly.geoms:
                 current_poly = max(current_poly.geoms, key=lambda g: g.area)
            else:
                 break
        if current_poly.geom_type != 'Polygon':
             break
             
    # 3. ドッグボーン追加 (最外周のみ)
    if dogbone and tool_paths:
        try:
            tool_paths[0] = add_dogbone_relief(Polygon(tool_paths[0]), diameter)
        except Exception:
             pass 
             
    return [LineString(p.coords) for p in tool_paths if p.geom_type in ('LineString', 'LinearRing')]


def generate_chamfer_paths(polygon: Polygon, chamfer_width: float) -> list[LineString]:
    """Vビット面取りパス生成"""
    if chamfer_width <= 0:
        return []
    try:
        # 面取り幅分だけ外側へオフセット
        chamfer_path_poly = polygon.buffer(chamfer_width, join_style=1) 
    except Exception:
        return []
        
    paths = []
    if chamfer_path_poly.geom_type == 'Polygon':
        paths.append(chamfer_path_poly.exterior)
    elif chamfer_path_poly.geom_type == 'MultiPolygon':
         for g in chamfer_path_poly.geoms:
             if g.geom_type == 'Polygon':
                 paths.append(g.exterior)
                 
    return [LineString(p.coords) for p in paths]


def generate_gcode(paths: list[LineString], z_start: float, z_final: float, feed_rate: float, tool_name: str) -> str:
    """パスリストからGコード文字列を生成"""
    gcode = [
        f"; --- {tool_name} Start ---",
        "G21 ; Metric",
        "G90 ; Absolute",
        "G00 Z10.0 ; Safe Z",
        f"T1 M06 ; Tool: {tool_name}",
        f"F{int(feed_rate)}",
        ""
    ]
    
    for path in paths:
        coords = np.array(path.coords)
        if len(coords) < 1: continue

        # アプローチ
        gcode.append(f"G00 X{coords[0, 0]:.3f} Y{coords[0, 1]:.3f}")
        gcode.append(f"G01 Z{z_start:.3f}")
        gcode.append(f"G01 Z{z_final:.3f}")
        
        # 切削移動
        for x, y in coords[1:]:
            gcode.append(f"G01 X{x:.3f} Y{y:.3f}")
            
        # リトラクト
        gcode.append("G00 Z10.0")

    gcode.extend([
        "",
        "M30 ; End",
        f"; --- {tool_name} End ---"
    ])
    return "\n".join(gcode)


# --- 2. Streamlit アプリケーション UI ---

st.set_page_config(page_title="Simple CAM", layout="wide")
st.title("🛠️ 簡易 CNC Gコードジェネレーター")
st.caption("DXFから治具ポケットと面取り加工のGコードを生成します")

# サイドバー: パラメーター入力
with st.sidebar:
    st.header("⚙️ 加工設定")
    
    st.subheader("エンドミル (ポケット加工)")
    tool_diameter = st.number_input("工具径 (mm)", value=3.0, step=0.1, format="%.1f")
    clearance = st.number_input("クリアランス (mm)", value=0.5, step=0.1, format="%.1f", help="治具と製品の隙間")
    pocket_depth = st.number_input("ポケット深さ (mm)", value=-5.0, step=0.1, format="%.1f", help="Z0からの深さ (負の値)")
    add_dogbone = st.checkbox("ドッグボーン逃げを追加", value=True)

    st.divider()

    st.subheader("Vビット (面取り加工)")
    acrylic_thickness = st.number_input("アクリル厚み (mm)", value=3.0, step=0.1, format="%.1f")
    chamfer_width = st.number_input("面取り幅 (mm)", value=0.5, step=0.1, format="%.1f")
    
    # 計算値の表示
    z_chamfer_start = pocket_depth + acrylic_thickness
    z_chamfer_final = z_chamfer_start - chamfer_width
    st.info(f"面取り開始Z: {z_chamfer_start:.2f}mm\n\n切込深さZ: {z_chamfer_final:.2f}mm")

    st.divider()
    feed_rate = st.number_input("送り速度 (mm/min)", value=300, step=10)


# メインエリア: ファイルアップロードと処理
st.header("1. DXFファイル入力")
uploaded_file = st.file_uploader("DXFファイルをアップロード", type=["dxf"])

if uploaded_file is not None:
    # ファイル読み込みと解析
    file_bytes = uploaded_file.getvalue()
    main_polygon = dxf_to_shapely_polygon(file_bytes)

    if main_polygon:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.success(f"解析成功: 面積 {main_polygon.area:.1f} mm²")
            
            # 元図形のプレビュー
            fig, ax = plt.subplots(figsize=(5, 5))
            x, y = main_polygon.exterior.xy
            ax.plot(x, y, color='blue', label='Original')
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)

        # パス計算とGコード生成
        with col2:
            st.header("2. 生成結果")
            
            # パス生成
            pocket_paths = generate_pocket_paths(main_polygon, tool_diameter, clearance, dogbone=add_dogbone)
            chamfer_paths = []
            if chamfer_width > 0:
                chamfer_paths = generate_chamfer_paths(main_polygon, chamfer_width)
            
            # プロット用
            fig_path, ax_path = plt.subplots(figsize=(5, 5))
            ax_path.plot(x, y, color='gray', linestyle='--', alpha=0.5, label='Original')
            
            # Gコード生成と描画
            gcode_pocket = None
            gcode_chamfer = None

            if pocket_paths:
                for idx, path in enumerate(pocket_paths):
                    px, py = path.xy
                    color = 'red' if idx == 0 else 'orange'
                    ax_path.plot(px, py, color=color, linewidth=1, label='Pocket' if idx == 0 else None)
                
                gcode_pocket = generate_gcode(pocket_paths, 0.0, pocket_depth, feed_rate, "Pocket_EM")

            if chamfer_paths:
                for idx, path in enumerate(chamfer_paths):
                    px, py = path.xy
                    ax_path.plot(px, py, color='green', linewidth=1, label='Chamfer')
                
                gcode_chamfer = generate_gcode(chamfer_paths, z_chamfer_start, z_chamfer_final, feed_rate, "Chamfer_Bit")

            ax_path.set_aspect('equal')
            ax_path.legend()
            ax_path.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig_path)
            
            # --- 修正箇所: Gコードを別々にダウンロード ---
            st.subheader("Gコード ダウンロード")
            
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                if gcode_pocket:
                    st.download_button(
                        "📥 ポケット加工 (.nc)",
                        gcode_pocket,
                        "pocket.nc",
                        mime="text/plain",
                        key="dl_pocket"
                    )
                    with st.expander("ポケット Gコード確認"):
                        st.code(gcode_pocket)
                else:
                    st.info("ポケット加工パスなし")

            with dl_col2:
                if gcode_chamfer:
                    st.download_button(
                        "📥 面取り加工 (.nc)",
                        gcode_chamfer,
                        "chamfer.nc",
                        mime="text/plain",
                        key="dl_chamfer"
                    )
                    with st.expander("面取り Gコード確認"):
                        st.code(gcode_chamfer)
                else:
                    st.info("面取り加工パスなし")

    else:
        st.error("有効な図形が見つかりませんでした。")
