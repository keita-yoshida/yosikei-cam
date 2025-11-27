import streamlit as st
import numpy as np
from shapely.geometry import Polygon, LineString, mapping, MultiPolygon
import matplotlib.pyplot as plt
import ezdxf 
from io import BytesIO

# --- 幾何学計算とGコード生成のコアロジック ---

def generate_gcode(paths, z_depth, feed_rate, tool_name="T1"):
    """工具パスからGコードを生成する関数"""
    gcode = []
    gcode.append(f"; --- {tool_name} G-Code Start ---")
    gcode.append("G21 ; Metric units")
    gcode.append("G90 ; Absolute positioning")
    gcode.append("G00 Z10.0 ; Safe Z height")
    gcode.append(f"T1 M06 ; Tool Change to {tool_name}")
    gcode.append(f"F{feed_rate} ; Set Feed Rate")
    gcode.append("")

    for i, path in enumerate(paths):
        coords = np.array(path.coords)
        
        # 最初の点への移動
        if i == 0:
            gcode.append(f"G00 X{coords[0, 0]:.4f} Y{coords[0, 1]:.4f}")
            gcode.append(f"G01 Z{z_depth:.4f}")
        else:
             gcode.append(f"G00 X{coords[0, 0]:.4f} Y{coords[0, 1]:.4f}")
             
        
        # パスを切削
        for x, y in coords[1:]:
            gcode.append(f"G01 X{x:.4f} Y{y:.4f}")
            
    # プログラム終了処理
    gcode.append("G00 Z10.0 ; Retract to safe Z")
    gcode.append("M30 ; Program end")
    gcode.append(f"; --- {tool_name} G-Code End ---")
    return "\n".join(gcode)

def add_dogbone_relief(polygon, diameter):
    """治具ポケットの内角にドッグボーン（直線延長）の逃げを追加する関数"""
    tool_r = diameter / 2.0
    relief_offset = tool_r * 0.4 
    
    coords = list(polygon.exterior.coords)
    new_coords = []
    num_points = len(coords) - 1 
    
    for i in range(num_points):
        current = coords[i]
        prev = coords[(i - 1 + num_points) % num_points]
        next_point = coords[(i + 1) % num_points]
        
        new_coords.append(current)

        v_in = np.array(prev) - np.array(current)
        v_out = np.array(next_point) - np.array(current)
        
        if np.linalg.norm(v_in) > 1e-6 and np.linalg.norm(v_out) > 1e-6:
            v_in_n = v_in / np.linalg.norm(v_in)
            v_out_n = v_out / np.linalg.norm(v_out)
            
            relief_pt1 = np.array(current) - v_in_n * relief_offset
            relief_pt2 = np.array(current) - v_out_n * relief_offset
            
            new_coords.append(tuple(relief_pt1))
            new_coords.append(tuple(relief_pt2))
            
    new_coords.append(new_coords[0])
    return LineString(new_coords)


def generate_pocket_paths(polygon, diameter, clearance, z_depth, dogbone=True):
    """治具ポケット加工の工具中心パスを生成する関数"""
    tool_r = diameter / 2.0
    
    boundary_offset = tool_r + clearance
    try:
        pocket_boundary = polygon.buffer(boundary_offset, join_style=2)
    except Exception:
        return [] 

    stepover = diameter * 0.7 
    current_poly = pocket_boundary
    tool_paths = []
    
    while current_poly.area > 0.001:
        if current_poly.exterior:
            tool_paths.append(current_poly.exterior)
            
        try:
            current_poly = current_poly.buffer(-stepover, join_style=2)
        except Exception:
            break 
            
        if current_poly.geom_type == 'MultiPolygon':
            current_poly = max(current_poly.geoms, key=lambda g: g.area)

        if not current_poly:
            break
            
    if dogbone and tool_paths:
        line_path = tool_paths[0]
        tool_paths[0] = add_dogbone_relief(Polygon(line_path), diameter)

    return tool_paths

def generate_chamfer_paths(polygon, chamfer_width, z_start):
    """Vビット面取り加工の工具中心パスを生成する関数"""
    z_final = z_start - chamfer_width
    
    try:
        chamfer_path = polygon.exterior.buffer(chamfer_width, join_style=2)
    except Exception:
        return [], z_final
        
    if chamfer_path.geom_type == 'Polygon':
        return [chamfer_path.exterior], z_final
    
    return [chamfer_path], z_final


def dxf_to_shapely_polygon(uploaded_file):
    """DXFファイルを読み込み、Shapely Polygonに変換する (PLINE, LWPOLYLINE, LINEのみ対応)"""
    
    if uploaded_file is None:
        return None, "ファイルがアップロードされていません。"
    
    try:
        dxf_bytes = uploaded_file.read()
        doc = ezdxf.read(BytesIO(dxf_bytes))
        msp = doc.modelspace()
        
        polylines = []
        
        for entity in msp:
            # ★★★ 修正箇所: 168行目（元のコードの行数）の構文エラーを修正 ★★★
            if entity.dxftype() == 'LWPOLYLINE' or entity.dxftype() == 'POLYLINE':
                coords = [(p[0], p[1]) for p in entity.vertices()]
                
                if entity.is_closed:
                    try:
                        polylines.append(Polygon(coords))
                    except Exception:
                        st.warning(f"Polygon変換に失敗したポリラインがあります。")
                else:
                    pass 
        
        if not polylines:
            return None, "DXFファイル内に閉じたポリライン (LWPOLYLINE/POLYLINE) が見つかりませんでした。"
        
        if len(polylines) > 1:
            main_polygon = max(polylines, key=lambda p: p.area)
            return main_polygon, f"複数の図形を検出。最大面積の図形（頂点数: {len(main_polygon.exterior.coords)}）を採用しました。"
        else:
            return polylines[0], f"図形を検出しました。（頂点数: {len(polylines[0].exterior.coords)}）"

    except ezdxf.DXFStructureError as e:
        return None, f"DXFファイルの構造エラーです: {e}"
    except Exception as e:
        # st.warning は関数内では使用できないため、ここではエラーメッセージを返す
        return None, f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}"


# --- Streamlit アプリケーション ---

st.set_page_config(layout="wide")
st.title("簡易 Web CAM (Python/Streamlit)")
st.caption("治具ポケット加工とVビット面取りのパス生成プロトタイプ")

# --- サイドバーでのパラメーター設定 ---
st.sidebar.header("📐 1. パラメーター設定")

# 治具ポケット設定
st.sidebar.subheader("治具ポケット加工 (エンドミル)")
d_em = st.sidebar.number_input("エンドミル工具径 $D_{\\text{EM}}$ (mm)", value=6.0, min_value=0.1)
clearance = st.sidebar.number_input("クリアランス $C$ (mm)", value=0.1, min_value=0.0)

# 治具ポケット深さ
z_pocket_input = st.sidebar.number_input("治具ポケット深さ $Z_{\\text{pocket}}$ (mm) (負の値で入力)", value=-1.0, max_value=0.0)
z_pocket = z_pocket_input

# アクリルの厚み
acrylic_thickness = st.sidebar.number_input("嵌めるアクリルの厚み $T$ (mm)", value=3.0, min_value=0.1)

# アクリル上面 Z_top を計算
z_acrylic_top = z_pocket + acrylic_thickness

st.sidebar.markdown(rf"> **ポケット深さ $Z_{{\text{{pocket}}}}$**: $\bf{{ {z_pocket:.2f} }}$ mm")
st.sidebar.markdown(rf"> **アクリル上面 (面取り基準) $Z_{{\text{{top}}}}$**: $\bf{{ {z_acrylic_top:.2f} }}$ mm")


# Vビット面取り設定
st.sidebar.subheader("Vビット面取り加工")
w_chamfer = st.sidebar.number_input("面取り幅 $W$ (mm)", value=0.5, min_value=0.01)

# 面取り最終深さを計算し表示
z_chamfer_final = z_acrylic_top - w_chamfer

st.sidebar.markdown(rf"> **面取り開始点**: $\bf{{ {z_acrylic_top:.2f} }}$ mm")
st.sidebar.markdown(rf"> **面取り最終深さ $Z_{{\text{{final}}}}$**: $\bf{{ {z_chamfer_final:.2f} }}$ mm")


# 共通設定
st.sidebar.subheader("共通設定")
feed_rate = st.sidebar.number_input("送り速度 $F$ (mm/min)", value=1000, min_value=100)
add_dogbone = st.sidebar.checkbox("治具に角の逃げ (Dogbone) を追加", value=True)


# --- 形状データの定義 (ファイルアップローダーを復活) ---
st.subheader("🛠️ 2. 部品形状データ (DXF/SVG 読み込み)")

uploaded_file = st.file_uploader(
    "DXF または SVG ファイルをアップロードしてください", 
    type=['dxf', 'svg']
)

original_polygon = None
file_status = "ファイルがアップロードされていません。"

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'dxf':
        original_polygon, file_status = dxf_to_shapely_polygon(uploaded_file)
    elif file_extension == 'svg':
        st.warning("現在、SVGファイルの複雑なパスの解析はサポートされていません。DXFファイルの使用を推奨します。")
        file_status = "SVGファイルは非対応です。"
    else:
        file_status = "サポートされていないファイル形式です。"
        
    if original_polygon is None:
        st.error(f"ファイル解析エラー: {file_status}")
    else:
        st.success(f"ファイル解析成功: {file_status}")
        
else:
    # ファイルがない場合はデモ用の四角形を使用 (動作確認用)
    st.info("ファイルがアップロードされていないため、デモ用の100mm x 50mmの四角形を使用します。")
    coords = [(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)]
    original_polygon = Polygon(coords)

st.code(f"採用された形状: {'デモ用四角形' if original_polygon and len(original_polygon.exterior.coords) == 5 and uploaded_file is None else uploaded_file.name if uploaded_file else 'なし'}")

# --- メイン処理 (すべて復活) ---

if st.button("🚀 Gコードを生成 & パスを計算"):
    
    if original_polygon is None:
        st.error("図形データが見つからないため、Gコードを生成できません。有効なファイルをアップロードするか、デモ用図形を使用してください。")
        st.stop()
        
    col1, col2 = st.columns(2)

    # 1. 治具ポケット加工
    pocket_paths = generate_pocket_paths(
        original_polygon, 
        diameter=d_em, 
        clearance=clearance, 
        z_depth=z_pocket, 
        dogbone=add_dogbone
    )
    pocket_gcode = generate_gcode(pocket_paths, z_pocket, feed_rate, "Pocket_EM_T1")

    with col1:
        st.header("1️⃣ 治具ポケット加工パス")
        st.subheader(f"Gコード (工具径: {d_em}mm, 深さ: {z_pocket:.2f}mm)")
        st.code(pocket_gcode)
        
        st.download_button(
            label="Gコードをダウンロード (治具ポケット)",
            data=pocket_gcode,
            file_name="pocket_gcode.nc",
            mime="text/plain",
            key="download_pocket"
        )
        
        # パスの描画
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(*original_polygon.exterior.xy, color='gray', linestyle='--', label='Original Shape')
        
        if pocket_paths:
            for i, path in enumerate(pocket_paths):
                if path.geom_type == 'LineString' or path.geom_type == 'LinearRing':
                    color = 'blue' if i == 0 else 'lightblue'
                    label = 'Tool Path (Boundary)' if i == 0 else None
                    ax.plot(*path.xy, color=color, linewidth=1, label=label)

            ax.set_title("治具ポケット加工パス")
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            st.pyplot(fig)
        else:
            st.error("ポケットパスの計算に失敗しました。パラメーターを確認してください。")


    # 2. Vビット面取り加工
    chamfer_paths, z_final = generate_chamfer_paths(original_polygon, w_chamfer, z_acrylic_top)
    chamfer_gcode = generate_gcode(chamfer_paths, z_final, feed_rate, "Chamfer_VBit_T2")

    with col2:
        st.header("2️⃣ Vビット面取り加工パス")
        st.subheader(f"Gコード (面取り幅: {w_chamfer}mm, 深さ: {z_final:.2f}mm)")
        st.code(chamfer_gcode)

        st.download_button(
            label="Gコードをダウンロード (Vビット面取り)",
            data=chamfer_gcode,
            file_name="chamfer_gcode.nc",
            mime="text/plain",
            key="download_chamfer"
        )

        # パスの描画
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(*original_polygon.exterior.xy, color='gray', linestyle='--', label='Original Shape')

        if chamfer_paths:
            for path in chamfer_paths:
                if path.geom_type == 'LineString' or path.geom_type == 'LinearRing':
                    ax2.plot(*path.xy, color='red', linewidth=2, label='V-Bit Path (TOC)')

            ax2.set_title("Vビット面取り加工パス")
            ax2.legend()
            ax2.set_aspect('equal', adjustable='box')
            st.pyplot(fig2)
        else:
            st.error("面取りパスの計算に失敗しました。パラメーターを確認してください。")
