import streamlit as st
import numpy as np
from shapely.geometry import Polygon, LineString, mapping
import matplotlib.pyplot as plt

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
             # 初期移動
            gcode.append(f"G00 X{coords[0, 0]:.4f} Y{coords[0, 1]:.4f}")
            # 切り込み
            gcode.append(f"G01 Z{z_depth:.4f}")
        else:
             # 前回の終了点から次のパスの開始点へ移動 (Zはそのまま)
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
    """
    治具ポケットの内角にドッグボーン（直線延長）の逃げを追加する関数
    ShapelyのPolygonの座標を直接変更する (単純な四角形のみ対応)
    """
    tool_r = diameter / 2.0
    # 逃げの深さ (工具半径より少し大きくする)
    relief_offset = tool_r * 0.4 
    
    # 座標を取得 (閉じたポリゴンなので最後の点は最初の点と同じ)
    coords = list(polygon.exterior.coords)
    
    new_coords = []
    num_points = len(coords) - 1 # 最後の点は最初の点と同じなので除く
    
    for i in range(num_points):
        # 現在の点、前の点、次の点を取得
        current = coords[i]
        prev = coords[(i - 1 + num_points) % num_points]
        next_point = coords[(i + 1) % num_points]
        
        new_coords.append(current)

        # ベクトルを計算
        v_in = np.array(prev) - np.array(current)
        v_out = np.array(next_point) - np.array(current)
        
        # ノルム（長さ）がゼロでないことを確認
        if np.linalg.norm(v_in) > 1e-6 and np.linalg.norm(v_out) > 1e-6:
            # 正規化
            v_in_n = v_in / np.linalg.norm(v_in)
            v_out_n = v_out / np.linalg.norm(v_out)
            
            # 逃げ処理の追加
            # 1. 逃げの点へ移動 (前方向へ)
            relief_pt1 = np.array(current) - v_in_n * relief_offset
            # 2. 逃げの点へ移動 (次方向へ)
            relief_pt2 = np.array(current) - v_out_n * relief_offset
            
            # 角を突き抜けるようにパスを挿入
            new_coords.append(tuple(relief_pt1))
            new_coords.append(tuple(relief_pt2))
            
    # 最後に閉じる
    new_coords.append(new_coords[0])
    return LineString(new_coords)


def generate_pocket_paths(polygon, diameter, clearance, z_depth, dogbone=True):
    """治具ポケット加工の工具中心パスを生成する関数"""
    tool_r = diameter / 2.0
    
    # 1. 境界線のオフセット (クリアランス分外側へ)
    boundary_offset = tool_r + clearance
    try:
        pocket_boundary = polygon.buffer(boundary_offset, join_style=2)
    except Exception:
        return [] # 失敗したら空リストを返す

    # 2. 穴埋めパスの生成 (ステップオーバーは工具径の70%とする)
    stepover = diameter * 0.7 
    current_poly = pocket_boundary
    tool_paths = []
    
    # ポケットパスの生成
    while current_poly.area > 0.001:
        # 現在のポリゴンの外周をパスとする
        if current_poly.exterior:
            tool_paths.append(current_poly.exterior)
            
        # 次のパス（内側のパス）を計算
        try:
            current_poly = current_poly.buffer(-stepover, join_style=2)
        except Exception:
            break # 小さくなりすぎたら終了
            
        # Multipolygonになった場合は、最大のものを次の対象とする
        if current_poly.geom_type == 'MultiPolygon':
            current_poly = max(current_poly.geoms, key=lambda g: g.area)

        if not current_poly:
            break
            
    # 3. 角の逃げ処理 (ドッグボーン型)
    if dogbone and tool_paths:
        # 最外周のパス (治具の境界線) に逃げを適用
        line_path = tool_paths[0]
        # LineStringの座標リストをPolygonに変換してから逃げ処理を適用
        tool_paths[0] = add_dogbone_relief(Polygon(line_path), diameter)

    return tool_paths

def generate_chamfer_paths(polygon, chamfer_width):
    """Vビット面取り加工の工具中心パスを生成する関数"""
    # Vビット 90度の場合、面取り幅 W = Z深さ D
    z_depth = -chamfer_width
    
    # 1. $X, Y$ 経路の決定 (外側に面取り幅 W だけオフセット)
    try:
        chamfer_path = polygon.exterior.buffer(chamfer_width, join_style=2)
    except Exception:
        return [], z_depth
        
    # 面取りパスは1本の線 (LineString) なので、その座標を返す
    if chamfer_path.geom_type == 'Polygon':
        return [chamfer_path.exterior], z_depth
    
    return [chamfer_path], z_depth


# --- Streamlit アプリケーション ---

st.set_page_config(layout="wide")
st.title("簡易 Web CAM (Python/Streamlit)")
st.caption("治具ポケット加工とVビット面取りのパス生成プロトタイプ")

# サイドバーでのパラメーター設定
st.sidebar.header("📐 1. パラメーター設定")

# 治具ポケット設定
st.sidebar.subheader("治具ポケット加工 (エンドミル)")
d_em = st.sidebar.number_input("エンドミル工具径 $D_{\\text{EM}}$ (mm)", value=6.0, min_value=0.1)
clearance = st.sidebar.number_input("クリアランス $C$ (mm)", value=0.1, min_value=0.0)
z_pocket = st.sidebar.number_input("ポケット深さ $Z_{\\text{depth}}$ (mm)", value=-5.0, max_value=0.0)

# Vビット面取り設定
st.sidebar.subheader("Vビット面取り加工")
w_chamfer = st.sidebar.number_input("面取り幅 $W$ (mm)", value=0.5, min_value=0.01)

# ★★★ 修正箇所 ★★★
# LaTeXの$Z_{\text{chamfer}}$の表記を、シンプルなMarkdownとf-stringの組み合わせに修正
st.sidebar.markdown(f"> **深さ $Z$**: **-{w_chamfer}** mm (90度Vビットのため)") 
# ★★★ 修正完了 ★★★

# 共通設定
st.sidebar.subheader("共通設定")
feed_rate = st.sidebar.number_input("送り速度 $F$ (mm/min)", value=1000, min_value=100)
add_dogbone = st.sidebar.checkbox("治具に角の逃げ (Dogbone) を追加", value=True)

# 形状データの定義 (DXF/SVGの代わりに手動で定義)
st.subheader("🛠️ 2. 部品形状データ (デモ用)")
st.info("通常はここでDXF/SVGファイルを読み込みます。今回は仮の四角形を使用します。")
# 外周 100x50 の四角形
coords = [(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)]
original_polygon = Polygon(coords)
st.code(f"元の形状 (四角形): 100mm x 50mm")

# --- メイン処理 ---

if st.button("🚀 Gコードを生成 & パスを計算"):
    
    col1, col2 = st.columns(2)

    # 1. 治具ポケット加工
    pocket_paths = generate_pocket_paths(
        original_polygon, 
        diameter=d_em, 
        clearance=clearance, 
        z_depth=z_pocket, 
        dogbone=add_dogbone
    )
    pocket_gcode = generate_gcode(pocket_paths, z_pocket, feed_rate, "Pocket Endmill")

    with col1:
        st.header("1️⃣ 治具ポケット加工パス")
        st.subheader(f"Gコード (工具径: {d_em}mm, 深さ: {z_pocket}mm)")
        st.code(pocket_gcode)
        
        # パスの描画
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(*original_polygon.exterior.xy, color='gray', linestyle='--', label='Original Shape')
        
        if pocket_paths:
            for i, path in enumerate(pocket_paths):
                # Shapely LineStringの座標を取得してプロット
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
    chamfer_paths, z_chamfer = generate_chamfer_paths(original_polygon, w_chamfer)
    chamfer_gcode = generate_gcode(chamfer_paths, z_chamfer, feed_rate, "Chamfer V-Bit")

    with col2:
        st.header("2️⃣ Vビット面取り加工パス")
        st.subheader(f"Gコード (面取り幅: {w_chamfer}mm, 深さ: {z_chamfer}mm)")
        st.code(chamfer_gcode)

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
