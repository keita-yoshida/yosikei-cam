import streamlit as st
import numpy as np
from shapely.geometry import Polygon, LineString, mapping, MultiPolygon
import matplotlib.pyplot as plt
import ezdxf 
from io import BytesIO

# Matplotlibの日本語フォント設定は引き続きコメントアウト
# ... (すべての関数定義は変更なし)

# --- Streamlit アプリケーション ---

st.set_page_config(layout="wide")
st.title("簡易 Web CAM (Python/Streamlit)")
st.caption("治具ポケット加工とVビット面取りのパス生成プロトタイプ")

# --- サイドバーでのパラメーター設定 (変更なし) ---
# ... (すべてのサイドバー設定コード) ...

# --- 形状データの定義をファイルアップロードに変更 (最終復活) ---
st.subheader("🛠️ 2. 部品形状データ (DXF/SVG 読み込み)")

# ★★★ 修正箇所: ファイルアップローダーを復活させる ★★★
uploaded_file = st.file_uploader(
    "DXF または SVG ファイルをアップロードしてください", 
    type=['dxf', 'svg']
)
# ★★★ 修正完了 ★★★

original_polygon = None
file_status = "ファイルがアップロードされていません。"

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'dxf':
        # dxf_to_shapely_polygon 関数はここで利用
        original_polygon, file_status = dxf_to_shapely_polygon(uploaded_file)
    elif file_extension == 'svg':
        file_status = "現在、SVGファイルの複雑なパスの解析はサポートされていません。DXFファイルの使用を推奨します。"
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

st.code(f"採用された形状: {'デモ用四角形' if original_polygon and len(original_polygon.exterior.coords) == 5 else uploaded_file.name if uploaded_file else 'なし'}")

# --- メイン処理 (変更なし、完全に復活した状態) ---

if st.button("🚀 Gコードを生成 & パスを計算"):
    
    if original_polygon is None:
        st.error("図形データが見つからないため、Gコードを生成できません。有効なファイルをアップロードしてください。")
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
        
        # ダウンロード機能
        st.download_button(
            label="Gコードをダウンロード (治具ポケット)",
            data=pocket_gcode,
            file_name="pocket_gcode.nc",
            mime="text/plain",
            key="download_pocket"
        )
        
        # パスの描画 (日本語フォント設定はコメントアウトしたままなので、警告が出る可能性があります)
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

        # ダウンロード機能
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
