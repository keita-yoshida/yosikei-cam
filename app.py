import streamlit as st
import numpy as np
from shapely.geometry import Polygon, LineString, mapping, MultiPolygon
import matplotlib.pyplot as plt
import ezdxf 
from io import BytesIO

# Matplotlibの日本語フォント設定は引き続きコメントアウト
# ... (generate_gcode から dxf_to_shapely_polygon までのすべての関数定義の全文)

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


# --- 形状データの定義 (ファイルアップローダーはまだコメントアウト) ---
st.subheader("🛠️ 2. 部品形状データ (DXF/SVG 読み込み)")

uploaded_file = None 

# original_polygon はここで定義
original_polygon = None
file_status = "ファイルがアップロードされていません。"

# ファイルがない場合はデモ用の四角形を使用 (動作確認用)
st.info("ファイルアップローダーは非表示です。デモ用の四角形を使用します。")
coords = [(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)]
original_polygon = Polygon(coords) # ★★★ 変数 original_polygon を正しく定義 ★★★

st.code(f"採用された形状: デモ用四角形")


# --- メイン処理 (すべて復活させ、 NameError を修正)
