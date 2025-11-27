import streamlit as st
import numpy as np
from shapely.geometry import Polygon, LineString, mapping, MultiPolygon
import matplotlib.pyplot as plt
import ezdxf 
from io import BytesIO

# ★★★ 修正箇所: Matplotlibの日本語フォント設定をコメントアウト ★★★
# import matplotlib.font_manager as fm
# import os

# try:
#     # IPAexGothicを優先的に使用する設定
#     plt.rcParams['font.family'] = 'sans-serif'
#     # Streamlit Cloud環境で利用可能なフォントにフォールバック設定
#     plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'IPA P Gothic', 'Noto Sans CJK JP', 'DejaVu Sans'] 
#     plt.rcParams['axes.unicode_minus'] = False # 負の記号 '-' を表示可能にする
# except Exception:
#     # フォント設定が失敗した場合のフォールバック
#     pass
# ★★★ 修正完了 ★★★


# --- 幾何学計算とGコード生成のコアロジック (変更なし) ---
# ... (以前に提示したコードの残りの部分)
