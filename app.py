import streamlit as st
import numpy as np
from shapely.geometry import Polygon, LineString, mapping, MultiPolygon
import matplotlib.pyplot as plt
import ezdxf 
from io import BytesIO

# Matplotlibの日本語フォント設定
import matplotlib.font_manager as fm
import os

try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'IPA P Gothic', 'Noto Sans CJK JP', 'DejaVu Sans'] 
    plt.rcParams['axes.unicode_minus'] = False 
except Exception:
    pass


# --- 幾何学計算とGコード生成のコアロジック ---

def generate_gcode(paths, z_depth, feed_rate, tool_name="T1"):
    """工具パスからGコードを生成する関数"""
# ... (変更なし)

# ... (generate_gcode 関数の残りの部分)

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
    
    # ★★★ 修正箇所: 74行目付近の括弧を修正 ★★★
    # 最後の点は最初の点と同じなので除く
    num_points = len(coords) - 1 
    # ★★★ 修正完了 ★★★
    
    for i in range(num_points):
        # 現在の点、前の点、次の点を取得
        current = coords[i]
# ... (add_dogbone_relief 関数の残りの部分)

# ... (generate_pocket_paths, generate_chamfer_paths, dxf_to_shapely_polygon 関数の残りの部分)

# --- Streamlit アプリケーション ---
# ... (Streamlit アプリケーションの残りの部分)
