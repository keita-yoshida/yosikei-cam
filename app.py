import streamlit as st
import numpy as np
from shapely.geometry import Polygon, LineString, mapping, MultiPolygon
import matplotlib.pyplot as plt
import ezdxf 
from io import BytesIO

# ★★★ 修正箇所: Matplotlibの日本語フォント設定を追加 ★★★
import matplotlib.font_manager as fm
import os

try:
    # IPAexGothicを優先的に使用する設定
    plt.rcParams['font.family'] = 'sans-serif'
    # Streamlit Cloud環境で利用可能なフォントにフォールバック設定
    plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'IPA P Gothic', 'Noto Sans CJK JP', 'DejaVu Sans'] 
    plt.rcParams['axes.unicode_minus'] = False # 負の記号 '-' を表示可能にする
except Exception:
    # フォント設定が失敗した場合のフォールバック
    pass
# ★★★ 修正完了 ★★★


# --- 幾何学計算とGコード生成のコアロジック (変更なし) ---

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
    num_points = len(
