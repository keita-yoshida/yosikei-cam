# ★★★ 修正箇所: 必要なインポートをすべて追加 ★★★
import streamlit as st
import numpy as np
from shapely.geometry import Polygon, LineString, mapping, MultiPolygon
import matplotlib.pyplot as plt
import ezdxf 
from io import BytesIO
# ★★★ 修正完了 ★★★

# Matplotlibの日本語フォント設定は引き続きコメントアウト

# --- 幾何学計算とGコード生成のコアロジック (すべて復活) ---

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
    # 最後の点は最初の点と同じなので除く
    num_points = len(coords) - 1 
    
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
            relief_pt1 = np.array(current) - v_in_n * relief_offset
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
        return [] 

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
            break 
            
        # Multipolygonになった場合は、最大のものを次の対象とする
        if current_poly.geom_type == 'MultiPolygon':
            current_poly = max(current_poly.geoms, key=lambda g: g.area)

        if not current_poly:
            break
            
    # 3. 角の逃げ処理 (ドッグボーン型)
    if dogbone and tool_paths:
        line_path = tool_paths[0]
        tool_paths[0] = add_dogbone_relief(Polygon(line_path), diameter)

    return tool_paths

def generate_chamfer_paths(polygon, chamfer_width, z_start):
    """Vビット面取り加工の工具中心パスを生成する関数"""
    z_final = z_start - chamfer_width
    
    # 1. $X, Y$ 経路の決定 (外側に面取り幅 W だけオフセット)
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
            if entity.dxftype() == 'LWPOLYLINE' or entity.dxftype
