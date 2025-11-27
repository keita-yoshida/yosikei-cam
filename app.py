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

    stepover = diameter
