import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile
import os
import math
import re
from collections import Counter

# 幾何学計算ライブラリ
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, GeometryCollection
from shapely.affinity import translate
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid

import ezdxf
import ezdxf.path

# Vカービング用
from scipy.spatial import Voronoi, cKDTree
# 3Dプレビュー用
from scipy.ndimage import grey_erosion

# --- 0. 多言語定義 (辞書) ---
LANG_DICT = {
    "Japanese": {
        "title": "yosikeiCAM 4.6",
        "origin_setting": "原点設定",
        "origin_option": ["左下 (Bottom-Left)", "中心 (Center)", "CAD座標 (Original)"],
        "process_setting": "加工設定",
        "tabs": ["ポケット", "面取り", "Vカーブ", "ドリル"],
        "enable": "有効",
        "tool_dia": "工具径 (mm)",
        "double_pass": "2回加工 (粗+仕上げ)",
        "finish_allowance": "仕上げ代 (mm)",
        "rough_feed": "送り速度 (粗)",
        "finish_feed": "仕上げ送り",
        "finish_depth_mode": "仕上げ深さ",
        "depth_options": ["ピッチ刻み", "最終一括"],
        "final_depth": "最終深さ Z",
        "pitch": "切込ピッチ",
        "chamfer_width": "幅 (mm)",
        "tip_offset": "刃先オフセット",
        "v_angle": "角度 (度)",
        "depth_limit": "深さ制限",
        "max_depth": "最大深さ (mm)",
        "precision": "精度",
        "drill_dia": "対象径 (mm)",
        "drill_depth": "深さ Z",
        "peck": "ペック量",
        "feed": "送り速度",
        "post_processor": "スタートコード/エンコード",
        "dxf_upload": "DXFアップロード",
        "path_select": "パス選択",
        "select_all": "すべて選択",
        "size_display": "加工サイズ",
        "gen_header": "パス生成",
        "download": ["VCARVE 保存", "POCKET 保存", "CHAMFER 保存", "DRILL 保存"],
        "error_no_geom": "図形が見つかりません。",
        "hole_detected": "検出された円",
        "out_tol": "出力公差 (円弧・直線近似, mm)",
        "file_size": "{name}: {kb:.1f} KB / {lines} 行",
        "preview_header": "3D 加工プレビュー",
        "preview_enable": "3Dプレビューを表示",
        "stock_thickness": "材料厚さ (mm)",
        "sim_resolution": "シミュレーション解像度 (mm)",
        "preview_running": "加工シミュレーション中...",
        "preview_none": "プレビューするパスがありません。",
        "rapid_warning": "早送り(G0)で材料内をXY移動している箇所が {n} 件あります。",
        "feed_warning": "送り速度 F{f} が機械の最大切削送り {m} mm/min を超えています。",
        "preview_note": "Gコードを読み直して工具形状で削った結果です (Vカーブ/面取り=Vビット, ポケット/ドリル=フラット)。"
    },
    "English": {
        "title": "yosikeiCAM 4.6",
        "origin_setting": "Origin Setup",
        "origin_option": ["Bottom-Left", "Center", "Original (CAD)"],
        "process_setting": "Machining Setup",
        "tabs": ["Pocket", "Chamfer", "V-Carve", "Drill"],
        "enable": "Enable",
        "tool_dia": "Tool Dia (mm)",
        "double_pass": "Double Pass (Rough+Finish)",
        "finish_allowance": "Finish Allowance (mm)",
        "rough_feed": "Rough Feed Rate",
        "finish_feed": "Finish Feed Rate",
        "finish_depth_mode": "Finish Depth Mode",
        "depth_options": ["Step-down", "Full-Depth"],
        "final_depth": "Final Depth Z",
        "pitch": "Z Step Pitch",
        "chamfer_width": "Width (mm)",
        "tip_offset": "Tip Offset",
        "v_angle": "V-Angle (deg)",
        "depth_limit": "Depth Limit",
        "max_depth": "Max Depth (mm)",
        "precision": "Precision",
        "drill_dia": "Hole Dia (mm)",
        "drill_depth": "Depth Z",
        "peck": "Peck Amount",
        "feed": "Feed Rate",
        "post_processor": "Start Code / End Code",
        "dxf_upload": "DXF Upload",
        "path_select": "Path Selection",
        "select_all": "Select All",
        "size_display": "Machining Size",
        "gen_header": "Toolpath Generation",
        "download": ["Save VCARVE", "Save POCKET", "Save CHAMFER", "Save DRILL"],
        "error_no_geom": "No valid geometry found.",
        "hole_detected": "Circles detected",
        "out_tol": "Output Tolerance (arc/line fitting, mm)",
        "file_size": "{name}: {kb:.1f} KB / {lines} lines",
        "preview_header": "3D Machining Preview",
        "preview_enable": "Show 3D preview",
        "stock_thickness": "Stock Thickness (mm)",
        "sim_resolution": "Simulation Resolution (mm)",
        "preview_running": "Simulating material removal...",
        "preview_none": "No toolpaths to preview.",
        "rapid_warning": "{n} rapid (G0) XY move(s) run inside the stock.",
        "feed_warning": "Feed F{f} exceeds the machine's max cutting feed of {m} mm/min.",
        "preview_note": "Simulated from the generated G-code with the tool shape (V-Carve/Chamfer = V-bit, Pocket/Drill = flat)."
    }
}

# --- 1. ポストプロセッサ定義 ---
# format  : G0/G1 か G00/G01 か
# comment : コメント形式 (";" または Fanuc 系の "()")
# tool_change : T1 M06 を出力するか (ATC なしの機械では False)
# canned_drill: ドリルを G81/G83 固定サイクルで出力するか
# max_feed: 機械の最大切削送り (mm/min)。超える設定には警告を出す (省略時はチェックしない)
POST_PROCESSORS = {
    "Generic (汎用)": {
        "start": "G21 ; Metric\nG90 ; Absolute\nG00 Z10.0 ; Safe Z\nM3 S10000 ; Spindle On",
        "end": "M5 ; Spindle Off\nG00 Z10.0\nM30 ; End",
        "format": "G00/G01", "comment": ";", "tool_change": True, "canned_drill": False
    },
    "GRBL / Candle": {
        "start": "G21 G90 G17\nG0 Z10.0\nM3 S10000",
        "end": "M5 G0 Z10.0\nM30",
        "format": "G0/G1", "comment": ";", "tool_change": True, "canned_drill": False
    },
    "Mach3 / Mach4": {
        "start": "G21 G90 G40 G80\nG00 Z10.0\nM03 S10000",
        "end": "M05\nG00 Z10.0\nM30",
        "format": "G00/G01", "comment": ";", "tool_change": True, "canned_drill": False
    },
    # 牧野 AVNC-74 (FANUC 0M) 取扱説明書に合わせた設定:
    # 主軸回転数はペンダントの変速ダイヤルと高低速レバーで手動設定するため S コードは出さない。
    # 使える M コードは M00-05, M07-09, M21-23, M30, M48, M49, M98, M99。切削送りは最大 2400mm/min。
    "Fanuc 0M (ATCなし)": {
        "start": "%\nO0001 (YOSIKEICAM)\nG17 G21 G40 G49 G80 G90\nG54\nG00 Z10.\n(SET SPINDLE SPEED WITH DIAL)\nM03",
        "end": "M05\nG91 G28 Z0.\nG90\nM30\n%",
        "format": "G00/G01", "comment": "()", "tool_change": False, "canned_drill": True,
        "max_feed": 2400
    }
}

# --- 2. 幾何学ユーティリティ ---

def ensure_list_of_polys(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        return list(geometry.geoms)
    elif geometry.geom_type == 'GeometryCollection':
        polys = []
        for g in geometry.geoms:
            if g.geom_type == 'Polygon':
                polys.append(g)
            elif g.geom_type == 'MultiPolygon':
                polys.extend(g.geoms)
        return polys
    return []

def dist_lseg(l1, l2, p):
    x0, y0 = l1[0], l1[1]
    xa, ya = l2[0], l2[1]
    xi, yi = p[0], p[1]
    dx, dy = xa-x0, ya-y0
    d2 = dx*dx + dy*dy
    if d2 == 0:
        return 0
    t = max(0, min(1, ((xi-x0)*dx + (yi-y0)*dy) / d2))
    return math.sqrt((xi - (x0 + t*dx))**2 + (yi - (y0 + t*dy))**2)

def douglas_peucker(points, tolerance):
    if len(points) < 3:
        return points
    dmax, index, end = 0, 0, len(points) - 1
    for i in range(1, end):
        d = dist_lseg(points[0], points[end], points[i])
        if d > dmax:
            index, dmax = i, d
    if dmax > tolerance:
        left = douglas_peucker(points[:index+1], tolerance)
        right = douglas_peucker(points[index:], tolerance)
        return left[:-1] + right
    else:
        return [points[0], points[end]]

def _to_polygons(pts):
    """点列から有効なポリゴンのリストを作る (自己交差は修復する)。"""
    if len(pts) < 3:
        return []
    poly = Polygon(pts)
    if poly.is_valid:
        return [poly] if poly.area > 0.0001 else []
    return [p for p in ensure_list_of_polys(make_valid(poly)) if p.area > 0.0001]

def join_open_paths(paths, tol=0.01):
    """端点が tol 以内で接する開いた線 (ARC/LINE など) をつなぎ、閉じた輪郭の点列を返す。
    paths: [(点列, 単独でも閉じてよいか)]。閉じなかった線の本数も返す。"""
    if not paths:
        return [], 0
    ends = np.array([p[0][i] for p in paths for i in (0, -1)], dtype=float)
    # 近い端点を同じノードにまとめる
    node = -np.ones(len(ends), dtype=int)
    tree = cKDTree(ends)
    for i in range(len(ends)):
        if node[i] < 0:
            node[tree.query_ball_point(ends[i], tol)] = i
    adj = {}
    for k in range(len(paths)):
        for n in (node[2 * k], node[2 * k + 1]):
            adj.setdefault(n, []).append(k)
    used, loops, dropped = set(), [], 0
    for start in range(len(paths)):
        if start in used:
            continue
        used.add(start)
        chain = list(paths[start][0])
        head, tail = node[2 * start], node[2 * start + 1]
        members = [start]
        # 末尾方向に、分岐のないノードを伝ってつなげていく
        while tail != head and len(adj[tail]) == 2:
            nxt = next((k for k in adj[tail] if k not in used), None)
            if nxt is None:
                break
            used.add(nxt)
            members.append(nxt)
            pts = paths[nxt][0]
            if node[2 * nxt] == tail:
                chain.extend(pts[1:]); tail = node[2 * nxt + 1]
            else:
                chain.extend(pts[-2::-1]); tail = node[2 * nxt]
        if tail == head and len(chain) > 3:
            loops.append(chain)
        elif len(members) == 1 and paths[start][1]:
            loops.append(chain)  # 単独の開いたポリラインは従来どおり自動で閉じる
        else:
            dropped += len(members)
    return loops, dropped

def dxf_to_shapely_list(dxf_bytes):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(dxf_bytes)
        tmp_path = tmp.name
    try:
        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()
        polys, open_paths = [], []
        for e in msp:
            kind = e.dxftype()
            if kind not in ('LWPOLYLINE', 'POLYLINE', 'SPLINE', 'CIRCLE', 'ARC', 'LINE', 'ELLIPSE'):
                continue
            try:
                if kind == 'CIRCLE':
                    # 法線が -Z の円 (Rhino 等) は OCS→WCS 変換しないと X が反転する
                    c = e.ocs().to_wcs(e.dxf.center)
                    polys.append(Point(c.x, c.y).buffer(e.dxf.radius, resolution=64))
                    continue
                pts = [(v.x, v.y) for v in ezdxf.path.make_path(e).flattening(0.01)]
                if len(pts) < 2:
                    continue
                if len(pts) > 3 and math.dist(pts[0], pts[-1]) < 1e-6:
                    polys.extend(_to_polygons(pts))
                else:
                    open_paths.append((pts, kind in ('LWPOLYLINE', 'POLYLINE', 'SPLINE')))
            except Exception:
                pass
        loops, dropped = join_open_paths(open_paths)
        for pts in loops:
            polys.extend(_to_polygons(pts))
        if dropped:
            st.warning(f"閉じていない線が {dropped} 本あり、無視しました。 / {dropped} open segment(s) could not be joined and were ignored.")
        polys.sort(key=lambda x: x.area, reverse=True)
        return polys
    except Exception as e:
        st.error(f"DXF Read Error: {e}")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def merge_polygons_xor(polys):
    if not polys:
        return None
    combined = Polygon()
    for p in polys:
        if combined.is_empty:
            combined = p
        else:
            combined = combined.symmetric_difference(p)
    return combined

def find_drill_points(geometry, target_dia, tolerance=0.2):
    polys = ensure_list_of_polys(geometry)
    drill_points = []
    def check_p(p):
        minx, miny, maxx, maxy = p.bounds
        w, h = maxx - minx, maxy - miny
        if abs(w - h) > tolerance:
            return None
        if not (target_dia - tolerance <= w <= target_dia + tolerance):
            return None
        return p.centroid
    for p in polys:
        pt = check_p(p)
        if pt:
            drill_points.append(pt)
        for interior in p.interiors:
            pt_h = check_p(Polygon(interior))
            if pt_h:
                drill_points.append(pt_h)
    return drill_points

def fmt_num(v):
    """座標値を小数点以下3桁までに丸め、末尾の0を省いて出力する (例: 5.0 -> "5.", -1.25 -> "-1.25")。
    Fanuc では小数点のない値が最小設定単位 (0.001mm) 扱いになるため、小数点は必ず残す。"""
    s = f"{v:.3f}".rstrip('0')
    return "0." if s in ("-0.", "0.") else s

class GcodeWriter:
    """モーダル (G コード・座標・送り) が変わらない語を省略してGコードを組み立てる。"""
    def __init__(self, post):
        self.post = post
        short = "G0/" in post["format"]
        self.codes = {0: "G0", 1: "G1", 2: "G2", 3: "G3"} if short else \
                     {0: "G00", 1: "G01", 2: "G02", 3: "G03"}
        self.lines = []
        self.reset_modal()

    def reset_modal(self):
        self.motion, self.pos, self.feed = None, {}, None

    def raw(self, text):
        for line in text.strip().splitlines():
            self.lines.append(line.rstrip())
        self.reset_modal()

    def comment(self, text):
        if self.post.get("comment") == "()":
            # 古い Fanuc は小文字・全角・括弧を受け付けないことがあるので英大文字のみにする
            text = re.sub(r'[^A-Z0-9 .,:+\-/=*#]', ' ', text.upper()).strip()
            self.lines.append(f"({text})")
        else:
            self.lines.append(f"; {text}")

    def move(self, motion, x=None, y=None, z=None, feed=None, i=None, j=None):
        words = []
        for axis, v in (("X", x), ("Y", y), ("Z", z)):
            if v is None:
                continue
            s = fmt_num(v)
            if self.pos.get(axis) != s:
                words.append(axis + s)
                self.pos[axis] = s
        if motion in (2, 3):
            words += [f"I{fmt_num(i)}", f"J{fmt_num(j)}"]
        elif not words:
            return
        if motion != 0 and feed is not None and feed != self.feed:
            words.append(f"F{int(feed)}")
            self.feed = feed
        if motion != self.motion:
            words.insert(0, self.codes[motion])
            self.motion = motion
        self.lines.append(" ".join(words))

    def current_xy(self):
        return float(self.pos["X"]), float(self.pos["Y"])

    def text(self):
        return "\n".join(self.lines) + "\n"

def _circle_through(a, b, c):
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = a @ a, b @ b, c @ c
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    return np.array([ux, uy])

def _arc_fit(pts, i, j, tol):
    """pts[i..j] が1本の円弧で公差内に収まれば (中心, CCWか) を返す。"""
    seg = pts[i:j + 1]
    center = _circle_through(seg[0], seg[len(seg) // 2], seg[-1])
    if center is None:
        return None
    r = np.hypot(*(seg[0] - center))
    if r > 2000.0 or r < 0.05:
        return None
    if np.max(np.abs(np.hypot(*(seg - center).T) - r)) > tol:
        return None
    # 各弦のふくらみ (サジッタ) も公差内か
    chord = np.hypot(*np.diff(seg, axis=0).T)
    if np.max(chord) ** 2 / (8.0 * r) > tol:
        return None
    ang = np.unwrap(np.arctan2(seg[:, 1] - center[1], seg[:, 0] - center[0]))
    d_ang = np.diff(ang)
    if not (np.all(d_ang > 0) or np.all(d_ang < 0)):
        return None
    if abs(ang[-1] - ang[0]) > np.radians(350):
        return None
    return center, bool(d_ang[0] > 0)

def fit_path(coords, tol):
    """2D 点列を直線と円弧の列 [('L', 終点), ('A', 終点, 中心, ccw)] に変換する。"""
    pts = np.asarray(coords, dtype=float)[:, :2]
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-6]
    pts = pts[keep]
    ops, line_run, i, n = [], [pts[0]], 0, len(pts)
    def flush_lines():
        if len(line_run) > 1:
            simple = LineString(line_run).simplify(tol, preserve_topology=False)
            for p in list(simple.coords)[1:]:
                ops.append(('L', np.array(p)))
    while i < n - 1:
        best, j = None, i + 3
        # 伸ばせるだけ伸ばし (倍々), 失敗したら二分探索で最長の円弧を探す
        if j < n and _arc_fit(pts, i, j, tol):
            lo, step = j, 4
            while True:
                hi = min(n - 1, lo + step)
                if hi == lo or not _arc_fit(pts, i, hi, tol):
                    break
                lo, step = hi, step * 2
            if hi != lo:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if _arc_fit(pts, i, mid, tol):
                        lo = mid
                    else:
                        hi = mid
            best = lo
        if best is None:
            line_run.append(pts[i + 1])
            i += 1
            continue
        center, ccw = _arc_fit(pts, i, best, tol)
        flush_lines()
        ops.append(('A', pts[best], center, ccw))
        line_run = [pts[best]]
        i = best
    flush_lines()
    return pts[0], ops

def write_path_2d(w, start, ops, feed):
    for op in ops:
        if op[0] == 'L':
            w.move(1, x=op[1][0], y=op[1][1], feed=feed)
        else:
            _, end, center, ccw = op
            sx, sy = w.current_xy()
            w.move(3 if ccw else 2, x=end[0], y=end[1], feed=feed,
                   i=center[0] - sx, j=center[1] - sy)

def generate_drill_gcode(points, z_start, z_final, peck_depth, feed, tool_name, header, footer, post):
    if not points:
        return None
    w = GcodeWriter(post)
    w.raw(header)
    w.comment(f"TOOL: {tool_name}")
    if post.get("tool_change"):
        w.raw("T1 M06")
    safe, r_plane = 5.0, z_start + 1.0
    w.move(0, z=safe)
    if post.get("canned_drill"):
        # G98: 穴と穴の間はイニシャル点 (安全高さ) まで戻る。
        # Q は小数点を受け付けない古い機種でも同じ意味になるよう 0.001mm 単位の整数で出す。
        peck = peck_depth < abs(z_final - z_start)
        cycle_words = f"Z{fmt_num(z_final)} R{fmt_num(r_plane)}" + \
            (f" Q{int(round(peck_depth * 1000))}" if peck else "")
        prev = (None, None)
        for k, pt in enumerate(points):
            xy = (fmt_num(pt.x), fmt_num(pt.y))
            words = [a + v for a, v, p in zip("XY", xy, prev) if v != p]
            if k == 0:
                w.lines.append(f"G98 {'G83' if peck else 'G81'} {' '.join(words)} {cycle_words} F{int(feed)}")
            elif words:
                w.lines.append(" ".join(words))
            prev = xy
        w.lines.append("G80")
        w.reset_modal()
        w.move(0, z=safe)
    else:
        for pt in points:
            w.comment(f"Drill X{fmt_num(pt.x)} Y{fmt_num(pt.y)}")
            w.move(0, x=pt.x, y=pt.y)
            w.move(0, z=r_plane)
            current_z = z_start
            while current_z > z_final:
                target_z = max(current_z - peck_depth, z_final)
                w.move(1, z=target_z, feed=feed)
                if target_z > z_final:
                    w.move(0, z=z_start + 0.5)
                    w.move(0, z=target_z + 0.5)
                current_z = target_z
            w.move(0, z=safe)
    w.raw(footer)
    return w.text()

def generate_pocket(geometry, tool_d, clearance, stepover):
    paths_rough, paths_finish = [], []
    r, step = tool_d / 2.0, tool_d * stepover
    offset_rough = -(r + clearance)
    try:
        current_rough = geometry.buffer(offset_rough, join_style=2)
    except:
        current_rough = Polygon()
    while not current_rough.is_empty and current_rough.area > 0.01:
        for p in ensure_list_of_polys(current_rough):
            paths_rough.append(p.exterior)
            paths_rough.extend(p.interiors)
        try:
            current_rough = current_rough.buffer(-step, join_style=2)
        except:
            break
    if clearance > 0:
        try:
            for p in ensure_list_of_polys(geometry.buffer(-r, join_style=2)):
                paths_finish.append(p.exterior)
                paths_finish.extend(p.interiors)
        except:
            pass
    return ([LineString(p.coords) for p in paths_rough if p.length > 0.1], 
            [LineString(p.coords) for p in paths_finish if p.length > 0.1])

def generate_chamfer_separated(geometry, tip_offset, finish_allowance=0.0):
    rough_paths, finish_paths = [], []
    if finish_allowance > 0:
        try:
            geom_r = geometry.buffer(tip_offset + finish_allowance, join_style=1)
            for poly in ensure_list_of_polys(geom_r):
                rough_paths.append(poly.exterior)
                rough_paths.extend(poly.interiors)
        except:
            pass
    try:
        geom_f = geometry.buffer(tip_offset, join_style=1)
        for poly in ensure_list_of_polys(geom_f):
            finish_paths.append(poly.exterior)
            finish_paths.extend(poly.interiors)
    except:
        pass
    return ([LineString(ls.coords) for ls in rough_paths], 
            [LineString(ls.coords) for ls in finish_paths])

class PathGraph:
    def __init__(self):
        self.adj = {}
    def add_edge(self, p1, p2):
        p1 = (round(p1[0], 3), round(p1[1], 3))
        p2 = (round(p2[0], 3), round(p2[1], 3))
        if p1 == p2: return
        if p1 not in self.adj: self.adj[p1] = set()
        if p2 not in self.adj: self.adj[p2] = set()
        self.adj[p1].add(p2)
        self.adj[p2].add(p1)
    def prune(self, min_len=4.0):
        for _ in range(10):
            leaves = [n for n, neigh in self.adj.items() if len(neigh) == 1]
            if not leaves: break
            for leaf in leaves:
                if leaf not in self.adj: continue
                neighbor = list(self.adj[leaf])[0]
                dist = math.sqrt((leaf[0]-neighbor[0])**2 + (leaf[1]-neighbor[1])**2)
                if dist < min_len and len(self.adj[neighbor]) > 2:
                    self.adj[neighbor].remove(leaf)
                    del self.adj[leaf]
    def get_chains(self):
        chains, visited = [], set()
        nodes = sorted(self.adj.keys(), key=lambda n: (len(self.adj[n])%2!=1, -len(self.adj[n])))
        for start in nodes:
            if start not in self.adj: continue
            for neighbor in list(self.adj[start]):
                edge = tuple(sorted((start, neighbor)))
                if edge in visited: continue
                chain, curr = [start, neighbor], neighbor
                visited.add(edge)
                while True:
                    cands = [n for n in self.adj[curr] if tuple(sorted((curr, n))) not in visited]
                    if len(cands) == 1:
                        nxt = cands[0]
                        visited.add(tuple(sorted((curr, nxt))))
                        chain.append(nxt)
                        curr = nxt
                        if len(self.adj[curr]) > 2: break
                    else: break
                if len(chain) > 1:
                    chains.append(chain)
        return chains

def generate_vcarve(geometry, angle_deg, use_limit, max_d, step_len=0.1, z_offset=0.0):
    polys = ensure_list_of_polys(geometry)
    if not polys:
        return []
    tan_a, graph = np.tan(np.radians(angle_deg/2)), PathGraph()
    combined_geom = unary_union(polys)
    boundary = combined_geom.boundary
    for poly in polys:
        smooth = poly.simplify(0.02, preserve_topology=True)
        line = smooth.exterior
        sample_res = 0.2
        num = max(50, min(8000, int(line.length / sample_res)))
        pts = [line.interpolate(i * line.length / (num-1)) for i in range(num)]
        coords = np.array([(p.x, p.y) for p in pts])
        try:
            vor_obj = Voronoi(coords)
            for (p1_idx, p2_idx), (v1_idx, v2_idx) in zip(vor_obj.ridge_points, vor_obj.ridge_vertices):
                if v1_idx < 0 or v2_idx < 0: continue
                v1, v2 = vor_obj.vertices[v1_idx], vor_obj.vertices[v2_idx]
                if combined_geom.contains(Point(v1)) and combined_geom.contains(Point(v2)):
                    if np.linalg.norm(vor_obj.points[p1_idx] - vor_obj.points[p2_idx]) > sample_res * 5.0:
                        graph.add_edge(v1, v2)
        except: continue
    graph.prune(min_len=4.0)
    chains, all_paths = graph.get_chains(), []
    for chain in chains:
        path_3d, ls = [], LineString(chain)
        pts_count = max(2, int(ls.length / step_len) + 1)
        for i in range(pts_count):
            pt = ls.interpolate(i * ls.length / (pts_count - 1))
            z = -(pt.distance(boundary) / tan_a) + z_offset
            if z > 0: z = 0
            if use_limit and z < max_d: z = max_d
            path_3d.append((pt.x, pt.y, z))
        if len(path_3d) > 1:
            all_paths.append(douglas_peucker(path_3d, 0.02))
    return all_paths

def make_gcode_phases_advanced(phases, tool_name, header, footer, post, is_3d=False, tol=0.01):
    w = GcodeWriter(post)
    w.raw(header)
    w.comment(f"TOOL: {tool_name}")
    if post.get("tool_change"):
        w.raw("T1 M06")
    safe = 5.0
    w.move(0, z=safe)
    for i, phase in enumerate(phases):
        p_list = phase.get('paths', [])
        if not p_list: continue
        feed, z_start, z_final = int(phase.get('feed', 300)), phase.get('z_start', 0), phase.get('z_final', 0)
        z_step = max(0.01, phase.get('z_step', abs(z_final - z_start)))
        w.comment(f"Phase {i+1}: {phase.get('name','')}")
        if is_3d:
            for p_pts in p_list:
                w.move(0, x=p_pts[0][0], y=p_pts[0][1])
                w.move(1, z=p_pts[0][2], feed=feed)
                for p in p_pts[1:]:
                    w.move(1, x=p[0], y=p[1], z=p[2], feed=feed)
                w.move(0, z=safe)
        else:
            fitted = [fit_path(path.coords, tol) for path in p_list]
            cur_z = z_start
            while cur_z > z_final + 1e-9:
                tgt_z = max(cur_z - z_step, z_final)
                for start, ops in fitted:
                    w.move(0, x=start[0], y=start[1])
                    w.move(0, z=cur_z + 1.0)
                    w.move(1, z=tgt_z, feed=feed)
                    write_path_2d(w, start, ops, feed)
                    w.move(0, z=safe)
                cur_z = tgt_z
    w.raw(footer)
    return w.text()

def analyze_holes(geometry):
    polys = ensure_list_of_polys(geometry)
    sizes = []
    def check_p(p):
        minx, miny, maxx, maxy = p.bounds
        w, h = maxx - minx, maxy - miny
        if abs(w - h) > 0.1: return None
        if abs(p.area - math.pi * ((w/2)**2)) / (math.pi * ((w/2)**2)) > 0.2: return None
        return w
    for p in polys:
        d = check_p(p)
        if d: sizes.append(round(d, 2))
        for interior in p.interiors:
            d_h = check_p(Polygon(interior))
            if d_h: sizes.append(round(d_h, 2))
    return Counter(sizes)

# --- 2.5 加工シミュレーション (Zマップ / ハイトマップ方式) ---
# 生成したGコードを読み直し、工具形状を材料表面に押し当てて削り残り形状を計算する。
# 「意図した形状」ではなく「Gコードが実際に削る形状」を表示するので、パスの検証にも使える。

GCODE_WORD = re.compile(r'([A-Z])\s*([-+]?\d*\.?\d*)')

def _arc_points(x0, y0, z0, x1, y1, z1, cx, cy, ccw, max_seg=0.2):
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    sweep = a1 - a0
    if ccw and sweep <= 1e-9:
        sweep += 2 * math.pi
    elif not ccw and sweep >= -1e-9:
        sweep -= 2 * math.pi
    r = math.hypot(x0 - cx, y0 - cy)
    n = max(2, int(math.ceil(abs(sweep) * r / max_seg)))
    t = np.linspace(0.0, 1.0, n + 1)
    a = a0 + sweep * t
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a), z0 + (z1 - z0) * t]

def parse_gcode_moves(gcode, safe_z=5.0):
    """切削移動 (G1/G2/G3/穴あけ固定サイクル) を (N, 6) の線分配列 [x0,y0,z0,x1,y1,z1] で返す。
    XY方向の早送りで材料内にいる回数も返す。"""
    segs, rapid_hits = [], 0
    x = y = None
    z = safe_z
    mode, cycle, cyc_z, cyc_r = 0, None, None, None
    for raw in gcode.splitlines():
        line = re.sub(r'\(.*?\)', '', raw.split(';')[0]).upper()
        words = [(w, v) for w, v in GCODE_WORD.findall(line) if v not in ('', '+', '-', '.')]
        gs = [float(v) for w, v in words if w == 'G']
        if 28 in gs or 30 in gs or 92 in gs:
            continue  # 原点復帰・座標系設定はシミュレーション対象外
        nx, ny, nz, has_xy, has_z = x, y, z, False, False
        ij, new_r = [0.0, 0.0], None
        for w, v in words:
            v = float(v)
            if w == 'X': nx, has_xy = v, True
            elif w == 'Y': ny, has_xy = v, True
            elif w == 'Z': nz, has_z = v, True
            elif w == 'I': ij[0] = v
            elif w == 'J': ij[1] = v
            elif w == 'R': new_r = v
        for g in gs:
            if g in (0, 1, 2, 3):
                mode, cycle = int(g), None
            elif g in (73, 81, 82, 83, 85, 86, 89):
                cycle = int(g)
            elif g == 80:
                cycle = None
        if cycle is not None:
            if has_z: cyc_z = nz
            if new_r is not None: cyc_r = new_r
            if has_xy:
                # イニシャル点で XY 位置決めしてから R 点→Z 点まで穴あけ
                if x is not None and min(z, z) < 0 and (nx, ny) != (x, y):
                    rapid_hits += 1
                if cyc_z is not None and cyc_r is not None:
                    segs.append((nx, ny, cyc_r, nx, ny, cyc_z))
                x, y = nx, ny
            continue
        if not (has_xy or has_z):
            continue
        if x is not None and y is not None and nx is not None and ny is not None:
            if mode == 1:
                segs.append((x, y, z, nx, ny, nz))
            elif mode in (2, 3):
                pts = _arc_points(x, y, z, nx, ny, nz, x + ij[0], y + ij[1], mode == 3)
                segs.extend(np.c_[pts[:-1], pts[1:]].tolist())
            elif (nx, ny) != (x, y) and min(z, nz) < 0:
                rapid_hits += 1
        x, y, z = nx, ny, nz
    return np.array(segs, dtype=float).reshape(-1, 6), rapid_hits

def tool_center_field(segs, x0, y0, cell, nx, ny):
    """線分を細分化し、各セルでの工具先端の最低Z (未通過セルは +inf) を返す。"""
    field = np.full((ny, nx), np.inf)
    if len(segs) == 0:
        return field
    p0, p1 = segs[:, :3], segs[:, 3:]
    length = np.hypot(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1])
    n = np.maximum(1, np.ceil(length / (cell * 0.5)).astype(int))
    idx = np.repeat(np.arange(len(segs)), n + 1)
    starts = np.repeat(np.cumsum(n + 1) - (n + 1), n + 1)
    t = (np.arange(len(idx)) - starts) / np.repeat(n, n + 1)
    pts = p0[idx] + (p1[idx] - p0[idx]) * t[:, None]
    ix = np.round((pts[:, 0] - x0) / cell).astype(int)
    iy = np.round((pts[:, 1] - y0) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.minimum.at(field, (iy[ok], ix[ok]), pts[ok, 2])
    return field

def cut_with_tool(height, centers, cell, radius, v_angle=None):
    """工具先端の軌跡フィールドを工具形状で膨らませ (グレースケール収縮)、材料面に反映する。
    v_angle=None ならフラットエンドミル、角度指定ならVビット。"""
    if not np.isfinite(centers).any():
        return height
    k = max(1, int(math.ceil(radius / cell)))
    dy, dx = np.mgrid[-k:k + 1, -k:k + 1] * cell
    d = np.hypot(dx, dy)
    footprint = d <= max(radius, cell * 0.5)
    profile = np.zeros_like(d) if v_angle is None else d / np.tan(np.radians(v_angle / 2.0))
    big = 1e6
    eroded = grey_erosion(np.where(np.isfinite(centers), centers, big),
                          footprint=footprint, structure=-profile, mode='constant', cval=big)
    return np.minimum(height, eroded)

@st.cache_data(show_spinner=False, max_entries=8)
def simulate_stock(jobs, bounds, cell):
    """jobs: (gcode, 工具種別, 工具径, V角度) のタプル。上面 Z=0 の材料を削ったハイトマップを返す。"""
    minx, miny, maxx, maxy = bounds
    nx = int(math.ceil((maxx - minx) / cell)) + 1
    ny = int(math.ceil((maxy - miny) / cell)) + 1
    height = np.zeros((ny, nx))
    rapid_hits = 0
    for gcode, kind, dia, angle in jobs:
        segs, hits = parse_gcode_moves(gcode)
        rapid_hits += hits
        if len(segs) == 0:
            continue
        centers = tool_center_field(segs, minx, miny, cell, nx, ny)
        if kind == 'vbit':
            # Vビットは最深点での切削半径まで考慮する
            depth = max(0.0, -float(np.min(segs[:, [2, 5]])))
            radius = depth * np.tan(np.radians(angle / 2.0))
            height = cut_with_tool(height, centers, cell, radius, angle)
        else:
            height = cut_with_tool(height, centers, cell, dia / 2.0)
    height = np.minimum(height, 0.0)
    xs = minx + np.arange(nx) * cell
    ys = miny + np.arange(ny) * cell
    return xs, ys, height, rapid_hits

def make_preview_figure(xs, ys, height, thickness):
    z = height.copy()
    z[z <= -thickness + 1e-6] = np.nan  # 貫通部は穴として表示
    fig = go.Figure(go.Surface(
        x=xs, y=ys, z=z, cmin=-thickness, cmax=0.0,
        colorscale=[[0.0, '#5b3a1a'], [0.6, '#b07a3e'], [1.0, '#e8c48f']],
        colorbar=dict(title='Z (mm)', thickness=12),
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.15, roughness=0.8),
        lightposition=dict(x=100, y=200, z=1000),
        contours=dict(z=dict(show=False)),
        hovertemplate='X %{x:.2f}<br>Y %{y:.2f}<br>Z %{z:.2f}<extra></extra>'))
    span = max(xs[-1] - xs[0], ys[-1] - ys[0], 1e-6)
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=(xs[-1] - xs[0]) / span, y=(ys[-1] - ys[0]) / span,
                             z=max(0.08, min(1.0, thickness / span * 2))),
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            zaxis=dict(range=[-thickness, 0.5]),
            camera=dict(eye=dict(x=0.6, y=-1.0, z=0.8))))
    return fig

# --- 3. UIロジック ---

st.set_page_config(page_title="yosikeiCAM", layout="wide")

with st.sidebar:
    lang_opt = list(LANG_DICT.keys())
    lang = st.selectbox("Language / 言語", lang_opt)
    T = LANG_DICT[lang]

st.title(T["title"])
st.caption("Ver 4.6: 3D Machining Preview")

with st.sidebar:
    st.header(T["origin_setting"])
    origin = st.radio("", T["origin_option"], index=0)
    st.divider()
    st.header(T["process_setting"])
    tab1, tab2, tab3, tab4 = st.tabs(T["tabs"])
    
    with tab1:
        enable_p = st.checkbox(T["enable"], value=True, key="cp")
        dia = st.number_input(T["tool_dia"], value=3.0, step=0.1)
        ucp = st.checkbox(T["double_pass"], value=False, key="uccp")
        clear = 0.0
        f_p_r = st.number_input(T["rough_feed"], value=300, min_value=1, key="fp_r")
        f_p_f, f_mode = f_p_r, "Step-down"
        if ucp:
            clear = st.number_input(T["finish_allowance"], value=0.2, step=0.1)
            c1, c2 = st.columns(2)
            with c1: f_p_f = st.number_input(T["finish_feed"], value=300, min_value=1, key="fp_f")
            with c2: 
                f_o = st.radio(T["finish_depth_mode"], T["depth_options"], index=0, key="fmp")
                f_mode = "Full-Depth" if f_o == T["depth_options"][1] else "Step-down"
        c3, c4 = st.columns(2)
        with c3: dep_p = st.number_input(T["final_depth"], value=-1.0, step=0.1)
        with c4: stp_p = st.number_input(T["pitch"], value=1.0, min_value=0.01)
    
    with tab2:
        enable_c = st.checkbox(T["enable"], value=True, key="cc")
        cw = st.number_input(T["chamfer_width"], value=0.5, step=0.1)
        to = st.number_input(T["tip_offset"], value=1.0, step=0.1)
        z_c = -(cw + to)
        st.caption(f"Z: {z_c:.2f}mm")
        fc_r = st.number_input(T["rough_feed"], value=300, key="fc_r")
        ucf = st.checkbox(T["double_pass"], value=False, key="uccf")
        cfa, fc_f = 0.0, fc_r
        if ucf:
            c5, c6 = st.columns(2)
            with c5: cfa = st.number_input(T["finish_allowance"], value=0.2, step=0.1)
            with c6: fc_f = st.number_input(T["finish_feed"], value=300, key="fc_f")
        
    with tab3:
        enable_v = st.checkbox(T["enable"], value=False, key="cv")
        va = st.number_input(T["v_angle"], value=60.0, step=10.0)
        ucv = st.checkbox(T["double_pass"], value=False, key="uccv")
        v_cl, fvf = 0.0, 300
        fv = st.number_input(T["rough_feed"], value=300, key="fv")
        if ucv:
            v_cl = st.number_input(T["finish_allowance"], value=0.2, step=0.1)
            fvf = st.number_input(T["finish_feed"], value=300, key="fvf")
        uvl = st.checkbox(T["depth_limit"], value=False)
        vl = st.number_input(T["max_depth"], value=-3.0) if uvl else -100.0
        vr = st.slider(T["precision"], value=0.05, min_value=0.02, max_value=0.2)
        
    with tab4:
        enable_d = st.checkbox(T["enable"], value=False, key="cd")
        ddt = st.number_input(T["drill_dia"], value=3.0)
        ddz = st.number_input(T["drill_depth"], value=-5.0)
        pck = st.number_input(T["peck"], value=2.0, min_value=0.1)
        fd = st.number_input(T["feed"], value=200, key="fd")
        
    st.divider()
    ppn = st.selectbox(T["post_processor"], list(POST_PROCESSORS.keys()))
    pp = POST_PROCESSORS[ppn]
    h_c = st.text_area("Header", pp["start"])
    f_c = st.text_area("Footer", pp["end"])
    out_tol = st.number_input(T["out_tol"], value=0.01, min_value=0.001, max_value=0.1, step=0.005, format="%.3f")

    st.divider()
    st.header(T["preview_header"])
    show_3d = st.checkbox(T["preview_enable"], value=True, key="p3d")
    stock_t = st.number_input(T["stock_thickness"], value=10.0, min_value=0.1, step=1.0)
    sim_res = st.select_slider(T["sim_resolution"], options=[0.05, 0.1, 0.2, 0.3, 0.5], value=0.2)

st.header(T["dxf_upload"])
f = st.file_uploader("", type=["dxf"])

if f:
    bn = os.path.splitext(f.name)[0]
    polys_raw = dxf_to_shapely_list(f.getvalue())
    if polys_raw:
        tu = unary_union(polys_raw)
        minx, miny, maxx, maxy = tu.bounds
        w_size, h_size = maxx-minx, maxy-miny
        ox, oy = 0, 0
        if origin.startswith(T["origin_option"][0].split(" ")[0]): 
            ox, oy = -minx, -miny
        elif origin.startswith(T["origin_option"][1].split(" ")[0]): 
            ox, oy = -(minx+w_size/2), -(miny+h_size/2)
            
        polys_moved = [translate(p, ox, oy) for p in polys_raw]
        
        st.sidebar.divider()
        st.sidebar.subheader(T["path_select"])
        all_c = st.sidebar.checkbox(T["select_all"], value=True, key="sa")
        selected = [i for i, p in enumerate(polys_moved) if st.sidebar.checkbox(f"Path #{i+1} (Area:{p.area:.1f})", value=all_c, key=f"p{i}")]
                
        target_polys = [polys_moved[i] for i in selected]
        gfc = merge_polygons_xor(target_polys)
        
        # 穴情報の解析
        drill_points_found = []
        if gfc:
            hs = analyze_holes(gfc)
            if hs:
                st.sidebar.info(f"{T['hole_detected']}: " + ", ".join([f"φ{d}({c})" for d, c in hs.items()]))
            if enable_d:
                drill_points_found = find_drill_points(gfc, ddt)
        
        st.success(f"{T['size_display']}: {w_size:.1f} x {h_size:.1f} mm")
        st.header(T["gen_header"])
        
        c_left, c_right = st.columns(2)
        with c_left:
            fig, ax = plt.subplots(figsize=(5,5))
            ax.plot(0, 0, 'r+', markersize=20)
            ax.axhline(0, color='red', alpha=0.3)
            ax.axvline(0, color='red', alpha=0.3)
            for i, p in enumerate(polys_moved):
                style = 'k-' if i in selected else 'k:'
                alpha = 1.0 if i in selected else 0.1
                ax.plot(*p.exterior.xy, style, alpha=alpha, linewidth=1)
                for interior in p.interiors:
                    ax.plot(*interior.xy, style, alpha=alpha, linewidth=1)
            for pt in drill_points_found:
                ax.plot(pt.x, pt.y, 'x', color='tab:purple', markersize=8)
            ax.axis('equal')
            ax.grid(True, linestyle=':')
            st.pyplot(fig)
            
        with c_right:
            gc_v, gc_p, gc_c, gc_d = None, None, None, None
            v_d, p_r_d, p_f_d, c_d = [], [], [], []
            if gfc and not gfc.is_empty:
                if enable_v:
                    with st.spinner("VCarve..."):
                        v_r_ps = generate_vcarve(gfc, va, uvl, vl, vr, z_offset=v_cl) if ucv else []
                        v_f_ps = generate_vcarve(gfc, va, uvl, vl, vr, z_offset=0.0)
                        v_d = v_r_ps + v_f_ps
                        phs_v = []
                        if v_r_ps: phs_v.append({'name':'V-Rough','paths':v_r_ps,'z_start':0,'z_final':0,'feed':fv})
                        if v_f_ps: phs_v.append({'name':'V-Finish','paths':v_f_ps,'z_start':0,'z_final':0,'feed':fvf if ucv else fv})
                        if phs_v: gc_v = make_gcode_phases_advanced(phs_v, "VBit", h_c, f_c, pp, True)
                if enable_p:
                    p_r, p_f = generate_pocket(gfc, dia, clear if ucp else 0.0, 0.5)
                    p_r_d, p_f_d = p_r, p_f
                    phs_p = []
                    if p_r: phs_p.append({'name':'Rough','paths':p_r,'z_start':0,'z_final':dep_p,'feed':f_p_r,'z_step':stp_p})
                    if p_f and ucp: phs_p.append({'name':'Finish','paths':p_f,'z_start':0,'z_final':dep_p,'feed':f_p_f,'z_step':(abs(dep_p) if f_mode==T["depth_options"][1] else stp_p)})
                    if phs_p: gc_p = make_gcode_phases_advanced(phs_p, "EndMill", h_c, f_c, pp, tol=out_tol)
                if enable_c:
                    rp_c, fp_c = generate_chamfer_separated(gfc, to, cfa if ucf else 0.0)
                    c_d = rp_c + fp_c
                    phs_c = []
                    if rp_c and ucf: phs_c.append({'name':'Rough','paths':rp_c,'z_start':0,'z_final':z_c,'feed':fc_r})
                    if fp_c: phs_c.append({'name':'Finish','paths':fp_c,'z_start':0,'z_final':z_c,'feed':fc_f if ucf else fc_r})
                    if phs_c: gc_c = make_gcode_phases_advanced(phs_c, "Chamfer", h_c, f_c, pp, tol=out_tol)
                if enable_d:
                    gc_d = generate_drill_gcode(drill_points_found, 0, ddz, pck, fd, f"Drill {ddt}mm", h_c, f_c, pp)

            fig2, ax2 = plt.subplots(figsize=(5,5))
            ax2.plot(0, 0, 'r+', markersize=20)
            ax2.axhline(0, color='red', alpha=0.3); ax2.axvline(0, color='red', alpha=0.3)
            for p in polys_moved:
                ax2.plot(*p.exterior.xy, 'k--', alpha=0.05)
            for ls in p_r_d: ax2.plot(*ls.xy, color='tab:blue', alpha=0.3)
            for ls in p_f_d: ax2.plot(*ls.xy, color='tab:cyan', alpha=0.8)
            for ls in c_d: ax2.plot(*ls.xy, color='tab:green', alpha=0.8)
            for pts in v_d: ax2.plot([p[0] for p in pts], [p[1] for p in pts], 'r-', linewidth=0.8)
            for pt in drill_points_found:
                ax2.plot(pt.x, pt.y, 'x', color='tab:purple', markersize=8)
            ax2.axis('equal'); ax2.grid(True, linestyle=':')
            st.pyplot(fig2)
            
            dl_c1, dl_c2 = st.columns(2)
            if gc_v: dl_c1.download_button(T["download"][0], gc_v, f"{bn}_vcarve.nc")
            if gc_p: dl_c2.download_button(T["download"][1], gc_p, f"{bn}_pocket.nc")
            if gc_c: dl_c1.download_button(T["download"][2], gc_c, f"{bn}_chamfer.nc")
            if gc_d: dl_c2.download_button(T["download"][3], gc_d, f"{bn}_drill.nc")
            max_feed = pp.get("max_feed")
            if max_feed:
                used = {int(v) for gc in (gc_v, gc_p, gc_c, gc_d) if gc for v in re.findall(r'F(\d+)', gc)}
                for fv_used in sorted(v for v in used if v > max_feed):
                    st.warning(T["feed_warning"].format(f=fv_used, m=max_feed))
            # 内蔵メモリ運転の目安になるよう、各ファイルのサイズを表示する
            for name, gc in (("VCARVE", gc_v), ("POCKET", gc_p), ("CHAMFER", gc_c), ("DRILL", gc_d)):
                if gc:
                    st.caption(T["file_size"].format(name=name, kb=len(gc.encode()) / 1024, lines=gc.count("\n")))

        if show_3d:
            st.header(T["preview_header"])
            jobs = []
            if gc_p: jobs.append((gc_p, 'flat', dia, 0.0))
            if gc_d: jobs.append((gc_d, 'flat', ddt, 0.0))
            if gc_c: jobs.append((gc_c, 'vbit', 0.0, 90.0))
            if gc_v: jobs.append((gc_v, 'vbit', 0.0, va))
            if jobs:
                bx0, by0, bx1, by1 = unary_union(polys_moved).bounds
                margin = max(5.0, dia / 2 + 2.0, ddt / 2 + 2.0, abs(z_c) + 2.0)
                sim_bounds = (bx0 - margin, by0 - margin, bx1 + margin, by1 + margin)
                # 大きな図面ではセル数が増えすぎないように解像度を自動で粗くする
                cell = max(sim_res, max(sim_bounds[2] - sim_bounds[0], sim_bounds[3] - sim_bounds[1]) / 800)
                with st.spinner(T["preview_running"]):
                    xs, ys, hmap, rapid_hits = simulate_stock(tuple(jobs), sim_bounds, cell)
                if rapid_hits:
                    st.warning(T["rapid_warning"].format(n=rapid_hits))
                st.plotly_chart(make_preview_figure(xs, ys, hmap, stock_t), use_container_width=True)
                st.caption(T["preview_note"] + f" (cell {cell:.2f} mm)")
            else:
                st.info(T["preview_none"])
    else:
        st.error(T["error_no_geom"])
