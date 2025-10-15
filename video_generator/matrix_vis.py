from __future__ import annotations
from typing import Iterable, List
import numpy as np
from manim import *

# ------------------------------------------------------------
# 配置：你可以在这里直接修改矩阵与可视化参数
# ------------------------------------------------------------
DATA = np.array([
    [ 0.2, -0.1,  0.0,  0.7],
    [-0.3,  0.5, -0.8,  0.1],
    [ 0.9,  0.2, -0.4, -0.6],
])

# 3D 柱子的 z 轴高度范围（会根据数据自动缩放到这个范围内）
Z_MIN, Z_MAX = 0.1, 2.5

# 单元格/柱子在 x,y 平面上的占位宽度（<1 时柱子之间会有缝隙）
CELL_SIZE = 0.95

# 颜色映射（正/负/零）
POS_COLOR = GREEN
NEG_COLOR = RED
ZERO_COLOR = GREY

# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def normalize_to_range(vals: np.ndarray, low: float, high: float) -> np.ndarray:
    vmin, vmax = float(vals.min()), float(vals.max())
    if np.isclose(vmin, vmax):
        return np.full_like(vals, (low + high) / 2.0)
    # 线性归一化到 [low, high]
    return (vals - vmin) / (vmax - vmin) * (high - low) + low


def pick_color(v: float) -> Color:
    if np.isclose(v, 0.0):
        return ZERO_COLOR
    return POS_COLOR if v > 0 else NEG_COLOR


# 尝试兼容不同 Manim 版本的 3D 方柱 API
# 优先使用 Cuboid，如果没有 Cuboid 则退化到 Prism
try:
    from manim.mobject.three_dimensions import Cuboid as _Cuboid  # ManimCE >= 0.18 常见
    def make_bar(dx: float, dy: float, dz: float, color: Color) -> Mobject:
        bar = _Cuboid(width=dx, depth=dy, height=dz)
        bar.set_fill(color, opacity=0.8).set_stroke(WHITE, 0.5, opacity=0.35)
        return bar
except Exception:
    try:
        from manim.mobject.three_dimensions import Prism as _Prism
        def make_bar(dx: float, dy: float, dz: float, color: Color) -> Mobject:
            base = Polygon(
                ORIGIN + np.array([ -dx/2, -dy/2, 0.0 ]),
                ORIGIN + np.array([  dx/2, -dy/2, 0.0 ]),
                ORIGIN + np.array([  dx/2,  dy/2, 0.0 ]),
                ORIGIN + np.array([ -dx/2,  dy/2, 0.0 ]),
            )
            bar = _Prism(base_polygon=base, height=dz)
            bar.set_fill(color, opacity=0.8).set_stroke(WHITE, 0.5, opacity=0.35)
            return bar
    except Exception:
        # 最后兜底：用 VGroup 拼六个面（简化版），避免版本差异导致报错
        def make_bar(dx: float, dy: float, dz: float, color: Color) -> Mobject:
            x, y, z = dx/2, dy/2, dz/2
            # 8 个顶点
            p = {
                'lbf': np.array([-x, -y, -z]), 'rbf': np.array([ x, -y, -z]),
                'rtf': np.array([ x,  y, -z]), 'ltf': np.array([-x,  y, -z]),
                'lbb': np.array([-x, -y,  z]), 'rbb': np.array([ x, -y,  z]),
                'rtb': np.array([ x,  y,  z]), 'ltb': np.array([-x,  y,  z]),
            }
            faces = [
                Polygon(p['lbf'], p['rbf'], p['rtf'], p['ltf']), # bottom
                Polygon(p['lbb'], p['rbb'], p['rtb'], p['ltb']), # top
                Polygon(p['lbf'], p['rbf'], p['rbb'], p['lbb']), # front
                Polygon(p['rbf'], p['rtf'], p['rtb'], p['rbb']), # right
                Polygon(p['rtf'], p['ltf'], p['ltb'], p['rtb']), # back
                Polygon(p['ltf'], p['lbf'], p['lbb'], p['ltb']), # left
            ]
            vg = VGroup(*faces).set_fill(color, 0.8).set_stroke(WHITE, 0.5, opacity=0.35)
            return vg


# ------------------------------------------------------------
# 2D 平面矩阵（带数字）
# ------------------------------------------------------------
class MatrixPlaneView(Scene):
    """展示一个带数字的矩阵平面视图，用 Table 实现。"""
    def construct(self):
        mat = DATA
        rows, cols = mat.shape

        # 将数值转成字符串放入 Table
        entries: List[List[str]] = [[f"{mat[r, c]:.2f}" for c in range(cols)] for r in range(rows)]
        table = Table(entries,
                      include_outer_lines=True,
                      v_buff=0.5, h_buff=0.8)
        table.scale(0.8)

        # 给正负数上色（文本颜色）
        for r in range(rows):
            for c in range(cols):
                cell = table.get_entries((r+1, c+1))
                val = mat[r, c]
                cell.set_color(pick_color(val))
        title = Text("Matrix (2D plane view)", weight=BOLD).scale(0.6).to_edge(UP)

        self.play(FadeIn(title))
        self.play(Create(table))
        self.wait(1.5)


# ------------------------------------------------------------
# 3D 柱状矩阵视图（柱高反映数值大小）
# ------------------------------------------------------------
class MatrixBars3D(ThreeDScene):
    def construct(self):
        mat = DATA
        rows, cols = mat.shape

        # 将数值绝对值或原值映射到 [Z_MIN, Z_MAX]
        # 如果你想用绝对值高度，改成 np.abs(mat)
        heights = normalize_to_range(mat, Z_MIN, Z_MAX)
        zmax = float(heights.max()) * 1.15

        # 3D 坐标轴（x: 列，y: 行，z: 高度）
        axes = ThreeDAxes(
            x_range=[0, cols, 1],
            y_range=[0, rows, 1],
            z_range=[0, zmax, zmax/5],
            x_length=6, y_length=6, z_length=4.5,
            axis_config={"include_numbers": True, "include_ticks": True}
        )

        # 让 (0,0) 在左下角直觉上对应矩阵 [行自下而上, 列自左而右]
        # 我们把 y 轴正向视为“从下到上”的行索引
        bars = VGroup()
        labels = VGroup()

        for r in range(rows):
            for c in range(cols):
                h = float(heights[r, c])
                color = pick_color(float(mat[r, c]))

                # 在网格中心放柱子：x = c + 0.5, y = r + 0.5
                x, y = c + 0.5, r + 0.5

                bar = make_bar(CELL_SIZE, CELL_SIZE, h, color)
                # 让柱子底部落在 z=0 平面上
                bar.move_to(axes.c2p(x, y, h/2))

                # 顶部数值标签
                label = DecimalNumber(float(mat[r, c]), num_decimal_places=2)
                label.scale(0.35)
                label.next_to(bar, OUT, buff=0.02)
                label.set_color(color)

                bars.add(bar)
                labels.add(label)

        title = Text("Matrix as 3D bars (height = value)", weight=BOLD).scale(0.6).to_edge(UP)

        # 3D 相机角度
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES, zoom=1.1)

        self.play(FadeIn(title))
        self.play(Create(axes))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.03))
        self.play(FadeIn(labels, shift=OUT*0.2), run_time=0.8)
        self.wait(1.5)

        # 小旋转浏览
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(2.5)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)


# ------------------------------------------------------------
# 组合场景：左侧 2D，右侧 3D（可选）
# ------------------------------------------------------------
class Matrix2DTo3D(ThreeDScene):
    def construct(self):
        # 2D 表
        mat = DATA
        rows, cols = mat.shape
        entries = [[f"{mat[r, c]:.2f}" for c in range(cols)] for r in range(rows)]
        table = Table(entries, include_outer_lines=True, v_buff=0.5, h_buff=0.8).scale(0.7)
        for r in range(rows):
            for c in range(cols):
                table.get_entries((r+1, c+1)).set_color(pick_color(mat[r, c]))

        title = Text("2D → 3D mapping", weight=BOLD).scale(0.6).to_edge(UP)

        # 3D 轴与柱
        heights = normalize_to_range(mat, Z_MIN, Z_MAX)
        zmax = float(heights.max()) * 1.15
        axes = ThreeDAxes(
            x_range=[0, cols, 1], y_range=[0, rows, 1], z_range=[0, zmax, zmax/5],
            x_length=5.5, y_length=5.5, z_length=4.0,
            axis_config={"include_numbers": False}
        )
        bars = VGroup()
        for r in range(rows):
            for c in range(cols):
                h = float(heights[r, c])
                color = pick_color(float(mat[r, c]))
                x, y = c + 0.5, r + 0.5
                bar = make_bar(CELL_SIZE, CELL_SIZE, h, color)
                bar.move_to(axes.c2p(x, y, h/2))
                bars.add(bar)

        # 布局：左 2D，右 3D
        left = VGroup(table).to_edge(LEFT).shift(LEFT*0.2)
        right = VGroup(axes, bars).to_edge(RIGHT).shift(RIGHT*0.1)

        # 相机
        self.set_camera_orientation(phi=60*DEGREES, theta=-50*DEGREES, zoom=1.0)

        self.play(FadeIn(title))
        self.play(Create(left))
        self.wait(0.5)
        self.play(Create(axes))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.03))
        self.wait(1.0)

        # 可选：画出从 2D 单元格到 3D 柱子顶部的连线，帮助认知映射
        connectors = VGroup()
        for r in range(rows):
            for c in range(cols):
                cell = table.get_cell((r+1, c+1))
                cell_center = cell.get_center()
                x, y = c + 0.5, r + 0.5
                z = float(heights[r, c])
                bar_top = axes.c2p(x, y, z)
                line = DashedLine(cell_center, bar_top, dash_length=0.08)
                line.set_stroke(color=pick_color(mat[r, c]), width=1.5, opacity=0.7)
                connectors.add(line)
        self.play(LaggedStart(*[Create(l) for l in connectors], lag_ratio=0.01, run_time=1.5))
        self.wait(2)


# 使用方法（终端）：
#   manim -pqh this_file.py MatrixPlaneView    # 仅 2D 平面
#   manim -pqh this_file.py MatrixBars3D       # 仅 3D 柱状
#   manim -pqh this_file.py Matrix2DTo3D       # 2D + 3D 联动
# 取决于你的 manim 版本，如果遇到 Cuboid/Prism 不存在，请升级 manim 或使用上面的兜底实现。
