from manim import *
from typing import List, Literal

# =====================
# 基础单元：带数值的正方形
# =====================

class SquareWithValue(VGroup):
    def __init__(self, value: float, size: float = 1.0, color=BLUE, **kwargs) -> None:
        super().__init__(**kwargs)
        """创建一个带数值显示的正方形，返回 VGroup(square, text)，并在 group 上记录 .value"""
        self.value = value
        self.size = size
        self.font_size = int(self.size * 20)
        # self.color = color

        self.square = Square(side_length=size, color=color)
        self.text = Text(f"{value:.2f}", font_size=self.font_size)
        self.text.move_to(self.square.get_center())
        self.add(self.square)
        self.add(self.text)

    def update_value(self, new_value: float):
        """立即更新方块的数值显示（无动画）。"""
        new_text = Text(f"{new_value:.2f}", font_size=self.font_size)
        new_text.move_to(self.square.get_center())
        self.remove(self.text)
        self.text = new_text
        self.value = new_value
        self.add(self.text)


    def animate_update_value(self, new_value: float, color=YELLOW, run_time: float = 0.8) -> AnimationGroup:
        """
        生成“数值变化”的动画。
        使用 Transform(old_text -> new_text) 避免 CallFunc 依赖。
        """
        # square, old_text = square_group
        square = self.square
        old_text = self.text
        self.remove(old_text)
        target_text = Text(f"{new_value:.2f}", font_size=self.font_size, color=color)
        target_text.move_to(square.get_center())
        self.text = target_text
        self.add(self.text)

        # 先给旧文本一个轻微变灰/降透明的过渡，再把它 Transform 成新文本
        text_fade = old_text.animate.set_opacity(0.2).set_color(GRAY)
        text_switch = Transform(old_text, target_text)  # 旧对象变形为新文本

        # 方块闪烁
        flash_on = square.animate.set_fill(color, opacity=0.2)
        flash_off = square.animate.set_fill(opacity=0)

        # 组合：文字过渡 -> 形变到新文本（两者并行更自然），再闪烁
        return Succession(
            AnimationGroup(text_fade, text_switch, run_time=run_time * 0.6, lag_ratio=0),
            AnimationGroup(flash_on, run_time=run_time * 0.2),
            AnimationGroup(flash_off, run_time=run_time * 0.2),
        )


# =====================
# 一维向量组件（不持有 Scene，避免 pickling 问题）
# =====================
class Vector1D(VGroup):
    """
    表示一维向量的可视化：由多个“带数值方块”按行或列排布而成。
    注意：本类不持有 Scene 引用，避免 `TypeError: cannot pickle '_thread.lock' object`。
    """
    def __init__(
        self,
        values: List[float],
        orientation: Literal["row", "col"] = "row",
        cell_size: float = 1.0,
        gap: float = 0,
        color=BLUE,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert len(values) > 0, "values 不能为空"
        self.orientation = orientation
        self.cell_size = cell_size
        self.gap = gap
        self.color = color

        # 创建每个单元格
        self.cells: List[VGroup] = []
        for v in values:
            cell = SquareWithValue(v, size=cell_size, color=color)
            self.cells.append(cell)

        # 排布
        self._arrange_cells()
        self.add(*self.cells)

    def _arrange_cells(self):
        """根据 orientation 进行水平（row）或垂直（col）排列"""
        if self.orientation == "row":
            for i, cell in enumerate(self.cells):
                if i == 0:
                    continue
                cell.next_to(self.cells[i - 1], RIGHT, buff=self.gap)
        else:
            for i, cell in enumerate(self.cells):
                if i == 0:
                    continue
                cell.next_to(self.cells[i - 1], DOWN, buff=self.gap)

    # ---- 覆盖所有值（无动画） ----
    def set_values(self, values: List[float]):
        assert len(values) == len(self.cells), "values 大小必须与单元格数量一致"
        for cell, v in zip(self.cells, values):
            cell.update_value(v)

    # ---- 获取当前数值 ----
    def get_values(self) -> List[float]:
        return [cell.value for cell in self.cells]

    # ---- 设置整条 Vector1D 的颜色（边框/文字/填充不透明度 可选）----
    def set_color(self, border_color=None, text_color=None, fill_opacity: float | None = None):
        for cell in self.cells:
            square, text = cell
            if border_color is not None:
                square.set_color(border_color)
                square.set_stroke(border_color)
            if text_color is not None:
                text.set_color(text_color)
            if fill_opacity is not None:
                square.set_fill(opacity=fill_opacity)

    # ---- 生成用于动画的整体 Animation（可与 LaggedStart/AnimationGroup 搭配）----
    def get_set_values_animation(self, values: List[float], color=YELLOW, run_time_each: float = 0.6) -> AnimationGroup:
        assert len(values) == len(self.cells), "values 大小必须与单元格数量一致"
        per_cell_anims = [cell.animate_update_value(v, color=color, run_time=run_time_each) for cell, v in zip(self.cells, values)]
        return AnimationGroup(*per_cell_anims, lag_ratio=0)


# =====================
# Matrix2D：由多个 Vector1D 组成的矩阵
# =====================
class Matrix2D(VGroup):
    """
    以 Vector1D 为行（或列）构建的二维矩阵。
    - major='row'  : rows 为 Vector1D(orientation='row')
    - major='col'  : cols 为 Vector1D(orientation='col')
    提供：
      * set_row_color / set_col_color
      * set_values / get_values_2d
      * get_set_values_animation（整矩阵动画）
      * split_by_indices(indices, axis='row'|'col') 返回 (A, B) 两个新矩阵
    """
    def __init__(
        self,
        values_2d: List[List[float]],
        major: Literal['row', 'col'] = 'row',
        cell_size: float = 1.0,
        hgap: float = 0.12,
        vgap: float = 0.12,
        color=BLUE,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert len(values_2d) > 0 and len(values_2d[0]) > 0, 'values_2d 不能为空'
        self.major = major
        self.cell_size = cell_size
        self.hgap = hgap
        self.vgap = vgap
        self.color = color

        self.vectors: List[Vector1D] = []
        if major == 'row':
            for row in values_2d:
                vec = Vector1D(row, orientation='row', cell_size=cell_size, gap=hgap, color=color)
                self.vectors.append(vec)
            # 垂直排列每一行
            for i, vec in enumerate(self.vectors):
                if i == 0:
                    continue
                vec.next_to(self.vectors[i-1], DOWN, buff=vgap)
            # 左对齐
            for vec in self.vectors:
                vec.align_to(self.vectors[0], LEFT)
        else:
            for col in values_2d:
                vec = Vector1D(col, orientation='col', cell_size=cell_size, gap=vgap, color=color)
                self.vectors.append(vec)
            # 水平排列每一列
            for i, vec in enumerate(self.vectors):
                if i == 0:
                    continue
                vec.next_to(self.vectors[i-1], RIGHT, buff=hgap)
            # 顶对齐
            for vec in self.vectors:
                vec.align_to(self.vectors[0], UP)

        self.add(*self.vectors)

    # ---- 颜色接口 ----
    def set_row_color(self, idx: int, border_color=None, text_color=None, fill_opacity: float | None = None):
        assert self.major == 'row', '当前矩阵以列为主（major=col），请使用 set_col_color 或在 row-major 下调用'
        self.vectors[idx].set_color(border_color, text_color, fill_opacity)

    def set_col_color(self, idx: int, border_color=None, text_color=None, fill_opacity: float | None = None):
        assert self.major == 'col', '当前矩阵以行为主（major=row），请使用 set_row_color 或在 col-major 下调用'
        self.vectors[idx].set_color(border_color, text_color, fill_opacity)

    # ---- 数值读写 ----
    def get_values_2d(self) -> List[List[float]]:
        return [vec.get_values() for vec in self.vectors]

    def set_values(self, values_2d: List[List[float]]):
        assert len(values_2d) == len(self.vectors), '行/列数量不匹配'
        for vec, vals in zip(self.vectors, values_2d):
            assert len(vals) == len(vec.cells), '每行/列长度不匹配'
            vec.set_values(vals)

    def get_set_values_animation(
        self,
        values_2d: List[List[float]],
        color=YELLOW,
        run_time_each: float = 0.5,
        axis: Literal['all','row','col'] = 'all',
        lag_ratio_outer: float = 0.1,
    ) -> AnimationGroup:
        """
        返回一个可播放的整矩阵动画。
        - axis='all'：所有向量并行；
        - axis='row'：逐行交错；
        - axis='col'：逐列交错（当 major='col' 时语义等同“逐向量交错”）。
        """
        assert len(values_2d) == len(self.vectors), '行/列数量不匹配'
        per_vec_anim = []
        for vec, vals in zip(self.vectors, values_2d):
            assert len(vals) == len(vec.cells), '每行/列长度不匹配'
            per_vec_anim.append(vec.get_set_values_animation(vals, color=color, run_time_each=run_time_each))

        if axis == 'all':
            return AnimationGroup(*per_vec_anim, lag_ratio=0)
        else:
            return LaggedStart(*per_vec_anim, lag_ratio=lag_ratio_outer)

    # ---- 根据索引拆分为两个新矩阵；支持移动 or 替换式转场 ----
    def split_by_indices(
        self,
        indices: list[int],
        axis: 'Literal["row","col"]' = 'row',
        with_move_anim: bool = False,
        transfer: bool = False,              # << 新增：使用 ReplacementTransform，转场后原处不会残留
        place_A_below_buff: float = 0.6,
        place_B_right_buff: float = 0.8,
    ):
        """
        indices 归为 A，其余归为 B；返回 (A, B) 两个全新 Matrix2D（数据拷贝重建）。

        参数
        - axis: 'row' 或 'col'，必须与 self.major 一致（与 Vector1D 维度对齐）。
        - with_move_anim: True 时，返回把“原矩阵中对应行/列整体平移到 A/B 位置”的动画。
        - transfer: True 时，返回使用 ReplacementTransform 的转场动画（推荐，不留残影）。
        注意：transfer=True 会优先于 with_move_anim（即忽略 with_move_anim）。
        - place_*: 新矩阵 A/B 的摆放缓冲。

        返回
        - 默认: (A, B)
        - with_move_anim 或 transfer 为 True: (A, B, anim)
        """
        assert axis in ('row','col'), "axis 必须是 'row' 或 'col'"
        assert axis == self.major, "axis 必须与当前矩阵 major 相同，便于按 Vector1D 维度拆分"

        n = len(self.vectors)
        sel = sorted(set(i for i in indices if 0 <= i < n))
        rest = [i for i in range(n) if i not in sel]

        # 原矩阵数值
        values = self.get_values_2d()
        a_vals = [values[i] for i in sel] if sel else []
        b_vals = [values[i] for i in rest] if rest else []

        # 构造新矩阵 A/B（全新对象，作“落位模板”和最终展示）
        A = Matrix2D(a_vals, major=self.major, cell_size=self.cell_size,
                    hgap=self.hgap, vgap=self.vgap, color=self.color) if a_vals else None
        B = Matrix2D(b_vals, major=self.major, cell_size=self.cell_size,
                    hgap=self.hgap, vgap=self.vgap, color=self.color) if b_vals else None

        # 若无需动画，直接返回
        if not (with_move_anim or transfer):
            return A, B

        # —— 为 A、B 安排位置（A 在原矩阵下方，B 在 A 右侧；若 A 为空则 B 放在原矩阵下方）——
        if A is not None:
            A.next_to(self, DOWN, buff=place_A_below_buff)
        if B is not None:
            if A is not None:
                B.next_to(A, RIGHT, buff=place_B_right_buff)
            else:
                B.next_to(self, DOWN, buff=place_A_below_buff)

        # ========== 方案一：ReplacementTransform（transfer=True，推荐） ==========
        if transfer:
            from manim import ReplacementTransform, AnimationGroup
            transforms = []

            # helper：把一行/一列中的每个单元（SquareWithValue 里的 cell/text）一一映射替换
            def per_vector_transfer(src_vec, dst_vec):
                # 假设 Vector1D 内部有 .cells（或等价列表）与 dst_vec 对齐；若你的实现是别的名字，请改这里
                # 如果你的单元是一个 VGroup（方块+文字一起），直接对该组做一次 ReplacementTransform 也可以
                if hasattr(src_vec, "cells") and hasattr(dst_vec, "cells"):
                    for src_cell, dst_cell in zip(src_vec.cells, dst_vec.cells):
                        transforms.append(ReplacementTransform(src_cell, dst_cell))
                else:
                    # 兜底：直接替换整个 Vector1D（视觉上会一行整体替换）
                    transforms.append(ReplacementTransform(src_vec, dst_vec))

            if axis == 'row':  # 行主
                if A is not None:
                    for k, i in enumerate(sel):
                        per_vector_transfer(self.vectors[i], A.vectors[k])
                if B is not None:
                    for k, i in enumerate(rest):
                        per_vector_transfer(self.vectors[i], B.vectors[k])
            else:              # 列主（与行主同理）
                if A is not None:
                    for k, i in enumerate(sel):
                        per_vector_transfer(self.vectors[i], A.vectors[k])
                if B is not None:
                    for k, i in enumerate(rest):
                        per_vector_transfer(self.vectors[i], B.vectors[k])

            transfer_anim = AnimationGroup(*transforms, lag_ratio=0.05)
            return A, B, transfer_anim

        # ========== 方案二：平移动画（with_move_anim=True，可能会留残影，需要后续手动 FadeOut 原件） ==========
        from manim import AnimationGroup
        move_anims = []
        if axis == 'row':
            if A is not None:
                for k, i in enumerate(sel):
                    move_anims.append(self.vectors[i].animate.move_to(A.vectors[k].get_center()))
            if B is not None:
                for k, i in enumerate(rest):
                    move_anims.append(self.vectors[i].animate.move_to(B.vectors[k].get_center()))
        else:
            if A is not None:
                for k, i in enumerate(sel):
                    move_anims.append(self.vectors[i].animate.move_to(A.vectors[k].get_center()))
            if B is not None:
                for k, i in enumerate(rest):
                    move_anims.append(self.vectors[i].animate.move_to(B.vectors[k].get_center()))
        move_anim = AnimationGroup(*move_anims, lag_ratio=0.1)
        return A, B, move_anim


    # # ---- 根据索引拆分为两个新矩阵 ----
    # def split_by_indices(
    #     self,
    #     indices: List[int],
    #     axis: Literal['row','col'] = 'row',
    #     with_move_anim: bool = False,
    #     place_A_below_buff: float = 0.6,
    #     place_B_right_buff: float = 0.8,
    # ):
    #     """
    #     indices 归为 A，其余归为 B；返回 (A, B) 两个 Matrix2D（全新对象，数据拷贝重建）。
    #     当 major='row' 且 axis='row'：按行拆；
    #     当 major='col' 且 axis='col'：按列拆。
    #     其他组合暂不直接支持（可先转置或用相同 major 的矩阵）。

    #     当 with_move_anim=True 时，同时返回一个 AnimationGroup，
    #     该动画会将“原矩阵中的被拆分部分”平移到新矩阵 A/B 对应位置。
    #     返回：
    #         - with_move_anim=False: (A, B)
    #         - with_move_anim=True : (A, B, move_anim)
    #     """
    #     assert axis in ('row','col')
    #     assert axis == self.major, 'axis 必须与当前矩阵 major 相同，便于按 Vector1D 维度拆分'

    #     n = len(self.vectors)
    #     sel = sorted(set(i for i in indices if 0 <= i < n))
    #     rest = [i for i in range(n) if i not in sel]

    #     values = self.get_values_2d()
    #     a_vals = [values[i] for i in sel]
    #     b_vals = [values[i] for i in rest]

    #     A = Matrix2D(a_vals, major=self.major, cell_size=self.cell_size,
    #                  hgap=self.hgap, vgap=self.vgap, color=self.color) if a_vals else None
    #     B = Matrix2D(b_vals, major=self.major, cell_size=self.cell_size,
    #                  hgap=self.hgap, vgap=self.vgap, color=self.color) if b_vals else None

    #     if not with_move_anim:
    #         return A, B

    #     # —— 为 A、B 安排位置（A 在原矩阵下方，B 在 A 右侧；若 A 为空则 B 放在原矩阵下方）——
    #     if A is not None:
    #         A.next_to(self, DOWN, buff=place_A_below_buff)
    #     if B is not None:
    #         if A is not None:
    #             B.next_to(A, RIGHT, buff=place_B_right_buff)
    #         else:
    #             B.next_to(self, DOWN, buff=place_A_below_buff)

    #     # —— 生成把“原矩阵中的行/列”移动到 A/B 对应槽位的动画 —— 
    #     move_anims = []
    #     if axis == 'row':  # 行主，逐行对齐
    #         if A is not None:
    #             for k, i in enumerate(sel):
    #                 move_anims.append(self.vectors[i].animate.move_to(A.vectors[k].get_center()))
    #         if B is not None:
    #             for k, i in enumerate(rest):
    #                 move_anims.append(self.vectors[i].animate.move_to(B.vectors[k].get_center()))
    #     else:  # axis == 'col'（列主，逐列对齐）
    #         if A is not None:
    #             for k, i in enumerate(sel):
    #                 move_anims.append(self.vectors[i].animate.move_to(A.vectors[k].get_center()))
    #         if B is not None:
    #             for k, i in enumerate(rest):
    #                 move_anims.append(self.vectors[i].animate.move_to(B.vectors[k].get_center()))

    #     move_anim = AnimationGroup(*move_anims, lag_ratio=0.1)
    #     return A, B, move_anim

    # # ---- 根据索引拆分为两个新矩阵 ----
    # def split_by_indices(self, indices: List[int], axis: Literal['row','col'] = 'row'):
    #     """
    #     indices 归为 A，其余归为 B；返回 (A, B) 两个 Matrix2D（全新对象，数据拷贝重建）。
    #     当 major='row' 且 axis='row'：按行拆；
    #     当 major='col' 且 axis='col'：按列拆。
    #     其他组合暂不直接支持（可先转置或用相同 major 的矩阵）。
    #     """
    #     assert axis in ('row','col')
    #     assert axis == self.major, 'axis 必须与当前矩阵 major 相同，便于按 Vector1D 维度拆分'
    #     n = len(self.vectors)
    #     sel = sorted(set(i for i in indices if 0 <= i < n))
    #     rest = [i for i in range(n) if i not in sel]
    #     values = self.get_values_2d()
    #     a_vals = [values[i] for i in sel]
    #     b_vals = [values[i] for i in rest]
    #     A = Matrix2D(a_vals, major=self.major, cell_size=self.cell_size, hgap=self.hgap, vgap=self.vgap, color=self.color) if a_vals else None
    #     B = Matrix2D(b_vals, major=self.major, cell_size=self.cell_size, hgap=self.hgap, vgap=self.vgap, color=self.color) if b_vals else None
    #     return A, B


# =====================
# 演示场景：展示 row/col 排布与三种触发方式
# =====================
class Vector1DDemo(Scene):
    def construct(self):
        init_vals = [3.14, -1.0, 0.25, 7.8, 2.0]

        # 行向量
        row_vec = Vector1D(init_vals, orientation="row", cell_size=1.0, gap=0.12, color=BLUE)
        row_vec.to_edge(UP).shift(DOWN * 0.5)
        self.play(FadeIn(row_vec))
        self.wait(0.3)

        # 列向量
        col_vec = Vector1D(init_vals[:3], orientation="col", cell_size=0.9, gap=0.12, color=GREEN)
        col_vec.next_to(row_vec, DOWN, buff=1.0)
        self.play(FadeIn(col_vec))
        self.wait(0.3)

        # 覆盖数值（无动画）
        new_vals_row = [4.0, -2.5, 0.0, 6.66, -3.3]
        row_vec.set_values(new_vals_row)
        self.wait(0.3)

        # 1) 同时更新（真正并行）
        next_vals_row = [8.0, -1.5, 2.25, 1.23, 0.5]
        self.play(row_vec.get_set_values_animation(next_vals_row, color=YELLOW, run_time_each=0.5))
        self.wait(0.3)

        # 2) 交错启动（LaggedStart）
        next_vals_col = [-0.5, 9.9, -7.7]
        col_anims = [anim_square_value_change(cell, v, color=RED, run_time=0.6) for cell, v in zip(col_vec.cells, next_vals_col)]
        self.play(LaggedStart(*col_anims, lag_ratio=0.2))
        self.wait(0.3)

        # 3) 串行（Succession）
        seq_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        seq_anims = [anim_square_value_change(cell, v, color=BLUE, run_time=0.4) for cell, v in zip(row_vec.cells, seq_vals)]
        self.play(Succession(*seq_anims))
        self.wait(0.6)


# =====================
# Matrix 演示
# =====================
class MatrixDemo(Scene):
    def construct(self):
        # 创建一个 3x4 行主矩阵
        mat_vals = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [-1.5, 0.0, 2.25, -3.14],
        ]
        M = Matrix2D(mat_vals, major='row', cell_size=0.8, hgap=0.12, vgap=0.12, color=BLUE)
        M.to_edge(UP + LEFT)
        self.play(FadeIn(M))
        self.wait(0.3)

        # 功能 1：对某一行设置颜色
        M.set_row_color(1, border_color=GREEN, text_color=GREEN, fill_opacity=0.1)
        self.wait(0.3)

        # 覆盖矩阵数值（动画，逐行交错）
        # new_vals = [
        #     [1.1, 2.2, 3.3, 4.4],
        #     [5.5, 6.6, 7.7, 8.8],
        #     [-2.0, 0.5, 2.5, -3.0],
        # ]
        # self.play(M.get_set_values_animation(new_vals, color=YELLOW, run_time_each=0.4, axis='row', lag_ratio_outer=0.2))
        # self.wait(0.3)

        # 功能 2：按行索引拆分为两个新矩阵（A: 选 [0,2]；B: 其余）
        A, B = M.split_by_indices([0, 2], axis='row')
        if A is not None:
            A.next_to(M, DOWN, buff=0.6)
        if B is not None:
            if A is not None:
                B.next_to(A, RIGHT, buff=0.8)
            else:
                B.next_to(M, DOWN, buff=0.6)
        # self.play(*(FadeIn(x) for x in [a for a in [A, B] if a is not None]))
        # self.wait(0.6)

        # —— 带“把原矩阵行移动到新矩阵”的拆分动画演示 ——
        # 仍按行索引 [0, 2] 拆成 A/B，但这次让原矩阵中的行移动到新矩阵位置
        # A2, B2, move_anim = M.split_by_indices([0, 2], axis='row', with_move_anim=True)

        # # 先把 A2/B2 放上舞台（作为定位与最终落位的可视参考）
        # self.add(*(x for x in [A2, B2] if x is not None))

        # # 播放移动动画：原矩阵被选中的行/剩余的行，分别移动到 A2 / B2 的对应槽位
        # self.play(move_anim)
        # self.wait(0.6)

        # —— 使用 transfer=True 的拆分演示（不会在原矩阵留下残影） ——
        A2, B2, transfer_anim = M.split_by_indices([0, 2], axis='row', transfer=True)
        self.add(*(x for x in [A2, B2] if x is not None))  # 先把 A2/B2 放上舞台作为落位模板
        self.play(transfer_anim)
        self.wait(0.6)


