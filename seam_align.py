"""缝合对位：把两片织物的边缘按每次 (1,1)、(1,2) 或 (2,1) 的步进缝到一起。

模型（代价与比较全程用 Fraction 精确计算，不用浮点）：
  A 侧共 A 行、B 侧共 B 行，起点 (0,0) 与终点 (A,B) 自动视为对位标记；
  另可给出若干标记对 (a_i, b_i)，在两侧都严格递增且位于起终点之间。每次
  缝合让两侧累计行号前进 (1,1)、(1,2) 或 (2,1)：分别表示 1 行对 1 行、
  1 行对 2 行、2 行对 1 行。路径必须恰好经过全部标记并在终点结束。
  标记把整条路径分成独立区间：第 i 段从 (a_{i-1}, b_{i-1}) 到
  (a_i, b_i)，记 da_i = a_i−a_{i-1}、db_i = b_i−b_{i-1}。段内每个落点
  (u, v)（相对段首）两侧累计进度比例分别为 u/da_i 与 v/db_i，该段代价为
  各内部落点 (u/da_i − v/db_i)² 之和（段首段尾比例差为 0，不计无影响），
  总目标为各段代价之和最小。段间没有耦合，故各段独立取最优后拼接即全局
  最优；同分按整条步进序列的字典序取最小（顺序 (1,1)＜(1,2)＜(2,1)），
  每段正向回溯时按此顺序取第一个能达成最优的步进即可。

单段可行性：n 次缝合中 (1,1)/(1,2)/(2,1) 的次数 n11/n12/n21 满足
  n11 = 3n − da − db, n12 = db − n, n21 = da − n，均须非负，即
  ceil((da+db)/3) ≤ n ≤ min(da, db)。该整数区间非空当且仅当
  max(da, db) ≤ 2·min(da, db)（两侧行数比例不超过 2:1）。不可行时
  指出首个无法衔接的区间，且不构造任何步进（不返回部分路线）。

求解用反向动态规划：g[u][v] 为段内从相对点 (u,v) 走到段尾的最小剩余
平方差和，g[da][db] = 0，自右上向左下递推后正向按步进固定顺序回溯。
A、B 各至多 60 行，网格至多 61×61，分数运算规模很小。
"""

from fractions import Fraction

# 同分字典序：(1,1) 最先、(1,2) 次之、(2,1) 最后
MOVES = ((1, 1), (1, 2), (2, 1))


def _segment_feasible(da, db):
    """该区间行数差能否只靠 (1,1)/(1,2)/(2,1) 走完（da, db ≥ 1）。"""
    return max(da, db) <= 2 * min(da, db)


def _solve_segment(da, db):
    """求单个区间 (0,0)→(da,db) 的最优步进序列。

    返回 (moves, cost)：moves 为 MOVES 中步进的序列，cost 为精确 Fraction。
    不可行时返回 (None, None)。
    """
    if not _segment_feasible(da, db):
        return None, None

    # 反向 DP：g[u][v] 为从 (u,v) 到 (da,db) 的最小剩余平方差和（分数）
    g = [[None for _v in range(db + 1)] for _u in range(da + 1)]
    g[da][db] = Fraction(0)
    for u in range(da, -1, -1):
        for v in range(db, -1, -1):
            if u == da and v == db:
                continue
            best = None
            for ma, mb in MOVES:
                u2, v2 = u + ma, v + mb
                if u2 > da or v2 > db:
                    continue
                tail = g[u2][v2]
                if tail is None:
                    continue
                # 落点 (u2,v2) 的段内进度比例差平方（段尾为 1−1=0）
                diff = Fraction(u2, da) - Fraction(v2, db)
                value = diff * diff + tail
                if best is None or value < best:
                    best = value
            g[u][v] = best

    # 正向按固定步进顺序回溯：首位能达成最优的步进即字典序最小
    moves = []
    u = v = 0
    while (u, v) != (da, db):
        for ma, mb in MOVES:
            u2, v2 = u + ma, v + mb
            if u2 > da or v2 > db or g[u2][v2] is None:
                continue
            diff = Fraction(u2, da) - Fraction(v2, db)
            if diff * diff + g[u2][v2] == g[u][v]:
                moves.append((ma, mb))
                u, v = u2, v2
                break
        else:  # pragma: no cover - g[0][0] 可行时理论上不可能发生
            raise RuntimeError("缝合对位回溯失败")
    return moves, g[0][0]


def solve_seam(rows_a, rows_b, markers):
    """求解缝合对位。

    参数（调用方已完成校验）：
      rows_a   A 侧行数，int，1~60
      rows_b   B 侧行数，int，1~60
      markers  标记对列表 [(a, b), ...]：1 ≤ a ≤ rows_a−1、
               1 ≤ b ≤ rows_b−1，且两侧均严格递增

    可行时返回：
      {"ok": True, "rows_a", "rows_b", "step_count", "cost",
       "anchors": [{"index", "a", "b", "kind"}, ...],
       "segments": [{"index", "from_anchor", "to_anchor",
                     "a_from", "a_to", "b_from", "b_to",
                     "step_from", "step_to", "cost",
                     "moves": [(ma, mb), ...]}, ...],
       "steps": [{"index", "move", "a_from", "a_to",
                  "b_from", "b_to", "anchor"}, ...]}
      cost 为各段进度比例差平方和之和（精确 Fraction）。anchors 按顺序为
      起点（index=0, kind="start"）、各用户标记（kind="marker"）、终点
      （kind="end"）；steps[].anchor 为该步落点对应的锚点下标，未落在
      锚点上为 None；a_from/a_to、b_from/b_to 均为 1 起的实际配对行号。
    无解时返回：
      {"ok": False, "fail_segment",
       "segments": [{"index", "from_anchor", "to_anchor",
                     "a_from", "a_to", "b_from", "b_to",
                     "feasible"}, ...]}
      fail_segment 为首个（下标 1 起）无法衔接的区间；返回全部区间的
      行数边界与可行性供诊断，但不含任何 steps（不返回部分路线）。
    """
    anchors = [(0, 0)] + list(markers) + [(rows_a, rows_b)]
    anchor_kinds = ["start"] + ["marker"] * len(markers) + ["end"]

    segment_bounds = []
    for i in range(len(anchors) - 1):
        a0, b0 = anchors[i]
        a1, b1 = anchors[i + 1]
        segment_bounds.append({
            "index": i + 1,
            "from_anchor": i,
            "to_anchor": i + 1,
            "a_from": a0,
            "a_to": a1,
            "b_from": b0,
            "b_to": b1,
            "da": a1 - a0,
            "db": b1 - b0,
        })

    fail_segment = None
    segment_plans = []
    for seg in segment_bounds:
        moves, cost = _solve_segment(seg["da"], seg["db"])
        if moves is None and fail_segment is None:
            fail_segment = seg["index"]
        segment_plans.append((moves, cost))

    if fail_segment is not None:
        # 不构造任何步进：只返回各区间边界与可行性诊断
        return {
            "ok": False,
            "rows_a": rows_a,
            "rows_b": rows_b,
            "fail_segment": fail_segment,
            "segments": [{
                "index": seg["index"],
                "from_anchor": seg["from_anchor"],
                "to_anchor": seg["to_anchor"],
                "a_from": seg["a_from"],
                "a_to": seg["a_to"],
                "b_from": seg["b_from"],
                "b_to": seg["b_to"],
                "feasible": plan[0] is not None,
            } for seg, plan in zip(segment_bounds, segment_plans)],
        }

    anchors_out = [{
        "index": i,
        "a": a,
        "b": b,
        "kind": anchor_kinds[i],
    } for i, (a, b) in enumerate(anchors)]

    steps = []
    segments_out = []
    total_cost = Fraction(0)
    step_index = 0
    x = y = 0                       # 两侧当前累计行号
    for seg, (moves, cost) in zip(segment_bounds, segment_plans):
        step_from = step_index + 1
        for ma, mb in moves:
            step_index += 1
            x2, y2 = x + ma, y + mb
            anchor = None
            if (x2, y2) == (seg["a_to"], seg["b_to"]):
                anchor = seg["to_anchor"]     # 区间端点即锚点（标记或终点）
            steps.append({
                "index": step_index,
                "move": [ma, mb],
                "a_from": x + 1,
                "a_to": x2,
                "b_from": y + 1,
                "b_to": y2,
                "anchor": anchor,
            })
            x, y = x2, y2
        total_cost += cost
        segments_out.append({
            "index": seg["index"],
            "from_anchor": seg["from_anchor"],
            "to_anchor": seg["to_anchor"],
            "a_from": seg["a_from"],
            "b_from": seg["b_from"],
            "a_to": seg["a_to"],
            "b_to": seg["b_to"],
            "step_from": step_from,
            "step_to": step_index,
            "cost": cost,
            "moves": [list(m) for m in moves],
        })

    return {
        "ok": True,
        "rows_a": rows_a,
        "rows_b": rows_b,
        "step_count": step_index,
        "cost": total_cost,
        "anchors": anchors_out,
        "segments": segments_out,
        "steps": steps,
    }
