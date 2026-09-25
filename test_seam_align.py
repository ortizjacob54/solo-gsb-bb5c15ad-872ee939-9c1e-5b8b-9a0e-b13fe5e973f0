import itertools
import random
import unittest
from fractions import Fraction

from seam_align import MOVES, solve_seam

MOVE_INDEX = {move: i for i, move in enumerate(MOVES)}


def brute_segment(da, db):
    """暴力枚举单区间所有步进序列，返回 (最小代价, 字典序最小序列) 或 None。"""
    best = None

    def dfs(u, v, seq, cost):
        nonlocal best
        if u == da and v == db:
            if best is None or (cost, seq) < best:
                best = (Fraction(cost), seq)
            return
        for k, (ma, mb) in enumerate(MOVES):
            u2, v2 = u + ma, v + mb
            if u2 > da or v2 > db:
                continue
            extra = Fraction(0)
            if (u2, v2) != (da, db):  # 区间端点比例差为 0，不计
                diff = Fraction(u2, da) - Fraction(v2, db)
                extra = diff * diff
            dfs(u2, v2, seq + (k,), cost + extra)

    dfs(0, 0, (), Fraction(0))
    return best


def step_sequence(result):
    """从求解结果的 steps 中取出整条步进序列（用 MOVES 下标表示）。"""
    return tuple(MOVE_INDEX[tuple(step['move'])] for step in result['steps'])


def assert_ranges_contiguous(self, result):
    """逐步行号范围必须连续覆盖 1..A 与 1..B，且每步宽度为步进值。"""
    a = b = 0
    for step in result['steps']:
        ma, mb = step['move']
        self.assertEqual(step['a_from'], a + 1)
        self.assertEqual(step['a_to'], a + ma)
        self.assertEqual(step['b_from'], b + 1)
        self.assertEqual(step['b_to'], b + mb)
        self.assertEqual(step['a_to'] - step['a_from'] + 1, ma)
        self.assertEqual(step['b_to'] - step['b_from'] + 1, mb)
        a, b = a + ma, b + mb
    self.assertEqual((a, b), (result['rows_a'], result['rows_b']))


class TestSolveSeam(unittest.TestCase):
    def test_example_segment(self):
        # A=10 B=8 标记 (4,5)：第一段 (1,1)(1,1)(1,2)(1,1) 代价 3/200，
        # 第二段只能三个 (2,1)，代价 0
        result = solve_seam(10, 8, [(4, 5)])
        self.assertTrue(result['ok'])
        self.assertEqual(result['step_count'], 7)
        self.assertEqual(result['cost'], Fraction(3, 200))
        self.assertEqual(
            step_sequence(result),
            (0, 0, 1, 0, 2, 2, 2))
        seg_costs = [s['cost'] for s in result['segments']]
        self.assertEqual(seg_costs, [Fraction(3, 200), Fraction(0)])
        # 第一段 3 步、第二段 3 步的步号范围
        self.assertEqual(
            [(s['step_from'], s['step_to']) for s in result['segments']],
            [(1, 4), (5, 7)])

    def test_anchors_start_marker_end(self):
        result = solve_seam(10, 8, [(4, 5)])
        self.assertEqual(
            [(a['a'], a['b'], a['kind']) for a in result['anchors']],
            [(0, 0, 'start'), (4, 5, 'marker'), (10, 8, 'end')])
        # 第 4 步落在标记（锚点 1）、第 7 步落在终点（锚点 2）
        self.assertEqual(
            [step['anchor'] for step in result['steps']],
            [None, None, None, 1, None, None, 2])

    def test_step_ranges_example(self):
        result = solve_seam(6, 4, [])
        self.assertTrue(result['ok'])
        assert_ranges_contiguous(self, result)
        # 6 行对 4 行：字典序最小的最优序列 (1,1)(2,1)(1,1)(2,1)
        self.assertEqual(step_sequence(result), (0, 2, 0, 2))
        self.assertEqual(result['cost'], Fraction(1, 72))

    def test_equal_rows_all_diagonal(self):
        result = solve_seam(7, 7, [])
        self.assertTrue(result['ok'])
        self.assertEqual(step_sequence(result), (0,) * 7)
        self.assertEqual(result['cost'], 0)
        assert_ranges_contiguous(self, result)

    def test_single_step(self):
        self.assertEqual(step_sequence(solve_seam(1, 1, [])), (0,))
        self.assertEqual(step_sequence(solve_seam(1, 2, [])), (1,))
        self.assertEqual(step_sequence(solve_seam(2, 1, [])), (2,))

    def test_infeasible_whole_edge(self):
        # 1 行对 3 行：比例超过 2:1，没有任何步进序列
        result = solve_seam(1, 3, [])
        self.assertFalse(result['ok'])
        self.assertEqual(result['fail_segment'], 1)
        self.assertEqual(result['segments'][0]['feasible'], False)
        self.assertNotIn('steps', result)

    def test_infeasible_first_segment_is_reported(self):
        # 标记 (2,7)：第 1 段 2 对 7 不可行，第 2 段 8 对 1 也不可行，报第 1 段
        result = solve_seam(10, 8, [(2, 7)])
        self.assertFalse(result['ok'])
        self.assertEqual(result['fail_segment'], 1)
        seg = result['segments'][0]
        self.assertEqual((seg['a_from'], seg['a_to']), (0, 2))
        self.assertEqual((seg['b_from'], seg['b_to']), (0, 7))
        self.assertFalse(any(s['feasible'] for s in result['segments']))

    def test_infeasible_later_segment(self):
        # 标记 (8,1)：第 1 段 8 对 1 不可行、第 2 段 2 对 7 不可行，报第 1 段；
        # 换成 (1,1) 后第 1 段可行，第 2 段 9 对 7 可行（比例 ≤2:1）
        result = solve_seam(10, 8, [(8, 1)])
        self.assertFalse(result['ok'])
        self.assertEqual(result['fail_segment'], 1)
        result2 = solve_seam(10, 8, [(1, 1)])
        self.assertTrue(result2['ok'])

    def test_feasibility_boundary_two_to_one(self):
        # 恰好 2:1 可行：只能全走 (2,1)（或 (1,2)）
        result = solve_seam(6, 3, [])
        self.assertTrue(result['ok'])
        self.assertEqual(step_sequence(result), (2, 2, 2))
        result = solve_seam(3, 6, [])
        self.assertTrue(result['ok'])
        self.assertEqual(step_sequence(result), (1, 1, 1))

    def test_segment_costs_sum_to_total(self):
        result = solve_seam(10, 8, [(4, 5)])
        self.assertEqual(
            result['cost'], sum(s['cost'] for s in result['segments']))

    def test_random_against_brute_force_single(self):
        # 无标记单区间：代价与整条步进序列都必须与暴力枚举一致
        rng = random.Random(20260925)
        for _ in range(200):
            rows_a = rng.randint(1, 8)
            rows_b = rng.randint(1, 8)
            result = solve_seam(rows_a, rows_b, [])
            best = brute_segment(rows_a, rows_b)
            msg = 'rows_a={} rows_b={}'.format(rows_a, rows_b)
            if best is None:
                self.assertFalse(result['ok'], msg=msg)
            else:
                self.assertTrue(result['ok'], msg=msg)
                self.assertEqual(result['cost'], best[0], msg=msg)
                self.assertEqual(step_sequence(result), best[1], msg=msg)
                assert_ranges_contiguous(self, result)

    def test_random_against_brute_force_markers(self):
        # 带标记：各区间独立取最优后拼接，与逐区间暴力结果一致；
        # 任一区间不可行时整题无解且不返回 steps
        rng = random.Random(99173)
        for _ in range(200):
            rows_a = rng.randint(2, 10)
            rows_b = rng.randint(2, 10)
            count = rng.randint(0, 2)
            if count:
                aa = sorted(rng.sample(range(1, rows_a),
                                       k=min(count, rows_a - 1)))
                bb = sorted(rng.sample(range(1, rows_b),
                                       k=min(len(aa), rows_b - 1)))
                markers = list(zip(aa[:len(bb)], bb))
            else:
                markers = []
            result = solve_seam(rows_a, rows_b, markers)
            points = [(0, 0)] + markers + [(rows_a, rows_b)]
            total = Fraction(0)
            seq = ()
            feasible = True
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                best = brute_segment(x1 - x0, y1 - y0)
                if best is None:
                    feasible = False
                    break
                total += best[0]
                seq += best[1]
            msg = 'A={} B={} markers={}'.format(rows_a, rows_b, markers)
            if not feasible:
                self.assertFalse(result['ok'], msg=msg)
                self.assertNotIn('steps', result, msg=msg)
                continue
            self.assertTrue(result['ok'], msg=msg)
            self.assertEqual(result['cost'], total, msg=msg)
            self.assertEqual(step_sequence(result), seq, msg=msg)
            assert_ranges_contiguous(self, result)

    def test_max_size_runs(self):
        # 60 对 60 为最大输入，瞬间完成且全程对角线
        result = solve_seam(60, 60, [])
        self.assertTrue(result['ok'])
        self.assertEqual(result['step_count'], 60)
        self.assertEqual(step_sequence(result), (0,) * 60)


if __name__ == '__main__':
    unittest.main()
