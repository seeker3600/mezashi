from __future__ import annotations

import math
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely import affinity

import medetect.datagen.placement as placement_mod
from medetect.datagen.ship import MIN_SHIP_BEAM_PX
from shapely.geometry import box

from medetect.datagen.placement import (
    _geometry_projection_extents,
    _place_cluster,
    _stamp_occupancy,
    find_water_position,
)


class TestFindWaterPosition:
    def test_all_water_finds_position(self) -> None:
        """全面水域なら位置が見つかる。"""
        mask = np.ones((100, 100), dtype=bool)
        rng = random.Random(42)
        pos = find_water_position(mask, ship_w=10, ship_h=20, angle_rad=0.0, rng=rng)
        assert pos is not None
        cx, cy = pos
        assert 0 <= cx < 100
        assert 0 <= cy < 100

    def test_all_land_returns_none(self) -> None:
        """全面陸地なら None を返す。"""
        mask = np.zeros((100, 100), dtype=bool)
        rng = random.Random(42)
        pos = find_water_position(mask, ship_w=10, ship_h=20, angle_rad=0.0, rng=rng)
        assert pos is None

    def test_small_water_region_avoided(self) -> None:
        """小さな水域にはサイズの大きい船は置けない。"""
        mask = np.zeros((100, 100), dtype=bool)
        mask[48:52, 48:52] = True
        rng = random.Random(42)
        pos = find_water_position(mask, ship_w=20, ship_h=40, angle_rad=0.0, rng=rng)
        assert pos is None

    def test_occupied_area_avoided(self) -> None:
        """占有済みエリアには配置されない。"""
        mask = np.ones((100, 100), dtype=bool)
        mask[10:90, 10:90] = False
        rng = random.Random(42)
        pos = find_water_position(mask, ship_w=5, ship_h=5, angle_rad=0.0, rng=rng)
        assert pos is not None


class TestStampOccupancy:
    def test_marks_center_occupied(self) -> None:
        """船の中心が占有済みになる。"""
        occupancy = np.zeros((100, 100), dtype=bool)
        _stamp_occupancy(occupancy, cx=50, cy=50, w=10, h=20, angle_rad=0.0)
        assert occupancy[50, 50]

    def test_corners_outside_are_free(self) -> None:
        """小さい船をスタンプしても遠い角は未占有のまま。"""
        occupancy = np.zeros((100, 100), dtype=bool)
        _stamp_occupancy(occupancy, cx=50, cy=50, w=10, h=20, angle_rad=0.0)
        assert not occupancy[0, 0]

    def test_prevents_second_placement(self) -> None:
        """スタンプ後の占有マスクで同位置への再配置ができない。"""
        water = np.ones((100, 100), dtype=bool)
        occupancy = np.zeros((100, 100), dtype=bool)
        _stamp_occupancy(occupancy, cx=50, cy=50, w=30, h=60, angle_rad=0.0)
        available = water & ~occupancy
        rng = random.Random(42)
        pos = find_water_position(available, ship_w=30, ship_h=60, angle_rad=0.0, rng=rng)
        if pos is not None:
            cx, cy = pos
            assert not (40 <= cx <= 60 and 30 <= cy <= 70)


class TestOverlapPrevention:
    def test_find_water_position_avoids_occupied(self) -> None:
        """占有マスクが考慮されて既存船との重複を避ける。"""
        water = np.ones((200, 200), dtype=bool)
        occupancy = np.zeros((200, 200), dtype=bool)
        _stamp_occupancy(occupancy, cx=100, cy=100, w=60, h=120, angle_rad=0.0)
        available = water & ~occupancy
        rng = random.Random(0)
        positions = []
        for _ in range(50):
            pos = find_water_position(
                available,
                ship_w=10,
                ship_h=20,
                angle_rad=0.0,
                rng=rng,
            )
            if pos is not None:
                positions.append(pos)
        for cx, cy in positions:
            assert not (70 <= cx <= 130 and 40 <= cy <= 160)


def _obb_area(label: str, img_size: int = 200) -> float:
    """YOLO OBB ラベル文字列からピクセル面積を計算する。"""
    parts = label.split()
    coords = [float(value) * img_size for value in parts[1:]]
    x1, y1, x2, y2, x3, y3, x4, y4 = coords
    return 0.5 * abs(
        (x1 * y2 - x2 * y1)
        + (x2 * y3 - x3 * y2)
        + (x3 * y4 - x4 * y3)
        + (x4 * y1 - x1 * y4)
    )


def _parse_obb_polygon(label: str, img_size: int) -> list[tuple[float, float]]:
    """YOLO OBB ラベルから4頂点のポリゴンを返す。"""
    parts = label.split()
    coords = [float(value) * img_size for value in parts[1:]]
    return [(coords[index], coords[index + 1]) for index in range(0, 8, 2)]


def _cross_2d(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    area = 0.0
    for index in range(len(poly)):
        next_index = (index + 1) % len(poly)
        area += poly[index][0] * poly[next_index][1] - poly[next_index][0] * poly[index][1]
    return abs(area) / 2.0


def _segment_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """2本の線分の交点を返す。"""
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p4[0] - p3[0], p4[1] - p3[1])
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0] + t * d1[0], p1[1] + t * d1[1])
    return None


def _polygon_intersection(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman による凸ポリゴンのクリッピング。"""
    output = list(poly_a)
    for index in range(len(poly_b)):
        if not output:
            return []
        edge_start = poly_b[index]
        edge_end = poly_b[(index + 1) % len(poly_b)]
        current = output
        output = []
        for point_index in range(len(current)):
            curr = current[point_index]
            prev = current[point_index - 1]
            curr_inside = _cross_2d(edge_start, edge_end, curr) >= 0
            prev_inside = _cross_2d(edge_start, edge_end, prev) >= 0
            if curr_inside:
                if not prev_inside:
                    intersect = _segment_intersect(prev, curr, edge_start, edge_end)
                    if intersect:
                        output.append(intersect)
                output.append(curr)
            elif prev_inside:
                intersect = _segment_intersect(prev, curr, edge_start, edge_end)
                if intersect:
                    output.append(intersect)
    return output


def _polygon_iou(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> float:
    """2つの凸ポリゴンの IoU を計算する。"""
    inter = _polygon_intersection(poly_a, poly_b)
    if len(inter) < 3:
        return 0.0
    inter_area = _polygon_area(inter)
    area_a = _polygon_area(poly_a)
    area_b = _polygon_area(poly_b)
    union = area_a + area_b - inter_area
    if union < 1e-12:
        return 0.0
    return inter_area / union


def _point_to_segment_dist(
    pt: tuple[float, float],
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> float:
    """点から線分までの最短距離。"""
    dx, dy = seg_b[0] - seg_a[0], seg_b[1] - seg_a[1]
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(pt[0] - seg_a[0], pt[1] - seg_a[1])
    t = max(0.0, min(1.0, ((pt[0] - seg_a[0]) * dx + (pt[1] - seg_a[1]) * dy) / len_sq))
    proj_x = seg_a[0] + t * dx
    proj_y = seg_a[1] + t * dy
    return math.hypot(pt[0] - proj_x, pt[1] - proj_y)


def _polygon_min_distance(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> float:
    """2つの凸ポリゴンの最短距離。"""
    inter = _polygon_intersection(poly_a, poly_b)
    if len(inter) >= 3 and _polygon_area(inter) > 1e-6:
        return 0.0
    best = float("inf")
    for poly_x, poly_y in [(poly_a, poly_b), (poly_b, poly_a)]:
        for pt in poly_x:
            for index in range(len(poly_y)):
                dist = _point_to_segment_dist(pt, poly_y[index], poly_y[(index + 1) % len(poly_y)])
                if dist < best:
                    best = dist
    return best


def _count_connected_components(mask: np.ndarray, min_size: int = 1) -> int:
    """8近傍連結成分を数える。"""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            count = 0
            while stack:
                cx, cy = stack.pop()
                count += 1
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if count >= min_size:
                components += 1
    return components


class _ForcedLayoutRandom(random.Random):
    """特定の cluster layout を強制する Random 派生。"""

    def __init__(
        self,
        seed: int,
        layout: str,
        base_angle: float = 0.0,
        stagger: bool = False,
        zero_centered_uniform_max: float = 2.0,
    ) -> None:
        super().__init__(seed)
        self._layout = layout
        self._base_angle = base_angle
        self._stagger = stagger
        self._zero_centered_uniform_max = zero_centered_uniform_max
        self._uniform_calls = 0
        self._random_calls = 0

    def random(self):
        call_idx = self._random_calls
        self._random_calls += 1
        # raft_tight のみ: 3 番目の random() 呼び出しが tight_stagger_mode 判定
        # (call 0: randint 内部, call 1: mixed, call 2: tight_stagger_mode)
        # stagger=False → 1.0 を返して stagger を無効化 (RNG 状態を消費しない)
        # stagger=True  → 0.0 を返して stagger を有効化
        if self._layout == "flush" and call_idx == 2:
            return 0.0 if self._stagger else 1.0
        return super().random()

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        pop_list = list(population)
        layout_map = {
            "flush": "raft_tight",
            "partial": "raft_open",
        }
        if set(pop_list) == {"raft_tight", "raft_open"} and k == 1:
            return [layout_map.get(self._layout, self._layout)]
        return super().choices(population, weights=weights, cum_weights=cum_weights, k=k)

    def uniform(self, a, b):
        if self._uniform_calls == 0 and a == 0 and b == 360:
            self._uniform_calls += 1
            return self._base_angle
        if a < 0 < b and max(abs(a), abs(b)) <= self._zero_centered_uniform_max:
            self._uniform_calls += 1
            return 0.0
        self._uniform_calls += 1
        return super().uniform(a, b)


class _ForcedOpenGapRandom(_ForcedLayoutRandom):
    """旧 raft_open gap branch を最大化する向きの RNG。"""

    def random(self):
        if self._layout == "partial":
            self._random_calls += 1
            return 0.75
        return super().random()

    def uniform(self, a, b):
        if a == 0.0 and b == 1.0:
            self._uniform_calls += 1
            return 1.0
        return super().uniform(a, b)


def _make_tapered_hull_rgba(beam_px: int, length_px: int) -> np.ndarray:
    """tight placement 用の簡単な船体マスクを作る。"""
    img = Image.new("RGBA", (beam_px, length_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    tip = max(1, round(length_px * 0.16))
    half_x = (beam_px - 1) / 2.0
    points = [
        (half_x, 0),
        (beam_px - 1, tip),
        (beam_px - 1, max(tip + 1, length_px - tip - 1)),
        (half_x, length_px - 1),
        (0, max(tip + 1, length_px - tip - 1)),
        (0, tip),
    ]
    draw.polygon(points, fill=(220, 220, 220, 255))
    return np.array(img, dtype=np.uint8)


def _resolve_ship_dimensions_sequence_factory(sizes: list[tuple[int, int]]):
    """定義済みサイズ列を返す _resolve_ship_dimensions モックを作る。"""
    calls = {"count": 0}

    def _mock_resolve_ship_dimensions(
        svg_text: str,
        resolution_m: float,
        rng: random.Random,
        length_range: tuple[float, float] | None = None,
        length_exponent: float = 1.0,
    ) -> tuple[str, int, int, float]:
        index = min(calls["count"], len(sizes) - 1)
        beam_px, length_px = sizes[index]
        calls["count"] += 1
        lb_ratio = length_px / max(beam_px, 1)
        return "mock_hull", beam_px, length_px, lb_ratio

    return _mock_resolve_ship_dimensions


def _capture_vector_cluster(monkeypatch: pytest.MonkeyPatch) -> list:
    """vector cluster renderer に渡される船配置を捕捉する。"""
    captured: list = []

    def _mock_render_vector_cluster(ships, image_size, blur_sigma, scene_scale, **kwargs):
        captured[:] = list(ships)
        return np.zeros((image_size, image_size, 4), dtype=np.uint8)

    monkeypatch.setattr(placement_mod, "_render_vector_raft_cluster", _mock_render_vector_cluster)
    return captured


def _rect_local_hull_geometry(
    svg_text: str,
    bw: int,
    lh: int,
    angle_deg: float,
):
    """軸整列矩形 hull を ship local geometry として返す。"""
    del svg_text
    geometry = box(-bw / 2.0, -lh / 2.0, bw / 2.0, lh / 2.0)
    return affinity.rotate(geometry, angle_deg, origin=(0.0, 0.0), use_radians=False)


def _shore_x_at_y(
    points: list[tuple[float, float]],
    y: float,
) -> float:
    if y <= points[0][1]:
        return points[0][0]
    if y >= points[-1][1]:
        return points[-1][0]

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 <= y <= y1 or y1 <= y <= y0:
            if y1 == y0:
                return float(x1)
            t = (y - y0) / (y1 - y0)
            return float(x0 + (x1 - x0) * t)
    return float(points[-1][0])


def _count_land_boundary_samples(
    geometry,
    points: list[tuple[float, float]],
    samples: int = 96,
) -> int:
    hits = 0
    boundary = geometry.exterior
    for distance in np.linspace(0.0, float(boundary.length), samples, endpoint=False):
        sample = boundary.interpolate(float(distance))
        if sample.x < _shore_x_at_y(points, sample.y) - 0.25:
            hits += 1
    return hits


def _svg_sequence_factory(svg_texts: list[str]):
    """呼び出しごとに異なる SVG 文字列を返す _pick_svg モックを作る。"""
    calls = {"count": 0}

    def _mock_pick_svg(*args, **kwargs) -> str:
        del args, kwargs
        index = min(calls["count"], len(svg_texts) - 1)
        calls["count"] += 1
        return svg_texts[index]

    return _mock_pick_svg


class TestTightClusterBridgeGeometry:
    def test_bridge_geometry_only_fills_internal_gap(self) -> None:
        """tight cluster bridge は外周リングではなく内部ギャップだけを埋める。"""
        ships = [
            placement_mod._RaftShipPlacement(
                svg_text="",
                cx=13.0,
                cy=22.0,
                bw=6,
                lh=24,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=0,
                hull_geom=box(10.0, 10.0, 16.0, 34.0),
                hull_fill=(220, 40, 40, 255),
            ),
            placement_mod._RaftShipPlacement(
                svg_text="",
                cx=19.3,
                cy=22.0,
                bw=6,
                lh=24,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=0,
                hull_geom=box(16.6, 10.0, 22.6, 34.0),
                hull_fill=(40, 80, 220, 255),
            ),
        ]

        bridge = placement_mod._tight_cluster_bridge_geometry(ships, 0.75)

        assert bridge is not None
        min_x, min_y, max_x, max_y = bridge.bounds
        assert min_x >= 15.9
        assert max_x <= 16.7
        assert min_y >= 10.0
        assert max_y <= 34.0


class TestPlaceCluster:
    """_place_cluster の均一モード / 混合モードの動作検証。"""

    _IMAGE_SIZE = 200

    @pytest.fixture()
    def scene(self):
        """全面水域の 200x200 シーン。"""
        size = self._IMAGE_SIZE
        return {
            "water_mask": np.ones((size, size), dtype=bool),
            "occupancy": np.zeros((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
        }

    def test_uniform_cluster_produces_labels(self, scene) -> None:
        """均一クラスター (mixed_prob=0) がラベルを生成する。"""
        rng = random.Random(42)
        labels = _place_cluster(
            scene["water_mask"],
            scene["occupancy"],
            None,
            resolution_m=10.0,
            rng=rng,
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"],
            length_range=(50.0, 80.0),
            mixed_prob=0.0,
        )
        assert len(labels) > 0

    def test_mixed_cluster_produces_labels(self, scene) -> None:
        """混合クラスター (mixed_prob=1) がラベルを生成する。"""
        rng = random.Random(42)
        labels = _place_cluster(
            scene["water_mask"],
            scene["occupancy"],
            None,
            resolution_m=10.0,
            rng=rng,
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"],
            length_range=(50.0, 80.0),
            mixed_prob=1.0,
        )
        assert len(labels) > 0

    def test_uniform_cluster_ships_similar_size(self, scene) -> None:
        """均一クラスターでは各船の OBB 面積が大きく乖離しない。"""
        rng = random.Random(7)
        labels = _place_cluster(
            scene["water_mask"],
            scene["occupancy"],
            None,
            resolution_m=10.0,
            rng=rng,
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"],
            length_range=(70.0, 90.0),
            mixed_prob=0.0,
        )
        assert len(labels) >= 2
        areas = [_obb_area(label, self._IMAGE_SIZE) for label in labels]
        ratio = max(areas) / min(areas)
        assert ratio < 2.0

    def test_mixed_cluster_shows_size_variety_over_runs(self, scene) -> None:
        """混合クラスターを繰り返すと OBB 面積に広い分散が出る。"""
        all_areas: list[float] = []
        for seed in range(15):
            wm = scene["water_mask"].copy()
            oc = np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool)
            rng = random.Random(seed)
            labels = _place_cluster(
                wm,
                oc,
                None,
                resolution_m=5.0,
                rng=rng,
                cluster_size_range=(3, 3),
                blur_sigma=0.0,
                alpha_range=(0.8, 0.9),
                class_id=0,
                image_size=self._IMAGE_SIZE,
                background=scene["background"].copy(),
                length_range=None,
                mixed_prob=1.0,
            )
            all_areas.extend(_obb_area(label, self._IMAGE_SIZE) for label in labels)
        assert len(all_areas) >= 4
        ratio = max(all_areas) / min(all_areas)
        assert ratio > 2.0

    def test_no_significant_overlap(self, scene) -> None:
        """クラスター内の船同士が大きく重ならない。"""
        max_iou = 0.0
        for seed in range(30):
            wm = scene["water_mask"].copy()
            oc = np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool)
            rng = random.Random(seed)
            labels = _place_cluster(
                wm,
                oc,
                None,
                resolution_m=5.0,
                rng=rng,
                cluster_size_range=(4, 4),
                blur_sigma=0.0,
                alpha_range=(0.8, 0.9),
                class_id=0,
                image_size=self._IMAGE_SIZE,
                background=scene["background"].copy(),
                length_range=(30.0, 80.0),
                mixed_prob=0.5,
            )
            if len(labels) < 2:
                continue
            polys = [_parse_obb_polygon(label, self._IMAGE_SIZE) for label in labels]
            for first in range(len(polys)):
                for second in range(first + 1, len(polys)):
                    iou = _polygon_iou(polys[first], polys[second])
                    if iou > max_iou:
                        max_iou = iou
        assert max_iou < 0.35

    def test_gap_variety(self, scene) -> None:
        """contact-only cluster でも密着と軽い開きの両方が出現する。"""
        min_gaps: list[float] = []
        for seed in range(50):
            wm = scene["water_mask"].copy()
            oc = np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool)
            rng = random.Random(seed)
            labels = _place_cluster(
                wm,
                oc,
                None,
                resolution_m=5.0,
                rng=rng,
                cluster_size_range=(3, 3),
                blur_sigma=0.0,
                alpha_range=(0.8, 0.9),
                class_id=0,
                image_size=self._IMAGE_SIZE,
                background=scene["background"].copy(),
                length_range=(30.0, 80.0),
                mixed_prob=0.5,
            )
            if len(labels) < 2:
                continue
            polys = [_parse_obb_polygon(label, self._IMAGE_SIZE) for label in labels]
            for index in range(len(polys) - 1):
                gap = _polygon_min_distance(polys[index], polys[index + 1])
                min_gaps.append(gap)

        assert len(min_gaps) >= 10
        touching = sum(1 for gap in min_gaps if gap <= 0.2)
        slightly_open = sum(1 for gap in min_gaps if 0.2 < gap <= 1.5)
        assert touching > 0
        assert slightly_open > 0

    def test_uniform_tight_cluster_scaled_ship_respects_beam_floor(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """均一 tight クラスターの後続船も beam 下限を下回らない。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        captured = _capture_vector_cluster(monkeypatch)
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            lambda *args, **kwargs: ("mock_hull", 2, 18, 9.0),
        )

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=0.0,
        )

        assert len(labels) == 2
        assert len(captured) == 2
        assert captured[0].bw == 2
        assert captured[1].bw == MIN_SHIP_BEAM_PX

    def test_uniform_open_cluster_scaled_ship_respects_beam_floor(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """均一 raft_open クラスターの後続船も beam 下限を下回らない。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        captured = _capture_vector_cluster(monkeypatch)

        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            lambda *args, **kwargs: ("mock_hull", 2, 18, 9.0),
        )

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(11, "partial", base_angle=0.0),
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=0.0,
        )

        assert len(labels) == 2
        assert len(captured) == 2
        assert captured[0].bw == 2
        assert captured[1].bw == MIN_SHIP_BEAM_PX

    def test_uniform_open_cluster_reuses_reference_svg(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """均一 raft_open クラスターは先頭船の SVG を後続船でも再利用する。"""
        captured = _capture_vector_cluster(monkeypatch)
        svg_ref = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4" data-ship-class="ref">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        svg_other = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4" data-ship-class="other">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(160,160,160)"/>'
            '</svg>'
        )

        monkeypatch.setattr(
            placement_mod,
            "_pick_svg",
            _svg_sequence_factory([svg_ref, svg_other, svg_other]),
        )
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            lambda *args, **kwargs: ("mock_hull", 6, 24, 4.0),
        )

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(11, "partial", base_angle=0.0),
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=0.0,
        )

        assert len(labels) == 3
        assert [ship.svg_text for ship in captured] == [svg_ref, svg_ref, svg_ref]

    def test_uniform_tight_cluster_reuses_reference_svg(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """均一 tight クラスターは先頭船の SVG を後続船でも再利用する。"""
        captured = _capture_vector_cluster(monkeypatch)

        monkeypatch.setattr(
            placement_mod,
            "_pick_svg",
            _svg_sequence_factory(["svg-ref", "svg-second", "svg-third"]),
        )
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            lambda *args, **kwargs: ("mock_hull", 6, 24, 4.0),
        )
        monkeypatch.setattr(
            placement_mod,
            "_local_hull_geometry",
            lambda svg_text, bw, lh, angle_deg: box(0.0, 0.0, float(bw), float(lh)),
        )
        monkeypatch.setattr(placement_mod, "extract_hull_fill", lambda svg_text: (200, 200, 200, 255))

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=0.0,
        )

        assert len(labels) == 3
        assert len(captured) == 3
        assert [ship.svg_text for ship in captured] == ["svg-ref", "svg-ref", "svg-ref"]

    @pytest.mark.parametrize(
        ("sizes", "description"),
        [
            ([(4, 18), (4, 18), (5, 20)], "small-small"),
            ([(14, 72), (14, 72), (16, 80)], "large-large"),
            ([(4, 18), (4, 18), (14, 72)], "small-large"),
        ],
    )
    def test_tight_cluster_hulls_touch_without_deep_overlap(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
        sizes: list[tuple[int, int]],
        description: str,
    ) -> None:
        """tight クラスターで船腹が接触しつつ過度にめり込まない。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <defs>'
            '    <clipPath id="h">'
            '      <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1"/>'
            '    </clipPath>'
            '  </defs>'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)" stroke="rgb(20,20,20)"/>'
            '</svg>'
        )
        captured = _capture_vector_cluster(monkeypatch)
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory(sizes),
        )

        rng = _ForcedLayoutRandom(7, "flush", base_angle=0.5)
        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=rng,
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2
        assert len(captured) == 2

        gap = captured[0].hull_geom.distance(captured[1].hull_geom)
        overlap_area = captured[0].hull_geom.intersection(captured[1].hull_geom).area
        _min_a, max_a = _geometry_projection_extents(captured[0].hull_geom, 1.0, 0.0)
        min_b, _max_b = _geometry_projection_extents(captured[1].hull_geom, 1.0, 0.0)
        penetration_px = max_a - min_b

        assert gap <= 1e-6 or overlap_area > 0.0, description
        assert penetration_px <= 1.0, description

    @pytest.mark.parametrize(
        ("sizes", "description"),
        [
            ([(6, 24), (6, 24)], "equal-length"),
            ([(4, 18), (14, 72)], "mixed-length"),
        ],
    )
    def test_raft_open_contact_heavy_hulls_touch_with_mixed_lengths(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
        sizes: list[tuple[int, int]],
        description: str,
    ) -> None:
        """contact-heavy raft_open では長さ差があっても hull 接触を維持する。"""
        captured = _capture_vector_cluster(monkeypatch)
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: "mock-svg")
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory(sizes),
        )
        monkeypatch.setattr(
            placement_mod,
            "_local_hull_geometry",
            lambda svg_text, bw, lh, angle_deg: box(0.0, 0.0, float(bw), float(lh)),
        )
        monkeypatch.setattr(placement_mod, "extract_hull_fill", lambda svg_text: (200, 200, 200, 255))

        base_angle = 0.5
        rng = _ForcedLayoutRandom(
            7,
            "partial",
            base_angle=base_angle,
            zero_centered_uniform_max=20.0,
        )
        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=rng,
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2
        assert len(captured) == 2

        gap = captured[0].hull_geom.distance(captured[1].hull_geom)
        overlap_area = captured[0].hull_geom.intersection(captured[1].hull_geom).area
        axis_x = math.cos(math.radians(base_angle))
        axis_y = math.sin(math.radians(base_angle))
        _min_a, max_a = _geometry_projection_extents(captured[0].hull_geom, axis_x, axis_y)
        min_b, _max_b = _geometry_projection_extents(captured[1].hull_geom, axis_x, axis_y)
        penetration_px = max_a - min_b

        assert gap <= 1e-6 or overlap_area > 0.0, description
        assert penetration_px <= 1.0, description

    def test_raft_open_does_not_reintroduce_gap_after_contact(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """raft_open は接触候補を見つけた後に gap を入れ直さない。"""
        captured = _capture_vector_cluster(monkeypatch)
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: "mock-svg")
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24)]),
        )
        monkeypatch.setattr(
            placement_mod,
            "_local_hull_geometry",
            lambda svg_text, bw, lh, angle_deg: box(0.0, 0.0, float(bw), float(lh)),
        )
        monkeypatch.setattr(placement_mod, "extract_hull_fill", lambda svg_text: (200, 200, 200, 255))

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedOpenGapRandom(7, "partial", base_angle=0.0, zero_centered_uniform_max=20.0),
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2
        assert len(captured) == 2
        assert captured[0].hull_geom.distance(captured[1].hull_geom) <= 1e-6

    def test_partial_open_cluster_failure_returns_no_labels(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """raft_open cluster が 1 隻しか成立しない場合は ship_c を返さない。"""
        _capture_vector_cluster(monkeypatch)
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: "mock-svg")
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24)]),
        )
        monkeypatch.setattr(
            placement_mod,
            "_local_hull_geometry",
            lambda svg_text, bw, lh, angle_deg: box(0.0, 0.0, float(bw), float(lh)),
        )
        monkeypatch.setattr(placement_mod, "extract_hull_fill", lambda svg_text: (200, 200, 200, 255))

        call_count = {"count": 0}
        original_obb_on_water = placement_mod._obb_on_water

        def _fail_second_ship_water_check(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] >= 2:
                return False
            return original_obb_on_water(*args, **kwargs)

        monkeypatch.setattr(placement_mod, "_obb_on_water", _fail_second_ship_water_check)

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(7, "partial", base_angle=0.0),
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=0.0,
        )

        assert labels == []

    def test_tight_cluster_labels_keep_subpixel_offsets(self, scene, monkeypatch: pytest.MonkeyPatch) -> None:
        """tight クラスターのラベル座標がサブピクセル位置を保持する。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <defs>'
            '    <clipPath id="h">'
            '      <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1"/>'
            '    </clipPath>'
            '  </defs>'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)" stroke="rgb(20,20,20)"/>'
            '</svg>'
        )
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))

        def _mock_rasterize(svg_text, bw, lh, angle_deg=0.0, supersample=4, exclude_hull=False):
            if exclude_hull:
                return np.zeros((lh, bw, 4), dtype=np.uint8)
            return _make_tapered_hull_rgba(bw, lh)

        monkeypatch.setattr(placement_mod, "rasterize_ship_svg", _mock_rasterize)
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(4, 18), (4, 18)]),
        )

        rng = _ForcedLayoutRandom(7, "flush", base_angle=0.5)
        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=rng,
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2
        second_poly = _parse_obb_polygon(labels[1], self._IMAGE_SIZE)
        fractions = [abs(coord - round(coord)) for x, y in second_poly for coord in (x, y)]
        assert max(fractions) >= 0.05

    def test_tight_cluster_final_render_has_no_background_slit(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tight クラスターの最終描画が背景スリットで分断されない。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <defs>'
            '    <clipPath id="h">'
            '      <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1"/>'
            '    </clipPath>'
            '  </defs>'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)" stroke="rgb(20,20,20)"/>'
            '</svg>'
        )
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24)]),
        )

        background = scene["background"].copy()
        rng = _ForcedLayoutRandom(7, "flush", base_angle=0.0)
        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=rng,
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=background,
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2
        ship_mask = np.any(background != scene["background"], axis=2)
        assert _count_connected_components(ship_mask, min_size=20) == 1

    def test_tight_cluster_forwards_sampled_water_tint(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tight クラスターは sampled water tint を最終ブレンドに渡す。"""
        sampled_tint = np.array([12.0, 34.0, 56.0], dtype=np.float32)
        recorded_tints: list[np.ndarray | None] = []
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )

        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24)]),
        )
        monkeypatch.setattr(
            placement_mod,
            "_render_vector_raft_cluster",
            lambda *args, **kwargs: (
                None,
                placement_mod.RgbaLayerPatch(0, 0, np.zeros((10, 10, 4), dtype=np.uint8)),
            ),
        )
        monkeypatch.setattr(placement_mod, "_sample_water_tint", lambda *args, **kwargs: sampled_tint)

        def _mock_blend_rgba_patch(
            background: np.ndarray,
            patch: object,
            alpha_factor: float,
            water_tint: np.ndarray | None,
        ) -> None:
            del background, patch, alpha_factor
            recorded_tints.append(None if water_tint is None else water_tint.copy())

        monkeypatch.setattr(placement_mod, "_blend_rgba_patch", _mock_blend_rgba_patch)

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2
        assert len(recorded_tints) == 1
        assert recorded_tints[0] is not None
        np.testing.assert_array_equal(recorded_tints[0], sampled_tint)

    def test_tight_cluster_shares_shadow_length(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tight クラスター内では全船が同じ影長パラメータを使う。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        shadow_lengths: list[float] = []
        shadow_patch_biases: list[float] = []
        darken_factors: list[float] = []

        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24)]),
        )
        monkeypatch.setattr(
            placement_mod,
            "_shadow_offset_pixels",
            lambda beam_px, length_px, azimuth_rad, shadow_length, *, scene_scale=1: shadow_lengths.append(shadow_length) or (3, 1),
        )
        monkeypatch.setattr(placement_mod, "_shadow_blur_sigma", lambda *args, **kwargs: 1.0)
        monkeypatch.setattr(placement_mod, "_shadow_alpha_for_ship", lambda *args, **kwargs: 1.05)

        def _mock_make_shadow_rgba(
            ship_rgba: np.ndarray,
            *,
            offset_x: int,
            offset_y: int,
            blur_sigma: float,
            alpha_scale: float,
        ) -> np.ndarray:
            del offset_x, offset_y, blur_sigma
            shadow_patch_biases.append(alpha_scale)
            return np.zeros((ship_rgba.shape[0], ship_rgba.shape[1], 4), dtype=np.uint8)

        def _mock_darken_rgba_patch(
            background: np.ndarray,
            patch: object,
            alpha_factor: float,
            clip_mask: np.ndarray | None = None,
        ) -> None:
            del background, patch, clip_mask
            darken_factors.append(alpha_factor)

        monkeypatch.setattr(placement_mod, "_make_shadow_rgba", _mock_make_shadow_rgba)
        monkeypatch.setattr(placement_mod, "_darken_rgba_patch", _mock_darken_rgba_patch)

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
            shadow_azimuth_rad=math.pi / 6.0,
            shadow_length=2.0,
            shadow_alpha=0.12,
            shadow_alpha_scale=2.0,
        )

        assert len(labels) == 2
        assert len(shadow_lengths) == 2
        assert shadow_patch_biases == [pytest.approx(1.05), pytest.approx(1.05)]
        assert len({round(value, 6) for value in shadow_lengths}) == 1
        assert shadow_lengths[0] == pytest.approx(2.0)
        assert darken_factors == [pytest.approx(0.24)]

    def test_tight_cluster_downsamples_only_local_patch(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tight クラスターは全画面ではなく局所パッチだけを縮小する。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        recorded_shapes: list[tuple[int, int]] = []

        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24)]),
        )

        def _mock_downsample_cluster_patch(
            layer: np.ndarray,
            scene_x0: int,
            scene_y0: int,
            scene_scale: int,
        ):
            recorded_shapes.append(layer.shape[:2])
            return placement_mod.RgbaLayerPatch(
                scene_x0 // scene_scale,
                scene_y0 // scene_scale,
                np.zeros(
                    (layer.shape[0] // scene_scale, layer.shape[1] // scene_scale, 4),
                    dtype=np.uint8,
                ),
            )

        monkeypatch.setattr(placement_mod, "_downsample_cluster_patch", _mock_downsample_cluster_patch)

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2
        assert recorded_shapes
        full_scene_size = self._IMAGE_SIZE * placement_mod._CLUSTER_SCENE_SUPERSAMPLE
        assert all(height < full_scene_size for height, _width in recorded_shapes)
        assert all(width < full_scene_size for _height, width in recorded_shapes)


class TestBerthPlacement:
    """岸線に沿う berth 配置の姿勢を検証する。"""

    _IMAGE_SIZE = 200

    @pytest.fixture()
    def vertical_shore_scene(self):
        """左が陸、右が水の縦岸線シーン。"""
        size = self._IMAGE_SIZE
        water_mask = np.zeros((size, size), dtype=bool)
        water_mask[:, 40:] = True
        return {
            "water_mask": water_mask,
            "occupancy": np.zeros((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
        }

    @pytest.fixture()
    def horizontal_shore_scene(self):
        """上が陸、下が水の横岸線シーン。"""
        size = self._IMAGE_SIZE
        water_mask = np.zeros((size, size), dtype=bool)
        water_mask[40:, :] = True
        return {
            "water_mask": water_mask,
            "occupancy": np.zeros((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
        }

    @pytest.fixture()
    def curved_shore_scene(self):
        """緩く右へ曲がる岸線シーン。"""
        size = self._IMAGE_SIZE
        water_mask = np.zeros((size, size), dtype=bool)
        points = [(40.0, 20.0), (46.0, 100.0), (56.0, 180.0)]
        shore_x = np.full(size, points[0][0], dtype=float)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            y_start = max(0, int(math.floor(min(y0, y1))))
            y_stop = min(size - 1, int(math.ceil(max(y0, y1))))
            for y in range(y_start, y_stop + 1):
                if y1 == y0:
                    shore_x[y] = x1
                    continue
                t = (y - y0) / (y1 - y0)
                shore_x[y] = x0 + (x1 - x0) * t
        shore_x[: int(points[0][1])] = points[0][0]
        shore_x[int(points[-1][1]) :] = points[-1][0]
        for y, x in enumerate(shore_x):
            water_mask[y, max(0, int(math.ceil(x))) :] = True
        return {
            "water_mask": water_mask,
            "occupancy": np.zeros((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
            "segments": [
                ((40.0, 20.0), (46.0, 100.0)),
                ((46.0, 100.0), (56.0, 180.0)),
            ],
            "points": points,
        }

    @pytest.fixture()
    def bulged_shore_scene(self):
        """中間で海側へ張り出す岸線シーン。"""
        size = self._IMAGE_SIZE
        water_mask = np.zeros((size, size), dtype=bool)
        points = [(40.0, 20.0), (50.0, 100.0), (56.0, 180.0)]
        shore_x = np.full(size, points[0][0], dtype=float)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            y_start = max(0, int(math.floor(min(y0, y1))))
            y_stop = min(size - 1, int(math.ceil(max(y0, y1))))
            for y in range(y_start, y_stop + 1):
                if y1 == y0:
                    shore_x[y] = x1
                    continue
                t = (y - y0) / (y1 - y0)
                shore_x[y] = x0 + (x1 - x0) * t
        shore_x[: int(points[0][1])] = points[0][0]
        shore_x[int(points[-1][1]) :] = points[-1][0]
        for y, x in enumerate(shore_x):
            water_mask[y, max(0, int(math.ceil(x))) :] = True
        return {
            "water_mask": water_mask,
            "occupancy": np.zeros((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
            "segments": [
                ((40.0, 20.0), (50.0, 100.0)),
                ((50.0, 100.0), (56.0, 180.0)),
            ],
            "points": points,
        }

    @pytest.fixture()
    def dock_touch_scene(self):
        """dock 側 corner が少し陸へはみ出す berth シーン。"""
        size = self._IMAGE_SIZE
        water_mask = np.zeros((size, size), dtype=bool)
        water_mask[:, 44:] = True
        return {
            "water_mask": water_mask,
            "occupancy": np.zeros((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
        }

    def _setup_mocks(self, monkeypatch: pytest.MonkeyPatch) -> list:
        captured = _capture_vector_cluster(monkeypatch)
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: "mock-svg")
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(8, 32), (8, 32), (8, 32)]),
        )
        monkeypatch.setattr(placement_mod, "_local_hull_geometry", _rect_local_hull_geometry)
        monkeypatch.setattr(placement_mod, "extract_hull_fill", lambda svg_text: (200, 200, 200, 255))
        return captured

    def test_berth_alongside_cluster_aligns_with_vertical_shore(
        self,
        vertical_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """横づけ berth は lead ship を岸へ付け、残りが沖側へ tight に連なる。"""
        captured = self._setup_mocks(monkeypatch)

        labels = _place_cluster(
            vertical_shore_scene["water_mask"],
            vertical_shore_scene["occupancy"],
            None,
            resolution_m=5.0,
            rng=random.Random(3),
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=vertical_shore_scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
            berth_prob=1.0,
            berth_stern_prob=0.0,
            berth_water_mask=vertical_shore_scene["water_mask"],
            berth_segments=[((40.0, 20.0), (40.0, 180.0))],
        )

        assert len(labels) == 3
        assert len(captured) == 3

        xs = [ship.cx for ship in captured]
        ys = [ship.cy for ship in captured]
        assert xs == sorted(xs)
        assert max(xs) - min(xs) > 12.0
        assert max(ys) - min(ys) < 3.0

        tangent = np.array([0.0, 1.0])
        for ship in captured:
            stern_dir = np.array([-math.sin(ship.angle_rad), math.cos(ship.angle_rad)])
            assert abs(float(np.dot(stern_dir, tangent))) > 0.95

        shoremost = min(captured, key=lambda ship: ship.cx)
        min_x, _min_y, _max_x, _max_y = shoremost.hull_geom.bounds
        assert 39.0 <= min_x <= 45.0

    def test_berth_alongside_followers_stack_offshore(
        self,
        vertical_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """横づけ berth の follower は lead ship の沖側へ接触気味に積まれる。"""
        captured = self._setup_mocks(monkeypatch)

        labels = placement_mod._place_berthed_cluster(
            vertical_shore_scene["water_mask"],
            vertical_shore_scene["occupancy"],
            [((40.0, 20.0), (40.0, 180.0))],
            None,
            resolution_m=5.0,
            rng=random.Random(7),
            n_ships=3,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=vertical_shore_scene["background"].copy(),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=False,
            blur_sigma=0.0,
        )

        assert len(labels) == 3
        assert len(captured) == 3

        ordered = sorted(captured, key=lambda ship: ship.cx)
        assert all(later.cx > earlier.cx for earlier, later in zip(ordered, ordered[1:]))
        assert max(abs(later.cy - earlier.cy) for earlier, later in zip(ordered, ordered[1:])) < 2.0
        gaps = [
            placement_mod._signed_geometry_gap(earlier.hull_geom, later.hull_geom)
            for earlier, later in zip(ordered, ordered[1:])
        ]
        assert all(abs(gap) <= 1.0 for gap in gaps)
        assert all(int(label.split()[0]) == 1 for label in labels)

    def test_berth_cluster_partial_single_returns_no_labels(
        self,
        vertical_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """berth cluster が 1 隻しか成立しない場合は cluster として返さない。"""
        captured = self._setup_mocks(monkeypatch)
        monkeypatch.setattr(placement_mod, "_offshore_contact_candidate", lambda *args, **kwargs: None)

        labels = placement_mod._place_berthed_cluster(
            vertical_shore_scene["water_mask"],
            vertical_shore_scene["occupancy"],
            [((40.0, 20.0), (40.0, 180.0))],
            None,
            resolution_m=5.0,
            rng=random.Random(7),
            n_ships=3,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=vertical_shore_scene["background"].copy(),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=False,
            blur_sigma=0.0,
        )

        assert labels == []
        assert len(captured) == 0

    def test_single_berthed_ship_uses_solo_class_id(
        self,
        vertical_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """単船 berth は cluster helper を通っても solo class_id を使う。"""
        captured = self._setup_mocks(monkeypatch)

        labels = placement_mod._place_berthed_cluster(
            vertical_shore_scene["water_mask"],
            vertical_shore_scene["occupancy"],
            [((40.0, 20.0), (40.0, 180.0))],
            None,
            resolution_m=5.0,
            rng=random.Random(19),
            n_ships=1,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=vertical_shore_scene["background"].copy(),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=False,
            blur_sigma=0.0,
        )

        assert len(labels) == 1
        assert len(captured) == 1
        assert int(labels[0].split()[0]) == 0

    def test_berth_stern_to_cluster_points_stern_toward_horizontal_shore(
        self,
        horizontal_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ともづけ berth は船尾が岸へ向き、岸線に沿って並ぶ。"""
        captured = self._setup_mocks(monkeypatch)

        labels = _place_cluster(
            horizontal_shore_scene["water_mask"],
            horizontal_shore_scene["occupancy"],
            None,
            resolution_m=5.0,
            rng=random.Random(5),
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=horizontal_shore_scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
            berth_prob=1.0,
            berth_stern_prob=1.0,
            berth_water_mask=horizontal_shore_scene["water_mask"],
            berth_segments=[((20.0, 40.0), (180.0, 40.0))],
        )

        assert len(labels) == 3
        assert len(captured) == 3

        xs = [ship.cx for ship in captured]
        ys = [ship.cy for ship in captured]
        assert max(xs) - min(xs) > 20.0
        assert max(ys) - min(ys) < 3.0

        land_dir = np.array([0.0, -1.0])
        for ship in captured:
            stern_dir = np.array([-math.sin(ship.angle_rad), math.cos(ship.angle_rad)])
            assert float(np.dot(stern_dir, land_dir)) > 0.95
            _min_x, min_y, _max_x, _max_y = ship.hull_geom.bounds
            assert 39.0 <= min_y <= 45.0

    def test_berth_alongside_cluster_accepts_connected_curved_segments(
        self,
        curved_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """緩く曲がる連続 shore run でも横づけ lead と沖側 tight stack を置ける。"""
        captured = self._setup_mocks(monkeypatch)

        labels = placement_mod._place_berthed_cluster(
            curved_shore_scene["water_mask"],
            curved_shore_scene["occupancy"],
            curved_shore_scene["segments"],
            None,
            resolution_m=5.0,
            rng=random.Random(13),
            n_ships=3,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=curved_shore_scene["background"].copy(),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=False,
            blur_sigma=0.0,
        )

        assert len(labels) == 3
        assert len(captured) == 3

        ordered = sorted(captured, key=lambda ship: ship.cx)
        xs = [ship.cx for ship in ordered]
        ys = [ship.cy for ship in ordered]
        assert xs[-1] - xs[0] > 8.0
        assert max(ys) - min(ys) < 6.0

        lead_stern_dir = np.array([-math.sin(ordered[0].angle_rad), math.cos(ordered[0].angle_rad)])
        for ship in ordered[1:]:
            stern_dir = np.array([-math.sin(ship.angle_rad), math.cos(ship.angle_rad)])
            assert float(np.dot(stern_dir, lead_stern_dir)) > 0.95

    def test_berth_alongside_cluster_stops_at_narrow_water_band(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """横づけ follower は沖側の水域が尽きた位置で打ち切られる。"""
        captured = self._setup_mocks(monkeypatch)

        size = self._IMAGE_SIZE
        water_mask = np.zeros((size, size), dtype=bool)
        water_mask[:, 40:64] = True
        occupancy = np.zeros((size, size), dtype=bool)
        background = np.full((size, size, 3), 60, dtype=np.uint8)

        labels = placement_mod._place_berthed_cluster(
            water_mask,
            occupancy,
            [((40.0, 64.0), (40.0, 136.0))],
            None,
            resolution_m=5.0,
            rng=random.Random(17),
            n_ships=5,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=background.copy(),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=False,
            blur_sigma=0.0,
        )

        assert len(labels) == 2
        assert len(captured) == 2
        xs = [ship.cx for ship in captured]
        assert xs == sorted(xs)

    def test_berth_stern_to_cluster_trims_invalid_tail_ship(
        self,
        horizontal_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ともづけ berth は末尾だけ入らない場合に prefix を残して短縮する。"""
        captured = self._setup_mocks(monkeypatch)

        def _resolve_until_short_prefix(
            water_mask: np.ndarray,
            hull_geom,
            cx: float,
            cy: float,
            water_nx: float,
            water_ny: float,
            *,
            max_shift_px: float,
            step_px: float = 0.25,
        ):
            del water_mask, water_nx, water_ny, max_shift_px, step_px
            if cx > 46.0:
                return None
            return cx, cy, hull_geom

        monkeypatch.setattr(
            placement_mod,
            "_resolve_berth_land_intrusion",
            _resolve_until_short_prefix,
        )

        labels = placement_mod._place_berthed_cluster(
            horizontal_shore_scene["water_mask"],
            horizontal_shore_scene["occupancy"],
            [((20.0, 40.0), (56.0, 40.0))],
            None,
            resolution_m=5.0,
            rng=random.Random(23),
            n_ships=3,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=horizontal_shore_scene["background"].copy(),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=True,
            blur_sigma=0.0,
        )

        assert len(labels) == 2
        assert len(captured) == 2

        xs = [ship.cx for ship in captured]
        ys = [ship.cy for ship in captured]
        assert max(xs) - min(xs) > 8.0
        assert max(ys) - min(ys) < 3.0

        land_dir = np.array([0.0, -1.0])
        for ship in captured:
            stern_dir = np.array([-math.sin(ship.angle_rad), math.cos(ship.angle_rad)])
            assert float(np.dot(stern_dir, land_dir)) > 0.95

    def test_berth_curved_run_avoids_land_intrusion(
        self,
        bulged_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """曲がった shore run でも hull は land 側へ侵入しない。"""
        captured = self._setup_mocks(monkeypatch)

        labels = placement_mod._place_berthed_cluster(
            bulged_shore_scene["water_mask"],
            bulged_shore_scene["occupancy"],
            bulged_shore_scene["segments"],
            None,
            resolution_m=5.0,
            rng=random.Random(21),
            n_ships=1,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=bulged_shore_scene["background"].copy(),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=False,
            blur_sigma=0.0,
        )

        assert len(labels) == 1
        assert len(captured) == 1
        assert _count_land_boundary_samples(captured[0].hull_geom, bulged_shore_scene["points"]) == 0

    def test_berth_pushes_dockside_intrusion_out_to_contact(
        self,
        dock_touch_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """dock 側へ侵入する berth 候補は沖側へ逃がして接触へ戻す。"""
        captured = self._setup_mocks(monkeypatch)

        labels = placement_mod._place_berthed_cluster(
            dock_touch_scene["water_mask"],
            dock_touch_scene["occupancy"],
            [((40.0, 20.0), (40.0, 180.0))],
            None,
            resolution_m=5.0,
            rng=random.Random(19),
            n_ships=1,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=dock_touch_scene["background"].copy(),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=False,
            blur_sigma=0.0,
        )

        assert len(labels) == 1
        assert len(captured) == 1
        min_x, _min_y, _max_x, _max_y = captured[0].hull_geom.bounds
        assert min_x >= 43.75

    def test_berth_short_run_limits_mixed_svg_generation(
        self,
        vertical_shore_scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """lead ship すら収まらない短い岸線では mixed SVG を生成しない。"""
        captured = _capture_vector_cluster(monkeypatch)
        pick_calls: list[int] = []

        def _capture_pick_svg(*args, **kwargs) -> str:
            del args, kwargs
            pick_calls.append(len(pick_calls))
            return f"mock-svg-{len(pick_calls)}"

        monkeypatch.setattr(placement_mod, "_pick_svg", _capture_pick_svg)
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(8, 18)]),
        )
        monkeypatch.setattr(placement_mod, "_local_hull_geometry", _rect_local_hull_geometry)
        monkeypatch.setattr(placement_mod, "extract_hull_fill", lambda svg_text: (200, 200, 200, 255))

        labels = placement_mod._place_berthed_cluster(
            vertical_shore_scene["water_mask"],
            vertical_shore_scene["occupancy"],
            [((40.0, 93.0), (40.0, 107.0))],
            None,
            resolution_m=5.0,
            rng=random.Random(31),
            n_ships=10,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=vertical_shore_scene["background"].copy(),
            length_range=(90.0, 90.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=True,
            berth_stern=False,
            blur_sigma=0.0,
        )

        assert len(labels) == 0
        assert len(captured) == 0
        assert len(pick_calls) == 0

    def test_resolve_berth_land_intrusion_uses_coarse_to_fine_search(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """land intrusion 解決は細かい線形探索より少ない判定で済ませる。"""
        calls: list[float] = []

        def _mock_intrudes_land(
            water_mask: np.ndarray,
            hull_geom,
            clearance_px: float = placement_mod._BERTH_LAND_CLEARANCE_PX,
        ) -> bool:
            del water_mask, clearance_px
            calls.append(float(hull_geom.centroid.x))
            return float(hull_geom.centroid.x) < 5.0

        monkeypatch.setattr(placement_mod, "_geometry_intrudes_land", _mock_intrudes_land)

        result = placement_mod._resolve_berth_land_intrusion(
            np.ones((16, 16), dtype=bool),
            box(-1.0, -1.0, 1.0, 1.0),
            0.0,
            0.0,
            1.0,
            0.0,
            max_shift_px=6.0,
            step_px=0.25,
        )

        assert result is not None
        cx, cy, _shifted = result
        assert cx == pytest.approx(5.0, abs=0.25)
        assert cy == pytest.approx(0.0)
        assert len(calls) < 10


class TestClusterRenderHelpers:
    """cluster renderer と area cluster の補助挙動を検証する。"""

    _IMAGE_SIZE = 200

    @pytest.fixture()
    def scene(self):
        size = self._IMAGE_SIZE
        return {
            "water_mask": np.ones((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
        }

    def test_vector_cluster_zero_blur_sigma_skips_extra_blur(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cluster blur sigma が 0 なら追加 blur を掛けない。"""
        blur_calls: list[float] = []
        svg_text = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        ships = [
            placement_mod._RaftShipPlacement(
                svg_text=svg_text,
                cx=30.0,
                cy=40.0,
                bw=6,
                lh=24,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=0,
                hull_geom=box(27.0, 28.0, 33.0, 52.0),
                hull_fill=(190, 190, 190, 255),
            ),
            placement_mod._RaftShipPlacement(
                svg_text=svg_text,
                cx=37.0,
                cy=40.0,
                bw=6,
                lh=24,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=0,
                hull_geom=box(34.0, 28.0, 40.0, 52.0),
                hull_fill=(190, 190, 190, 255),
            ),
        ]

        def _capture_blur(layer: np.ndarray, sigma: float) -> np.ndarray:
            del layer
            blur_calls.append(sigma)
            return np.zeros((1, 1, 4), dtype=np.uint8)

        monkeypatch.setattr(placement_mod, "gaussian_blur_rgba_premultiplied", _capture_blur)

        placement_mod._render_vector_raft_cluster(
            ships,
            image_size=self._IMAGE_SIZE,
            blur_sigma=0.0,
            scene_scale=placement_mod._CLUSTER_SCENE_SUPERSAMPLE,
        )

        assert blur_calls == []

    def test_vector_cluster_blur_sigma_scales_extra_blur(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cluster blur sigma は scene-scale blur にそのまま反映される。"""
        blur_calls: list[float] = []
        svg_text = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        ships = [
            placement_mod._RaftShipPlacement(
                svg_text=svg_text,
                cx=30.0,
                cy=40.0,
                bw=6,
                lh=24,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=0,
                hull_geom=box(27.0, 28.0, 33.0, 52.0),
                hull_fill=(190, 190, 190, 255),
            ),
            placement_mod._RaftShipPlacement(
                svg_text=svg_text,
                cx=37.0,
                cy=40.0,
                bw=6,
                lh=24,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=0,
                hull_geom=box(34.0, 28.0, 40.0, 52.0),
                hull_fill=(190, 190, 190, 255),
            ),
        ]

        def _capture_blur(layer: np.ndarray, sigma: float) -> np.ndarray:
            blur_calls.append(sigma)
            return layer

        monkeypatch.setattr(placement_mod, "gaussian_blur_rgba_premultiplied", _capture_blur)

        placement_mod._render_vector_raft_cluster(
            ships,
            image_size=self._IMAGE_SIZE,
            blur_sigma=0.4,
            scene_scale=placement_mod._CLUSTER_SCENE_SUPERSAMPLE,
        )

        assert blur_calls == [pytest.approx(1.6)]

    def test_vector_cluster_shadow_source_keeps_hull_alpha(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """shadow source は detail が空でも hull alpha を保持する。"""
        observed_alpha_max: list[int] = []
        svg_text = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        ships = [
            placement_mod._RaftShipPlacement(
                svg_text=svg_text,
                cx=30.0,
                cy=40.0,
                bw=6,
                lh=24,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=0,
                hull_geom=box(27.0, 28.0, 33.0, 52.0),
                hull_fill=(190, 190, 190, 255),
            ),
        ]

        def _empty_detail_raster(
            svg_text: str,
            bw: int,
            lh: int,
            angle_deg: float = 0.0,
            supersample: int = 4,
            exclude_hull: bool = False,
        ) -> np.ndarray:
            del svg_text, angle_deg, supersample, exclude_hull
            return np.zeros((lh, bw, 4), dtype=np.uint8)

        def _capture_shadow_source(
            ship_rgba: np.ndarray,
            *,
            offset_x: int,
            offset_y: int,
            blur_sigma: float,
            alpha_scale: float,
        ) -> np.ndarray:
            del offset_x, offset_y, blur_sigma, alpha_scale
            observed_alpha_max.append(int(ship_rgba[:, :, 3].max()))
            return np.zeros_like(ship_rgba)

        monkeypatch.setattr(placement_mod, "rasterize_ship_svg", _empty_detail_raster)
        monkeypatch.setattr(placement_mod, "_make_shadow_rgba", _capture_shadow_source)
        monkeypatch.setattr(placement_mod, "_shadow_offset_pixels", lambda *args, **kwargs: (3, 1))
        monkeypatch.setattr(placement_mod, "_shadow_blur_sigma", lambda *args, **kwargs: 1.0)
        monkeypatch.setattr(placement_mod, "_shadow_alpha_for_ship", lambda *args, **kwargs: 1.0)

        placement_mod._render_vector_raft_cluster(
            ships,
            image_size=self._IMAGE_SIZE,
            blur_sigma=0.0,
            scene_scale=placement_mod._CLUSTER_SCENE_SUPERSAMPLE,
            shadow_azimuth_rad=math.pi / 6.0,
            shadow_length=2.0,
            shadow_alpha=0.12,
            shadow_alpha_scale=1.0,
        )

        assert observed_alpha_max == [255]

    def test_open_cluster_shares_shadow_length(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """raft_open クラスターでも全船が同じ影長パラメータを使う。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        shadow_lengths: list[float] = []
        shadow_patch_biases: list[float] = []
        darken_factors: list[float] = []

        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (100, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24), (6, 24)]),
        )
        monkeypatch.setattr(
            placement_mod,
            "_shadow_offset_pixels",
            lambda beam_px, length_px, azimuth_rad, shadow_length, *, scene_scale=1: shadow_lengths.append(shadow_length) or (3, 1),
        )
        monkeypatch.setattr(placement_mod, "_shadow_blur_sigma", lambda *args, **kwargs: 1.0)
        monkeypatch.setattr(placement_mod, "_shadow_alpha_for_ship", lambda *args, **kwargs: 1.05)

        def _mock_make_shadow_rgba(
            ship_rgba: np.ndarray,
            *,
            offset_x: int,
            offset_y: int,
            blur_sigma: float,
            alpha_scale: float,
        ) -> np.ndarray:
            del offset_x, offset_y, blur_sigma
            shadow_patch_biases.append(alpha_scale)
            return np.zeros((ship_rgba.shape[0], ship_rgba.shape[1], 4), dtype=np.uint8)

        def _mock_darken_rgba_patch(
            background: np.ndarray,
            patch: object,
            alpha_factor: float,
            clip_mask: np.ndarray | None = None,
        ) -> None:
            del background, patch, clip_mask
            darken_factors.append(alpha_factor)

        monkeypatch.setattr(placement_mod, "_make_shadow_rgba", _mock_make_shadow_rgba)
        monkeypatch.setattr(placement_mod, "_darken_rgba_patch", _mock_darken_rgba_patch)

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(11, "partial", base_angle=0.0),
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
            shadow_azimuth_rad=math.pi / 3.0,
            shadow_length=3.25,
            shadow_alpha=0.12,
            shadow_alpha_scale=2.0,
        )

        assert len(labels) >= 1
        assert len(shadow_lengths) == len(labels)
        assert len(shadow_patch_biases) == len(labels)
        assert all(value == pytest.approx(1.05) for value in shadow_patch_biases)
        assert len({round(value, 6) for value in shadow_lengths}) == 1
        assert shadow_lengths[0] == pytest.approx(3.25)
        assert darken_factors == [pytest.approx(0.24)]

    def test_open_cluster_downsamples_only_local_patch(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """raft_open クラスターも全画面ではなく局所パッチだけを縮小する。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
            '</svg>'
        )
        recorded_shapes: list[tuple[int, int]] = []

        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (100, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24), (6, 24)]),
        )

        def _mock_downsample_cluster_patch(
            layer: np.ndarray,
            scene_x0: int,
            scene_y0: int,
            scene_scale: int,
        ):
            recorded_shapes.append(layer.shape[:2])
            return placement_mod.RgbaLayerPatch(
                scene_x0 // scene_scale,
                scene_y0 // scene_scale,
                np.zeros(
                    (layer.shape[0] // scene_scale, layer.shape[1] // scene_scale, 4),
                    dtype=np.uint8,
                ),
            )

        monkeypatch.setattr(placement_mod, "_downsample_cluster_patch", _mock_downsample_cluster_patch)

        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=_ForcedLayoutRandom(11, "partial", base_angle=0.0),
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) >= 1
        assert recorded_shapes
        full_scene_size = self._IMAGE_SIZE * placement_mod._CLUSTER_SCENE_SUPERSAMPLE
        assert all(height < full_scene_size for height, _width in recorded_shapes)
        assert all(width < full_scene_size for _height, width in recorded_shapes)


class TestVisibleClusterComponents:
    """visible silhouette の connected component による cluster 判定を検証する。"""

    def test_isolated_ship_in_event_gets_solo_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """接触ペアと孤立船が混在すると、孤立船だけ solo 扱いになる。"""
        monkeypatch.setattr(
            placement_mod,
            "rasterize_ship_svg",
            lambda *args, **kwargs: np.zeros((30, 10, 4), dtype=np.uint8),
        )

        ships = [
            placement_mod._RaftShipPlacement(
                svg_text="mock-a",
                cx=5.0,
                cy=15.0,
                bw=10,
                lh=30,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=1,
                hull_geom=box(0.0, 0.0, 10.0, 30.0),
                hull_fill=(200, 200, 200, 255),
            ),
            placement_mod._RaftShipPlacement(
                svg_text="mock-b",
                cx=15.0,
                cy=15.0,
                bw=10,
                lh=30,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=1,
                hull_geom=box(10.0, 0.0, 20.0, 30.0),
                hull_fill=(200, 200, 200, 255),
            ),
            placement_mod._RaftShipPlacement(
                svg_text="mock-c",
                cx=45.0,
                cy=15.0,
                bw=10,
                lh=30,
                angle_deg=0.0,
                angle_rad=0.0,
                class_id=1,
                hull_geom=box(40.0, 0.0, 50.0, 30.0),
                hull_fill=(200, 200, 200, 255),
            ),
        ]

        flags = placement_mod._cluster_component_flags(ships, scene_scale=1)

        assert flags == [True, True, False]


class TestClusterClassId:
    """クラスター種別（raft_tight / raft_open）による class_id 分岐の検証。"""

    _IMAGE_SIZE = 200
    _MOCK_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
        '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)"/>'
        '</svg>'
    )

    @pytest.fixture()
    def scene(self):
        """全面水域の 200x200 シーン。"""
        size = self._IMAGE_SIZE
        return {
            "water_mask": np.ones((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
        }

    def _base_args(self, scene, *, size_thresholds=None):
        return dict(
            water_mask=scene["water_mask"],
            occupancy=np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            svg_metas=None,
            resolution_m=1.0,
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            size_thresholds=size_thresholds,
            mixed_prob=0.0,
        )

    def _setup_mocks(self, monkeypatch: pytest.MonkeyPatch) -> list:
        """共通モックを設定し、_RaftShipPlacement リストを返す。"""
        captured = _capture_vector_cluster(monkeypatch)
        monkeypatch.setattr(placement_mod, "_pick_svg", lambda *args, **kwargs: self._MOCK_SVG)
        monkeypatch.setattr(placement_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            placement_mod,
            "_resolve_ship_dimensions",
            lambda *args, **kwargs: ("mock_hull", 6, 24, 4.0),
        )
        return captured

    def test_raft_tight_uses_cluster_class_id(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """raft_tight クラスターの船は cluster class_id (1 = ship_c) を使う。"""
        self._setup_mocks(monkeypatch)
        labels = _place_cluster(
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            **self._base_args(scene),
        )
        assert len(labels) == 2
        assert all(int(lbl.split()[0]) == 1 for lbl in labels)

    def test_raft_open_uses_cluster_class_id(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """raft_open クラスターの船は cluster class_id (1 = ship_c) を使う。"""
        self._setup_mocks(monkeypatch)
        labels = _place_cluster(
            rng=_ForcedLayoutRandom(7, "partial", base_angle=0.0),
            **self._base_args(scene),
        )
        assert len(labels) > 0
        assert all(int(lbl.split()[0]) == 1 for lbl in labels)

    def test_raft_open_isolated_ship_in_event_uses_solo_class_id(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """同一 event 内で孤立した raft_open 船は solo class_id を使う。"""
        self._setup_mocks(monkeypatch)
        monkeypatch.setattr(placement_mod, "_cluster_component_flags", lambda ships, scene_scale: [True, False])

        labels = _place_cluster(
            rng=_ForcedLayoutRandom(7, "partial", base_angle=0.0),
            **self._base_args(scene),
        )

        assert [int(lbl.split()[0]) for lbl in labels] == [1, 0]

    def test_partial_cluster_failure_returns_no_labels(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cluster event が 1 隻しか成立しない場合は失敗扱いでラベルを返さない。"""
        self._setup_mocks(monkeypatch)

        call_count = {"count": 0}
        original_obb_on_water = placement_mod._obb_on_water

        def _fail_after_first_ship(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] >= 2:
                return False
            return original_obb_on_water(*args, **kwargs)

        monkeypatch.setattr(placement_mod, "_obb_on_water", _fail_after_first_ship)

        labels = _place_cluster(
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            **self._base_args(scene),
        )

        assert labels == []

    def test_raft_tight_with_size_threshold_uses_cluster_offset(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """size_threshold ありの raft_tight は n_solo 分オフセットされた class_id を使う。

        threshold=100m, resolution=1m/px, 船長=24px→24m < 100m
        → small bucket(0), n_solo=2, cluster_id = 0+2 = 2 (ship_small_c)
        """
        self._setup_mocks(monkeypatch)
        labels = _place_cluster(
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            **self._base_args(scene, size_thresholds=(100.0,)),
        )
        assert len(labels) == 2
        assert all(int(lbl.split()[0]) == 2 for lbl in labels)

    def test_raft_tight_large_ship_with_size_threshold(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """size_threshold ありの raft_tight で大きい船は large_c class_id を使う。

        threshold=10m, resolution=1m/px, 船長=24px→24m ≥ 10m
        → large bucket(1), n_solo=2, cluster_id = 1+2 = 3 (ship_large_c)
        """
        self._setup_mocks(monkeypatch)
        labels = _place_cluster(
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0),
            **self._base_args(scene, size_thresholds=(10.0,)),
        )
        assert len(labels) == 2
        assert all(int(lbl.split()[0]) == 3 for lbl in labels)

    def test_raft_tight_stagger_mode_applies_longitudinal_offset(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tight_stagger_mode が有効なとき、2隻目以降が前後方向にずれて配置される。

        stagger_step は1回だけサンプルされるため、2隻目は base_cy からずれる。
        """
        captured = self._setup_mocks(monkeypatch)
        _place_cluster(
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0, stagger=True),
            **self._base_args(scene),
        )
        assert len(captured) == 2
        base_cy = 100.0  # find_water_position mock が返す cy
        assert captured[0].cy == pytest.approx(base_cy)
        assert abs(captured[1].cy - base_cy) > 0.01

    def test_raft_tight_stagger_mode_creates_monotonic_drift(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tight_stagger_mode が有効なとき、cy が単調に変化する（累積ドリフト）。

        stagger_step を1回サンプルして累積するため、cy のずれは等差的に増える。
        3隻で ship1.cy < ship2.cy < ship3.cy または逆順になることを検証する。
        """
        captured = self._setup_mocks(monkeypatch)
        _place_cluster(
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0, stagger=True),
            **{**self._base_args(scene), "cluster_size_range": (3, 3)},
        )
        assert len(captured) == 3
        cy_values = [ship.cy for ship in captured]
        diffs = [cy_values[k + 1] - cy_values[k] for k in range(len(cy_values) - 1)]
        # 全差分が同符号 → 単調増加 or 単調減少（累積ドリフトの証拠）
        assert all(d > 0.01 for d in diffs) or all(d < -0.01 for d in diffs)

    def test_raft_tight_aligned_mode_has_no_longitudinal_offset(
        self, scene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tight_stagger_mode が無効なとき、全船が前後方向に揃って配置される。

        base_angle=0.0 では縦方向が前後軸。stagger なしなので全隻の cy が base_cy に一致する。
        """
        captured = self._setup_mocks(monkeypatch)
        _place_cluster(
            rng=_ForcedLayoutRandom(7, "flush", base_angle=0.0, stagger=False),
            **self._base_args(scene),
        )
        assert len(captured) == 2
        base_cy = 100.0
        assert captured[0].cy == pytest.approx(base_cy)
        assert captured[1].cy == pytest.approx(base_cy)