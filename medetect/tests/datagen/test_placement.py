from __future__ import annotations

import math
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

import medetect.datagen.placement as placement_mod

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

    def __init__(self, seed: int, layout: str, base_angle: float = 0.0) -> None:
        super().__init__(seed)
        self._layout = layout
        self._base_angle = base_angle
        self._uniform_calls = 0

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        pop_list = list(population)
        layout_map = {
            "flush": "raft_tight",
            "partial": "raft_open",
            "gapped": "area_scattered",
        }
        if set(pop_list) == {"raft_tight", "raft_open", "area_scattered"} and k == 1:
            return [layout_map.get(self._layout, self._layout)]
        return super().choices(population, weights=weights, cum_weights=cum_weights, k=k)

    def uniform(self, a, b):
        if self._uniform_calls == 0 and a == 0 and b == 360:
            self._uniform_calls += 1
            return self._base_angle
        if a < 0 < b and max(abs(a), abs(b)) <= 2.0:
            self._uniform_calls += 1
            return 0.0
        self._uniform_calls += 1
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
        """クラスター間に隙間・接触の両方が出現する。"""
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
        tight = sum(1 for gap in min_gaps if gap <= 1.5)
        gapped = sum(1 for gap in min_gaps if gap > 3.0)
        assert tight > 0
        assert gapped > 0

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