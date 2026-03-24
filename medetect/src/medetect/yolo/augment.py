from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

_SUPPORTED_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


class RandomCloudOverlay(ImageOnlyTransform):
    """指定フォルダから雲画像をランダムに選び、学習画像全体に重畳する Augmentation。

    雲画像は入力画像全体を必ずカバーするように拡大したうえで、
    ランダムな回転・平行移動を施してから alpha ブレンドされる。
    アルファチャンネルを持つ PNG であればそのまま利用し、
    RGB/グレースケール画像の場合は輝度をアルファとして扱う。

    Parameters
    ----------
    cloud_dir:
        雲画像（PNG / JPG / TIF）が格納されたディレクトリ。
    alpha_range:
        重畳時の不透明度の範囲 (min, max)。0.0 = 完全透明、1.0 = 完全不透明。
    scale_range:
        全体カバーに必要な最小スケールに対する追加倍率の範囲。
        1.0 = ちょうど全体をカバー、2.0 = 2 倍大きい雲。値は 1.0 以上を推奨。
    rotation_range:
        回転角度の範囲 (min_deg, max_deg)。
    p:
        このTransformが適用される確率。
    """

    def __init__(
        self,
        cloud_dir: str | Path,
        alpha_range: tuple[float, float] = (0.2, 0.6),
        scale_range: tuple[float, float] = (1.0, 1.5),
        rotation_range: tuple[float, float] = (0.0, 360.0),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.cloud_dir = str(cloud_dir)
        self.alpha_range = alpha_range
        self.scale_range = scale_range
        self.rotation_range = rotation_range

        cloud_paths = sorted(
            path
            for path in Path(cloud_dir).iterdir()
            if path.suffix.lower() in _SUPPORTED_EXTS
        )
        if not cloud_paths:
            raise ValueError(f"cloud_dir に画像が見つかりません: {cloud_dir}")
        self._cloud_paths: list[str] = [str(p) for p in cloud_paths]

    # ------------------------------------------------------------------
    # Albumentations interface
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, Any]:
        return {
            "cloud_idx": random.randint(0, len(self._cloud_paths) - 1),
            "rotation": random.uniform(*self.rotation_range),
            "scale": random.uniform(*self.scale_range),
            "opacity": random.uniform(*self.alpha_range),
            # 平行移動量を正規化値（0〜1）で持つ
            "rel_tx": random.uniform(0.0, 1.0),
            "rel_ty": random.uniform(0.0, 1.0),
        }

    def apply(
        self,
        img: np.ndarray,
        cloud_idx: int = 0,
        rotation: float = 0.0,
        scale: float = 1.0,
        opacity: float = 0.4,
        rel_tx: float = 0.5,
        rel_ty: float = 0.5,
        **params: Any,
    ) -> np.ndarray:
        cloud_raw = cv2.imread(self._cloud_paths[cloud_idx], cv2.IMREAD_UNCHANGED)
        if cloud_raw is None:
            return img

        h, w = img.shape[:2]
        ch, cw = cloud_raw.shape[:2]

        # ---- スケール ----
        # 任意の角度に回転しても入力画像全体をカバーできるよう、
        # 雲の短辺が画像対角線 × √2 以上になるようにスケーリングする。
        # （正方形を 45° 回転したとき内接する正方形の一辺 = 元の辺 / √2）
        diagonal = math.hypot(h, w)
        base_scale = diagonal * math.sqrt(2) / min(ch, cw)
        total_scale = base_scale * scale  # scale は全体カバー最小値に対する追加倍率
        new_h = max(1, round(ch * total_scale))
        new_w = max(1, round(cw * total_scale))
        cloud_resized = cv2.resize(cloud_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # ---- 回転 ----
        M = cv2.getRotationMatrix2D((new_w / 2.0, new_h / 2.0), rotation, 1.0)
        cloud_rotated = cv2.warpAffine(
            cloud_resized, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # ---- 平行移動（全体カバーを保ちながらランダムにずらす） ----
        max_dx = (new_w - w) // 2
        max_dy = (new_h - h) // 2
        dx = round((rel_tx - 0.5) * 2.0 * max_dx)
        dy = round((rel_ty - 0.5) * 2.0 * max_dy)
        x0 = max(0, min(new_w - w, new_w // 2 - w // 2 + dx))
        y0 = max(0, min(new_h - h, new_h // 2 - h // 2 + dy))
        cloud_cropped = cloud_rotated[y0 : y0 + h, x0 : x0 + w]

        # ---- アルファブレンド ----
        cloud_rgb, cloud_alpha = _split_cloud(cloud_cropped)
        cloud_alpha = cloud_alpha * opacity
        result = img.astype(np.float32)
        if img.ndim == 2:
            cloud_gray = cv2.cvtColor(
                cloud_rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY
            ).astype(np.float32)
            result = result * (1.0 - cloud_alpha) + cloud_gray * cloud_alpha
        else:
            result = result * (1.0 - cloud_alpha[:, :, np.newaxis]) + cloud_rgb * cloud_alpha[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("cloud_dir", "alpha_range", "scale_range", "rotation_range")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _split_cloud(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """雲画像を float32 BGR と float32 alpha (0–1) に分割して返す。"""
    if cloud.ndim == 2:
        rgb = cv2.cvtColor(cloud, cv2.COLOR_GRAY2BGR).astype(np.float32)
        alpha = cloud.astype(np.float32) / 255.0
    elif cloud.shape[2] == 4:
        rgb = cloud[:, :, :3].astype(np.float32)
        alpha = cloud[:, :, 3].astype(np.float32) / 255.0
    else:
        rgb = cloud.astype(np.float32)
        gray = cv2.cvtColor(cloud, cv2.COLOR_BGR2GRAY)
        alpha = gray.astype(np.float32) / 255.0
    return rgb, alpha
