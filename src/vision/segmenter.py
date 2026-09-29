"""Spec A — ROI 내 표적 분할, 면적·등가직경 측정.

색상·형상 기반은 초기 후보 방법이며, 실제 표적 색 확인 전 고정 요구조건이 아니다.
여기서는 '밝은 표적 vs 어두운 배경' 그레이 임계를 기본 후보로 사용한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegResult:
    mask_area_px2: float
    equivalent_diameter_px: float
    center: tuple[float, float] | None   # 마스크 무게중심(px, 전체 프레임 좌표)
    bbox_measured: tuple[int, int, int, int] | None  # 마스크 경계 bbox(전체 좌표)
    touches_border: bool                  # ROI 경계에 마스크가 닿았는가(잘림 후보)
    found: bool


def _clamp_roi(x, y, w, h, W, H):
    x0 = max(0, int(x)); y0 = max(0, int(y))
    x1 = min(W, int(x + w)); y1 = min(H, int(y + h))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def segment(image: np.ndarray, track_bbox, cfg) -> SegResult:
    """track_bbox=(x,y,w,h) 주변 ROI에서 밝은 표적 마스크를 만든다."""
    H, W = image.shape[:2]
    if track_bbox is None:
        return SegResult(0.0, 0.0, None, None, False, False)

    x, y, w, h = track_bbox
    mx = int(w * cfg.roi_margin)
    my = int(h * cfg.roi_margin)
    rx, ry, rw, rh = _clamp_roi(x - mx, y - my, w + 2 * mx, h + 2 * my, W, H)
    if rw <= 0 or rh <= 0:
        return SegResult(0.0, 0.0, None, None, False, False)

    roi = image[ry:ry + rh, rx:rx + rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if cfg.use_otsu:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, cfg.threshold, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return SegResult(0.0, 0.0, None, None, False, False)

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 0:
        return SegResult(0.0, 0.0, None, None, False, False)

    eq_d = float(math.sqrt(4.0 * area / math.pi))
    M = cv2.moments(c)
    cx = rx + M["m10"] / M["m00"]
    cy = ry + M["m01"] / M["m00"]
    bx, by, bw, bh = cv2.boundingRect(c)
    # 마스크가 ROI 경계에 닿으면 잘림 후보
    touches = (bx <= 1 or by <= 1 or bx + bw >= rw - 1 or by + bh >= rh - 1)
    return SegResult(
        mask_area_px2=area,
        equivalent_diameter_px=eq_d,
        center=(cx, cy),
        bbox_measured=(rx + bx, ry + by, bw, bh),
        touches_border=touches,
        found=True,
    )
