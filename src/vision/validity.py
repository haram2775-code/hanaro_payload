"""Spec A — track_valid / size_valid / state 판정.

공통 규칙 §5: 중심 추적 유효성과 크기 측정 유효성을 분리한다.
가림·팽창·화면 잘림에서는 크기 기반 값을 무효(size_valid=False) 처리한다.
"""
from __future__ import annotations

from dataclasses import dataclass

STATE_SEARCH = "SEARCH"
STATE_TRACK = "TRACK"
STATE_OCCLUDED = "OCCLUDED"
STATE_CLIPPED = "CLIPPED"
STATE_INFLATING = "INFLATING"
STATE_LOST = "LOST"


@dataclass
class Judgement:
    track_valid: bool
    size_valid: bool
    state: str


def judge(
    *,
    frame_w: int,
    frame_h: int,
    track_ok: bool,
    center,
    seg,                    # SegResult
    prev_area: float | None,
    cfg,
) -> Judgement:
    # --- 중심 추적 유효성 ---
    track_valid = bool(track_ok and center is not None)
    if track_valid:
        cx, cy = center
        if not (0 <= cx < frame_w and 0 <= cy < frame_h):
            track_valid = False

    if not track_valid:
        return Judgement(False, False, STATE_LOST)

    # --- 크기 측정 유효성 ---
    frame_area = float(frame_w * frame_h)
    area = seg.mask_area_px2 if seg.found else 0.0
    size_valid = True
    state = STATE_TRACK

    # 마스크 없음/너무 작음 → 가림 또는 소실
    if not seg.found or area < cfg.min_area_px2:
        size_valid = False
        state = STATE_OCCLUDED
    else:
        # 화면/ROI 경계 접촉 → 잘림
        if seg.touches_border or _bbox_touches_frame(seg.bbox_measured, frame_w, frame_h, cfg.clip_margin_px):
            size_valid = False
            state = STATE_CLIPPED
        # 면적 상한 초과
        elif area > frame_area * cfg.max_area_fraction:
            size_valid = False
            state = STATE_INFLATING
        elif prev_area is not None and prev_area > 0:
            ratio = area / prev_area
            if ratio >= cfg.inflate_ratio:
                size_valid = False
                state = STATE_INFLATING
            elif ratio <= cfg.occlude_ratio:
                size_valid = False
                state = STATE_OCCLUDED

    return Judgement(track_valid, size_valid, state)


def _bbox_touches_frame(bbox, W, H, margin) -> bool:
    if bbox is None:
        return False
    x, y, w, h = bbox
    return (x <= margin or y <= margin or x + w >= W - margin or y + h >= H - margin)
