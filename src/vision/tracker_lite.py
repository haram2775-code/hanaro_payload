"""Pi Zero 2 W용 경량 낙하산 추적기 (Spec: 짐벌 연동 · 원거리 · 상대속도).

기존 vision.tracker.Tracker(CSRT/폴백)는 그대로 두고, 여기에 Pi Zero 2 W의
RAM(512MB)·연산 제약에서 4608×2592 @ 14fps를 감당하도록 설계한 별도 경량
추적기를 **추가**한다. 절대 거리/m/s는 확정 출력하지 않는다(공통 규칙 §6):
표적 간 상대운동은 픽셀 크기 변화율(접근/이격)로 낸다.

핵심 설계 (근거):
  1. 풀프레임을 상시 메모리에 두지 않는다. 짐벌(github.com/smchoi02/gimbal_control)이
     표적을 화면 중앙 근처에 유지하므로, **중앙 ROI만 크롭**해 처리한다.
     4608×2592 BGR 1장 = ~34MB → 상시 보관 불가. 512×512 ROI = ~0.75MB.
  2. ROI 안에서 **다운스케일 + 밝기/색 임계 세그먼트**로 표적을 잡는다(CSRT 불필요).
     짐벌 잔여 오차(gimbal.py, ±3° ≈ 화면 중앙에서 수백 px)를 ROI 크기로 흡수한다.
  3. 원거리 표적(작은 px)까지: ROI를 다운스케일하되 최소 표적폭(~8px)이 유지되는
     스케일을 고른다. camera_geometry.md 기준 main은 ~600m, drogue는 ~200m까지.
  4. 상대속도: 두 표적(또는 순차 단계)의 등가직경 시계열에서 **크기 변화율**을 내고,
     '접근/유지/이격'과 상대 스케일 속도(1/s, 부호)로 보고한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


# ── ROI 크롭 (풀프레임 상시 보관 회피) ─────────────────────────────
@dataclass
class RoiConfig:
    roi_size: int = 512          # 중앙 크롭 한 변(px). 짐벌 잔여 오차 흡수 여유.
    proc_max: int = 256          # 처리용 다운스케일 상한(px). 연산 예산 절감.
    min_target_px: float = 8.0   # 이보다 작으면 추적 불가로 본다(원거리 한계).


def center_roi(frame_shape, cfg: RoiConfig, gimbal_offset=(0.0, 0.0)):
    """짐벌이 맞춘 표적 위치(중앙+오차) 주변 ROI 사각형을 반환.

    반환: (rx, ry, rw, rh) — 원본 프레임 좌표. 풀프레임을 복사하지 않고
    이 사각형만 크롭해 처리한다.
    """
    H, W = frame_shape[:2]
    cx = W / 2.0 + gimbal_offset[0]
    cy = H / 2.0 + gimbal_offset[1]
    s = cfg.roi_size
    rx = int(round(cx - s / 2.0))
    ry = int(round(cy - s / 2.0))
    rx = max(0, min(W - 1, rx))
    ry = max(0, min(H - 1, ry))
    rw = min(s, W - rx)
    rh = min(s, H - ry)
    return rx, ry, rw, rh


# ── 경량 세그먼트 (ROI 다운스케일 + HSV 채도/밝기) ────────────────
@dataclass
class LiteDetection:
    found: bool
    center: tuple[float, float] | None      # 원본 프레임 좌표(px)
    equivalent_diameter_px: float
    area_px2: float
    bbox: tuple[int, int, int, int] | None   # 원본 프레임 좌표


def detect_in_roi(frame, roi, cfg: RoiConfig) -> LiteDetection:
    """중앙 ROI 안에서 낙하산(채도 높은 밝은 물체)을 1개 잡는다.

    IREC 하늘 배경은 채도 높은 파랑이고 낙하산은 주황/흰색이므로,
    HSV에서 '파랑이 아닌' + '충분히 밝은' 화소를 표적 후보로 쓴다.
    다운스케일 후 처리해 Pi 연산을 줄인다.
    """
    rx, ry, rw, rh = roi
    if rw <= 0 or rh <= 0:
        return LiteDetection(False, None, 0.0, 0.0, None)
    sub = frame[ry:ry + rh, rx:rx + rw]

    # 다운스케일 (연산 예산 절감). scale = proc / max(rw,rh)
    scale = min(1.0, cfg.proc_max / float(max(rw, rh)))
    if scale < 1.0:
        small = cv2.resize(sub, (max(1, int(rw * scale)), max(1, int(rh * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = sub

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # 하늘(파랑, Hue ~100-130 in OpenCV 0-179)을 배제, 나머지 유채색/밝은 것만.
    sky = (h >= 95) & (h <= 135) & (s > 60)
    bright = v > 110
    colored = s > 70
    mask = ((bright & colored) | (bright & (v > 200))) & (~sky)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return LiteDetection(False, None, 0.0, 0.0, None)
    c = max(cnts, key=cv2.contourArea)
    area_small = float(cv2.contourArea(c))
    if area_small <= 1.0:
        return LiteDetection(False, None, 0.0, 0.0, None)

    M = cv2.moments(c)
    cx_s = M["m10"] / M["m00"]
    cy_s = M["m01"] / M["m00"]
    bx, by, bw, bh = cv2.boundingRect(c)

    # 원본 스케일로 역변환
    inv = 1.0 / scale
    cx = rx + cx_s * inv
    cy = ry + cy_s * inv
    area = area_small * inv * inv
    eq_d = math.sqrt(4.0 * area / math.pi)
    bbox = (int(rx + bx * inv), int(ry + by * inv), int(bw * inv), int(bh * inv))

    if eq_d < cfg.min_target_px:
        # 잡았으나 원거리 한계 이하 — 위치는 주되 크기 신뢰 낮음
        return LiteDetection(True, (cx, cy), eq_d, area, bbox)
    return LiteDetection(True, (cx, cy), eq_d, area, bbox)


# ── 상대속도 (픽셀 크기 변화율 → 접근/유지/이격) ──────────────────
def relative_scale_rate(diam_prev: float, diam_now: float, dt_s: float) -> float:
    """등가직경의 상대 변화율 [1/s]. +면 팽창(접근), −면 축소(이격).

    절대 거리/속도가 아니라 스케일 속도. d(ln diameter)/dt ≈ (Δd/d)/dt.
    """
    if diam_prev <= 0 or dt_s <= 0:
        return 0.0
    return (diam_now - diam_prev) / diam_prev / dt_s


def classify_relative_motion(scale_rate: float, deadband: float = 0.02) -> str:
    """상대운동 라벨. deadband[1/s] 안이면 '유지'."""
    if scale_rate > deadband:
        return "APPROACH"   # 커짐 = 가까워짐
    if scale_rate < -deadband:
        return "RECEDE"     # 작아짐 = 멀어짐
    return "HOLD"


@dataclass
class LiteTrackerState:
    diam_ema: float | None = None
    last_center: tuple[float, float] | None = None
    misses: int = 0


class LiteTracker:
    """짐벌이 중앙에 둔 표적을 ROI 안에서 잡아 위치·크기·상대운동을 낸다.

    프레임당: ROI 크롭 → 다운스케일 → HSV 세그먼트 → 중심/직경 → EMA 평활.
    풀프레임 상시 보관 없음. CSRT 불필요.
    """

    def __init__(self, roi_cfg: RoiConfig | None = None, ema_alpha: float = 0.4,
                 max_misses: int = 5):
        self.cfg = roi_cfg or RoiConfig()
        self.ema_alpha = ema_alpha
        self.max_misses = max_misses
        self.state = LiteTrackerState()

    def step(self, frame, gimbal_offset=(0.0, 0.0)):
        """returns (found, center, eq_diam_px, bbox, roi)."""
        roi = center_roi(frame.shape, self.cfg, gimbal_offset)
        det = detect_in_roi(frame, roi, self.cfg)
        if not det.found or det.center is None:
            self.state.misses += 1
            return False, self.state.last_center, self.state.diam_ema or 0.0, None, roi
        self.state.misses = 0
        self.state.last_center = det.center
        # EMA로 직경 평활(지터 억제)
        if self.state.diam_ema is None:
            self.state.diam_ema = det.equivalent_diameter_px
        else:
            a = self.ema_alpha
            self.state.diam_ema = a * det.equivalent_diameter_px + (1 - a) * self.state.diam_ema
        return True, det.center, self.state.diam_ema, det.bbox, roi

    @property
    def lost(self) -> bool:
        return self.state.misses > self.max_misses
