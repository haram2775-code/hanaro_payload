"""낙하산 캐노피 형상 렌더러 (Spec B 확장).

단순 원(circle) 대신 물리적으로 타당한 낙하산 형상을 그린다:
- 반구형/타원형 캐노피(투영 시 타원 + 반경방향 음영)
- 전개 단계: reefed → inflating → full
- shroud line(현삭)이 confluence point로 수렴
- 관측각(elevation)에 따른 투영 종횡비 변화

주의(공통 규칙): 정확한 형상 모델링이 실제 낙하산 **검출 성공**을 보장하지는 않는다.
이 렌더러는 검출기가 실영상에서 마주칠 어려움(비대칭, 팽창 변화, 관측각)을 더 잘 재현하기 위한 것이다.
모든 치수는 픽셀 단위이며, 절대 거리/속도는 다루지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Canopy:
    """낙하산 캐노피 렌더 파라미터. 모든 크기는 픽셀."""
    kind: str = "hemispherical"       # hemispherical | elliptical | round
    color: tuple[int, int, int] = (60, 160, 230)   # BGR (기본: 주황 계열)
    alt_color: tuple[int, int, int] = (230, 230, 230)  # 교대 패널 색
    n_panels: int = 8                  # gore(패널) 수
    shroud_color: tuple[int, int, int] = (200, 200, 200)
    n_shroud: int = 8                  # 표시할 현삭 수
    shroud_len_ratio: float = 1.3      # confluence까지 거리 / 캐노피 반경
    edge_darken: float = 0.55          # 가장자리 음영 계수(구형감)


def _aspect_for_kind(kind: str) -> float:
    """정면 관측 시 캐노피 투영의 세로/가로 비(높이/폭)."""
    if kind == "hemispherical":
        return 0.62      # 반구는 위에서 눌린 돔
    if kind == "elliptical":
        return 0.55      # semi-ellipsoid는 더 납작
    return 0.72          # round/flat


def render_canopy(
    img: np.ndarray,
    center: tuple[float, float],
    diameter_px: float,
    canopy: Canopy,
    *,
    inflation: float = 1.0,     # 0=reefed, 1=full 팽창
    view_elev: float = 0.0,     # 관측 고도각(rad). 0=정면(옆), pi/2=바로 아래
    occlude_top_frac: float = 0.0,  # 상단 가림 비율
) -> None:
    """캐노피 + shroud line을 img에 그린다(in place)."""
    cx, cy = center
    r = max(1.0, diameter_px / 2.0)

    # 전개 단계: reefed면 폭이 좁고 세로로 길쭉
    infl = float(np.clip(inflation, 0.05, 1.0))
    width = r * (0.35 + 0.65 * infl)          # reefed일수록 좁음
    base_aspect = _aspect_for_kind(canopy.kind)
    # 관측각: 바로 아래에서 볼수록 원형(aspect→1), 옆에서 볼수록 납작
    view_factor = base_aspect + (1.0 - base_aspect) * math.sin(min(max(view_elev, 0.0), math.pi / 2))
    height = width * view_factor * (0.7 + 0.3 * infl)

    ax = int(round(width))
    ay = int(round(height))
    if ax < 1 or ay < 1:
        return

    # --- 캐노피 채우기: 교대 패널로 gore 표현 ---
    # 캐노피는 상단이 볼록한 돔. y는 위로 볼록하도록 중심을 약간 위로.
    dome_cy = int(round(cy - height * 0.15))
    seg = 360 / max(1, canopy.n_panels)
    for i in range(canopy.n_panels):
        a0 = 180 + i * seg      # 상단 반원(180~360)만 캐노피
        a1 = 180 + (i + 1) * seg
        if a0 >= 360:
            break
        col = canopy.color if i % 2 == 0 else canopy.alt_color
        cv2.ellipse(img, (int(cx), dome_cy), (ax, ay), 0,
                    a0, min(a1, 360), col, thickness=-1, lineType=cv2.LINE_AA)

    # 하단 개구부(스커트) 살짝 둥글게
    cv2.ellipse(img, (int(cx), dome_cy), (ax, int(ay * 0.32)), 0,
                0, 180, _scale_color(canopy.color, 0.85), thickness=-1, lineType=cv2.LINE_AA)

    # --- 가장자리 음영(구형감) ---
    overlay = img.copy()
    cv2.ellipse(overlay, (int(cx), dome_cy), (ax, ay), 0, 180, 360,
                _scale_color(canopy.color, canopy.edge_darken), thickness=max(1, ax // 8),
                lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # --- shroud lines: 스커트 가장자리 → confluence point ---
    conf = (cx, dome_cy + height * canopy.shroud_len_ratio)
    for k in range(canopy.n_shroud):
        t = k / max(1, canopy.n_shroud - 1)
        ex = cx - ax + 2 * ax * t
        ey = dome_cy + ay * 0.30
        cv2.line(img, (int(ex), int(ey)), (int(conf[0]), int(conf[1])),
                 canopy.shroud_color, 1, lineType=cv2.LINE_AA)

    # --- 상단 가림(occlusion) ---
    if occlude_top_frac > 0:
        cover_h = int(round(2 * ay * occlude_top_frac))
        top = dome_cy - ay
        # 배경색으로 덮는 것은 호출자가 배경색을 알아야 하므로 여기선 회색 막대
        cv2.rectangle(img, (int(cx - ax) - 4, top - 4),
                      (int(cx + ax) + 4, top + cover_h), (40, 40, 40), thickness=-1)


def canopy_projected_bbox(center, diameter_px, canopy: Canopy, *,
                          inflation=1.0, view_elev=0.0) -> tuple[int, int, int, int]:
    """렌더된 캐노피(현삭 제외)의 투영 bbox와 등가 면적 계산용 축."""
    cx, cy = center
    r = max(1.0, diameter_px / 2.0)
    infl = float(np.clip(inflation, 0.05, 1.0))
    width = r * (0.35 + 0.65 * infl)
    base_aspect = _aspect_for_kind(canopy.kind)
    view_factor = base_aspect + (1.0 - base_aspect) * math.sin(min(max(view_elev, 0.0), math.pi / 2))
    height = width * view_factor * (0.7 + 0.3 * infl)
    dome_cy = cy - height * 0.15
    x = int(round(cx - width))
    y = int(round(dome_cy - height))
    w = int(round(2 * width))
    h = int(round(1.32 * height))   # 돔 + 스커트
    return x, y, w, h


def canopy_projected_area(diameter_px, canopy: Canopy, *, inflation=1.0, view_elev=0.0) -> float:
    """투영 면적(px^2) 추정 = 반타원 돔 면적."""
    r = max(1.0, diameter_px / 2.0)
    infl = float(np.clip(inflation, 0.05, 1.0))
    width = r * (0.35 + 0.65 * infl)
    base_aspect = _aspect_for_kind(canopy.kind)
    view_factor = base_aspect + (1.0 - base_aspect) * math.sin(min(max(view_elev, 0.0), math.pi / 2))
    height = width * view_factor * (0.7 + 0.3 * infl)
    return math.pi * width * height * 0.5 + math.pi * width * (height * 0.32) * 0.5


def _scale_color(c, f):
    return tuple(int(max(0, min(255, v * f))) for v in c)
