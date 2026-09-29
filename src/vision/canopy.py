"""낙하산 캐노피 형상 렌더러 (Spec B 확장) — 실제 Rocketman 제품 형상 반영.

형상 출처 (웹 확인 2026-09-29, configs/parachute_specs.md에 상세 인용):
- Main = Rocketman High-Performance CD 2.2 → 제품 페이지 "Type: Toroidal / Annular / Iris".
  반구 돔이 아니라 apex를 캐노피 안으로 당긴 도넛형(pull-down apex)으로,
  중앙에 큰 spill hole이 있다. 10ft = gore 12개, spill hole 21.12"/120" ≈ 17.6%.
  (the-rocketman.com/products/rocketman-high-performance-cd-2-2-parachutes)
- Drogue = Rocketman Pro-Experimental (Cd 0.97) → 4-line 평면/원형 캐노피, spill hole 없음.
  (the-rocketman.com/products/pro-experimental-drogue-parachutes)
- 토로이달 개념 참고: Fruity Chutes Iris Ultra ("doughnut cut in half").

모델링:
- toroidal : 중앙 구멍이 뚫린 링형 캐노피 + 교대 gore 색 + spill hole 테두리 loop
- round    : 단순 평면/원형 캐노피(음영 있는 원반), 소수 gore
- hemispherical/elliptical : (하위호환) 볼록 돔

주의(공통 규칙): 정확한 형상 모델링이 실제 낙하산 **검출 성공**을 보장하지 않는다.
이 렌더러는 검출기가 실영상에서 마주칠 어려움(중앙 구멍, 비대칭, 팽창 변화,
관측각)을 더 잘 재현하기 위한 근사다. 색 배치·정확한 투영각은 추정이다.
모든 치수는 픽셀 단위이며, 절대 거리/속도는 다루지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Canopy:
    """낙하산 캐노피 렌더 파라미터. 모든 크기는 픽셀."""
    kind: str = "toroidal"             # toroidal | round | hemispherical | elliptical
    color: tuple[int, int, int] = (60, 160, 230)   # BGR (기본: 주황 계열)
    alt_color: tuple[int, int, int] = (230, 230, 230)  # 교대 gore 색
    n_panels: int = 12                 # gore(패널) 수
    shroud_color: tuple[int, int, int] = (200, 200, 200)
    n_shroud: int = 12                 # 표시할 현삭 수
    shroud_len_ratio: float = 1.3      # confluence까지 거리 / 캐노피 반경
    edge_darken: float = 0.55          # 가장자리 음영 계수(구형감)
    spill_hole_ratio: float = 0.176    # 중앙 spill hole 지름 / 캐노피 지름 (toroidal)
    render_backend: str = "2d"         # "2d"(기존) | "mesh"(3D). 기본 2d → 동작 보존.


def _aspect_for_kind(kind: str) -> float:
    """정면 관측 시 캐노피 투영의 세로/가로 비(높이/폭)."""
    if kind == "toroidal":
        return 0.70      # 토로이달 링을 비스듬히 본 눌린 타원
    if kind == "hemispherical":
        return 0.62
    if kind == "elliptical":
        return 0.55
    return 0.78          # round/flat (거의 원반)


def _shape_geometry(diameter_px, canopy: Canopy, inflation, view_elev):
    """공통 투영 형상(폭·높이·중심오프셋)을 계산해 렌더·bbox·면적이 일치하도록 한다."""
    r = max(1.0, diameter_px / 2.0)
    infl = float(np.clip(inflation, 0.05, 1.0))
    width = r * (0.35 + 0.65 * infl)
    base_aspect = _aspect_for_kind(canopy.kind)
    view_factor = base_aspect + (1.0 - base_aspect) * math.sin(
        min(max(view_elev, 0.0), math.pi / 2))
    height = width * view_factor * (0.7 + 0.3 * infl)
    return width, height, infl


def render_canopy(
    img: np.ndarray,
    center: tuple[float, float],
    diameter_px: float,
    canopy: Canopy,
    *,
    inflation: float = 1.0,     # 0=reefed, 1=full 팽창
    view_elev: float = 0.0,     # 관측 고도각(rad). 0=정면(옆), pi/2=바로 아래
    occlude_top_frac: float = 0.0,  # 상단 가림 비율
    roll_deg: float = 0.0,      # 캐노피 면내 자전(도). 사실성 옵션.
    hole_color: tuple[int, int, int] | None = None,  # toroidal spill hole 색(하늘 통과)
    tilt_x_deg: float = 0.0,    # 진자 흔들림 기울임 X(도) — mesh 백엔드에서 사용
    tilt_y_deg: float = 0.0,    # 진자 흔들림 기울임 Y(도) — mesh 백엔드에서 사용
) -> None:
    """캐노피 + shroud line을 img에 그린다(in place)."""
    if canopy.render_backend == "mesh":
        _render_canopy_mesh(img, center, diameter_px, canopy,
                            inflation=inflation, view_elev=view_elev,
                            roll_deg=roll_deg, tilt_x_deg=tilt_x_deg,
                            tilt_y_deg=tilt_y_deg)
        return
    cx, cy = center
    width, height, infl = _shape_geometry(diameter_px, canopy, inflation, view_elev)
    ax = int(round(width))
    ay = int(round(height))
    if ax < 1 or ay < 1:
        return

    if canopy.kind == "toroidal":
        _render_toroidal(img, cx, cy, ax, ay, canopy, infl, roll_deg, hole_color)
    elif canopy.kind == "round":
        _render_round(img, cx, cy, ax, ay, canopy, roll_deg)
    else:
        _render_dome(img, cx, cy, ax, ay, canopy, height)

    # --- shroud lines: 캐노피 하단 가장자리 → confluence point ---
    conf = (cx, cy + height * canopy.shroud_len_ratio)
    skirt_y = cy + ay * 0.15
    for k in range(canopy.n_shroud):
        t = k / max(1, canopy.n_shroud - 1)
        ex = cx - ax + 2 * ax * t
        cv2.line(img, (int(ex), int(skirt_y)), (int(conf[0]), int(conf[1])),
                 canopy.shroud_color, 1, lineType=cv2.LINE_AA)

    # --- 상단 가림(occlusion) ---
    if occlude_top_frac > 0:
        cover_h = int(round(2 * ay * occlude_top_frac))
        top = int(cy) - ay
        cv2.rectangle(img, (int(cx - ax) - 4, top - 4),
                      (int(cx + ax) + 4, top + cover_h), (40, 40, 40), thickness=-1)


def _render_toroidal(img, cx, cy, ax, ay, canopy: Canopy, infl, roll_deg=0.0, hole_color=None):
    """도넛형(annular) 캐노피: 교대 gore 링 + 중앙 spill hole.

    apex가 안으로 당겨져 위에서/비스듬히 보면 가운데가 뚫린 링으로 보인다.
    roll_deg만큼 gore 배치를 회전시켜 자전을 표현한다. hole_color가 주어지면
    spill hole을 그 색(=뒤의 하늘)으로 칠해 '구멍으로 하늘이 보이는' 실제 형상을 낸다.
    """
    cxi, cyi = int(cx), int(cy)
    # 바깥 링을 gore별 부채꼴로 채운다(전체 360도, 링 모양).
    seg = 360.0 / max(1, canopy.n_panels)
    for i in range(canopy.n_panels):
        a0 = roll_deg + i * seg
        a1 = roll_deg + (i + 1) * seg
        col = canopy.color if i % 2 == 0 else canopy.alt_color
        cv2.ellipse(img, (cxi, cyi), (ax, ay), 0, a0, a1, col,
                    thickness=-1, lineType=cv2.LINE_AA)

    # 가장자리 음영(둥근 볼륨감)
    overlay = img.copy()
    cv2.ellipse(overlay, (cxi, cyi), (ax, ay), 0, 0, 360,
                _scale_color(canopy.color, canopy.edge_darken),
                thickness=max(1, ax // 8), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)

    # 중앙 spill hole: 실제로는 '뒤의 하늘'이 보인다. hole_color(하늘색)가 주어지면
    # 그 색으로, 없으면(기존 동작) 캐노피보다 어두운 색으로 구멍을 표현한다.
    hx = max(1, int(round(ax * canopy.spill_hole_ratio)))
    hy = max(1, int(round(ay * canopy.spill_hole_ratio)))
    fill = hole_color if hole_color is not None else (35, 35, 40)
    cv2.ellipse(img, (cxi, cyi), (hx, hy), 0, 0, 360, fill,
                thickness=-1, lineType=cv2.LINE_AA)
    # spill hole 테두리 loop(밝은 링)
    cv2.ellipse(img, (cxi, cyi), (hx, hy), 0, 0, 360,
                _scale_color(canopy.alt_color, 0.9), thickness=1, lineType=cv2.LINE_AA)


def _render_round(img, cx, cy, ax, ay, canopy: Canopy, roll_deg=0.0):
    """단순 평면/원형 캐노피(drogue). 소수 gore + 반경 음영, spill hole 없음."""
    cxi, cyi = int(cx), int(cy)
    seg = 360.0 / max(1, canopy.n_panels)
    for i in range(canopy.n_panels):
        a0 = roll_deg + i * seg
        a1 = roll_deg + (i + 1) * seg
        col = canopy.color if i % 2 == 0 else canopy.alt_color
        cv2.ellipse(img, (cxi, cyi), (ax, ay), 0, a0, a1, col,
                    thickness=-1, lineType=cv2.LINE_AA)
    overlay = img.copy()
    cv2.ellipse(overlay, (cxi, cyi), (ax, ay), 0, 0, 360,
                _scale_color(canopy.color, canopy.edge_darken),
                thickness=max(1, ax // 6), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)


def _render_dome(img, cx, cy, ax, ay, canopy: Canopy, height):
    """하위호환: 볼록 반구/타원 돔(상단 반원 + 스커트)."""
    dome_cy = int(round(cy - height * 0.15))
    seg = 360 / max(1, canopy.n_panels)
    for i in range(canopy.n_panels):
        a0 = 180 + i * seg
        a1 = 180 + (i + 1) * seg
        if a0 >= 360:
            break
        col = canopy.color if i % 2 == 0 else canopy.alt_color
        cv2.ellipse(img, (int(cx), dome_cy), (ax, ay), 0,
                    a0, min(a1, 360), col, thickness=-1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (int(cx), dome_cy), (ax, int(ay * 0.32)), 0,
                0, 180, _scale_color(canopy.color, 0.85), thickness=-1, lineType=cv2.LINE_AA)
    overlay = img.copy()
    cv2.ellipse(overlay, (int(cx), dome_cy), (ax, ay), 0, 180, 360,
                _scale_color(canopy.color, canopy.edge_darken),
                thickness=max(1, ax // 8), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)


def canopy_projected_bbox(center, diameter_px, canopy: Canopy, *,
                          inflation=1.0, view_elev=0.0,
                          roll_deg=0.0, tilt_x_deg=0.0, tilt_y_deg=0.0,
                          img_shape=(480, 640)) -> tuple[int, int, int, int]:
    """렌더된 캐노피(현삭 제외)의 투영 bbox. mesh 백엔드는 실루엣에서 계산."""
    if canopy.render_backend == "mesh":
        _mesh, m, R, cam, scale = _build_mesh_and_pose(
            center, diameter_px, canopy, inflation=inflation, view_elev=view_elev,
            roll_deg=roll_deg, tilt_x_deg=tilt_x_deg, tilt_y_deg=tilt_y_deg)
        met = _mesh.mesh_projected_metrics(m, R, center, scale, cam, img_shape)
        return met["bbox"]
    cx, cy = center
    width, height, _ = _shape_geometry(diameter_px, canopy, inflation, view_elev)
    if canopy.kind in ("toroidal", "round"):
        x = int(round(cx - width))
        y = int(round(cy - height))
        w = int(round(2 * width))
        h = int(round(2 * height))
        return x, y, w, h
    # 돔 계열: 중심을 위로 올린 반원 + 스커트
    dome_cy = cy - height * 0.15
    x = int(round(cx - width))
    y = int(round(dome_cy - height))
    w = int(round(2 * width))
    h = int(round(1.32 * height))
    return x, y, w, h


def canopy_projected_area(diameter_px, canopy: Canopy, *, inflation=1.0, view_elev=0.0,
                          center=None, roll_deg=0.0, tilt_x_deg=0.0, tilt_y_deg=0.0,
                          img_shape=(480, 640)) -> float:
    """투영 면적(px^2). mesh 백엔드는 실루엣 면적(구멍 자동 반영)."""
    if canopy.render_backend == "mesh":
        c = center if center is not None else (img_shape[1] / 2, img_shape[0] / 2)
        _mesh, m, R, cam, scale = _build_mesh_and_pose(
            c, diameter_px, canopy, inflation=inflation, view_elev=view_elev,
            roll_deg=roll_deg, tilt_x_deg=tilt_x_deg, tilt_y_deg=tilt_y_deg)
        met = _mesh.mesh_projected_metrics(m, R, c, scale, cam, img_shape)
        return met["area"]
    width, height, _ = _shape_geometry(diameter_px, canopy, inflation, view_elev)
    if canopy.kind == "toroidal":
        outer = math.pi * width * height
        hole = math.pi * (width * canopy.spill_hole_ratio) * (height * canopy.spill_hole_ratio)
        return outer - hole
    if canopy.kind == "round":
        return math.pi * width * height
    # 돔 계열: 반타원 돔 + 스커트
    return math.pi * width * height * 0.5 + math.pi * width * (height * 0.32) * 0.5


def _scale_color(c, f):
    return tuple(int(max(0, min(255, v * f))) for v in c)


# =====================================================================
# 3D mesh 백엔드 (M1~M4) — vision.mesh 모듈로 위임. 2D 경로는 이 코드를 타지 않는다.
# =====================================================================
def _build_mesh_and_pose(center, diameter_px, canopy: Canopy, *,
                         inflation, view_elev, roll_deg, tilt_x_deg, tilt_y_deg):
    """메시·자세·스케일·카메라를 렌더/메트릭이 공유하도록 한 곳에서 만든다."""
    from . import mesh as _mesh  # 지연 임포트(2D 경로엔 불필요)
    kind = canopy.kind if canopy.kind in ("toroidal", "round") else "round"
    m = _mesh.build_canopy_mesh(
        kind, radius=1.0, n_panels=canopy.n_panels,
        color=canopy.color, alt_color=canopy.alt_color,
        inflation=inflation, spill_hole_ratio=canopy.spill_hole_ratio,
        n_shroud=canopy.n_shroud, shroud_len_ratio=canopy.shroud_len_ratio)
    R = _mesh.pose_matrix(view_elev, roll=math.radians(roll_deg),
                          tilt_x=math.radians(tilt_x_deg),
                          tilt_y=math.radians(tilt_y_deg))
    cam = _mesh.Camera(cx=center[0], cy=center[1])
    scale = max(1.0, diameter_px / 2.0)
    return _mesh, m, R, cam, scale


def _render_canopy_mesh(img, center, diameter_px, canopy: Canopy, *,
                        inflation, view_elev, roll_deg, tilt_x_deg, tilt_y_deg):
    _mesh, m, R, cam, scale = _build_mesh_and_pose(
        center, diameter_px, canopy, inflation=inflation, view_elev=view_elev,
        roll_deg=roll_deg, tilt_x_deg=tilt_x_deg, tilt_y_deg=tilt_y_deg)
    _mesh.render_mesh(img, m, R, center, scale, cam,
                      shroud_color=canopy.shroud_color)
