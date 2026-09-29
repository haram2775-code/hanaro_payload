"""IREC 회수 영상 사실성 레이어 (Spec B 확장) — 배경·조명·대기·센서.

목적: 합성 장면을 실제 IREC(Intercollegiate Rocket Engineering Competition,
뉴멕시코 Spaceport America) 낙하산 회수 영상에 가깝게 만들어, 추적 검증이
실제 난이도를 대표하도록 한다.

주의(공통 규칙 §1·§2·§3):
- 여기의 IREC 조건(하늘색, 태양 방향, 노이즈 세기, 촬영 거리 등)은 실측이 아니라
  **가정/설정값**이다. 확정 임무 요구조건이 아니다.
- 사실성이 높아진다고 실제 검출 성공이 보장되지 않는다. 이 레이어는 검출기가
  실영상에서 마주칠 어려움을 더 잘 재현하기 위한 근사다.
- 모든 치수는 픽셀 단위이며 절대 거리/속도는 다루지 않는다.

기존 동작 보존: RealismConfig.enabled=False(기본)이면 이 레이어는 아무것도 하지
않고, synth의 기존 회색-하늘 경로가 그대로 쓰인다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RealismConfig:
    """IREC 사실성 파라미터. 전부 설정값/가정. enabled=False면 완전 비활성(기존 동작)."""
    enabled: bool = False

    # --- 하늘 (BGR) ---
    sky_top: tuple[int, int, int] = (150, 95, 45)      # 천정 진파랑
    sky_horizon: tuple[int, int, int] = (205, 180, 150)  # 지평선 옅은 파랑/회청
    clouds: float = 0.25          # 권운 강도 0..1 (0=맑음)
    cloud_seed: int = 7

    # --- 지형 (하단 사막 밴드) ---
    terrain_frac: float = 0.0     # 화면 하단 지형 비율 0..1 (0=지형 없음)
    terrain_color: tuple[int, int, int] = (110, 150, 185)  # 황갈색 사막(BGR)

    # --- 조명 ---
    sun_dir_deg: float = 60.0     # 태양 방위(도). 캐노피 상단 하이라이트 방향
    sun_intensity: float = 0.35   # 방향성 조명 세기 0..1
    backlight: float = 0.20       # 백라이트 반투과(캐노피가 밝아짐) 0..1

    # --- 대기/광학 ---
    heat_shimmer: float = 0.0     # 열 아지랑이 warp 진폭(px) 0=없음
    camera_jitter_px: float = 0.0  # 프레임별 카메라 흔들림 표준편차(px)
    motion_blur: int = 0          # 모션블러 커널 길이(px) 0=없음

    # --- 센서 ---
    sensor_noise: float = 4.0     # 가우시안 노이즈 표준편차
    shot_noise: float = 0.0       # 광자(shot) 노이즈 세기 0..1
    jpeg_quality: int = 0         # 0=압축 없음, 1..100=JPEG 재인코딩(블록 아티팩트)


def make_sky(width: int, height: int, cfg: RealismConfig, rng) -> np.ndarray:
    """IREC 파란 하늘 그라디언트 + 선택적 구름 + 선택적 사막 지형."""
    top = np.array(cfg.sky_top, dtype=np.float32)
    hor = np.array(cfg.sky_horizon, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # 0=위, 1=아래
    grad = top[None, :] * (1.0 - ramp) + hor[None, :] * ramp         # (H,3)
    frame = np.repeat(grad[:, None, :], width, axis=1)               # (H,W,3)

    if cfg.clouds > 0:
        frame = _add_clouds(frame, cfg, rng)

    if cfg.terrain_frac > 0:
        th = int(round(height * cfg.terrain_frac))
        if th > 0:
            terr = np.array(cfg.terrain_color, dtype=np.float32)
            band = frame[height - th:, :, :]
            # 지형은 위쪽 경계에서 하늘과 살짝 섞이도록.
            blendrows = np.linspace(0.4, 1.0, th, dtype=np.float32)[:, None, None]
            frame[height - th:, :, :] = band * (1.0 - blendrows) + terr[None, None, :] * blendrows
            # 지형 질감(관목/토양)
            frame[height - th:, :, :] += rng.normal(0.0, 6.0, (th, width, 3))

    return frame


def _add_clouds(frame: np.ndarray, cfg: RealismConfig, rng) -> np.ndarray:
    """저주파 노이즈를 흐려 얇은 권운을 만든다(밝게 더한다)."""
    h, w = frame.shape[:2]
    small = rng.random((max(2, h // 32), max(2, w // 32))).astype(np.float32)
    cloud = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    cloud = cv2.GaussianBlur(cloud, (0, 0), sigmaX=w / 40.0)
    cloud = np.clip((cloud - 0.5) * 2.0, 0, 1) ** 1.5  # 상위만 남겨 성긴 구름
    add = (cloud * (cfg.clouds * 60.0))[:, :, None]
    return frame + add


def apply_canopy_lighting(img: np.ndarray, center, ax: int, ay: int,
                          cfg: RealismConfig) -> None:
    """캐노피 영역에 방향성 하이라이트/그림자 + 백라이트를 in-place로 적용."""
    if cfg.sun_intensity <= 0 and cfg.backlight <= 0:
        return
    cx, cy = int(center[0]), int(center[1])
    x0, y0 = max(0, cx - ax), max(0, cy - ay)
    x1, y1 = min(img.shape[1], cx + ax), min(img.shape[0], cy + ay)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1].astype(np.float32)
    hh, ww = roi.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    # 태양 방향 성분
    ang = np.deg2rad(cfg.sun_dir_deg)
    nx = (xx - ww / 2.0) / max(1.0, ww / 2.0)
    ny = (yy - hh / 2.0) / max(1.0, hh / 2.0)
    lit = -(nx * np.cos(ang) + ny * np.sin(ang))  # 태양쪽 밝고 반대쪽 어둡게
    lit = np.clip(lit, -1, 1)
    # 캐노피 타원 footprint로 마스킹(사각형 halo 방지). 중심 1 → 가장자리 0으로 페더.
    rad = np.sqrt(nx * nx + ny * ny)
    mask = np.clip(1.0 - rad, 0.0, 1.0)          # 타원 밖은 0
    mask = mask ** 0.6                            # 가장자리 부드럽게
    delta = (lit * (cfg.sun_intensity * 70.0) + cfg.backlight * 40.0) * mask
    roi += delta[:, :, None]
    img[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(img.dtype)


def sky_color_at(cfg: RealismConfig, height: int, y: float) -> tuple[int, int, int]:
    """세로 위치 y에서의 하늘색(BGR). toroidal spill hole을 '하늘 통과'로 칠할 때 사용."""
    r = float(np.clip(y / max(1, height - 1), 0.0, 1.0))
    top = np.array(cfg.sky_top, dtype=np.float32)
    hor = np.array(cfg.sky_horizon, dtype=np.float32)
    c = top * (1.0 - r) + hor * r
    return tuple(int(v) for v in c)


def apply_atmosphere_and_sensor(frame: np.ndarray, cfg: RealismConfig, rng,
                                *, motion_dir_deg: float = 90.0) -> np.ndarray:
    """열 아지랑이 → 모션블러 → 노이즈 → JPEG 순으로 광학/센서 결함을 적용."""
    f = frame.astype(np.float32)

    if cfg.heat_shimmer > 0:
        f = _heat_shimmer(f, cfg, rng)

    if cfg.motion_blur and cfg.motion_blur >= 2:
        f = _motion_blur(f, cfg.motion_blur, motion_dir_deg)

    if cfg.sensor_noise > 0:
        f += rng.normal(0.0, cfg.sensor_noise, f.shape)
    if cfg.shot_noise > 0:
        # 밝기 비례 노이즈(광자 통계 근사)
        f += rng.normal(0.0, 1.0, f.shape) * np.sqrt(np.clip(f, 0, 255)) * cfg.shot_noise

    out = np.clip(f, 0, 255).astype(np.uint8)

    if cfg.jpeg_quality and 1 <= cfg.jpeg_quality <= 100:
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def _heat_shimmer(f: np.ndarray, cfg: RealismConfig, rng) -> np.ndarray:
    h, w = f.shape[:2]
    dx = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    dy = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=8.0) * cfg.heat_shimmer
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=8.0) * cfg.heat_shimmer
    map_x = (np.tile(np.arange(w, dtype=np.float32), (h, 1)) + dx)
    map_y = (np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) + dy)
    return cv2.remap(f, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _motion_blur(f: np.ndarray, length: int, dir_deg: float) -> np.ndarray:
    length = int(length)
    kernel = np.zeros((length, length), dtype=np.float32)
    ang = np.deg2rad(dir_deg)
    cx = cy = (length - 1) / 2.0
    for i in range(length):
        t = i - (length - 1) / 2.0
        x = int(round(cx + t * np.cos(ang)))
        y = int(round(cy + t * np.sin(ang)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s <= 0:
        return f
    kernel /= s
    return cv2.filter2D(f, -1, kernel)


# ---------------------------------------------------------------------------
# 추적 판정(trackability) 지표 — 개선 효과를 정량 확인.
# ---------------------------------------------------------------------------
@dataclass
class TrackabilityResult:
    frames: int = 0
    trackable_frames: int = 0
    mean_michelson: float = 0.0
    mean_target_width_px: float = 0.0
    detail: list = field(default_factory=list)

    @property
    def trackable_ratio(self) -> float:
        return self.trackable_frames / self.frames if self.frames else 0.0


def frame_trackability(frame: np.ndarray, center, ax: int, ay: int,
                       *, min_width_px: float = 8.0,
                       min_michelson: float = 0.12) -> dict:
    """한 프레임의 추적 가능성: 표적/국소배경 Michelson 대비 + 표적 유효폭.

    판정 기준(가정): 표적 유효폭 >= min_width_px 이고 대비 >= min_michelson 이면 trackable.
    """
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    ax = max(1, int(ax)); ay = max(1, int(ay))

    tx0, ty0 = max(0, cx - ax), max(0, cy - ay)
    tx1, ty1 = min(w, cx + ax), min(h, cy + ay)
    if tx1 <= tx0 or ty1 <= ty0:
        return {"trackable": False, "michelson": 0.0, "width_px": 0.0}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target = gray[ty0:ty1, tx0:tx1]

    # 국소 배경 링(표적 bbox를 1.6배 확장한 테두리)
    bx0, by0 = max(0, cx - int(ax * 1.6)), max(0, cy - int(ay * 1.6))
    bx1, by1 = min(w, cx + int(ax * 1.6)), min(h, cy + int(ay * 1.6))
    ring = gray[by0:by1, bx0:bx1].copy()
    # 중앙(표적) 부분을 링 통계에서 제외
    iy0, ix0 = ty0 - by0, tx0 - bx0
    iy1, ix1 = iy0 + (ty1 - ty0), ix0 + (tx1 - tx0)
    mask = np.ones(ring.shape, dtype=bool)
    mask[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = False
    bg_vals = ring[mask]
    if bg_vals.size == 0:
        return {"trackable": False, "michelson": 0.0, "width_px": float(2 * ax)}

    t_mean = float(target.mean())
    b_mean = float(bg_vals.mean())
    denom = t_mean + b_mean
    michelson = abs(t_mean - b_mean) / denom if denom > 0 else 0.0
    width_px = float(2 * ax)
    trackable = (width_px >= min_width_px) and (michelson >= min_michelson)
    return {"trackable": bool(trackable), "michelson": michelson, "width_px": width_px}
