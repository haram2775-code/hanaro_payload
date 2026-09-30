"""Spec B — 합성영상·정답 데이터 생성기 (다중 낙하산 표적 + 정확한 형상).

설정과 seed가 주어지면 결정론적으로 시험 영상과 표적별 정답값(GT) CSV를 생성한다.
표적은 실제 Rocketman 낙하산 사양(출처: the-rocketman.com 제품 페이지, 2026-09-29,
configs/parachute_specs.md 참조)을 기반으로 형상을 모델링한다:
Main=Toroidal/Annular(중앙 spill hole), Drogue=평면-원형(4-line). 전개 단계·현삭·관측각 포함.

주의(공통 규칙): 형상 모델링의 정확도가 실제 검출 성공을 보장하지 않는다.
절대 거리/m/s는 다루지 않으며 모든 크기·중심은 픽셀 단위다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from . import code_version
from .canopy import Canopy, render_canopy, canopy_projected_bbox, canopy_projected_area
from .scene import (
    RealismConfig, make_sky, apply_canopy_lighting, apply_atmosphere_and_sensor,
    sky_color_at, frame_trackability, TrackabilityResult,
)

# GT 상태 어휘는 Spec A와 정합한다.
STATE_TRACK = "TRACK"
STATE_OCCLUDED = "OCCLUDED"
STATE_CLIPPED = "CLIPPED"
STATE_INFLATING = "INFLATING"
STATE_LOST = "LOST"


@dataclass
class Target:
    """합성 표적 하나. 픽셀 단위. 모든 값은 설정값/가정."""
    name: str = "main"
    kind: str = "toroidal"                 # 캐노피 형상 (Main=toroidal, Drogue=round)
    color: tuple[int, int, int] = (40, 150, 240)   # BGR
    alt_color: tuple[int, int, int] = (240, 240, 240)
    n_panels: int = 12
    n_shroud: int = 12
    spill_hole_ratio: float = 0.176        # toroidal 중앙 구멍 비율
    start_diameter: float = 120.0          # 시작 등가직경(px)
    start_center: tuple[float, float] = (320.0, 200.0)
    velocity_px: tuple[float, float] = (0.0, 0.0)  # 프레임당 중심 이동
    diameter_growth_per_frame: float = 1.0  # 크기 배율(접근>1, 이격<1)
    view_elev_deg: float = 25.0            # 관측 고도각(도, 시작)
    view_elev_deg_end: float | None = None # 끝 관측각(도). None이면 고정각(기존 동작).
    # 전개: [start,end] 구간에서 inflation 0.2→1.0
    inflate_range: tuple[int, int] | None = None
    inflate_from: float = 0.2
    # 가림
    occlude_range: tuple[int, int] | None = None
    occlude_fraction: float = 0.6
    # 상실
    lost_range: tuple[int, int] | None = None
    # --- 낙하 동역학(사실성, 옵션). 기본 0 → 기존 동작 보존. 값은 설정값/가정. ---
    swing_amp_px: float = 0.0        # 진자 흔들림 진폭(px)
    swing_period_frames: float = 40.0  # 진자 주기(프레임)
    rotate_deg_per_frame: float = 0.0  # 캐노피 자전(도/프레임)
    wind_drift_px: float = 0.0       # 프레임당 수평 바람 표류(px)
    render_backend: str = "2d"       # "2d"(기존) | "mesh"(3D). 기본 2d → 동작 보존.

    def canopy(self) -> Canopy:
        return Canopy(kind=self.kind, color=self.color, alt_color=self.alt_color,
                      n_panels=self.n_panels, n_shroud=self.n_shroud,
                      spill_hole_ratio=self.spill_hole_ratio,
                      render_backend=self.render_backend)


# =====================================================================
# 순차 전개 시퀀스 (1급 개념) — 기존 다중표적 모델과 별개의 추가 경로.
# =====================================================================
# 물리 흐름: drogue가 먼저 전개되고, 전개 완료가 확인되면 main 전개로 전이한다.
# "두 개의 동시 표적"이 아니라 "시간에 따라 이어지는 단일 전개 시퀀스"이며,
# 매 순간 활성 단계(Stage) 1개만 추적/기록한다.
#
# 상태 전이(각 Stage): DEPLOYING → DEPLOYED → (HANDOFF) → 다음 Stage.DEPLOYING
#   - DEPLOYING : inflation < 1.0 (전개 중)
#   - DEPLOYED  : inflation == 1.0 도달 후 hold_frames 동안 안정 (= "전개 완료 확인")
#   - HANDOFF   : overlap_frames > 0일 때 다음 단계로 넘어가는 전환 구간
#   - RELEASED  : 전환이 끝나 더 이상 활성이 아닌(사라진) 단계
# 전개 완료 판정 기준(가정): inflation이 1.0에 도달하고 hold_frames 만큼 유지되면
# "전개 완료(DEPLOYED)"로 확정하고 다음 단계로 전이한다.

STATE_DEPLOYING = "DEPLOYING"
STATE_DEPLOYED = "DEPLOYED"
STATE_HANDOFF = "HANDOFF"
STATE_RELEASED = "RELEASED"


@dataclass
class Stage:
    """전개 시퀀스의 한 단계. 활성 캐노피 1개 + 그 단계의 운동/전개 파라미터."""
    target: Target                          # 이 단계에서 보이는 캐노피(형상·색·gore)
    inflate_frames: int = 20                # 전개(팽창)에 걸리는 프레임 수
    inflate_from: float = 0.2               # 전개 시작 시 팽창률
    hold_frames: int = 10                   # 전개 완료 후 "완료 확인" 유지 프레임 수
    overlap_frames: int = 0                 # 다음 단계와 겹쳐 보이는 전환 프레임(0=겹침 없음)

    def duration(self) -> int:
        """이 단계가 활성인 총 프레임 수(전개 + 완료 유지)."""
        return max(1, self.inflate_frames + self.hold_frames)


@dataclass
class DeploymentSequence:
    """순차 전개 시퀀스: 순서 있는 Stage 리스트. drogue → main 흐름을 표현."""
    stages: list[Stage] = field(default_factory=list)

    def timeline(self) -> list[dict]:
        """각 단계의 [start, deployed_at, end] 프레임 경계를 계산한다.

        start        : 단계 활성 시작 프레임
        deploying_end: 전개 완료(DEPLOYED 도달) 프레임
        end          : 단계 활성 종료(다음 단계 시작 직전). overlap이 있으면 겹친다.
        """
        marks = []
        cursor = 0
        for st in self.stages:
            start = cursor
            deploying_end = start + st.inflate_frames
            end = start + st.duration()
            marks.append({"start": start, "deploying_end": deploying_end, "end": end})
            # 다음 단계는 이 단계 종료에서 overlap 만큼 앞당겨 시작.
            cursor = max(start + 1, end - st.overlap_frames)
        return marks

    def total_frames(self) -> int:
        marks = self.timeline()
        return max((m["end"] for m in marks), default=1)


def _seq_inflation(i: int, start: int, inflate_frames: int, inflate_from: float) -> float:
    """단계 로컬 프레임 기준 팽창률(0..1)."""
    if inflate_frames <= 0:
        return 1.0
    if i < start:
        return inflate_from
    frac = (i - start) / inflate_frames
    if frac >= 1.0:
        return 1.0
    return inflate_from + (1.0 - inflate_from) * frac


def _seq_stage_state(i: int, mark: dict, stage: Stage, active_next_start: int | None) -> str:
    """시퀀스 단계의 프레임별 상태(전개 상태 어휘)를 판정한다."""
    if i < mark["deploying_end"]:
        return STATE_DEPLOYING
    # 전개 완료 이후: 다음 단계가 이미 시작했으면(겹침) HANDOFF, 아니면 DEPLOYED
    if active_next_start is not None and i >= active_next_start:
        return STATE_HANDOFF
    return STATE_DEPLOYED


@dataclass
class SceneConfig:
    """합성 시나리오 설정. 모든 값은 설정값/가정이며 확정 요구조건이 아니다."""
    name: str = "main_only"
    width: int = 640
    height: int = 480
    fps: float = 30.0
    frames: int = 90
    seed: int = 1234
    bg_gray: int = 70            # 하늘 배경 밝기
    bg_noise: float = 3.0
    sky_gradient: bool = True    # 위→아래 밝기 그라디언트(하늘)
    blur_ksize: int = 0
    targets: list[Target] = field(default_factory=lambda: [Target()])
    # 순차 전개 시퀀스(1급 개념). None이면 기존 다중표적 경로를 쓴다(동작 보존).
    sequence: "DeploymentSequence | None" = None
    # IREC 사실성 레이어(옵션). None이면 기존 회색-하늘 경로 그대로(동작 보존).
    realism: "RealismConfig | None" = None


def _in_range(i: int, rng: tuple[int, int] | None) -> bool:
    return rng is not None and rng[0] <= i <= rng[1]


def _inflation(i: int, t: Target) -> float:
    if t.inflate_range is None:
        return 1.0
    s, e = t.inflate_range
    if i < s:
        return t.inflate_from
    if i > e:
        return 1.0
    frac = (i - s) / max(1, e - s)
    return t.inflate_from + (1.0 - t.inflate_from) * frac


def _make_background(cfg: SceneConfig, rng) -> np.ndarray:
    # 사실성 레이어가 켜져 있으면 IREC 하늘/지형을 만든다(기존 경로 대체 X, 분기).
    if cfg.realism is not None and cfg.realism.enabled:
        return np.clip(make_sky(cfg.width, cfg.height, cfg.realism, rng), 0, 255).astype(np.uint8)
    frame = np.full((cfg.height, cfg.width, 3), cfg.bg_gray, dtype=np.float32)
    if cfg.sky_gradient:
        grad = np.linspace(cfg.bg_gray + 40, cfg.bg_gray - 20, cfg.height, dtype=np.float32)
        frame = np.repeat(grad[:, None, None], cfg.width, axis=1)
        frame = np.repeat(frame, 3, axis=2)
    if cfg.bg_noise > 0:
        frame += rng.normal(0.0, cfg.bg_noise, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _dynamics_offset(i: int, t: Target) -> tuple[float, float, float, float, float]:
    """프레임 i에서 진자 흔들림/바람 표류/자전을 계산한다(사실성 옵션).

    반환: (dx, dy, roll_deg, tilt_x_deg, tilt_y_deg). 기본값이면 (0,0,0,0,0)이라
    기존 동작과 동일. tilt_*는 mesh 백엔드에서 진자 기울임으로 쓰인다(2D는 무시).
    """
    dx = 0.0
    dy = 0.0
    tilt_x = 0.0
    tilt_y = 0.0
    if t.swing_amp_px > 0 and t.swing_period_frames > 0:
        phase = 2.0 * math.pi * i / t.swing_period_frames
        dx += t.swing_amp_px * math.sin(phase)
        dy += 0.3 * t.swing_amp_px * math.sin(2.0 * phase)  # 수직은 2배 주파수 소진폭
        # 진자 기울임: 흔들림 위상에 비례해 캐노피가 기운다(최대 ~18도).
        tilt_y += 18.0 * math.sin(phase) * min(1.0, t.swing_amp_px / 12.0)
        tilt_x += 6.0 * math.sin(2.0 * phase) * min(1.0, t.swing_amp_px / 12.0)
    if t.wind_drift_px:
        dx += t.wind_drift_px * i
    roll = t.rotate_deg_per_frame * i
    return dx, dy, roll, tilt_x, tilt_y


def _view_elev_rad(t: Target, local_i: int, span: int) -> float:
    """관측각을 프레임에 따라 보간(도→rad). view_elev_deg_end가 None이면 고정각.

    span은 이 표적/단계가 활성인 총 프레임 수(0..span-1 동안 시작→끝 선형 보간).
    """
    a0 = t.view_elev_deg
    if t.view_elev_deg_end is None or span <= 1:
        return math.radians(a0)
    frac = min(1.0, max(0.0, local_i / (span - 1)))
    a = a0 + (t.view_elev_deg_end - a0) * frac
    return math.radians(a)


def generate(cfg: SceneConfig, out_dir: Path) -> dict:
    """영상 + 표적별 GT CSV + 메타데이터를 생성한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    video_path = out_dir / f"{cfg.name}.mp4"
    gt_path = out_dir / f"{cfg.name}_gt.csv"
    meta_path = out_dir / f"{cfg.name}_meta.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, cfg.fps, (cfg.width, cfg.height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 열기 실패: {video_path}")

    # 표적별 동적 상태
    state = [{"cx": t.start_center[0], "cy": t.start_center[1],
              "d": t.start_diameter} for t in cfg.targets]

    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        gt = csv.writer(f)
        gt.writerow([
            "frame_id", "timestamp_s", "target_name",
            "gt_center_x_px", "gt_center_y_px",
            "gt_area_px2", "gt_diameter_px", "gt_inflation",
            "gt_visible", "gt_state",
        ])

        if cfg.sequence is not None:
            track = _render_sequence(cfg, writer, gt, rng)
        else:
            track = _render_targets(cfg, writer, gt, rng, state)

    writer.release()

    realism_on = cfg.realism is not None and cfg.realism.enabled
    meta = {
        "spec": "B-synthetic",
        "code_version": code_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "source_kind": "synthetic",
        "video_id": cfg.name,
        "mode": "sequence" if cfg.sequence is not None else "targets",
        "realism": "irec" if realism_on else "none",
        "n_targets": len(cfg.targets),
        "target_names": [t.name for t in cfg.targets],
        "sequence_stages": (
            [s.target.name for s in cfg.sequence.stages] if cfg.sequence else None
        ),
        "trackability": {
            "frames_measured": track.frames,
            "trackable_frames": track.trackable_frames,
            "trackable_ratio": round(track.trackable_ratio, 4),
            "mean_michelson_contrast": round(track.mean_michelson, 4),
            "mean_target_width_px": round(track.mean_target_width_px, 2),
            "criteria": "trackable = (유효폭>=8px) and (Michelson 대비>=0.12). 가정치.",
        },
        "config": _cfg_to_dict(cfg),
        "video_path": str(video_path),
        "gt_path": str(gt_path),
        "note": "형상 정확도가 실제 검출 성공을 보장하지 않음(공통 규칙). 절대 거리/속도 없음.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _render_targets(cfg: SceneConfig, writer, gt, rng, state) -> TrackabilityResult:
    """기존 다중표적 경로(동작 보존): 매 프레임 모든 표적을 동시에 렌더/기록.

    realism이 켜지면 동역학(흔들림/자전/바람)·조명·하늘색 spill hole·광학/센서
    결함을 추가로 적용한다(기본값이면 아무 영향 없음).
    """
    rl = cfg.realism
    tr = TrackabilityResult()
    for i in range(cfg.frames):
        frame = _make_background(cfg, rng)

        for ti, t in enumerate(cfg.targets):
            st = state[ti]
            st["d"] = max(2.0, st["d"] * t.diameter_growth_per_frame)
            st["cx"] += t.velocity_px[0]
            st["cy"] += t.velocity_px[1]

            dxo, dyo, roll, tx, ty = _dynamics_offset(i, t)
            infl = _inflation(i, t)
            view_elev = _view_elev_rad(t, i, cfg.frames)
            cx, cy, d = st["cx"] + dxo, st["cy"] + dyo, st["d"]

            visible = 1
            tstate = STATE_TRACK
            occ_frac = 0.0

            if _in_range(i, t.inflate_range):
                tstate = STATE_INFLATING
            if _in_range(i, t.occlude_range):
                occ_frac = t.occlude_fraction
                tstate = STATE_OCCLUDED

            if _in_range(i, t.lost_range):
                visible = 0
                tstate = STATE_LOST
            else:
                cnp = t.canopy()
                hole = sky_color_at(rl, cfg.height, cy) if (rl and rl.enabled) else None
                render_canopy(frame, (cx, cy), d, cnp,
                              inflation=infl, view_elev=view_elev,
                              occlude_top_frac=occ_frac, roll_deg=roll, hole_color=hole,
                              tilt_x_deg=tx, tilt_y_deg=ty)
                bx, by, bw, bh = canopy_projected_bbox(
                    (cx, cy), d, cnp, inflation=infl, view_elev=view_elev,
                    roll_deg=roll, tilt_x_deg=tx, tilt_y_deg=ty,
                    img_shape=(cfg.height, cfg.width))
                if (bx < 0 or by < 0 or bx + bw >= cfg.width or by + bh >= cfg.height):
                    tstate = STATE_CLIPPED
                if rl and rl.enabled:
                    apply_canopy_lighting(frame, (cx, cy), bw // 2, bh // 2, rl)
                    # 대표 표적(첫 표적)의 추적성 지표 수집
                    if ti == 0:
                        res = frame_trackability(frame, (cx, cy), bw // 2, bh // 2)
                        tr.frames += 1
                        tr.trackable_frames += int(res["trackable"])
                        tr.mean_michelson += res["michelson"]
                        tr.mean_target_width_px += res["width_px"]

            area = canopy_projected_area(d, t.canopy(), inflation=infl, view_elev=view_elev,
                                         center=(cx, cy), roll_deg=roll,
                                         tilt_x_deg=tx, tilt_y_deg=ty,
                                         img_shape=(cfg.height, cfg.width))
            gt.writerow([
                i, f"{i / cfg.fps:.6f}", t.name,
                f"{cx:.3f}", f"{cy:.3f}",
                f"{area:.3f}", f"{d:.3f}", f"{infl:.3f}",
                visible, tstate,
            ])

        if rl and rl.enabled:
            frame = apply_atmosphere_and_sensor(frame, rl, rng)
        elif cfg.blur_ksize and cfg.blur_ksize >= 3 and cfg.blur_ksize % 2 == 1:
            frame = cv2.GaussianBlur(frame, (cfg.blur_ksize, cfg.blur_ksize), 0)

        writer.write(frame)

    if tr.frames:
        tr.mean_michelson /= tr.frames
        tr.mean_target_width_px /= tr.frames
    return tr


def _render_sequence(cfg: SceneConfig, writer, gt, rng) -> TrackabilityResult:
    """순차 전개 시퀀스 경로(1급 개념): 매 프레임 활성 단계만 렌더/기록.

    각 단계는 자신의 활성 구간 [start, end) 동안만 등장한다. 전개 완료(DEPLOYED)가
    확인되면 다음 단계로 전이하며, overlap_frames > 0인 단계에서만 다음 단계와
    잠깐 겹쳐 보인다(HANDOFF). 그 외에는 매 순간 활성 단계 1개만 존재한다.
    realism이 켜지면 동역학/조명/하늘 spill hole/광학·센서 결함을 추가한다.
    """
    seq = cfg.sequence
    marks = seq.timeline()
    total = cfg.frames
    rl = cfg.realism
    tr = TrackabilityResult()

    # 단계별 동적 상태(중심·직경) — 활성 시작 시점에 초기화.
    st = [{"cx": s.target.start_center[0], "cy": s.target.start_center[1],
           "d": s.target.start_diameter, "started": False} for s in seq.stages]

    for i in range(total):
        frame = _make_background(cfg, rng)
        active_idx = None

        for si, stage in enumerate(seq.stages):
            m = marks[si]
            if not (m["start"] <= i < m["end"]):
                continue  # 이 단계는 아직/이미 비활성

            t = stage.target
            s = st[si]
            if not s["started"]:
                s["started"] = True  # 이미 초기값으로 세팅됨
            else:
                s["d"] = max(2.0, s["d"] * t.diameter_growth_per_frame)
                s["cx"] += t.velocity_px[0]
                s["cy"] += t.velocity_px[1]

            local_i = i - m["start"]
            dxo, dyo, roll, tx, ty = _dynamics_offset(local_i, t)
            infl = _seq_inflation(i, m["start"], stage.inflate_frames, stage.inflate_from)
            view_elev = _view_elev_rad(t, local_i, m["end"] - m["start"])
            cx, cy, d = s["cx"] + dxo, s["cy"] + dyo, s["d"]

            next_start = marks[si + 1]["start"] if si + 1 < len(marks) else None
            stage_state = _seq_stage_state(i, m, stage, next_start)

            cnp = t.canopy()
            hole = sky_color_at(rl, cfg.height, cy) if (rl and rl.enabled) else None
            render_canopy(frame, (cx, cy), d, cnp, inflation=infl, view_elev=view_elev,
                          roll_deg=roll, hole_color=hole, tilt_x_deg=tx, tilt_y_deg=ty)
            bx, by, bw, bh = canopy_projected_bbox(
                (cx, cy), d, cnp, inflation=infl, view_elev=view_elev,
                roll_deg=roll, tilt_x_deg=tx, tilt_y_deg=ty,
                img_shape=(cfg.height, cfg.width))
            if (bx < 0 or by < 0 or bx + bw >= cfg.width or by + bh >= cfg.height):
                stage_state = STATE_CLIPPED
            if rl and rl.enabled:
                apply_canopy_lighting(frame, (cx, cy), bw // 2, bh // 2, rl)
                if active_idx is None:  # 첫(주) 활성 단계의 추적성만 집계
                    active_idx = si
                    res = frame_trackability(frame, (cx, cy), bw // 2, bh // 2)
                    tr.frames += 1
                    tr.trackable_frames += int(res["trackable"])
                    tr.mean_michelson += res["michelson"]
                    tr.mean_target_width_px += res["width_px"]

            area = canopy_projected_area(d, cnp, inflation=infl, view_elev=view_elev,
                                         center=(cx, cy), roll_deg=roll,
                                         tilt_x_deg=tx, tilt_y_deg=ty,
                                         img_shape=(cfg.height, cfg.width))
            gt.writerow([
                i, f"{i / cfg.fps:.6f}", t.name,
                f"{cx:.3f}", f"{cy:.3f}",
                f"{area:.3f}", f"{d:.3f}", f"{infl:.3f}",
                1, stage_state,
            ])

        if rl and rl.enabled:
            frame = apply_atmosphere_and_sensor(frame, rl, rng)
        elif cfg.blur_ksize and cfg.blur_ksize >= 3 and cfg.blur_ksize % 2 == 1:
            frame = cv2.GaussianBlur(frame, (cfg.blur_ksize, cfg.blur_ksize), 0)

        writer.write(frame)

    if tr.frames:
        tr.mean_michelson /= tr.frames
        tr.mean_target_width_px /= tr.frames
    return tr


def _cfg_to_dict(cfg: SceneConfig) -> dict:
    d = asdict(cfg)
    return d


# --- Rocketman 사양 기반 표적 프리셋 (출처: the-rocketman.com 제품 페이지, 2026-09-29) ---
def drogue_target(**kw) -> Target:
    """Drogue: Rocketman Pro-Experimental 3 ft, Cd 0.97.

    형상 출처: Pro Experimental Drogue 제품 페이지 → 4-line 평면/원형 캐노피,
    spill hole 없음. → kind=round, gore/현삭 4개. 고속 구간이라 작고 빠름.
    """
    base = dict(
        name="drogue", kind="round", color=(50, 90, 210), alt_color=(230, 230, 230),
        n_panels=4, n_shroud=4, spill_hole_ratio=0.0,
        start_diameter=42.0, view_elev_deg=20.0,
    )
    base.update(kw)
    return Target(**base)


def main_target(**kw) -> Target:
    """Main: Rocketman High-Performance CD 2.2, 10 ft.

    형상 출처: HPC CD 2.2 제품 페이지(120in 행) → Type "Toroidal/Annular/Iris",
    gore 12, spill hole 21.12"/120"≈17.6%. → kind=toroidal, 중앙 구멍 있음.
    저속 구간이라 크고 느림.
    """
    base = dict(
        name="main", kind="toroidal", color=(40, 150, 240), alt_color=(240, 240, 240),
        n_panels=12, n_shroud=12, spill_hole_ratio=0.176,
        start_diameter=130.0, view_elev_deg=28.0,
    )
    base.update(kw)
    return Target(**base)


# --- 순차 전개 시퀀스 프리셋 (1급 개념) ---
def drogue_then_main_sequence() -> DeploymentSequence:
    """물리 흐름: drogue 먼저 전개 → 완료 확인 → main 전개.

    가정/설정값(configs/parachute_specs.md): drogue는 고속 구간이라 작고 빠르게
    전개, main은 저속 구간이라 크고 천천히 전개된다. 기본은 겹침 없는 깔끔한
    단계 전환(overlap_frames=0). main 단계에서만 살짝 겹치게 하려면 그 값을 키운다.
    """
    return DeploymentSequence(stages=[
        Stage(
            target=drogue_target(start_center=(340.0, 150.0), velocity_px=(-0.4, 1.4),
                                 diameter_growth_per_frame=1.002),
            inflate_frames=12, inflate_from=0.25, hold_frames=18, overlap_frames=0,
        ),
        Stage(
            target=main_target(start_center=(300.0, 250.0), start_diameter=48.0,
                               velocity_px=(0.2, 0.5), diameter_growth_per_frame=1.006),
            inflate_frames=28, inflate_from=0.2, hold_frames=22, overlap_frames=0,
        ),
    ])


def sequence_scenes() -> list[SceneConfig]:
    """시퀀스 기반 시나리오. 기존 regression_scenes()와 별개(추가 경로)."""
    seq = drogue_then_main_sequence()
    return [
        SceneConfig(name="seq_drogue_then_main", frames=seq.total_frames(),
                    sequence=seq),
    ]


# --- IREC 사실성 프리셋 (가정/설정값, configs/irec_conditions.md 참조) ---
def irec_realism(**kw) -> RealismConfig:
    """IREC(Spaceport America, 정오, 사막) 회수 영상 조건 프리셋. 모두 가정치."""
    base = dict(
        enabled=True,
        sky_top=(150, 95, 45), sky_horizon=(205, 180, 150),   # 진파랑→옅은 지평선
        clouds=0.22, terrain_frac=0.0,
        sun_dir_deg=60.0, sun_intensity=0.35, backlight=0.18,
        heat_shimmer=0.6, camera_jitter_px=0.0, motion_blur=0,
        sensor_noise=5.0, shot_noise=0.03, jpeg_quality=85,
    )
    base.update(kw)
    return RealismConfig(**base)


def _irec_sequence() -> DeploymentSequence:
    """IREC 시퀀스: 추적 가능하도록 표적을 더 크게, 동역학(흔들림/자전/바람) 부여."""
    return DeploymentSequence(stages=[
        Stage(
            target=drogue_target(
                start_center=(330.0, 140.0), start_diameter=70.0,
                velocity_px=(-0.3, 1.3), diameter_growth_per_frame=1.003,
                swing_amp_px=6.0, swing_period_frames=26.0,
                rotate_deg_per_frame=2.2, wind_drift_px=0.15,
            ),
            inflate_frames=12, inflate_from=0.25, hold_frames=18, overlap_frames=0,
        ),
        Stage(
            target=main_target(
                start_center=(300.0, 240.0), start_diameter=90.0,
                velocity_px=(0.15, 0.45), diameter_growth_per_frame=1.006,
                swing_amp_px=10.0, swing_period_frames=44.0,
                rotate_deg_per_frame=1.1, wind_drift_px=0.1,
            ),
            inflate_frames=28, inflate_from=0.2, hold_frames=26, overlap_frames=0,
        ),
    ])


def irec_scenes() -> list[SceneConfig]:
    """IREC 사실성 시나리오(추가 경로). 기존 시나리오·시퀀스는 그대로 보존."""
    seq = _irec_sequence()
    return [
        # 순차 전개(drogue→main), 진한 파란 하늘 + 대기/센서 결함
        SceneConfig(name="irec_seq_drogue_then_main", frames=seq.total_frames(),
                    sequence=seq, realism=irec_realism()),
        # 근접 하강: 큰 main, 진자 흔들림/자전, 지평선 근처 지형 밴드
        SceneConfig(name="irec_main_descent", frames=120,
                    realism=irec_realism(terrain_frac=0.18, clouds=0.15),
                    targets=[main_target(
                        start_center=(300.0, 150.0), start_diameter=150.0,
                        velocity_px=(0.3, 1.6), diameter_growth_per_frame=1.004,
                        swing_amp_px=14.0, swing_period_frames=48.0,
                        rotate_deg_per_frame=0.9, wind_drift_px=0.2)]),
        # 원거리 회수: 표적이 작고(추적 난이도↑) 아지랑이/노이즈 강함
        SceneConfig(name="irec_main_far", frames=100,
                    realism=irec_realism(heat_shimmer=1.1, sensor_noise=7.0,
                                         jpeg_quality=72, clouds=0.3),
                    targets=[main_target(
                        start_center=(360.0, 170.0), start_diameter=54.0,
                        velocity_px=(-0.4, 0.7), diameter_growth_per_frame=1.006,
                        swing_amp_px=5.0, swing_period_frames=40.0,
                        rotate_deg_per_frame=1.4, wind_drift_px=0.25)]),
    ]


# --- 3D mesh 백엔드 시나리오 (M1~M4). 기존 2D 시나리오·시퀀스는 그대로 보존. ---
def _mesh_sequence() -> DeploymentSequence:
    """IREC 시퀀스와 동일 구성이되 캐노피를 3D mesh 백엔드로 렌더."""
    seq = _irec_sequence()
    for stg in seq.stages:
        stg.target.render_backend = "mesh"
    return seq


def mesh_scenes() -> list[SceneConfig]:
    """3D mesh 렌더 시나리오(추가 경로). 기존 2D/시퀀스/IREC는 무변경."""
    mseq = _mesh_sequence()
    md = main_target(render_backend="mesh", start_center=(300.0, 150.0),
                     start_diameter=150.0, velocity_px=(0.3, 1.4),
                     diameter_growth_per_frame=1.004, swing_amp_px=14.0,
                     swing_period_frames=48.0, rotate_deg_per_frame=0.9,
                     wind_drift_px=0.2, view_elev_deg=55.0)
    return [
        SceneConfig(name="mesh_seq_drogue_then_main", frames=mseq.total_frames(),
                    sequence=mseq, realism=irec_realism()),
        SceneConfig(name="mesh_main_descent", frames=120, realism=irec_realism(),
                    targets=[md]),
    ]


# --- 관측각이 하강하며 변하는 시나리오 (camera_geometry.md 기반, 추가 경로) ---
def descend_view_scenes() -> list[SceneConfig]:
    """하강하며 시선각이 변하는 시나리오. mesh 백엔드로 3D 자세가 실제로 기운다.

    가정(configs/camera_geometry.md):
    - descend_view_body: 페이로드 카메라가 더 빨리 멀어지는 본체(main 계열)를 올려다봄.
      관측각 85°→45°, 분리로 표적이 작아짐(diameter_growth<1).
    - descend_view_ground: 지상 카메라가 하강 표적을 봄. 관측각 30°→70°, 접근으로 커짐.
    기존 시퀀스/IREC/mesh/2D 시나리오는 그대로 보존.
    """
    body = main_target(
        render_backend="mesh", start_center=(320.0, 150.0), start_diameter=140.0,
        velocity_px=(0.1, 0.9), diameter_growth_per_frame=0.985,   # 멀어짐→작아짐
        view_elev_deg=85.0, view_elev_deg_end=45.0,
        swing_amp_px=9.0, swing_period_frames=46.0, rotate_deg_per_frame=0.8,
        wind_drift_px=0.15)
    ground = main_target(
        render_backend="mesh", start_center=(300.0, 210.0), start_diameter=70.0,
        velocity_px=(0.2, 0.3), diameter_growth_per_frame=1.012,   # 접근→커짐
        view_elev_deg=30.0, view_elev_deg_end=70.0,
        swing_amp_px=12.0, swing_period_frames=50.0, rotate_deg_per_frame=1.0,
        wind_drift_px=0.2)
    return [
        SceneConfig(name="descend_view_body", frames=120,
                    realism=irec_realism(), targets=[body]),
        SceneConfig(name="descend_view_ground", frames=120,
                    realism=irec_realism(terrain_frac=0.16), targets=[ground]),
    ]


# --- 짐벌 연동 추적 시나리오 (14 fps, Pi Zero 2W 목표. camera_geometry.md 기반) ---
# primary 추적은 짐벌(github.com/smchoi02/gimbal_control)이 GPS/IMU로 카메라를
# 표적 방향에 두고, 이 합성영상은 그 짐벌이 남긴 잔여 지향 오차 하에서 표적이
# 화면 중앙 근처(오차만큼 벗어난 위치)에 나타나도록 만든다. 14 fps로 촬영을 가정.
# 표적은 페이로드 자신의 drogue→main 순차 전개(본체와 동일 구속조건)를 재현.
def _gimbal_sequence(far: bool = False) -> DeploymentSequence:
    """페이로드의 drogue→main 순차 전개. far=True면 원거리(작은 표적).

    가정(configs/parachute_specs.md, camera_geometry.md): 페이로드도 본체와 동일하게
    3 ft drogue + main, 1.5 kg, drogue≥20 / main≤11 m/s. 낙하 속도가 빠를수록
    (drogue 단계) 더 빨리 멀어져 diameter_growth<1(축소)이 크고, main 단계는
    저속이라 축소가 완만하다.
    """
    drg_d = 34.0 if far else 64.0
    main_d = 40.0 if far else 84.0
    return DeploymentSequence(stages=[
        Stage(
            target=drogue_target(
                start_center=(300.0, 210.0), start_diameter=drg_d,
                velocity_px=(0.2, 0.4),
                diameter_growth_per_frame=0.992,   # 고속 → 빠르게 멀어짐(축소)
                swing_amp_px=5.0, swing_period_frames=22.0,
                rotate_deg_per_frame=2.6, wind_drift_px=0.1),
            inflate_frames=10, inflate_from=0.3, hold_frames=16, overlap_frames=0,
        ),
        Stage(
            target=main_target(
                start_center=(300.0, 230.0), start_diameter=main_d,
                velocity_px=(0.15, 0.3),
                diameter_growth_per_frame=0.998,   # 저속 → 완만하게 멀어짐
                swing_amp_px=9.0, swing_period_frames=42.0,
                rotate_deg_per_frame=1.0, wind_drift_px=0.12),
            inflate_frames=24, inflate_from=0.2, hold_frames=30, overlap_frames=0,
        ),
    ])


def gimbal_scenes() -> list[SceneConfig]:
    """짐벌 연동 추적용 14 fps 시나리오(추가 경로). 표적은 화면 중앙 부근.

    실제 센서는 4608×2592지만 합성/처리 부담을 줄이려 640×480으로 렌더한다.
    짐벌 잔여 오차·ROI 크롭·상대속도 로직은 해상도 무관하며, 메타에 실제 해상도
    가정을 기록한다. tracker_lite가 이 영상에서 표적을 잡아 상대운동을 낸다.
    """
    seq = _gimbal_sequence(far=False)
    seq_far = _gimbal_sequence(far=True)
    return [
        # 근~중거리: drogue→main 순차 전개, 표적 중앙 부근, 14fps
        SceneConfig(name="gimbal_seq_track", fps=14.0, frames=seq.total_frames(),
                    sequence=seq, realism=irec_realism()),
        # 원거리: 표적 작음(추적 한계 시험), 아지랑이/노이즈 강함, 14fps
        SceneConfig(name="gimbal_far_track", fps=14.0, frames=seq_far.total_frames(),
                    sequence=seq_far,
                    realism=irec_realism(heat_shimmer=1.1, sensor_noise=7.0,
                                         jpeg_quality=72, clouds=0.3)),
    ]


# --- 회귀/시연 시나리오 ---
def regression_scenes() -> list[SceneConfig]:
    return [
        SceneConfig(name="main_only", targets=[main_target()]),
        SceneConfig(name="main_approach",
                    targets=[main_target(diameter_growth_per_frame=1.015)]),
        SceneConfig(name="main_inflating",
                    targets=[main_target(inflate_range=(10, 35), start_diameter=130.0)]),
        SceneConfig(name="drogue_only",
                    targets=[drogue_target(velocity_px=(1.5, 2.5))]),
        # 두 표적 동시: drogue(작고 빠름, 위) + main(크고 느림, 아래)
        SceneConfig(name="dual_drogue_main", frames=110, targets=[
            drogue_target(start_center=(430.0, 120.0), velocity_px=(-0.8, 1.2),
                          diameter_growth_per_frame=0.995),
            main_target(start_center=(250.0, 300.0), velocity_px=(0.4, 0.6),
                        diameter_growth_per_frame=1.008,
                        inflate_range=(5, 30)),
        ]),
        # 전이 시연: drogue가 상실되고 main이 전개(단계 전환 모사)
        SceneConfig(name="dual_transition", frames=120, targets=[
            drogue_target(start_center=(360.0, 130.0), velocity_px=(-0.5, 1.0),
                          lost_range=(55, 119)),
            main_target(start_center=(300.0, 280.0), start_diameter=40.0,
                        inflate_range=(50, 85), diameter_growth_per_frame=1.004),
        ]),
        SceneConfig(name="main_occlusion",
                    targets=[main_target(occlude_range=(30, 55))]),
        SceneConfig(name="main_clip_out",
                    targets=[main_target(start_center=(140.0, 240.0),
                                         velocity_px=(-5.0, 0.0))]),
    ]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spec B 합성영상·정답 생성기 (다중 낙하산)")
    p.add_argument("--out", default="data/synthetic", help="출력 폴더")
    p.add_argument("--scene", default="all",
                   help="시나리오 이름 또는 'all'")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--frames", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out)
    # 기존 회귀 + 시퀀스 + IREC 사실성 + 3D mesh + 관측각-하강 + 짐벌 연동(모두 추가 경로).
    scenes = (regression_scenes() + sequence_scenes() + irec_scenes()
              + mesh_scenes() + descend_view_scenes() + gimbal_scenes())
    if args.scene != "all":
        scenes = [s for s in scenes if s.name == args.scene]
        if not scenes:
            print(f"알 수 없는 시나리오: {args.scene}")
            return 2
    for s in scenes:
        if args.seed is not None:
            s.seed = args.seed
        if args.frames is not None:
            s.frames = args.frames
        meta = generate(s, out_dir)
        print(f"생성: {meta['video_path']}  표적: {meta['target_names']}  GT: {meta['gt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
