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
    view_elev_deg: float = 25.0            # 관측 고도각(도)
    # 전개: [start,end] 구간에서 inflation 0.2→1.0
    inflate_range: tuple[int, int] | None = None
    inflate_from: float = 0.2
    # 가림
    occlude_range: tuple[int, int] | None = None
    occlude_fraction: float = 0.6
    # 상실
    lost_range: tuple[int, int] | None = None

    def canopy(self) -> Canopy:
        return Canopy(kind=self.kind, color=self.color, alt_color=self.alt_color,
                      n_panels=self.n_panels, n_shroud=self.n_shroud,
                      spill_hole_ratio=self.spill_hole_ratio)


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
    frame = np.full((cfg.height, cfg.width, 3), cfg.bg_gray, dtype=np.float32)
    if cfg.sky_gradient:
        grad = np.linspace(cfg.bg_gray + 40, cfg.bg_gray - 20, cfg.height, dtype=np.float32)
        frame = np.repeat(grad[:, None, None], cfg.width, axis=1)
        frame = np.repeat(frame, 3, axis=2)
    if cfg.bg_noise > 0:
        frame += rng.normal(0.0, cfg.bg_noise, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


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

        for i in range(cfg.frames):
            frame = _make_background(cfg, rng)

            for ti, t in enumerate(cfg.targets):
                st = state[ti]
                st["d"] = max(2.0, st["d"] * t.diameter_growth_per_frame)
                st["cx"] += t.velocity_px[0]
                st["cy"] += t.velocity_px[1]

                infl = _inflation(i, t)
                view_elev = math.radians(t.view_elev_deg)
                cx, cy, d = st["cx"], st["cy"], st["d"]

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
                    render_canopy(frame, (cx, cy), d, cnp,
                                  inflation=infl, view_elev=view_elev,
                                  occlude_top_frac=occ_frac)
                    bx, by, bw, bh = canopy_projected_bbox(
                        (cx, cy), d, cnp, inflation=infl, view_elev=view_elev)
                    if (bx < 0 or by < 0 or bx + bw >= cfg.width or by + bh >= cfg.height):
                        tstate = STATE_CLIPPED

                area = canopy_projected_area(d, t.canopy(), inflation=infl, view_elev=view_elev)
                gt.writerow([
                    i, f"{i / cfg.fps:.6f}", t.name,
                    f"{cx:.3f}", f"{cy:.3f}",
                    f"{area:.3f}", f"{d:.3f}", f"{infl:.3f}",
                    visible, tstate,
                ])

            if cfg.blur_ksize and cfg.blur_ksize >= 3 and cfg.blur_ksize % 2 == 1:
                frame = cv2.GaussianBlur(frame, (cfg.blur_ksize, cfg.blur_ksize), 0)

            writer.write(frame)

    writer.release()

    meta = {
        "spec": "B-synthetic",
        "code_version": code_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "source_kind": "synthetic",
        "video_id": cfg.name,
        "n_targets": len(cfg.targets),
        "target_names": [t.name for t in cfg.targets],
        "config": _cfg_to_dict(cfg),
        "video_path": str(video_path),
        "gt_path": str(gt_path),
        "note": "형상 정확도가 실제 검출 성공을 보장하지 않음(공통 규칙). 절대 거리/속도 없음.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


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
    scenes = regression_scenes()
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
