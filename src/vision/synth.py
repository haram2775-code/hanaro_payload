"""Spec B — 합성영상·정답 데이터 생성기.

설정과 seed가 주어지면 결정론적으로 시험 영상과 프레임별 정답값(GT) CSV를 생성한다.
단순 도형 합성이며, 성공이 실제 낙하산 검출 성공을 의미하지 않는다(요구사항 참조).

절대 거리/m/s는 다루지 않는다. 모든 크기·중심은 픽셀 단위다.
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

# GT 상태 어휘는 Spec A와 정합한다.
STATE_TRACK = "TRACK"
STATE_OCCLUDED = "OCCLUDED"
STATE_CLIPPED = "CLIPPED"
STATE_INFLATING = "INFLATING"
STATE_LOST = "LOST"


@dataclass
class SceneConfig:
    """합성 시나리오 설정. 모든 값은 설정값/가정이며 확정 요구조건이 아니다."""
    name: str = "static"
    width: int = 640
    height: int = 480
    fps: float = 30.0
    frames: int = 90
    seed: int = 1234
    bg_gray: int = 40           # 배경 밝기 (0-255)
    bg_noise: float = 3.0       # 배경 가우시안 노이즈 표준편차
    target_gray: int = 210      # 표적 밝기
    start_diameter: float = 60.0  # 시작 등가직경(px)
    # 크기 프로파일: 프레임 진행에 따른 직경 배율(1.0=정지). approach>1, recede<1
    diameter_growth_per_frame: float = 1.0
    start_center: tuple[float, float] = (320.0, 240.0)
    velocity_px: tuple[float, float] = (0.0, 0.0)  # 프레임당 중심 이동(px)
    blur_ksize: int = 0          # 0이면 블러 없음(홀수만 유효)
    # 가림: [start_frame, end_frame] 구간에서 표적 상단을 가림 막대로 덮음
    occlude_range: tuple[int, int] | None = None
    occlude_fraction: float = 0.6   # 가려지는 표적 높이 비율
    # 팽창: [start,end] 구간에서 직경을 빠르게 키움
    inflate_range: tuple[int, int] | None = None
    inflate_rate: float = 1.06      # 팽창 구간의 프레임당 직경 배율
    # 상실: [start,end] 구간에서 표적을 그리지 않음
    lost_range: tuple[int, int] | None = None


def _in_range(i: int, rng: tuple[int, int] | None) -> bool:
    return rng is not None and rng[0] <= i <= rng[1]


def generate(cfg: SceneConfig, out_dir: Path) -> dict:
    """영상 + GT CSV + 메타데이터를 생성한다. 요약 dict를 반환한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    video_path = out_dir / f"{cfg.name}.mp4"
    gt_path = out_dir / f"{cfg.name}_gt.csv"
    meta_path = out_dir / f"{cfg.name}_meta.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, cfg.fps, (cfg.width, cfg.height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 열기 실패: {video_path}")

    cx, cy = cfg.start_center
    diameter = cfg.start_diameter

    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        gt = csv.writer(f)
        gt.writerow([
            "frame_id", "timestamp_s", "gt_center_x_px", "gt_center_y_px",
            "gt_area_px2", "gt_diameter_px", "gt_visible", "gt_state",
        ])

        for i in range(cfg.frames):
            # 배경
            frame = np.full((cfg.height, cfg.width, 3), cfg.bg_gray, dtype=np.uint8)
            if cfg.bg_noise > 0:
                noise = rng.normal(0.0, cfg.bg_noise, frame.shape)
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # 크기 갱신
            growth = cfg.diameter_growth_per_frame
            if _in_range(i, cfg.inflate_range):
                growth = cfg.inflate_rate
            diameter = max(2.0, diameter * growth)

            # 중심 갱신
            cx += cfg.velocity_px[0]
            cy += cfg.velocity_px[1]

            radius = diameter / 2.0
            area = math.pi * radius * radius

            visible = 1
            state = STATE_TRACK
            if _in_range(i, cfg.inflate_range):
                state = STATE_INFLATING

            lost = _in_range(i, cfg.lost_range)
            if lost:
                visible = 0
                state = STATE_LOST
            else:
                # 표적을 원으로 그림
                cv2.circle(frame, (int(round(cx)), int(round(cy))),
                           int(round(radius)), (cfg.target_gray,) * 3, thickness=-1)

                # 가림 막대
                if _in_range(i, cfg.occlude_range):
                    top = int(round(cy - radius))
                    cover_h = int(round(2 * radius * cfg.occlude_fraction))
                    cv2.rectangle(
                        frame,
                        (int(round(cx - radius)) - 4, top - 4),
                        (int(round(cx + radius)) + 4, top + cover_h),
                        (cfg.bg_gray,) * 3, thickness=-1,
                    )
                    state = STATE_OCCLUDED

                # 화면 잘림 판정 (경계에 표적이 닿음)
                if (cx - radius < 0 or cx + radius >= cfg.width or
                        cy - radius < 0 or cy + radius >= cfg.height):
                    state = STATE_CLIPPED

            if cfg.blur_ksize and cfg.blur_ksize >= 3 and cfg.blur_ksize % 2 == 1:
                frame = cv2.GaussianBlur(frame, (cfg.blur_ksize, cfg.blur_ksize), 0)

            writer.write(frame)
            ts = i / cfg.fps
            gt.writerow([
                i, f"{ts:.6f}", f"{cx:.3f}", f"{cy:.3f}",
                f"{area:.3f}", f"{diameter:.3f}", visible, state,
            ])

    writer.release()

    meta = {
        "spec": "B-synthetic",
        "code_version": code_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "source_kind": "synthetic",
        "video_id": cfg.name,
        "config": asdict(cfg),
        "video_path": str(video_path),
        "gt_path": str(gt_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


# --- 최소 회귀 시나리오 (요구사항 R4) ---
def regression_scenes() -> list[SceneConfig]:
    return [
        SceneConfig(name="static", diameter_growth_per_frame=1.0),
        SceneConfig(name="approach", diameter_growth_per_frame=1.02),
        SceneConfig(name="recede", diameter_growth_per_frame=0.985),
        SceneConfig(name="small_target", start_diameter=18.0),
        SceneConfig(name="occlusion", occlude_range=(30, 55)),
        SceneConfig(name="inflating", inflate_range=(20, 45)),
        SceneConfig(name="lost_return", lost_range=(35, 50)),
        SceneConfig(name="clip_out", start_center=(120.0, 240.0),
                    velocity_px=(-6.0, 0.0)),
    ]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spec B 합성영상·정답 생성기")
    p.add_argument("--out", default="data/synthetic", help="출력 폴더")
    p.add_argument("--scene", default="all",
                   help="시나리오 이름 또는 'all'(모든 회귀 시나리오)")
    p.add_argument("--seed", type=int, default=None, help="seed 재정의")
    p.add_argument("--frames", type=int, default=None, help="frames 재정의")
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
        print(f"생성: {meta['video_path']}  GT: {meta['gt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
