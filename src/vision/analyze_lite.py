"""짐벌 연동 경량 추적 파이프라인 (Pi Zero 2 W · 14 fps 목표).

vision.tracker_lite.LiteTracker로 한 영상을 처리하여, 프레임별
중심(px)·등가직경(px)·상대운동(접근/유지/이격)·처리시간(ms)을 CSV로 낸다.
기존 vision.analyze(Spec A, CSRT)는 그대로 두고 이 경량 경로를 **추가**한다.

- 짐벌 잔여 오차: gimbal.GimbalErrorModel로 프레임별 화면 오프셋을 주고,
  ROI를 그 오프셋 위치에 둬 추적기가 오차를 흡수하는지 검증한다.
- 절대 거리/m/s 확정 출력 금지(§6): 상대운동은 픽셀 스케일 변화율로만 낸다.
- 두 표적/두 단계가 있으면 각각 처리해 표적 간 상대 스케일 속도차를 낸다.

사용:
  python -m vision.analyze_lite --video data/synthetic/gimbal_far.mp4 \
      --out output/run_gimbal --gimbal-error
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2

from . import code_version
from .video_io import VideoReader
from .tracker_lite import LiteTracker, RoiConfig, relative_scale_rate, classify_relative_motion
from .gimbal import GimbalErrorModel


def run_lite(video: str, out_dir: str, *, roi_size: int = 512, proc_max: int = 256,
             use_gimbal_error: bool = True, gimbal: GimbalErrorModel | None = None,
             seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    video_id = Path(video).stem

    reader = VideoReader(video)
    fps = reader.fps or 14.0
    dt = 1.0 / fps
    tracker = LiteTracker(RoiConfig(roi_size=roi_size, proc_max=proc_max))
    gmodel = gimbal or GimbalErrorModel(seed=seed)
    rng = random.Random(seed)

    csv_path = out / f"{video_id}_lite.csv"
    rows = []
    proc_ms = []
    prev_diam = None
    found_frames = 0
    n = 0

    for fd in reader:
        n += 1
        offset = gmodel.offset_px(fd.frame_id, fps, rng) if use_gimbal_error else (0.0, 0.0)
        t0 = time.perf_counter()
        found, center, diam, bbox, roi = tracker.step(fd.image, gimbal_offset=offset)
        # 상대운동(직전 대비 크기 변화율)
        rate = relative_scale_rate(prev_diam, diam, dt) if (found and prev_diam) else 0.0
        motion = classify_relative_motion(rate) if found else "LOST"
        dt_ms = (time.perf_counter() - t0) * 1000.0
        proc_ms.append(dt_ms)
        if found:
            found_frames += 1
            prev_diam = diam
        rows.append({
            "frame_id": fd.frame_id,
            "timestamp_s": round(fd.timestamp_s, 6),
            "found": int(found),
            "cx": round(center[0], 2) if center else "",
            "cy": round(center[1], 2) if center else "",
            "eq_diam_px": round(diam, 2),
            "gimbal_dx_px": round(offset[0], 1),
            "gimbal_dy_px": round(offset[1], 1),
            "scale_rate_per_s": round(rate, 4),
            "rel_motion": motion,
            "processing_ms": round(dt_ms, 3),
        })

    # CSV 쓰기
    import csv as _csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    proc_ms_sorted = sorted(proc_ms)
    p50 = proc_ms_sorted[len(proc_ms_sorted) // 2] if proc_ms else 0.0
    p95 = proc_ms_sorted[int(len(proc_ms_sorted) * 0.95)] if proc_ms else 0.0
    budget_ms = 1000.0 / fps
    meta = {
        "spec": "gimbal-lite-tracker",
        "code_version": code_version(),
        "opencv_version": cv2.__version__,
        "video_id": video_id,
        "fps_used": fps,
        "frame_count": n,
        "found_fraction": round(found_frames / max(1, n), 3),
        "gimbal_error_applied": use_gimbal_error,
        "gimbal_model": gmodel.as_dict(),
        "roi": {"roi_size": roi_size, "proc_max": proc_max},
        "timing_ms": {
            "p50": round(p50, 3), "p95": round(p95, 3),
            "max": round(max(proc_ms), 3) if proc_ms else 0.0,
            "budget_at_fps": round(budget_ms, 2),
            "meets_budget_p95": bool(p95 <= budget_ms),
            "note": "PC 측정치. Pi Zero 2W 실측 아님(공통 규칙 §3). 예산 판단은 상대 비교용.",
        },
        "outputs": {"csv": str(csv_path)},
    }
    (out / f"{video_id}_lite_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="짐벌 연동 경량 추적(Pi Zero 2W 목표)")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--roi-size", type=int, default=512)
    p.add_argument("--proc-max", type=int, default=256)
    p.add_argument("--gimbal-error", action="store_true", help="짐벌 잔여 오차 주입")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    meta = run_lite(args.video, args.out, roi_size=args.roi_size, proc_max=args.proc_max,
                    use_gimbal_error=args.gimbal_error, seed=args.seed)
    t = meta["timing_ms"]
    print(f"완료: {meta['frame_count']} 프레임, 검출률 {meta['found_fraction']}")
    print(f"처리시간 p50={t['p50']}ms p95={t['p95']}ms / 예산 {t['budget_at_fps']}ms "
          f"(p95 예산내: {t['meets_budget_p95']})")
    print(f"CSV: {meta['outputs']['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
