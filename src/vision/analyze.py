"""Spec A — CLI 파이프라인.

한 영상을 끝까지 처리하여 CSV, 주석 영상, 실행 메타데이터, 실패 구간 목록을 저장한다.
절대 거리/m/s는 출력하지 않는다(공통 규칙 §6). 픽셀 단위 중심·크기만 다룬다.

사용 예:
  python -m vision.analyze --video data/synthetic/approach.mp4 \
      --bbox 290 210 60 60 --out output/run_approach
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from . import code_version
from .config import AnalyzeConfig
from .video_io import VideoReader, AnnotatedVideoWriter
from .tracker import Tracker
from .segmenter import segment
from . import validity
from .writer import CsvWriter, write_metadata, write_failures


def run(video: str, out_dir: str, cfg: AnalyzeConfig, source_kind: str = "unknown") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    video_id = Path(video).stem

    reader = VideoReader(video, fps_override=cfg.fps_override)
    if cfg.init_bbox is None:
        raise ValueError("init_bbox가 필요합니다(수동 표적 초기화). --bbox 또는 설정으로 지정하세요.")

    ann = AnnotatedVideoWriter(out / f"{video_id}_annotated.mp4",
                               fps=reader.fps, size=(reader.width, reader.height))
    csvw = CsvWriter(out / f"{video_id}.csv")

    tracker = Tracker(cfg.tracker)
    prev_area: float | None = None
    failures: list[dict] = []
    n_frames = 0
    ts_estimated = False

    for fd in reader:
        n_frames += 1
        ts_estimated = fd.ts_is_estimated
        t0 = time.perf_counter()

        if fd.frame_id == 0:
            tracker.init(fd.image, cfg.init_bbox)

        track_ok, bbox, center = tracker.update(fd.image)
        seg = segment(fd.image, bbox, cfg)
        # 폴백 트래커면 마스크 중심으로 위치 보정
        if tracker.backend == "fallback" and seg.found and seg.center is not None:
            tracker.recenter(seg.center)
            center = seg.center

        # 크기 측정 중심이 있으면 그것을 중심으로 우선 사용(경계 정밀)
        report_center = seg.center if (seg.found and seg.center is not None) else center

        j = validity.judge(
            frame_w=reader.width, frame_h=reader.height,
            track_ok=track_ok, center=report_center, seg=seg,
            prev_area=prev_area, cfg=cfg,
        )

        area = seg.mask_area_px2 if seg.found else 0.0
        eq_d = seg.equivalent_diameter_px if seg.found else 0.0
        dt_ms = (time.perf_counter() - t0) * 1000.0

        csvw.write_row(
            frame_id=fd.frame_id, timestamp_s=fd.timestamp_s,
            center=report_center if j.track_valid else None,
            mask_area_px2=area, equivalent_diameter_px=eq_d,
            track_valid=j.track_valid, size_valid=j.size_valid,
            state=j.state, processing_ms=dt_ms,
        )
        ann.write(fd.image, report_center, bbox, j.state, j.track_valid, j.size_valid)

        if (not j.track_valid) or (not j.size_valid):
            failures.append({
                "frame_id": fd.frame_id,
                "timestamp_s": round(fd.timestamp_s, 6),
                "state": j.state,
                "track_valid": j.track_valid,
                "size_valid": j.size_valid,
            })

        if j.size_valid:
            prev_area = area

    csvw.close()
    ann.release()

    meta = {
        "spec": "A-video-analysis",
        "code_version": code_version(),
        "opencv_version": cv2.__version__,
        "source_kind": source_kind,
        "video_id": video_id,
        "video_path": str(video),
        "fps_used": reader.fps,
        "timestamp_is_estimated": ts_estimated,
        "frame_count": n_frames,
        "tracker_backend": tracker.backend,
        "tracker_requested": cfg.tracker,
        "config": cfg.as_dict(),
        "outputs": {
            "csv": str(out / f"{video_id}.csv"),
            "annotated_video": str(out / f"{video_id}_annotated.mp4"),
            "failures": str(out / f"{video_id}_failures.json"),
            "metadata": str(out / f"{video_id}_meta.json"),
        },
        "failure_frame_count": len(failures),
    }
    write_failures(out / f"{video_id}_failures.json", failures)
    write_metadata(out / f"{video_id}_meta.json", meta)
    return meta


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spec A 저장영상 낙하산 분석")
    p.add_argument("--video", required=True, help="입력 영상 파일")
    p.add_argument("--out", required=True, help="출력 폴더")
    p.add_argument("--config", default=None, help="설정 파일(JSON/YAML)")
    p.add_argument("--bbox", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   default=None, help="초기 표적 bbox(수동 초기화)")
    p.add_argument("--fps", type=float, default=None, help="FPS 재정의")
    p.add_argument("--tracker", default=None, help="CSRT|KCF|MOSSE")
    p.add_argument("--source-kind", default="unknown", help="real|synthetic|unknown")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = AnalyzeConfig.load(args.config)
    if args.bbox is not None:
        cfg.init_bbox = tuple(args.bbox)
    if args.fps is not None:
        cfg.fps_override = args.fps
    if args.tracker is not None:
        cfg.tracker = args.tracker
    cfg.validate()
    meta = run(args.video, args.out, cfg, source_kind=args.source_kind)
    print(f"완료: {meta['frame_count']} 프레임, 실패 {meta['failure_frame_count']}개")
    print(f"CSV: {meta['outputs']['csv']}")
    print(f"주석 영상: {meta['outputs']['annotated_video']}")
    print(f"트래커 백엔드: {meta['tracker_backend']}, timestamp 추정: {meta['timestamp_is_estimated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
