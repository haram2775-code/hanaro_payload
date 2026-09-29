"""Spec A — 영상 입력/주석 영상 출력.

원본에 실제 촬영시각이 없으면 FPS로 timestamp를 계산하고 추정값임을 표시한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameData:
    frame_id: int
    timestamp_s: float
    ts_is_estimated: bool
    image: np.ndarray


class VideoReader:
    def __init__(self, path: str | Path, fps_override: float | None = None):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"영상 열기 실패: {self.path}")
        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 합성/저장영상은 프레임별 실제 촬영시각 메타데이터가 없다 → FPS 기반 추정.
        self.ts_is_estimated = True
        self.fps = fps_override or (src_fps if src_fps and src_fps > 1e-3 else 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        idx = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                break
            yield FrameData(
                frame_id=idx,
                timestamp_s=idx / self.fps,
                ts_is_estimated=self.ts_is_estimated,
                image=img,
            )
            idx += 1
        self.cap.release()


class AnnotatedVideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"주석 영상 열기 실패: {path}")

    def write(self, image, center, bbox, state, track_valid, size_valid):
        vis = image.copy()
        color = (0, 255, 0) if (track_valid and size_valid) else (0, 165, 255)
        if not track_valid:
            color = (0, 0, 255)
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        if center is not None and track_valid:
            cv2.circle(vis, (int(center[0]), int(center[1])), 3, color, -1)
        cv2.putText(vis, f"{state} t={int(track_valid)} s={int(size_valid)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self.writer.write(vis)

    def release(self):
        self.writer.release()
