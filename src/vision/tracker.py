"""Spec A — 초기 bbox 추적.

OpenCV 트래커를 우선 사용하고, 미지원 빌드에서는 마지막 마스크 중심을 따라가는
간단한 색상/템플릿 폴백으로 대체한다. 자동 표적 탐색은 이 Spec 범위 밖이다.
"""
from __future__ import annotations

import cv2
import numpy as np


def _make_cv_tracker(name: str):
    name = name.upper()
    factories = {}
    # OpenCV 버전에 따라 위치가 다르다.
    for mod in (cv2, getattr(cv2, "legacy", None)):
        if mod is None:
            continue
        for key, attr in (
            ("CSRT", "TrackerCSRT_create"),
            ("KCF", "TrackerKCF_create"),
            ("MOSSE", "TrackerMOSSE_create"),
        ):
            fn = getattr(mod, attr, None)
            if fn is not None and key not in factories:
                factories[key] = fn
    fn = factories.get(name) or factories.get("CSRT") or factories.get("KCF")
    return fn() if fn else None


class Tracker:
    """center, bbox를 반환한다. ok=False면 track_valid=False로 처리한다."""

    def __init__(self, name: str = "CSRT"):
        self.name = name
        self._cv = None
        self._fallback_bbox = None

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self._cv = _make_cv_tracker(self.name)
        self._fallback_bbox = tuple(int(v) for v in bbox)
        if self._cv is not None:
            self._cv.init(frame, tuple(int(v) for v in bbox))

    def update(self, frame: np.ndarray):
        """returns (ok, bbox, center)."""
        if self._cv is not None:
            ok, box = self._cv.update(frame)
            if ok:
                x, y, w, h = box
                self._fallback_bbox = (int(x), int(y), int(w), int(h))
                return True, self._fallback_bbox, (x + w / 2.0, y + h / 2.0)
            return False, self._fallback_bbox, None
        # 폴백: bbox 유지(마스크 중심 갱신은 파이프라인이 담당)
        if self._fallback_bbox is None:
            return False, None, None
        x, y, w, h = self._fallback_bbox
        return True, self._fallback_bbox, (x + w / 2.0, y + h / 2.0)

    def recenter(self, center: tuple[float, float]) -> None:
        """마스크 중심으로 bbox 위치를 보정(폴백 트래커 품질 향상용)."""
        if self._fallback_bbox is None or center is None:
            return
        _, _, w, h = self._fallback_bbox
        self._fallback_bbox = (int(center[0] - w / 2), int(center[1] - h / 2), w, h)

    @property
    def backend(self) -> str:
        return "opencv" if self._cv is not None else "fallback"
