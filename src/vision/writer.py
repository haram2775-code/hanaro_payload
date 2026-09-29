"""Spec A — CSV(고정 스키마) + 실패 구간 + 실행 메타데이터 JSON.

CSV 스키마(요구사항 R5, 순서 고정):
frame_id, timestamp_s, center_x_px, center_y_px, mask_area_px2,
equivalent_diameter_px, track_valid, size_valid, state, processing_ms
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

CSV_COLUMNS = [
    "frame_id", "timestamp_s", "center_x_px", "center_y_px",
    "mask_area_px2", "equivalent_diameter_px",
    "track_valid", "size_valid", "state", "processing_ms",
]


class CsvWriter:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(CSV_COLUMNS)

    def write_row(self, *, frame_id, timestamp_s, center, mask_area_px2,
                  equivalent_diameter_px, track_valid, size_valid, state,
                  processing_ms):
        cx = "" if center is None else f"{center[0]:.3f}"
        cy = "" if center is None else f"{center[1]:.3f}"
        self._w.writerow([
            frame_id, f"{timestamp_s:.6f}", cx, cy,
            f"{mask_area_px2:.3f}", f"{equivalent_diameter_px:.3f}",
            int(bool(track_valid)), int(bool(size_valid)), state,
            f"{processing_ms:.3f}",
        ])

    def close(self):
        self._f.close()


def write_metadata(path: str | Path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def write_failures(path: str | Path, failures: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
