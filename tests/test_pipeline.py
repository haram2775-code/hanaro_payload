"""Spec A/B 단위 + end-to-end 스모크 테스트.

실행: (repo 루트에서)
  python -m pytest tests -q
또는 pytest 없이:
  python tests/test_pipeline.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.config import AnalyzeConfig                     # noqa: E402
from vision.writer import CSV_COLUMNS                       # noqa: E402
from vision import validity, synth, analyze                 # noqa: E402


def test_csv_schema_order():
    assert CSV_COLUMNS == [
        "frame_id", "timestamp_s", "center_x_px", "center_y_px",
        "mask_area_px2", "equivalent_diameter_px",
        "track_valid", "size_valid", "state", "processing_ms",
    ]


def test_validity_lost_when_track_fails():
    cfg = AnalyzeConfig()
    j = validity.judge(frame_w=640, frame_h=480, track_ok=False,
                       center=None, seg=_no_seg(), prev_area=None, cfg=cfg)
    assert j.state == "LOST"
    assert not j.track_valid and not j.size_valid


def test_validity_inflating_invalidates_size():
    cfg = AnalyzeConfig()
    seg = _seg(area=5000.0, touches=False)
    j = validity.judge(frame_w=640, frame_h=480, track_ok=True,
                       center=(320, 240), seg=seg, prev_area=1000.0, cfg=cfg)
    assert j.track_valid          # 중심은 유효
    assert not j.size_valid       # 크기는 무효
    assert j.state == "INFLATING"


def test_validity_track_but_size_separable():
    """공통 규칙 §5: 중심 유효 + 크기 무효(가림)를 분리 판정."""
    cfg = AnalyzeConfig()
    seg = _seg(area=0.0, touches=False, found=False)
    j = validity.judge(frame_w=640, frame_h=480, track_ok=True,
                       center=(320, 240), seg=seg, prev_area=1000.0, cfg=cfg)
    assert j.track_valid
    assert not j.size_valid
    assert j.state == "OCCLUDED"


def test_end_to_end_and_reproducible(tmp_path=None):
    base = Path(tmp_path) if tmp_path else ROOT / "output" / "_test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    scene = synth.SceneConfig(name="approach", diameter_growth_per_frame=1.02,
                              frames=40, seed=7)
    synth.generate(scene, base)
    video = base / "approach.mp4"
    assert video.exists()

    cfg = AnalyzeConfig(init_bbox=(290, 210, 60, 60), tracker="CSRT")
    out1 = base / "run1"
    out2 = base / "run2"
    analyze.run(str(video), str(out1), cfg, source_kind="synthetic")
    analyze.run(str(video), str(out2), cfg, source_kind="synthetic")

    csv1 = (out1 / "approach.csv").read_text(encoding="utf-8")
    csv2 = (out2 / "approach.csv").read_text(encoding="utf-8")
    # 재현성: 처리시간(processing_ms) 컬럼만 실행마다 다르므로 그 컬럼을 제외하고 비교
    assert _strip_timing(csv1) == _strip_timing(csv2), "동일 입력 재현 실패"

    rows = list(csv.DictReader(csv1.splitlines()))
    assert len(rows) == 40
    # 접근 시나리오: 마지막이 첫 유효 프레임보다 직경이 크다
    valid = [r for r in rows if r["size_valid"] == "1"]
    assert valid, "유효 크기 프레임이 없음"
    assert float(valid[-1]["equivalent_diameter_px"]) > float(valid[0]["equivalent_diameter_px"])


def _strip_timing(csv_text: str) -> list[list[str]]:
    out = []
    for i, line in enumerate(csv_text.strip().splitlines()):
        cols = line.split(",")
        out.append(cols[:-1])  # 마지막 processing_ms 제거
    return out


# --- helpers ---
class _FakeSeg:
    def __init__(self, area, touches, found):
        self.mask_area_px2 = area
        self.equivalent_diameter_px = (area) ** 0.5
        self.center = (320.0, 240.0) if found else None
        self.bbox_measured = (300, 220, 40, 40) if found else None
        self.touches_border = touches
        self.found = found


def _seg(area, touches, found=True):
    return _FakeSeg(area, touches, found)


def _no_seg():
    return _FakeSeg(0.0, False, False)


if __name__ == "__main__":
    test_csv_schema_order()
    test_validity_lost_when_track_fails()
    test_validity_inflating_invalidates_size()
    test_validity_track_but_size_separable()
    test_end_to_end_and_reproducible()
    print("ALL TESTS PASSED")
