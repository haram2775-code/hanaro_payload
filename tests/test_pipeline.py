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
    # 새 다중표적 API: main 표적 접근 시나리오
    scene = synth.SceneConfig(
        name="approach", frames=40, seed=7,
        targets=[synth.main_target(start_center=(320.0, 240.0),
                                   start_diameter=110.0,
                                   diameter_growth_per_frame=1.02)],
    )
    synth.generate(scene, base)
    video = base / "approach.mp4"
    assert video.exists()
    # GT 스키마에 target_name 컬럼이 포함되는지 확인
    gt_rows = list(csv.DictReader((base / "approach_gt.csv").read_text(encoding="utf-8").splitlines()))
    assert gt_rows and gt_rows[0]["target_name"] == "main"

    cfg = AnalyzeConfig(init_bbox=(280, 190, 100, 90), tracker="CSRT")
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
    # 접근 시나리오: 마지막 유효 직경이 첫 유효 직경보다 크다
    valid = [r for r in rows if r["size_valid"] == "1"]
    assert valid, "유효 크기 프레임이 없음"
    assert float(valid[-1]["equivalent_diameter_px"]) > float(valid[0]["equivalent_diameter_px"])


def test_dual_target_gt_has_both():
    """두 표적(drogue+main) 시나리오는 프레임당 GT 두 행을 남긴다."""
    base = ROOT / "output" / "_test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    scene = [s for s in synth.regression_scenes() if s.name == "dual_drogue_main"][0]
    scene.frames = 20
    synth.generate(scene, base)
    rows = list(csv.DictReader((base / "dual_drogue_main_gt.csv").read_text(encoding="utf-8").splitlines()))
    names = {r["target_name"] for r in rows}
    assert names == {"drogue", "main"}, f"두 표적이 모두 있어야 함: {names}"
    assert len(rows) == 20 * 2


def test_sequence_timeline_transitions():
    """시퀀스 타임라인: 단계가 순서대로, 겹침 없이 이어지고 각 단계가 전개완료 지점을 갖는다."""
    seq = synth.drogue_then_main_sequence()
    marks = seq.timeline()
    assert len(marks) == 2
    d, m = marks
    # drogue가 먼저, main이 그 뒤에 시작
    assert d["start"] == 0
    assert m["start"] >= d["end"], "겹침 없음(기본 overlap=0)이면 main은 drogue 종료 후 시작"
    # 각 단계에 전개완료(deploying_end) 경계가 활성 구간 안에 있음
    assert d["start"] < d["deploying_end"] <= d["end"]
    assert m["start"] < m["deploying_end"] <= m["end"]


def test_sequence_single_active_stage_and_state_flow():
    """시퀀스 생성물: 프레임당 활성 단계 1개만 기록되고, drogue가 DEPLOYED 도달 후 main이 등장."""
    base = ROOT / "output" / "_test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    scene = [s for s in synth.sequence_scenes() if s.name == "seq_drogue_then_main"][0]
    synth.generate(scene, base)
    rows = list(csv.DictReader((base / "seq_drogue_then_main_gt.csv").read_text(encoding="utf-8").splitlines()))
    assert rows, "시퀀스 GT가 비어있음"

    # 1) 프레임당 활성 단계 1개(겹침 없음): frame_id 중복 없음
    frame_ids = [r["frame_id"] for r in rows]
    assert len(frame_ids) == len(set(frame_ids)), "겹침 없는 시퀀스인데 같은 프레임에 2개 기록됨"

    # 2) drogue가 먼저, main이 나중 — 마지막 drogue 프레임 < 첫 main 프레임
    drogue_frames = [int(r["frame_id"]) for r in rows if r["target_name"] == "drogue"]
    main_frames = [int(r["frame_id"]) for r in rows if r["target_name"] == "main"]
    assert drogue_frames and main_frames, "두 단계 모두 나타나야 함"
    assert max(drogue_frames) < min(main_frames), "drogue 단계가 main 단계보다 먼저 끝나야 함"

    # 3) 상태 전이: drogue 단계에 DEPLOYING과 DEPLOYED가 모두 나타남(전개완료 확인)
    drogue_states = {r["gt_state"] for r in rows if r["target_name"] == "drogue"}
    assert synth.STATE_DEPLOYING in drogue_states, "drogue 전개 중 상태가 있어야 함"
    assert synth.STATE_DEPLOYED in drogue_states, "drogue 전개 완료(DEPLOYED) 상태가 있어야 함"
    # main도 전개 중 상태를 가짐
    main_states = {r["gt_state"] for r in rows if r["target_name"] == "main"}
    assert synth.STATE_DEPLOYING in main_states


def test_regression_scenes_preserved():
    """기존 진입점 보존: regression_scenes()에 기존 이름들이 그대로 있고 sequence 필드가 없다."""
    names = {s.name for s in synth.regression_scenes()}
    for expected in ("main_only", "dual_drogue_main", "dual_transition", "main_occlusion"):
        assert expected in names, f"기존 시나리오 {expected} 보존돼야 함"
    # 기존 시나리오는 시퀀스를 쓰지 않음(다중표적 경로 유지)
    assert all(s.sequence is None for s in synth.regression_scenes())
    # 기존 시나리오는 사실성 레이어를 쓰지 않음(회색-하늘 경로 유지)
    assert all(s.realism is None for s in synth.regression_scenes())


def test_realism_off_preserves_legacy_background():
    """realism=None이면 기존 회색-하늘 경로 그대로(채도 거의 0)."""
    import numpy as np
    cfg = synth.SceneConfig(name="_bgtest", width=64, height=64, realism=None)
    rng = np.random.default_rng(0)
    bg = synth._make_background(cfg, rng)
    hsv = _to_hsv(bg)
    # 회색 배경도 채널별 노이즈로 소량의 chroma가 생기지만(≈16), IREC 파란 하늘(≈118)과는
    # 확연히 구분된다. 무채색 상한을 넉넉히 30으로 둔다.
    assert hsv[..., 1].mean() < 30, "기존 배경은 무채색(회색)이어야 함"


def test_irec_realism_bluer_and_more_trackable():
    """IREC 사실성 배경은 회색 배경보다 채도가 높고(파란 하늘), 추적성이 개선된다."""
    import numpy as np
    # 파란 하늘 채도 확인
    rl = synth.irec_realism(heat_shimmer=0.0, clouds=0.0, jpeg_quality=0)
    cfg = synth.SceneConfig(name="_skytest", width=80, height=80, realism=rl)
    rng = np.random.default_rng(0)
    bg = synth._make_background(cfg, rng)
    hsv = _to_hsv(bg)
    assert hsv[..., 1].mean() > 40, "IREC 하늘은 유채색(파랑)이어야 함"

    # 추적성: IREC 시퀀스 결과 meta에 지표가 있고, 대비/폭 기준을 상당수 프레임이 충족
    base = ROOT / "output" / "_test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    scene = [s for s in synth.irec_scenes() if s.name == "irec_seq_drogue_then_main"][0]
    scene.frames = 24
    meta = synth.generate(scene, base)
    tk = meta["trackability"]
    assert tk["frames_measured"] > 0
    assert tk["mean_target_width_px"] >= 8.0, "표적이 추적 가능한 최소 크기 이상이어야 함"
    assert tk["trackable_ratio"] >= 0.5, f"IREC 프레임의 절반 이상은 추적 가능해야 함: {tk}"


def test_irec_scenes_additive():
    """IREC 시나리오가 존재하고, 기존/시퀀스 시나리오와 이름 충돌이 없다."""
    irec = {s.name for s in synth.irec_scenes()}
    assert "irec_seq_drogue_then_main" in irec
    reg = {s.name for s in synth.regression_scenes()}
    seq = {s.name for s in synth.sequence_scenes()}
    assert irec.isdisjoint(reg) and irec.isdisjoint(seq), "IREC 이름은 기존과 겹치면 안 됨"
    # 모든 IREC 시나리오는 사실성이 켜져 있어야 함
    assert all(s.realism and s.realism.enabled for s in synth.irec_scenes())


def test_dynamics_offset_default_is_zero():
    """동역학 파라미터 기본값이면 오프셋 0 — 기존 동작 보존."""
    t = synth.main_target()  # swing/rotate/wind 모두 기본 0
    assert synth._dynamics_offset(10, t) == (0.0, 0.0, 0.0)


def _to_hsv(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype("float32")


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
    test_dual_target_gt_has_both()
    test_sequence_timeline_transitions()
    test_sequence_single_active_stage_and_state_flow()
    test_regression_scenes_preserved()
    test_realism_off_preserves_legacy_background()
    test_irec_realism_bluer_and_more_trackable()
    test_irec_scenes_additive()
    test_dynamics_offset_default_is_zero()
    print("ALL TESTS PASSED")
