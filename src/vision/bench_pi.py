"""Pi Zero 2 W 실측 벤치마크 (경량 추적기).

실제 하드웨어(Raspberry Pi Zero 2 W)에서 vision.tracker_lite.LiteTracker의
프레임당 처리 시간·검출률·CPU/메모리를 측정하고, 4608×2592·14 fps
(71.4 ms/프레임) 예산 대비 결과를 CSV + 요약 리포트로 남긴다.

기존 모듈·인터페이스는 변경하지 않고 재사용한다:
  - tracker_lite.LiteTracker / RoiConfig / relative_scale_rate / classify_relative_motion
  - gimbal.GimbalErrorModel.offset_px(frame_id, fps, rng)
  - (선택) video_io.VideoReader — --video 모드에서 디코드 포함 측정

pytest 없이 독립 실행:
  python -m vision.bench_pi                      # 합성 프레임(디코드 제외) 순수 추적 연산
  python -m vision.bench_pi --full-res           # 4608×2592 프레임 실측(디코드 제외)
  python -m vision.bench_pi --video data/synthetic/gimbal_seq_track.mp4   # 디코드 포함
  python tests/../src/vision/bench_pi.py         # __main__ 러너로도 실행 가능

── 가정 명시 (측정 방식·지표) ──
  1. 입력 소스: 실제 카메라가 없어도 재현 가능하도록 기본은 **메모리 합성 프레임**
     (디코드 비용 제외, 순수 추적 연산 측정). --video는 디코드 포함 전체 파이프라인.
  2. 해상도: --full-res는 4608×2592 프레임을 매 프레임 새로 만들어 Pi RAM·연산을
     실측(34MB/프레임 → 링버퍼 없이 1장씩). 기본은 640×480(가벼운 스모크).
  3. CPU/메모리: psutil이 있으면 프로세스 CPU%/RSS, 없으면 표준 라이브러리 폴백
     (resource.getrusage 최대 RSS[Linux], time.process_time CPU). Pi(라즈비안)는
     resource 항상 존재 → psutil 미설치에서도 동작.
  4. 예산: 71.4 ms/프레임(14fps) 대비 p95. 이 스크립트는 Pi에서 실행되면 platform
     정보를 리포트에 기록하므로, PC→Pi 클럭 환산(공통 규칙 §3)이 아니라 실측이다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np

from . import code_version
from .tracker_lite import (
    LiteTracker, RoiConfig, relative_scale_rate, classify_relative_motion,
)
from .gimbal import GimbalErrorModel

# psutil 있으면 사용, 없으면 폴백.
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - 환경 의존
    psutil = None
    _HAS_PSUTIL = False

try:
    import resource  # Linux/Unix only
    _HAS_RESOURCE = True
except Exception:  # pragma: no cover - Windows
    resource = None
    _HAS_RESOURCE = False


BUDGET_FPS = 14.0
BUDGET_MS = 1000.0 / BUDGET_FPS  # 71.43 ms/프레임
FULL_RES = (4608, 2592)          # IMX708 유효 해상도 (camera_geometry.md)


# ── 합성 프레임 생성기 (디코드 비용 제외, 순수 추적 연산 측정용) ───
def _synth_frame(w: int, h: int, frame_id: int, rng: np.random.Generator,
                 target_offset=(0.0, 0.0)) -> np.ndarray:
    """파란 하늘 + 중앙 부근(오프셋)의 주황/흰 낙하산 1개를 그린 프레임.

    tracker_lite가 잡을 수 있는 최소한의 표적을 만든다(무거운 렌더 없이 빠르게).
    실제 합성 파이프라인(synth.py)과 별개 — 벤치는 디코드/렌더가 아니라 추적
    연산 시간을 재는 것이 목적이라 가벼운 표적으로 충분하다.
    """
    frame = np.empty((h, w, 3), dtype=np.uint8)
    # 하늘 그라디언트(BGR): 위 진파랑 → 아래 옅음
    top = np.array([150, 95, 45], dtype=np.float32)
    bot = np.array([205, 180, 150], dtype=np.float32)
    col = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    grad = (top[None, :] * (1 - col) + bot[None, :] * col)  # (h,3)
    frame[:] = grad[:, None, :].astype(np.uint8)
    # 표적: 중앙 + 오프셋
    cx = int(w / 2 + target_offset[0])
    cy = int(h / 2 + target_offset[1])
    r = max(6, int(min(w, h) * 0.03))
    cv2.circle(frame, (cx, cy), r, (40, 150, 240), -1)       # 주황 캐노피
    cv2.circle(frame, (cx, cy), r // 2, (240, 240, 240), -1)  # 흰 패널
    return frame


# ── 자원(CPU/메모리) 샘플러 ────────────────────────────────────────
class ResourceSampler:
    def __init__(self):
        self.backend = "psutil" if _HAS_PSUTIL else ("resource" if _HAS_RESOURCE else "none")
        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        if self._proc is not None:
            self._proc.cpu_percent(None)  # 첫 호출은 0 → 초기화
        self.rss_samples_mb: list[float] = []
        self.cpu_samples: list[float] = []
        self._t_wall0 = time.time()
        self._t_cpu0 = time.process_time()

    def sample(self) -> None:
        if self._proc is not None:
            try:
                self.rss_samples_mb.append(self._proc.memory_info().rss / (1024 * 1024))
                self.cpu_samples.append(self._proc.cpu_percent(None))
            except Exception:
                pass

    def summary(self) -> dict:
        out: dict = {"backend": self.backend}
        if self.rss_samples_mb:
            out["rss_mb_mean"] = round(sum(self.rss_samples_mb) / len(self.rss_samples_mb), 1)
            out["rss_mb_peak"] = round(max(self.rss_samples_mb), 1)
        if self.cpu_samples:
            # 첫 샘플(0) 제외 평균
            vals = [c for c in self.cpu_samples[1:]] or self.cpu_samples
            out["cpu_percent_mean"] = round(sum(vals) / len(vals), 1)
            out["cpu_percent_peak"] = round(max(self.cpu_samples), 1)
        # resource 폴백(최대 RSS)
        if _HAS_RESOURCE:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            # Linux: ru_maxrss는 KB. macOS: bytes. Linux 가정(Pi).
            out["max_rss_mb_rusage"] = round(ru.ru_maxrss / 1024.0, 1)
        # 전체 CPU 사용률(프로세스 CPU시간 / 벽시계)
        wall = max(1e-9, time.time() - self._t_wall0)
        cpu = time.process_time() - self._t_cpu0
        out["cpu_util_overall"] = round(100.0 * cpu / wall, 1)
        return out


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * q))
    return sorted_vals[idx]


# ── 벤치 실행 ──────────────────────────────────────────────────────
def run_bench(*, frames: int = 200, width: int = 640, height: int = 480,
              full_res: bool = False, roi_size: int = 512, proc_max: int = 256,
              use_gimbal_error: bool = True, video: str | None = None,
              warmup: int = 5, seed: int = 0, out_dir: str = "output/bench_pi") -> dict:
    """Pi에서 프레임당 처리 시간·검출률·자원을 측정한다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if full_res:
        width, height = FULL_RES

    tracker = LiteTracker(RoiConfig(roi_size=roi_size, proc_max=proc_max))
    gmodel = GimbalErrorModel(seed=seed)
    rng_off = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    fps = BUDGET_FPS
    dt = 1.0 / fps
    sampler = ResourceSampler()

    # 입력 소스 준비
    reader = None
    src = "synth-memory"
    if video is not None:
        from .video_io import VideoReader
        reader = VideoReader(video)
        width, height = reader.width, reader.height
        src = f"video:{Path(video).name}"

    per_frame_ms: list[float] = []
    rows = []
    found_frames = 0
    prev_diam = None
    n = 0

    def frame_iter():
        if reader is not None:
            for fd in reader:
                yield fd.frame_id, fd.image
        else:
            for i in range(frames + warmup):
                off = gmodel.offset_px(i, fps, rng_off) if use_gimbal_error else (0.0, 0.0)
                yield i, _synth_frame(width, height, i, rng_np, target_offset=off)

    for frame_id, image in frame_iter():
        offset = gmodel.offset_px(frame_id, fps, rng_off) if use_gimbal_error else (0.0, 0.0)
        t0 = time.perf_counter()
        found, center, diam, bbox, roi = tracker.step(image, gimbal_offset=offset)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # warmup 프레임은 통계에서 제외(캐시/JIT 워밍업)
        if n < warmup:
            n += 1
            continue

        per_frame_ms.append(elapsed_ms)
        rate = relative_scale_rate(prev_diam, diam, dt) if (found and prev_diam) else 0.0
        motion = classify_relative_motion(rate) if found else "LOST"
        if found:
            found_frames += 1
            prev_diam = diam
        sampler.sample()
        rows.append({
            "frame_id": frame_id,
            "found": int(found),
            "eq_diam_px": round(diam, 2),
            "gimbal_dx_px": round(offset[0], 1),
            "gimbal_dy_px": round(offset[1], 1),
            "rel_motion": motion,
            "processing_ms": round(elapsed_ms, 4),
        })
        n += 1

    if reader is not None:
        # VideoReader가 컨텍스트를 열어두면 닫는다(있으면).
        rel = getattr(reader, "release", None)
        if callable(rel):
            rel()

    measured = len(per_frame_ms)
    ms_sorted = sorted(per_frame_ms)
    p50 = _percentile(ms_sorted, 0.50)
    p95 = _percentile(ms_sorted, 0.95)
    p99 = _percentile(ms_sorted, 0.99)
    mx = max(per_frame_ms) if per_frame_ms else 0.0
    mean = sum(per_frame_ms) / measured if measured else 0.0
    achievable_fps = (1000.0 / p95) if p95 > 0 else 0.0

    # CSV
    csv_path = out / "bench_pi_frames.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    is_pi = _detect_pi()
    res = sampler.summary()
    summary = {
        "spec": "pi-zero-2w-bench",
        "code_version": code_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "is_raspberry_pi": is_pi,
            "cpu_count": os.cpu_count(),
            "note": ("Pi 실측" if is_pi else
                     "이 실행은 Pi가 아님 — PC 시간을 Pi 성능으로 환산 금지(공통 규칙 §3). "
                     "최종 14fps 판정은 Pi Zero 2W 실측으로만."),
        },
        "input": {
            "source": src,
            "resolution": [width, height],
            "full_res_4608x2592": bool(full_res or (width, height) == FULL_RES),
            "frames_measured": measured,
            "warmup_frames": warmup,
        },
        "roi": {"roi_size": roi_size, "proc_max": proc_max,
                "gimbal_error_applied": use_gimbal_error,
                "gimbal_sigma_deg": round(gmodel.sigma_total_deg(), 3)},
        "timing_ms": {
            "mean": round(mean, 3), "p50": round(p50, 3), "p95": round(p95, 3),
            "p99": round(p99, 3), "max": round(mx, 3),
            "budget_ms_at_14fps": round(BUDGET_MS, 2),
            "meets_budget_p95": bool(p95 <= BUDGET_MS),
            "meets_budget_max": bool(mx <= BUDGET_MS),
            "achievable_fps_from_p95": round(achievable_fps, 1),
        },
        "detection": {
            "found_fraction": round(found_frames / max(1, measured), 3),
        },
        "resources": res,
        "outputs": {"csv": str(csv_path)},
    }
    (out / "bench_pi_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(out / "bench_pi_report.txt", summary)
    return summary


def _detect_pi() -> bool:
    """라즈베리파이 여부를 best-effort로 판정(측정이 Pi 실측인지 리포트에 남기기 위함)."""
    if platform.system() != "Linux":
        return False
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            txt = Path(path).read_bytes().decode(errors="ignore").lower()
            if "raspberry pi" in txt:
                return True
        except Exception:
            pass
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(errors="ignore").lower()
        if "raspberry" in cpuinfo or "bcm2" in cpuinfo:
            return True
    except Exception:
        pass
    return False


def _write_report(path: Path, s: dict) -> None:
    t = s["timing_ms"]
    r = s["resources"]
    p = s["platform"]
    lines = [
        "=== Pi Zero 2 W 경량 추적기 벤치마크 ===",
        f"코드: {s['code_version']}  OpenCV: {s['opencv_version']}  NumPy: {s['numpy_version']}",
        f"플랫폼: {p['system']}/{p['machine']} python{p['python']} "
        f"cores={p['cpu_count']}  Raspberry Pi 실측={p['is_raspberry_pi']}",
        f"  {p['note']}",
        f"입력: {s['input']['source']}  해상도={s['input']['resolution']}  "
        f"4608x2592={s['input']['full_res_4608x2592']}  "
        f"측정 프레임={s['input']['frames_measured']}(워밍업 {s['input']['warmup_frames']} 제외)",
        f"ROI: {s['roi']['roi_size']}px → 다운스케일 {s['roi']['proc_max']}px  "
        f"짐벌오차={s['roi']['gimbal_error_applied']}(σ≈{s['roi']['gimbal_sigma_deg']}°)",
        "",
        "── 프레임당 처리 시간 [ms] ──",
        f"  mean={t['mean']}  p50={t['p50']}  p95={t['p95']}  p99={t['p99']}  max={t['max']}",
        f"  예산(14fps)={t['budget_ms_at_14fps']}ms  "
        f"p95 예산내={t['meets_budget_p95']}  max 예산내={t['meets_budget_max']}",
        f"  p95 기준 달성 가능 fps ≈ {t['achievable_fps_from_p95']}",
        "",
        f"검출률: {s['detection']['found_fraction']}",
        "",
        "── 자원 사용 ──",
        f"  backend={r.get('backend')}  "
        f"RSS mean={r.get('rss_mb_mean','n/a')}MB peak={r.get('rss_mb_peak','n/a')}MB "
        f"maxRSS(rusage)={r.get('max_rss_mb_rusage','n/a')}MB",
        f"  CPU% mean={r.get('cpu_percent_mean','n/a')} peak={r.get('cpu_percent_peak','n/a')}  "
        f"전체 CPU 사용률={r.get('cpu_util_overall','n/a')}%",
        "",
        f"CSV: {s['outputs']['csv']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pi Zero 2W 경량 추적기 실측 벤치마크")
    p.add_argument("--frames", type=int, default=200, help="측정 프레임 수(워밍업 제외)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--full-res", action="store_true",
                   help="4608x2592 실측(디코드 제외, 프레임 1장씩 생성)")
    p.add_argument("--roi-size", type=int, default=512)
    p.add_argument("--proc-max", type=int, default=256)
    p.add_argument("--video", default=None, help="영상 파일(디코드 포함 측정)")
    p.add_argument("--no-gimbal-error", action="store_true", help="짐벌 오차 주입 끔")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output/bench_pi")
    return p


def main(argv: list[str] | None = None) -> int:
    a = _build_parser().parse_args(argv)
    s = run_bench(frames=a.frames, width=a.width, height=a.height, full_res=a.full_res,
                  roi_size=a.roi_size, proc_max=a.proc_max,
                  use_gimbal_error=not a.no_gimbal_error, video=a.video,
                  warmup=a.warmup, seed=a.seed, out_dir=a.out)
    # 요약 리포트를 표준출력에도 인쇄
    print(Path(a.out, "bench_pi_report.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
