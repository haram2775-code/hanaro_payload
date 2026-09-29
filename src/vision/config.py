"""Spec A — 설정 로드/검증.

HSV/밝기 임계값 등은 모두 **설정값(가정)**이며 실제 표적 색 확인 전 고정 요구조건이 아니다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AnalyzeConfig:
    # 입력
    init_bbox: tuple[int, int, int, int] | None = None  # x, y, w, h (수동 초기화)
    fps_override: float | None = None                   # 원본 FPS가 없을 때 사용

    # 분할(segmentation) — 밝은 표적을 배경에서 분리하는 임계값
    threshold: int = 128           # ROI 내 그레이 임계값(설정값)
    use_otsu: bool = True          # Otsu 자동 임계 사용 여부
    roi_margin: float = 0.6        # bbox 대비 ROI 확장 비율(설정값)

    # 유효성 판정 임계값(모두 설정값, 실데이터로 재조정)
    min_area_px2: float = 30.0
    max_area_fraction: float = 0.9   # 프레임 면적 대비 상한
    inflate_ratio: float = 1.4       # 직전 대비 면적 급증 비율 → INFLATING
    occlude_ratio: float = 0.5       # 직전 대비 면적 급감 비율 → OCCLUDED
    clip_margin_px: int = 2          # 경계 접촉 여유

    # 추적기
    tracker: str = "CSRT"            # CSRT|KCF|MOSSE, 미지원 시 색상 폴백

    # 재현성
    seed: int = 0

    @staticmethod
    def load(path: str | Path | None) -> "AnalyzeConfig":
        cfg = AnalyzeConfig()
        if path is None:
            return cfg
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(text) or {}
            except ImportError as e:
                raise RuntimeError("YAML 설정을 읽으려면 PyYAML이 필요합니다") from e
        else:
            data = json.loads(text)
        for k, v in data.items():
            if hasattr(cfg, k):
                if k in ("init_bbox",) and v is not None:
                    v = tuple(int(x) for x in v)
                setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.init_bbox is not None:
            if len(self.init_bbox) != 4 or self.init_bbox[2] <= 0 or self.init_bbox[3] <= 0:
                raise ValueError(f"init_bbox는 (x,y,w,h)이고 w,h>0 이어야 함: {self.init_bbox}")
        if self.min_area_px2 <= 0:
            raise ValueError("min_area_px2는 양수여야 함")
        if not (0 < self.max_area_fraction <= 1.0):
            raise ValueError("max_area_fraction는 (0,1] 범위여야 함")

    def as_dict(self) -> dict:
        return asdict(self)
