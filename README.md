# HANARO Payload — Vision (Spec A + Spec B)

`KIRO_DEVELOPMENT_PROPOSAL.md`의 첫 두 개발 대상을 구현한 것이다.

- **Spec A** 저장영상 낙하산 분석: 영상 → 수동 표적 초기화 → 추적 → 중심·면적·등가직경 측정 → CSV + 주석 영상.
- **Spec B** 합성영상·정답(GT) 생성기: 실영상 없이 K1/K2를 검증할 시험 영상과 정답값 생성.

공통 규칙은 `.kiro/steering/개발-공통-규칙.md`, Spec 문서는 `.kiro/specs/`에 있다.
이 코드는 **픽셀 단위** 중심·크기만 출력한다. 절대 거리/m/s는 출력하지 않는다(카메라 보정·투영 치수 미확인).

## 설치
```powershell
pip install -r requirements.txt
```
> **트래커 참고:** 기본 `opencv-python`에는 CSRT/KCF/MOSSE 트래커가 없어, 그 경우 파이프라인은
> 마스크 중심을 따라가는 폴백으로 동작한다(메타데이터 `tracker_backend=fallback`).
> 실영상에서 강인한 추적이 필요하면 `pip install opencv-contrib-python`를 사용한다.

## 사용법

합성 시험 영상 + 정답값 생성 (모든 회귀 시나리오):
```powershell
$env:PYTHONPATH="src"
python -m vision.synth --scene all --out data\synthetic
```

한 영상 분석 (수동 bbox 초기화):
```powershell
python -m vision.analyze --video data\synthetic\approach.mp4 `
    --bbox 290 210 60 60 --out output\run_approach --source-kind synthetic
```

## 출력물
| 파일 | 내용 |
|---|---|
| `<name>.csv` | 프레임별 결과. 컬럼: `frame_id, timestamp_s, center_x_px, center_y_px, mask_area_px2, equivalent_diameter_px, track_valid, size_valid, state, processing_ms` |
| `<name>_annotated.mp4` | bbox·중심·상태를 그린 주석 영상 |
| `<name>_meta.json` | 코드 버전, OpenCV 버전, seed, 영상 식별자, 실제/합성 구분, FPS, timestamp 추정 여부, 트래커 백엔드 등 |
| `<name>_failures.json` | track/size 무효 프레임 목록(실패 구간) |

## 핵심 설계 결정
- **유효성 분리:** `track_valid`(중심)와 `size_valid`(크기)를 분리 판정한다. 가림·팽창·화면 잘림에서는 중심이 유효해도 크기를 무효(size_valid=0)로 기록한다.
- **상태:** `SEARCH | TRACK | OCCLUDED | CLIPPED | INFLATING | LOST`.
- **재현성:** 동일 영상·설정 재실행 시 CSV가 동일하다(processing_ms 제외). seed·버전을 메타데이터에 기록한다.
- **가정 분리:** HSV/밝기 임계값 등은 설정값(가정)이며 실제 표적 색 확인 전 고정 요구조건이 아니다.

## 시험
```powershell
python tests\test_pipeline.py      # 또는: python -m pytest tests -q
```

## 종료 조건 (K1)
한 영상을 끝까지 처리하고 CSV·주석 영상·메타데이터를 저장하며, 상실·잘림·가림을 상태로 표시한다.
자동 표적 탐색·재포착은 다음 단계이며 이 구현에 포함되지 않는다.
