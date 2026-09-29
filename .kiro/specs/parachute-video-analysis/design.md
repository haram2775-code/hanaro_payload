# Spec A — 설계

근거: `KIRO_DEVELOPMENT_PROPOSAL.md` §3.2, §3.3. 공통 규칙은 `.kiro/steering/개발-공통-규칙.md`.

## 1. 데이터 흐름

```
설정(YAML/JSON) + 영상 파일 + 초기 bbox
        │
        ▼
[VideoReader] ──frame, timestamp, ts_is_estimated──▶ [Tracker]
                                                        │ center, bbox
                                                        ▼
                                                   [Segmenter]
                                                        │ mask, area, eq_diameter
                                                        ▼
                                                 [ValidityJudge] ── track_valid, size_valid, state
                                                        │
                          ┌─────────────────────────────┼─────────────────────────────┐
                          ▼                             ▼                             ▼
                    [CsvWriter]                 [AnnotatedVideoWriter]           [RunMetadata]
```

## 2. 모듈

| 모듈 | 책임 | 비고 |
|---|---|---|
| `config.py` | 설정 로드·검증, 기본값 | HSV 범위 등은 **설정값**(가정), 고정 요구조건 아님 |
| `video_io.py` | 프레임 읽기, timestamp 산출, 주석 영상 쓰기 | 실제 촬영시각 없으면 `ts_is_estimated=True` |
| `tracker.py` | 초기 bbox로 추적, 중심·bbox 반환 | OpenCV 트래커(가능 시) 또는 색상 후보 |
| `segmenter.py` | bbox ROI에서 mask, 면적, 등가직경 | 색상·형상 기반. 실제 표적 색 미확인 → 설정으로 분리 |
| `validity.py` | track_valid, size_valid, state 판정 | 팽창·가림·잘림 → size_valid=False |
| `writer.py` | CSV, 실패 구간, 실행 메타데이터 JSON | 스키마는 요구사항 R5 고정 |
| `analyze.py` | CLI 진입점, 파이프라인 조립 | 결정론 보장(seed 고정) |

## 3. 유효성·상태 판정 규칙 (설계 결정, 튜닝 대상)

- `track_valid`: 트래커가 업데이트에 성공하고 중심이 프레임 안일 때 True.
- `size_valid`: mask 면적이 하한/상한 사이이고, bbox가 화면 경계에 닿지 않으며(=CLIPPED 아님), 면적 급증 비율이 임계 이하(=INFLATING 아님)일 때 True.
- 상태 우선순위: `LOST`(track 실패) → `CLIPPED`(경계 접촉) → `OCCLUDED`(면적 급감/마스크 소실) → `INFLATING`(면적 급증) → `TRACK`.
- 임계값(면적 하한/상한, 급변 비율, 경계 여유)은 모두 설정값이며 실데이터로 재조정한다.

## 4. 결정론
- 랜덤을 쓰지 않거나, 쓰면 seed를 설정에서 받아 고정한다.
- OpenCV 트래커 선택과 파라미터를 메타데이터에 기록해 동일 입력 재현을 보장한다.

## 5. 절대량 금지
- 이 Spec은 픽셀 단위 크기·중심만 출력한다. 절대 거리·m/s는 출력하지 않는다(공통 규칙 §6).

## 6. 실패·엣지 케이스
- 영상 열기 실패, 빈 프레임, ROI가 프레임 밖, mask 전무 → 해당 프레임을 유효성 False로 기록하고 실패 구간 목록에 추가(중단하지 않음).
