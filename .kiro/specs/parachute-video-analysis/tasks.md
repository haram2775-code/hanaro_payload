# Spec A — 구현 작업

각 작업은 요구사항(R#)과 설계 모듈에 대응한다. 완료 시 체크한다.

- [x] 1. 프로젝트 골격: `src/vision/`, `configs/`, `tests/`, `requirements.txt`, `README.md`
- [x] 2. `config.py` — 설정 로드/기본값/검증 (R1, R2)
- [x] 3. `video_io.py` — 프레임 읽기 + timestamp(추정 표시) + 주석 영상 쓰기 (R1, R5)
- [x] 4. `segmenter.py` — ROI mask, 면적, 등가직경 (R3)
- [x] 5. `tracker.py` — 초기 bbox 추적, 중심 반환 (R2, R3)
- [x] 6. `validity.py` — track_valid/size_valid/state 판정 (R3, R4)
- [x] 7. `writer.py` — CSV(고정 스키마) + 실패 구간 + 메타데이터 JSON (R5, R6)
- [x] 8. `analyze.py` — CLI 파이프라인 조립, 결정론 (R6)
- [x] 9. 단위 테스트: segmenter/validity/writer + 합성영상 end-to-end 스모크
- [x] 10. 재현성 확인: 동일 입력 2회 실행 → CSV 동일 (R6)

## 관련 Spec
- Spec B(합성영상·정답 생성기)를 먼저 구현해 실영상 없이 K1/K2를 검증할 수 있게 한다.
