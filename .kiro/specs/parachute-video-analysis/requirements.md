# Spec A — 저장영상 낙하산 분석 프로그램: 요구사항

근거: `KIRO_DEVELOPMENT_PROPOSAL.md` §3, §5(Spec A), §6(K1). 이 요구사항은 Kiro 생성물이며 팀 승인 임무 요구조건이 아니다.

## 범위
첫 버전은 영상을 열고, 첫 프레임에서 표적 영역을 **수동 선택**한 뒤, 이후 프레임의 추적 결과를 표시하고 CSV로 저장하는 수준으로 제한한다. 자동 표적 탐색과 재포착은 다음 단계로 둔다. 수동으로 초기화한 추적 성능을 자동 검출 성능으로 보고하지 않는다.

## User Story
개발자로서, 저장된(또는 합성) 낙하산 영상을 한 번 실행으로 끝까지 처리하고 프레임별 중심·크기·유효성·처리시간을 재현 가능한 CSV로 얻고 싶다. 그래야 상대운동 추정기의 입력 품질과 Pi 이식 우선순위를 판단할 수 있다.

## 요구사항 (EARS)

### R1 영상 입력
- WHEN 사용자가 영상 파일 경로와 설정 파일을 제공하면 THE SYSTEM SHALL 프레임을 순서대로 읽어 처리한다.
- IF 원본에 실제 촬영시각이 없으면 THE SYSTEM SHALL FPS로 timestamp를 계산하고 그 값이 추정값임을 실행 메타데이터에 표시한다.

### R2 표적 초기화 (수동)
- WHEN 초기 표적 영역(bbox)이 설정·인자·인터랙션 중 하나로 주어지면 THE SYSTEM SHALL 그 영역으로 추적을 초기화한다.
- THE SYSTEM SHALL 자동 표적 탐색을 이 Spec에서 수행하지 않는다.

### R3 추적과 측정
- THE SYSTEM SHALL 각 프레임에서 표적을 추적하고 중심(center_x, center_y)을 계산한다.
- THE SYSTEM SHALL 표적 경계에서 mask 면적과 등가직경을 측정한다.
- THE SYSTEM SHALL 중심 추적 유효성(`track_valid`)과 크기 측정 유효성(`size_valid`)을 분리해 판정한다.

### R4 상태 판정
- THE SYSTEM SHALL 각 프레임 상태를 `SEARCH|TRACK|OCCLUDED|CLIPPED|INFLATING|LOST` 중 하나로 기록한다.
- WHEN 표적이 화면 밖으로 나가거나 마스크를 얻지 못하면 THE SYSTEM SHALL 상실(LOST) 또는 잘림(CLIPPED)을 기록한다.

### R5 출력
- THE SYSTEM SHALL 프레임별로 CSV 한 행을 기록한다. 컬럼: `frame_id, timestamp_s, center_x_px, center_y_px, mask_area_px2, equivalent_diameter_px, track_valid, size_valid, state, processing_ms`.
- THE SYSTEM SHALL 주석(annotated) 영상, 실행 조건 메타데이터(JSON), 실패 구간 목록을 저장한다.

### R6 재현성
- WHEN 동일 영상·동일 설정으로 다시 실행하면 THE SYSTEM SHALL 동일한 CSV 결과를 생성한다.
- THE SYSTEM SHALL 실행 메타데이터에 코드 버전/commit, 설정·의존성 버전, seed, 영상 식별자, 실제/합성 구분을 기록한다.

## 종료 조건 (K1)
한 영상을 끝까지 처리하고 CSV·주석 영상·설정을 저장하며, 표적 상실·잘림을 표시한다. 자동 검출 완료를 뜻하지 않는다.
