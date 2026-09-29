# Spec B — 합성영상·정답 데이터 생성기: 요구사항

근거: `KIRO_DEVELOPMENT_PROPOSAL.md` §2(②), §4, §5(Spec B), §6(K2). Kiro 생성물이며 팀 승인 임무 요구조건이 아니다.

## 범위
표적 크기·이동·가림·밝기·블러를 조절한 시험 영상과 **정답값(ground truth)**을 함께 생성한다. 실제 영상 없이 시작할 수 있다. 단순 도형 합성의 성공은 실제 낙하산 검출 성공을 의미하지 않으므로 실영상 검증은 별도로 유지한다.

## 요구사항 (EARS)

### R1 시나리오 생성
- WHEN 설정(크기·이동·가림·밝기·블러·정지/접근/이격/팽창 프로파일, seed)이 주어지면 THE SYSTEM SHALL 해당 시험 영상을 생성한다.

### R2 정답값 저장
- THE SYSTEM SHALL 프레임별 정답 CSV를 저장한다. 컬럼 최소: `frame_id, timestamp_s, gt_center_x_px, gt_center_y_px, gt_area_px2, gt_diameter_px, gt_visible, gt_state`.
- `gt_state`는 Spec A 상태 어휘(`TRACK|OCCLUDED|CLIPPED|INFLATING|LOST` 등)와 정합한다.

### R3 재현성
- WHEN 동일 설정·동일 seed로 다시 실행하면 THE SYSTEM SHALL 동일한 영상과 동일한 정답값을 생성한다.

### R4 회귀 세트
- THE SYSTEM SHALL 최소 회귀 시나리오를 제공한다: 표적 상실·복귀, 작은 표적, 가림, 팽창, 화면 잘림.

## 종료 조건 (K2 준비)
동일 설정·seed에서 동일 영상·정답값을 생성하고, Spec A가 이를 입력으로 중심·크기오차·유효율을 계산할 수 있다.
