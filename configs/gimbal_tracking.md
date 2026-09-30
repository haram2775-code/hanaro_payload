# 짐벌 연동 추적 · 성능 예산 (Spec: 짐벌 연동 · 14 fps · Pi Zero 2 W)

근거 문서: `configs/camera_geometry.md`, `configs/parachute_specs.md`,
외부 레포 `github.com/smchoi02/gimbal_control`(README·SD_data.md·gps_module_design.md, 2026-09-30 확인).
공통 규칙 §1·§2·§3·§6 준수 — 모든 수치는 **가정/설정값**이며 실측·확정 요구조건이 아니다.

## 1. 짐벌 연동 (primary = GPS/IMU 짐벌, vision = 잔여 오차 보정)

### 1.1 짐벌 인터페이스 (레포 확인)
`gimbal_control`은 **GPS/IMU 기반 coarse 지향 제어기**다. 페이로드가 로켓 본체 방향으로
카메라를 돌려 표적을 화면 중앙 근처에 둔다.

- 출력(`gimbal_target_tracker` README, `SD_data.md`):
  - `target_yaw_deg` / `target_pitch_deg` — 계산된 표적 방향.
  - `cmd_yaw/pitch` — 급동작 억제 필터를 통과한 실제 모터 명령(≠ target).
  - `pos_yaw/pitch` — Dynamixel 엔코더 현재각.
  - `rel_n_m/rel_e_m/rel_d_m`, `range_m` — 페이로드 기준 로켓 NED 상대 위치.
  - 상태 FSM: `BOOT→WAIT→ALIGN→TRACK`, 실패 시 `HOLD`(마지막 시선 유지 + IMU 흔들림 보정), `FAULT`.
- 최소 조건: 수평 거리 ≥ 5 m, 초기 정렬 10 s 안정 필요.
- 갱신율: 짐벌 목표 명령 50 Hz, LoRa 로켓 수신 10 Hz, SD 로그 2 Hz.

→ **영상 추적기의 역할**: 짐벌이 남긴 **잔여 지향 오차를 화면 안에서 흡수·정밀 보정**한다.
1차 획득(acquisition)은 짐벌이 하므로, 영상 추적기는 화면 중앙 부근의 ROI만 보면 된다.

### 1.2 짐벌 지향 오차 가정 (오차원·근거·크기)
| 오차원 | 레포 근거 | 가정 크기(1σ) |
|---|---|---|
| GPS 수평 오차 | MAX-M10S `hacc_mm`; gps_module_design "GPS 수평 수 m" | 원거리에서 각도로 환산되는 저주파 편이 |
| BNO085 yaw bias | "Game Rotation Vector yaw는 절대 북쪽 기준 아님, bias 보정 필요" | yaw 정렬 잔차 ~1.4° |
| 기압 pitch bias | "정렬 못 맞추면 pitch에 기압 초기오차 잔존" | pitch 잔차 ~1.0° |
| 명령 필터 지연 | "급격한 동작 막는 필터"(cmd≠target) | 표적 이동 시 수 프레임 지연 |
| IMU/통신 지터 | `sens_age`, `age`(ms) 산포 | 프레임별 ~0.6° |

- **종합 지향 오차 σ ≈ 3.0° (1σ, 독립성분 RSS).** 실측 없음 → 설정값.
- 화면 환산(`camera_geometry.md`, f≈3548 px): **3° ≈ 186 px** 오프셋.
  → 표적은 프레임 중앙에서 최대 수백 px 벗어난 위치에 나타날 수 있다.
- 구현: `src/vision/gimbal.py`의 `GimbalErrorModel`
  = 정렬 잔차(상수) + GPS 저주파 편이(사인) + 프레임별 지터 + 명령 지연.
  합성영상에서 표적을 이 오프셋 위치에 두어, 추적기가 오차를 흡수하는지 검증한다.
- 흡수 방법: `src/vision/tracker_lite.py`가 **중앙 ROI(기본 512 px)** 를 오프셋 위치에
  두고 그 안에서만 세그먼트. ROI가 짐벌 잔여 오차(~186 px)를 넉넉히 덮는다.

## 2. 분리(멀어짐) 속도 가정 (극단값 배제, 근거 명시)

페이로드도 본체와 **동일 구속조건**: 3 ft drogue + main, 1.5 kg, drogue ≥20 m/s, main ≤11 m/s.
두 물체(본체·페이로드)의 **낙하 속도차 = 분리 속도**. 실측 없음 → 합리적 범위 가정.

| 단계 | 두 물체 속도대 | 가정 분리 속도(속도차) | 근거 |
|---|---|---|---|
| drogue | 20–24 m/s | **3–6 m/s** | 동일 3ft·1.5kg이라 종단속도차는 전개 타이밍·항력 미세차뿐 → 작음. 극단(20−9=11)은 지시대로 배제 |
| main | ≤11 m/s | **1.5–3 m/s** | 동일 낙하산·질량이면 저속 구간 속도차는 더 작음 |

- **낙하가 빠를수록 더 빨리 멀어진다** → drogue 단계에서 표적 축소율이 크고
  (`diameter_growth_per_frame` < 1이 더 낮음), main 단계는 완만.
- 합성 반영: `gimbal_scenes()`의 drogue 단계 `growth≈0.992`(빠른 축소),
  main 단계 `growth≈0.998`(완만). 원거리 변형(`gimbal_far_track`)은 표적을 더 작게 시작.
- **상대속도 산출**: 절대 m/s 확정 금지(§6). 대신 등가직경의 **상대 스케일 변화율**
  `d(diam)/diam/dt [1/s]`로 내고, `APPROACH/HOLD/RECEDE`로 분류
  (`tracker_lite.relative_scale_rate`, `classify_relative_motion`).
  카메라 보정·유효 투영치수가 확정되면 이 스케일 속도를 절대량으로 승격 가능.

## 3. 성능 예산 (Raspberry Pi Zero 2 W, 14 fps)

### 3.1 하드웨어 제약
- **RAM 512 MB**(공유, OS·드라이버 점유분 제외하면 가용 수백 MB).
- 4608×2592 BGR **원본 1장 = 4608·2592·3 ≈ 34 MB**. 링버퍼·복사본 몇 장이면 수백 MB →
  **풀프레임 상시 보관·다중 버퍼링은 위험**.
- CPU: 쿼드코어 Cortex-A53 1 GHz(저전력). 무거운 트래커(CSRT)·풀프레임 컨볼루션은 부담.

### 3.2 프레임당 예산
- 14 fps → **프레임당 1000/14 ≈ 71.4 ms**.
- 처리 파이프라인(프레임당):
  1. 프레임 획득(카메라 스트림). ROI만 관심 → 가능하면 센서/ISP 크롭 사용.
  2. **중앙 ROI 크롭**(짐벌이 표적을 중앙 부근에 둠) — 512×512 = ~0.75 MB.
  3. **다운스케일**(≤256 px) → 연산량 1/4 이하.
  4. HSV 채도/밝기 세그먼트 + 모폴로지 + 최대 컨투어 → 중심·등가직경.
  5. EMA 평활 + 상대 스케일 변화율.
- CSRT·옵티컬플로우 불필요: 짐벌이 큰 움직임을 흡수하므로 프레임별 재검출로 충분.

### 3.3 14 fps 감당 여부 — 검토 방법과 한계
- `src/vision/analyze_lite.py`가 프레임별 `processing_ms`를 기록하고 메타에
  `p50/p95/max`와 예산(71.4 ms) 대비 `meets_budget_p95`를 남긴다.
- **주의(공통 규칙 §3)**: 이 측정은 **PC 실측치**다. **PC 시간을 클럭비로 환산해
  Pi 성능으로 확정하지 않는다.** Pi 처리성능은 **실제 보드 실측으로만** 확정한다.
  여기 리포트는 알고리즘 간 상대 비교·병목 식별용이며, 최종 14 fps 감당 판정은
  Pi Zero 2 W 실측 Gate로만 내린다.
- 설계상 근거(정성): ROI 512→다운스케일 256에서 세그먼트는 256×256 화소에 대한
  색변환·임계·모폴로지·컨투어뿐 → 저사양 ARM에서도 수~수십 ms 대가 현실적 목표.
  풀 4608×2592 세그먼트(≈36배 화소)는 예산 초과 위험이 크므로 ROI 전략이 필수.

### 3.4 원거리 한계
- `camera_geometry.md`: main(3.05 m) ~600 m, drogue(0.91 m) ~200 m까지 최소 표적폭 유지.
- `tracker_lite.RoiConfig.min_target_px`(기본 8 px) 이하이면 크기 신뢰 낮음으로 표시.
- 더 먼 거리는 ROI 다운스케일을 줄이거나(연산↑) 망원 렌즈(협 FOV)가 필요 — 별도 과제.

## 4. 보존(기존 동작 무변경)
- 기존 `vision.tracker.Tracker`(CSRT/폴백), `vision.analyze`(Spec A), 시퀀스·IREC·
  mesh/2D·관측각-하강 시나리오·모든 테스트는 **그대로**. 이 Spec은 전부 **추가**:
  - 신규 모듈 `gimbal.py`, `tracker_lite.py`, `analyze_lite.py`.
  - 신규 시나리오 `gimbal_seq_track`, `gimbal_far_track`(14 fps).
  - 신규 테스트(오차 모델·상대운동·ROI·시나리오·경량 end-to-end).
