# 낙하산 사양 (사용자 제공) — 가정/설정값

> **분류(공통 규칙 §1·§2):** 아래 값은 사용자가 제공한 낙하산 사양이며 **가정/설정값**이다.
> 팀 승인 임무 요구조건이 아니며 `01_REQUIREMENTS_BASELINE.md`로 승격하지 않는다.
> 출처: 사용자 채팅 입력(2026-09-29) + Rocketman 제품 계열 확인(the-rocketman.com, 2026-09-29).

## 표적 정합

| 역할 | 낙하산 | 영상 추적 대상? | 비고 |
|---|---|---|---|
| Drogue | 3 ft, Cd 0.97 | ✅ (고속 구간) | apogee~main 전개 전. 페이로드 가변 낙하산과 이름이 겹치나 별개 |
| Main | 10 ft, Cd 2.2 | ✅ (저속 구간) | 상태 문서의 "약 10 ft 표적". main 전개 후 가시 |

**두 표적 모두 추적**이 목표(사용자 확인). 단일 표적 가정을 버리고 다중 표적으로 다룬다.

## 상세 사양

> **형상 출처(2026-09-29 웹 확인):**
> - Main: [Rocketman High Performance CD 2.2 Parachutes](https://www.the-rocketman.com/products/rocketman-high-performance-cd-2-2-parachutes) — 제품 페이지 스펙 시트(사이즈별 gore 수·spill hole·투영/캐노피 면적)
> - Drogue: [Pro Experimental Drogue Parachutes](https://www.the-rocketman.com/products/pro-experimental-drogue-parachutes) — 제품 페이지(4-line 구성·재질)
> - 토로이달 형상 개념 참고: [Fruity Chutes Iris Ultra](https://ssh.fruitychutes.com/parachute_recovery_systems/iris_ultra_parachutes) — "toroidal = doughnut cut in half, pull-down apex (apex를 캐노피 안으로 당김)"
> - 제품 사진 URL은 웹 확인했으나 저장소에 저장하지 않음(저작권). 형상 정보만 텍스트로 인용.

### Drogue — 출처: Pro Experimental Drogue 제품 페이지
| 항목 | 값 | 분류 |
|---|---|---|
| 제조사 / 제품 | Rocketman Pro-Experimental Drogue | 확인 |
| 공칭 직경 | 3 ft (0.9144 m) | 사용자 제공 |
| Cd | 0.97 (전 사이즈 공통) | 제품 페이지 명시 |
| **캐노피 형상** | **평면-원형(flat/round), 4-line 구성** ("cross-form·conical보다 우수"라고 명시) | **제품 페이지 확인** |
| gore/현삭 수 | **4** (4-line configuration) | 제품 페이지 확인 |
| spill hole | 명시 없음 → 없음으로 모델링 | 추정 |
| 재질 | 1.9 oz Mil-Spec 립스톱, 70 데니어, 제로 포로시티 | 제품 페이지 확인 |
| 3ft 표면적 | 5.78 sq ft, 라인 4 ft, 무게 4.5 oz | 제품 페이지 확인 |
| 색상 | "Colors may vary" (제품별 상이) → 대비 높은 임의색 | 제품 페이지 확인 |
| Shock cord | 8600 lbs / 30 ft | 사용자 제공 |
| Deployment bag | 3" × 8" | 사용자 제공 |

### Main — 출처: Rocketman HPC CD 2.2 제품 페이지 (120in/10ft 행)
| 항목 | 값 | 분류 |
|---|---|---|
| 제조사 / 제품 | Rocketman High-Performance CD 2.2 | 확인 |
| 공칭 직경 | 10 ft = 120 in (3.048 m) | 사용자 제공 / 페이지 일치 |
| Cd | 2.2 | 사용자 제공 / 페이지 일치. 기준면적 정의 확인 전 확정 금지 |
| **캐노피 형상** | **Toroidal / Annular / Iris** (반구 돔 아님. apex를 캐노피 안으로 당긴 도넛형, 중앙에 큰 spill hole) | **제품 페이지 "Type" 명시** |
| **gore 수 (10ft)** | **12** | 제품 페이지 확인 (사이즈별: 2–4ft=8, 5–12ft=12, 14ft=16, 16–18ft=18, 24ft+=24) |
| **spill hole 직경 (10ft)** | **21.12 in** → 캐노피폭 대비 **~17.6%** 중앙 구멍 | 제품 페이지 확인 |
| 투영/캐노피 면적 (10ft) | 투영 76.10 sq ft / 캐노피 135.47 sq ft → 투영≈**56%** (토로이달 특징) | 제품 페이지 확인 |
| 현삭 수 (10ft) | 12 라인, 1/4" 브레이드 나일론 250 lb | 제품 페이지 확인 |
| swivel | 3,000 lb (10ft), 스윙/진동 억제 | 제품 페이지 확인 |
| 재질 | 1.1 oz Mil-Spec 립스톱, 20 데니어, 제로 포로시티 | 제품 페이지 확인 |
| 색상 | 교대(alternating) 컬러 패널 | 제품 페이지 확인 |
| Shock cord | 8600 lbs / 50 ft | 사용자 제공 |
| Deployment bag | 4" × 12" | 사용자 제공 |

> **형상 모델링 반영(canopy.py):**
> - Main → `kind="toroidal"`, `n_panels=12`, `spill_hole_ratio≈0.176` (중앙 구멍), 투영 면적에서 구멍 제외.
> - Drogue → `kind="round"`, `n_panels=4`, spill hole 없음.
> - **여전히 추정인 부분:** 정확한 색 배치·gore별 색 순서(제품 사진마다 다름, "colors may vary"), 전개 중 실제 투영 형상의 시간 변화, 정확한 3D→2D 투영 각도. 이들은 형상 근사이며 실측이 아니다.

## 시뮬레이션 입력 (Spec F용)

| 항목 | 값 | 분류 |
|---|---|---|
| 낙하(회수) 질량 | **≥ 1.5 kg** (규정 하한). 기본 1.5 kg, 민감도 범위 1.5–2.5 kg | 사용자 제공(규정 하한) |
| 대기 밀도 | ISA 해수면 1.225 kg/m³ 기본, 고도별 프로파일은 별도 | 가정 |
| 중력 | 9.80665 m/s² | 표준 |

## 여전히 미확인 (영상 절대거리 환산 blocker)
- 카메라 내부 파라미터(intrinsics): Camera Module 3 Wide 초점거리·센서모드·crop
- 전개 중 실제 **투영** 형상·유효 직경 (10 ft/3 ft는 공칭 평면 직경)
- 관측 방향·자세, 시각 동기
→ 이 값들이 없으면 절대 거리·m/s를 확정 출력하지 않는다(공통 규칙 §6).
