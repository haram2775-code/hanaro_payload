# 3D Mesh 캐노피 모델 (가정/설정값) — M1~M4

근거: 개발 공통 규칙 §1·§2·§3·§6, `configs/parachute_specs.md`(형상 출처),
`configs/irec_conditions.md`(장면 조건). 구현: `src/vision/mesh.py`,
`src/vision/canopy.py`(`render_backend="mesh"` 분기).

이 문서의 수치는 **실측이 아니라 합성용 가정/설정값**이다. 실제 카메라 보정·
캐노피 형상 실측이 오면 대체한다. 3D 충실도가 실제 검출 성공을 보장하지 않으며,
절대 거리/m/s는 다루지 않는다(§6).

## 1. 좌표계·카메라
- 캐노피 중심이 원점. 표면은 +Z로 볼록, 스커트가 z≈0.
- 핀홀 카메라: 초점거리 `f`(기본 700px, **가정값**), 주점=표적 화면중심, 표적을
  카메라 앞 `dist`(기본 6, 상대 단위)에 둔다. `scale=diameter_px/2`로 화면 크기를 맞춘다.
- 투영: `p' = R·p`, 원근 배율 `f/(z+dist)` 적용 후 화면 좌표로.

## 2. 곡면 프로파일 (반경 u∈[hole..1] → 높이 z)
- **toroidal (Main)**: `z(u)=sin(uπ)·0.55·infl − (1−u)²·0.5·infl`.
  둘째 항이 apex를 안으로 당겨(pull-down) 중앙 spill hole과 뒤 내벽을 만든다.
  안쪽 시작 반경 `u_lo = spill_hole_ratio`(=0.176, 제품 페이지 근거).
- **round (Drogue)**: `z(u)=√(1−u²)·0.6·infl` 얕은 반구, 구멍 없음.
- `infl`(inflation 0.05..1)이 곡면 진폭을 보간 → 전개(reefed→full)가 3D로 표현.

## 3. 자세(회전) = 관측각 + 진자 + 자전
`pose_matrix(view_elev, roll, tilt_x, tilt_y)`:
- `view_elev`(rad): 0=옆에서(스커트 정면), π/2=바로 아래(정면 링).
- `roll`: 자전(광축 둘레). `rotate_deg_per_frame` 누적.
- `tilt_x/tilt_y`: **진자 흔들림 기울임**. `swing_amp_px` 위상에 비례(최대 ~18°).
  → 캐노피가 흔들리며 실제로 기운 형상이 보인다(2D 백엔드는 tilt 무시).

## 4. 래스터화 — OpenCV-only (의존성 0)
- 삼각형별 `fillConvexPoly` + **painter z-정렬**(먼 면 먼저). GPU/GL 불필요.
- **Lambert 법선 음영**: 면 법선·광원 방향으로 `ambient + (1−ambient)·max(n·L,0)`.
- 한계(범위 명시): painter 정렬은 **볼록 근사**에 유효. 심하게 접힌(자기교차)
  캐노피는 z-정렬이 부정확할 수 있어 범위 밖으로 둔다.

## 5. GT 일관성
`mesh_projected_metrics`가 **투영 실루엣 마스크**에서 중심(moments)·bbox·면적·
등가직경을 계산 → 렌더와 GT가 동일 기준(2D의 해석식 불일치 원천 제거).
toroidal의 spill hole은 실루엣에 자연히 반영된다.

## 6. 기존 동작 보존
- `Canopy.render_backend`/`Target.render_backend` 기본값 `"2d"` → 기존 2D 경로
  그대로. mesh 코드는 backend="mesh"일 때만 실행된다.
- 기존 시나리오·시퀀스·IREC 시나리오는 모두 2D 백엔드를 유지한다. mesh 시나리오
  (`mesh_seq_drogue_then_main`, `mesh_main_descent`)는 추가 경로다.

## 7. 성능(가정, 측정 필요)
삼각형 수 = `n_radial × n_circum × 2`. drogue~수백, main~수천. PC 기준 프레임당
수~수십 ms 예상. **PC 시간을 Pi 성능으로 환산 금지**(§3) — Pi 실측은 별도.
