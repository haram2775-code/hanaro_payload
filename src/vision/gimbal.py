"""짐벌 지향 오차 모델 (Spec: 짐벌 연동 추적).

primary 추적은 IMU/GPS 기반 짐벌 제어(github.com/smchoi02/gimbal_control)가
카메라를 표적 방향으로 돌려 표적을 화면 중앙 근처에 둔다는 가정.
영상 추적기의 역할은 그 **잔여 지향 오차**를 화면 안에서 흡수·정밀 보정하는 것이다.

이 모듈은 짐벌이 남기는 오차를 합성영상에서 재현한다: 표적을 프레임 중앙이 아니라
"짐벌이 맞춘 (오차 섞인) 위치"에 두도록, 프레임별 화면-공간 오프셋(px)을 생성한다.

── 오차원과 가정 크기 (gimbal_control 레포 확인 근거, 모두 가정/설정값) ──
  gimbal_target_tracker/README.md, gps_module_design.md, SD_data.md 확인:
  1. GPS 수평 오차 (MAX-M10S hacc): 원거리에서 각도로 환산되는 저주파 편이.
  2. BNO085 yaw: "Game Rotation Vector yaw는 절대 북쪽 기준 아님" → yaw bias.
  3. 기압 bias(pitch): "정렬 못 맞추면 pitch에 기압 초기오차 잔존".
  4. 명령 필터 지연: cmd ≠ target ("급격한 동작 막는 필터") → 표적 이동 시 지연.

  → 종합 지향 오차 σ ≈ 3.0° (1σ). 실측 없음 → 설정값. camera_geometry.md의
    f_px로 각→픽셀 환산: offset_px ≈ f_px · tan(err_rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class GimbalErrorModel:
    """짐벌 지향 오차 → 화면 공간 표적 오프셋(px) 생성기.

    구성 요소(모두 가정/설정값, gimbal_control 레포 오차원 근거):
      static_bias_deg   : 정렬 잔차(yaw bias + baro pitch bias). 클립 내 상수.
      drift_amp_deg     : GPS 저주파 편이 진폭.
      drift_period_s    : 그 편이의 주기.
      jitter_std_deg    : 프레임별 랜덤 지터(IMU 노이즈, 통신 지연 산포).
      lag_frames        : 명령 필터 지연(표적이 움직이면 오프셋이 지연 반영).
      f_px              : 카메라 초점거리[px] (camera_geometry.md, IMX708 ≈ 3548).
    """
    static_bias_deg: tuple[float, float] = (1.4, 1.0)   # (yaw, pitch) 정렬 잔차
    drift_amp_deg: float = 1.2                            # GPS 저주파 편이
    drift_period_s: float = 6.0
    jitter_std_deg: float = 0.6                           # 프레임별 지터(1σ)
    lag_frames: int = 3                                   # 명령 필터 지연
    f_px: float = 3548.0                                  # IMX708 f[px] (가정)
    seed: int = 0

    def sigma_total_deg(self) -> float:
        """대략적 종합 지향 오차 1σ[deg] (독립 성분 RSS)."""
        bx, by = self.static_bias_deg
        static_mag = math.hypot(bx, by)
        return math.sqrt(static_mag ** 2 + self.drift_amp_deg ** 2 + self.jitter_std_deg ** 2)

    def offset_px(self, frame_id: int, fps: float, rng) -> tuple[float, float]:
        """frame_id에서 짐벌이 남긴 표적 화면 오프셋(dx, dy)[px].

        표적은 (프레임 중앙 + 이 오프셋)에 나타난다 = 짐벌이 완벽히 중앙에
        맞추지 못한 잔차. 추적기는 이 오프셋 하에서도 표적을 잡아야 한다.
        """
        t = frame_id / max(1e-6, fps)
        # 정렬 잔차(상수) + GPS 저주파 편이(사인) — 각도[deg]
        bx, by = self.static_bias_deg
        drift = self.drift_amp_deg * math.sin(2 * math.pi * t / self.drift_period_s)
        drift_y = 0.6 * self.drift_amp_deg * math.cos(2 * math.pi * t / self.drift_period_s)
        # 프레임별 지터
        jx = rng.gauss(0.0, self.jitter_std_deg)
        jy = rng.gauss(0.0, self.jitter_std_deg)
        err_yaw_deg = bx + drift + jx
        err_pitch_deg = by + drift_y + jy
        # 각도 → 픽셀 (소각 근사에서 tan ≈ rad, 하지만 정확히 tan 사용)
        dx = self.f_px * math.tan(math.radians(err_yaw_deg))
        dy = self.f_px * math.tan(math.radians(err_pitch_deg))
        return dx, dy

    def as_dict(self) -> dict:
        d = asdict(self)
        d["sigma_total_deg"] = round(self.sigma_total_deg(), 3)
        # 대표 오프셋 크기(px) — 정렬 잔차만 기준
        bx, by = self.static_bias_deg
        d["static_offset_px"] = round(
            self.f_px * math.tan(math.radians(math.hypot(bx, by))), 1
        )
        return d
