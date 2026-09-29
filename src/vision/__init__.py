"""HANARO Payload — 저장영상 낙하산 분석 (Spec A) + 합성 생성기 (Spec B).

공통 규칙: .kiro/steering/개발-공통-규칙.md
이 패키지는 픽셀 단위 중심·크기만 출력하며, 절대 거리/m/s는 출력하지 않는다.
"""

__version__ = "0.1.0"


def code_version() -> str:
    """실행 메타데이터에 기록할 코드 버전. git commit이 있으면 덧붙인다."""
    import subprocess

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{__version__}+{sha}"
    except Exception:
        return __version__
