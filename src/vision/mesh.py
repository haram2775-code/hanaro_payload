"""3D mesh 기반 캐노피 렌더러 (Spec B 확장, OpenCV-only 백엔드) — M1~M4.

목적: 캐노피를 파라메트릭 3D 곡면 메시로 만들고, 핀홀 카메라로 투영해 임의
관측각·자세·팽창·자전·진자 기울임에서 물리적으로 정확한 실루엣을 렌더한다.
추가 GPU/GL 의존성 없이 numpy + OpenCV(fillConvexPoly + painter z-정렬)만 쓴다.

설계 근거: configs/mesh_model.md (곡면 프로파일·카메라 f 등 가정값 문서).
- Main(toroidal): apex를 안으로 당긴 pull-down 프로파일 → 중앙 spill hole + 뒤 내벽.
- Drogue(round): 얕은 반구 프로파일, 구멍 없음.

주의(공통 규칙 §1·§2·§3):
- 카메라 초점거리/주점, 곡면 프로파일 계수는 실측이 아니라 **가정/설정값**이다.
- 3D 충실도가 실제 검출 성공을 보장하지 않는다. 절대 거리/속도는 다루지 않는다.
- painter z-정렬은 볼록 근사에 유효하다. 심하게 접힌(자기교차) 캐노피는 범위 밖.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 카메라 (핀홀) — 합성용 가정값.
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    """핀홀 카메라. f/주점은 합성용 가정치(configs/mesh_model.md)."""
    f: float = 700.0                 # 초점거리(px) — 가정값
    cx: float = 320.0                # 주점 x
    cy: float = 240.0                # 주점 y
    # 카메라 자세(월드→카메라). 낙하산은 카메라 앞 +Z 거리 dist에 놓는다.
    dist: float = 6.0                # 카메라~표적 거리(월드 단위, 상대값)

    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx],
                         [0, self.f, self.cy],
                         [0, 0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# 메시 생성
# ---------------------------------------------------------------------------
@dataclass
class Mesh:
    verts: np.ndarray                 # (N,3) 월드 정점
    faces: np.ndarray                 # (M,3) 삼각형 정점 인덱스
    face_color: np.ndarray            # (M,3) BGR 면 색
    shroud_lines: list = field(default_factory=list)  # [(p0(3,), p1(3,)), ...]


def _profile_z(u: np.ndarray, kind: str, inflation: float,
               spill_hole_ratio: float) -> np.ndarray:
    """반경 방향 정규화 u∈[0..1]에서 캐노피 표면 높이 z(위로 볼록).

    inflation(0.05..1)으로 곡면 진폭을 보간한다. toroidal은 중앙을 안으로 당겨(음수)
    pull-down apex를 만든다(→ 위/비스듬히 보면 구멍).
    """
    infl = float(np.clip(inflation, 0.05, 1.0))
    if kind == "toroidal":
        # 바깥(u=1)은 스커트(z≈0), 중간이 볼록, 중앙 근처(u→hole)는 안으로 당김.
        dome = np.sin(np.clip(u, 0, 1) * math.pi) * (0.55 * infl)   # 링 볼륨
        pull = -(1.0 - np.clip(u / max(1e-3, 1.0), 0, 1)) ** 2 * (0.5 * infl)  # 중앙 당김
        # 중앙 구멍 안쪽은 표면이 없음(별도 마스크로 face 생성에서 제외)
        return dome + pull
    # round/hemispherical: 얕은 반구
    return np.sqrt(np.clip(1.0 - u * u, 0, 1)) * (0.6 * infl)


def build_canopy_mesh(kind: str, radius: float, n_panels: int,
                      color, alt_color, *, inflation: float = 1.0,
                      spill_hole_ratio: float = 0.0,
                      n_radial: int = 6, n_circum: int | None = None,
                      n_shroud: int = 12, shroud_len_ratio: float = 1.3) -> Mesh:
    """파라메트릭 캐노피 메시(월드 좌표, 반경 radius) + 현삭 3D 선분.

    좌표계: 캐노피 중심이 원점, 표면은 +Z(위)로 볼록, 스커트가 아래(z≈0).
    관측 편의상 '위'를 -Y로 두어 카메라(-Z에서 바라봄)에 자연스럽게 투영되게 한다.
    """
    n_circum = n_circum or max(n_panels * 3, 24)
    u_lo = spill_hole_ratio if kind == "toroidal" else 0.0
    us = np.linspace(u_lo, 1.0, n_radial + 1)
    vs = np.linspace(0.0, 2.0 * math.pi, n_circum, endpoint=False)

    col = np.array(color, dtype=np.float64)
    acol = np.array(alt_color, dtype=np.float64)

    verts = []
    vindex = {}
    for ri, u in enumerate(us):
        z = float(_profile_z(np.array([u]), kind, inflation, spill_hole_ratio)[0])
        r = u * radius
        for ci, v in enumerate(vs):
            x = r * math.cos(v)
            y = r * math.sin(v)
            # 표면을 카메라 쪽으로: 위(+Z) → 화면 위(-Y) 로 회전은 project에서 처리.
            vindex[(ri, ci)] = len(verts)
            verts.append((x, y, z))
    verts = np.array(verts, dtype=np.float64)

    faces = []
    fcols = []
    for ri in range(n_radial):
        for ci in range(n_circum):
            c0, c1 = ci, (ci + 1) % n_circum
            a = vindex[(ri, c0)]
            b = vindex[(ri, c1)]
            c = vindex[(ri + 1, c0)]
            d = vindex[(ri + 1, c1)]
            # gore 교대색: 원주를 n_panels로 나눠 결정
            gore = int((ci / n_circum) * n_panels) % 2
            fc = col if gore == 0 else acol
            faces.append((a, b, d)); fcols.append(fc)
            faces.append((a, d, c)); fcols.append(fc)
    faces = np.array(faces, dtype=np.int32)
    face_color = np.array(fcols, dtype=np.float64)

    # 현삭: 스커트(가장 바깥 링) 정점 일부 → confluence(중앙 아래).
    conf = np.array([0.0, 0.0, -radius * shroud_len_ratio], dtype=np.float64)
    shroud = []
    outer = n_radial
    step = max(1, n_circum // max(1, n_shroud))
    for ci in range(0, n_circum, step):
        p = verts[vindex[(outer, ci)]]
        shroud.append((p.copy(), conf.copy()))

    return Mesh(verts=verts, faces=faces, face_color=face_color, shroud_lines=shroud)


# ---------------------------------------------------------------------------
# 자세(회전) — 관측각 / 진자 기울임 / 자전
# ---------------------------------------------------------------------------
def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def pose_matrix(view_elev: float, roll: float = 0.0,
                tilt_x: float = 0.0, tilt_y: float = 0.0) -> np.ndarray:
    """캐노피 자세 회전행렬.

    view_elev: 관측 고도각(rad). 0=옆에서(스커트가 정면), pi/2=바로 아래에서(정면 링).
    roll: 자전(rad, 광축 둘레). tilt_x/tilt_y: 진자 흔들림 기울임(rad).
    """
    # 관측각: 지상에서 낙하산을 올려다보는 기하. view_elev가 클수록 캐노피 면을
    # 정면(카메라 쪽)으로 눕혀 채워진 원반으로 보이게 한다. X축으로 -(pi/2 - elev).
    R = _rot_x(-(math.pi / 2.0 - view_elev))
    R = R @ _rot_x(tilt_x) @ _rot_y(tilt_y)
    R = R @ _rot_z(roll)
    return R


def project(verts: np.ndarray, R: np.ndarray, center_px, scale: float,
            cam: Camera) -> np.ndarray:
    """월드 정점 → 화면 픽셀. 캐노피는 정규 반경(≈1)으로 만들고 scale(px)로 크기 맞춤.

    핀홀 원근을 쓰되, 기준 깊이(dist)에서 반경 1.0이 정확히 scale px가 되도록
    정규화한다(원근감은 유지, 전체 크기는 scale이 결정). 화면 y는 아래로 증가.
    """
    p = (R @ verts.T).T                    # (N,3) 회전
    z = p[:, 2] + cam.dist                 # 카메라 앞 거리
    z = np.where(np.abs(z) < 1e-3, 1e-3, z)
    persp = cam.dist / z                   # 기준 깊이 대비 원근 배율(중앙면=1)
    sx = center_px[0] + p[:, 0] * persp * scale
    sy = center_px[1] - p[:, 1] * persp * scale
    return np.stack([sx, sy, z], axis=1)   # z는 정렬용으로 보존


def render_mesh(img: np.ndarray, mesh: Mesh, R: np.ndarray, center_px,
                scale: float, cam: Camera, *,
                light_dir=(0.4, -0.6, 0.7), ambient: float = 0.55,
                shroud_color=(200, 200, 200)) -> None:
    """메시를 painter z-정렬로 래스터화(in place). Lambert 법선 음영 포함(M4 조명 토대)."""
    proj = project(mesh.verts, R, center_px, scale, cam)
    scr = proj[:, :2]
    zc = proj[:, 2]

    # 면 깊이(정렬)와 법선(음영·후면 컬링)
    tri = mesh.faces
    p0 = mesh.verts[tri[:, 0]]; p1 = mesh.verts[tri[:, 1]]; p2 = mesh.verts[tri[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    n = (R @ n.T).T
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(norm < 1e-9, 1e-9, norm)
    L = np.array(light_dir, dtype=np.float64)
    L = L / (np.linalg.norm(L) + 1e-9)
    shade = ambient + (1.0 - ambient) * np.clip(n @ L, 0, 1)   # (M,)

    face_z = zc[tri].mean(axis=1)
    order = np.argsort(-face_z)   # 먼 면 먼저(뒤→앞)

    for fi in order:
        a, b, c = tri[fi]
        pts = np.array([scr[a], scr[b], scr[c]], dtype=np.int32)
        col = np.clip(mesh.face_color[fi] * shade[fi], 0, 255)
        cv2.fillConvexPoly(img, pts, (int(col[0]), int(col[1]), int(col[2])),
                           lineType=cv2.LINE_AA)

    # 현삭
    for p0w, p1w in mesh.shroud_lines:
        pp = project(np.stack([p0w, p1w]), R, center_px, scale, cam)[:, :2].astype(int)
        cv2.line(img, tuple(pp[0]), tuple(pp[1]), shroud_color, 1, lineType=cv2.LINE_AA)


def mesh_projected_metrics(mesh: Mesh, R: np.ndarray, center_px, scale: float,
                           cam: Camera, img_shape) -> dict:
    """투영 실루엣에서 GT(중심·bbox·면적·등가직경)를 렌더와 동일 기준으로 계산."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    proj = project(mesh.verts, R, center_px, scale, cam)[:, :2]
    for a, b, c in mesh.faces:
        pts = np.array([proj[a], proj[b], proj[c]], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return {"center": center_px, "bbox": (0, 0, 0, 0), "area": 0.0, "diameter": 0.0}
    cx = m["m10"] / m["m00"]; cy = m["m01"] / m["m00"]
    area = float((mask > 0).sum())
    ys, xs = np.where(mask > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    diameter = 2.0 * math.sqrt(area / math.pi)
    return {"center": (cx, cy), "bbox": bbox, "area": area, "diameter": diameter}
