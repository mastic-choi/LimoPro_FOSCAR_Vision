"""LIMO 카메라 모델 (limo.yaml 내부 보정, 640x360으로 축소). 차체 좌표: 원점 뒷바퀴 축 아래 바닥, x 전방, y 왼쪽, z 위."""
import numpy as np, cv2
K1280 = np.array([[915.5676604376551, 0, 640.740180963267], [0, 915.886717746674, 304.9204190392459], [0, 0, 1]])
D = np.array([0.12412820056754428, -0.20032223539401411, -0.006012395226057269, -0.0029489188463282003, -0.005925371395923599])
K = K1280.copy(); K[:2] *= 0.5  # 640x360
B = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # 카메라(x오른쪽,y아래,z앞) -> 차체
def R_cam2car(pitch_deg, yaw_deg=0.0, roll_deg=0.0):
    p, y, r = np.radians([pitch_deg, yaw_deg, roll_deg])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return Rz @ Ry @ Rx @ B
def ipm(uv, pitch, h=0.19, cam_x=0.244, yaw=0.0, roll=0.0):
    """이미지 픽셀(640x360) -> 바닥 점 (차체 좌표 m). 지평선 위 픽셀은 nan."""
    n = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    d = np.c_[n, np.ones(len(n))] @ R_cam2car(pitch, yaw, roll).T
    t = np.where(d[:, 2] < -1e-6, h / -np.minimum(d[:, 2], -1e-6), np.nan)
    return np.c_[cam_x + t * d[:, 0], t * d[:, 1]]
def project(P, pitch, h=0.19, cam_x=0.244, yaw=0.0, roll=0.0):
    """바닥 점 (차체 좌표 m, z=0) -> 이미지 픽셀(640x360). 카메라 뒤쪽은 nan."""
    R = R_cam2car(pitch, yaw, roll); C = np.array([cam_x, 0, h])
    Pc = (np.c_[P, np.zeros(len(P))] - C) @ R  # = R^T (P - C)
    ok = Pc[:, 2] > 0.05
    uv = np.full((len(P), 2), np.nan)
    if ok.any():
        uv[ok] = cv2.projectPoints(Pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    return uv
