"""DXF 선 -> 트랙 BEV 지도(5mm/px). 칠해진 표시(차선 등)를 채운 마스크 + 거리변환."""
import os, json, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config; SP = sys.argv[1] if len(sys.argv) > 1 else config.WORK; RES = 5.0  # mm/px
polys = json.load(open(SP + "/map/lines.json"))
W, H = int(10000 / RES), int(6000 / RES)
def to_px(pts):  # 도면(1:30, 원점 128.5,122) -> 지도 px (y 위쪽이 + 이므로 뒤집음)
    mm = (np.array(pts) - (128.5, 122.0)) * 30.0
    return np.stack([mm[:, 0] / RES, (6000 - mm[:, 1]) / RES], 1)
edge = np.zeros((H, W), np.uint8)
for p in polys:
    a = to_px(p["pts"])
    if np.ptp(a[:, 0]) > W * 0.95 and np.ptp(a[:, 1]) > H * 0.95: continue  # 10m x 6m 바깥 테두리 제외
    cv2.polylines(edge, [np.round(a * 4).astype(np.int32)], False, 255, 1, cv2.LINE_AA, shift=2)
# 20~40mm 폭 윤곽선 쌍 사이를 채움 (닫힘 연산 9px = 45mm)
paint = cv2.morphologyEx((edge > 40).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
np.save(SP + "/map/paint.npy", paint)
dt = cv2.distanceTransform((1 - paint).astype(np.uint8), cv2.DIST_L2, 5) * RES  # mm
np.save(SP + "/map/dt_mm.npy", dt.astype(np.float32))
vis = cv2.cvtColor(paint * 255, cv2.COLOR_GRAY2BGR)
cv2.imwrite(SP + "/map/paint.png", cv2.resize(vis, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
crop = vis[int(4150 / RES):int(5300 / RES), int(1000 / RES):int(3200 / RES)]  # 왼쪽 위 굽은 길 확대
cv2.imwrite(SP + "/map/paint_crop.png", crop)
print("map", W, H, "painted px", int(paint.sum()), "=", round(paint.sum() * RES * RES / 1e6, 2), "m2")
