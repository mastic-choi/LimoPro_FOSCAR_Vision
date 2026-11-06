"""트랙 도면 DXF -> 선 목록(mm). 레이어/종류/색 요약과 확인용 그림."""
import sys, ezdxf, numpy as np, json
from ezdxf import path as ezpath
doc = ezdxf.readfile(sys.argv[1]); msp = doc.modelspace()
print("units", doc.header.get("$INSUNITS"), "extmin", doc.header.get("$EXTMIN"), "extmax", doc.header.get("$EXTMAX"))
from collections import Counter
c = Counter((e.dxftype(), e.dxf.layer, e.dxf.get("color", 256)) for e in msp)
for k, v in c.most_common(30): print(v, k)
polys = []
for e in msp:
    if e.dxf.layer not in ("Visible", "0"): continue
    t = e.dxftype()
    try:
        if t in ("LINE", "ARC", "SPLINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ELLIPSE"):
            p = ezpath.make_path(e); pts = np.array([(v.x, v.y) for v in p.flattening(2.0)])
            if len(pts) >= 2: polys.append({"t": t, "layer": e.dxf.layer, "color": e.dxf.get("color", 256), "pts": pts.round(1).tolist()})
        elif t == "SOLID":
            pts = np.array([(e.dxf.vtx0.x, e.dxf.vtx0.y), (e.dxf.vtx1.x, e.dxf.vtx1.y), (e.dxf.vtx3.x, e.dxf.vtx3.y), (e.dxf.vtx2.x, e.dxf.vtx2.y)])
            polys.append({"t": t, "layer": e.dxf.layer, "color": e.dxf.get("color", 256), "pts": pts.round(1).tolist()})
    except Exception as ex:
        print("skip", t, ex)
allp = np.concatenate([np.array(p["pts"]) for p in polys])
print("n", len(polys), "bbox", allp.min(0), allp.max(0))
# INSERT(블록 참조)도 있는지
print("INSERT", sum(1 for e in msp if e.dxftype() == "INSERT"))
json.dump(polys, open(sys.argv[2], "w"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 10))
for p in polys:
    a = np.array(p["pts"]); ax.plot(a[:, 0], a[:, 1], lw=0.6, color="C1" if p["layer"] == "Visible" else "C0")
ax.set_aspect("equal"); fig.savefig(sys.argv[3], dpi=110, bbox_inches="tight")
