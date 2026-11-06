"""Live driving view, one screen: the camera image warped to the ground (BEV) with da, the lane follower's
generated path, Ld circle and target point.

The BEV uses the follower's own homography, sent on /lane_follower/lookahead, so it always matches what the
controller sees (cam_* changes apply immediately). Camera frames and da masks are paired by header.stamp.
http_port > 0 serves the view as an MJPEG stream for a browser (http://<jetson>:<port>), no X display needed.
No torch here: this node runs on the system Python.
"""
import json
import math
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

DA_COLOR = np.array((0, 255, 0), np.float32)
ALPHA = 0.35
WINDOW = "lane_viz"
KEEP = 30           # frames kept waiting for their masks
LOOK_MAX_AGE = 1.0  # s: without a newer lookahead message the follower is gone
SCALE = 5           # display px per BEV grid cell (1 cm)
X_NEAR = 0.40       # m from the rear axle where the view starts: just inside Ld_min so the target always shows
JPEG_QUALITY = 75


def stamp_key(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def to_bgr(msg):
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported encoding: {msg.encoding}")
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img


def to_mask(msg):
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width] > 0


def text_bar(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render(frame, da, look, fps):
    """frame: camera BGR, da: bool mask, look: /lane_follower/lookahead dict with "bev"."""
    g = look["bev"]
    res, s = g["res"], SCALE
    out_w = int(round(2 * g["y_half"] / res * s))
    out_h = int(round((g["x_max"] - X_NEAR) / res * s))
    D = np.diag([s, s, 1.0])                                  # grid cell → display px
    M = D @ np.array(g["M"], np.float64)                      # mask px → display px
    mw, mh = look["mask_wh"]
    ch, cw = frame.shape[:2]
    k = mw / cw                                               # camera → mask: same width, centered crop/pad
    C = np.array([[k, 0, 0], [0, k, -(ch * k - mh) / 2.0], [0, 0, 1]])
    img = cv2.warpPerspective(frame, M @ C, (out_w, out_h), flags=cv2.INTER_LINEAR)
    if da.shape == (mh, mw):
        d = cv2.warpPerspective(da.astype(np.uint8), M, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
        f = img[d].astype(np.float32)
        img[d] = (f * (1 - ALPHA) + DA_COLOR * ALPHA).astype(np.uint8)
    # below the image bottom the camera sees nothing, and behind the camera the homography mirrors: blank it
    p = M @ np.array([mw / 2.0, mh - 1.0, 1.0])
    img[max(int(p[1] / p[2]), 0):] = 0

    px = lambda x, y: (int(round((g["y_half"] - y) / res * s)), int(round((g["x_max"] - x) / res * s)))
    axle = px(0.0, 0.0)
    if look.get("path"):
        cv2.polylines(img, [np.array([px(x, y) for x, y in look["path"]], np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
    if "Ld" in look:
        cv2.circle(img, axle, int(round(look["Ld"] / res * s)), (255, 255, 0), 1, cv2.LINE_AA)
        t = px(*look["target"])
        cv2.line(img, axle, t, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, t, 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, t, 7, (0, 0, 0), 1, cv2.LINE_AA)

    text = f"{look.get('mode', '?')}  v {look.get('v', 0):.2f}  steer {math.degrees(look.get('steer', 0)):+5.1f}"
    if "Ld" in look:
        text += f"  Ld {look['Ld']:.2f}"
    if "conf" in look:
        text += f"  conf {look['conf']:.2f}"
    text_bar(img, text + f"  {fps:4.1f} fps")
    return img


def waiting(text):
    img = np.zeros((300, 700, 3), np.uint8)
    cv2.putText(img, text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img


class Stream:
    """Latest JPEG for the browser. The node encodes only while someone is watching."""

    def __init__(self, port, logger):
        self.cond = threading.Condition()
        self.jpg, self.seq, self.clients = None, 0, 0
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    body = (b"<html><head><title>LIMO BEV</title></head><body style='margin:0;background:#000'>"
                            b"<img src='/stream' style='width:100vw;height:100vh;object-fit:contain;display:block'>"
                            b"</body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with stream.cond:
                    stream.clients += 1
                seen = -1
                try:
                    while True:
                        with stream.cond:
                            stream.cond.wait_for(lambda: stream.seq != seen, timeout=5.0)
                            jpg, seen = stream.jpg, stream.seq
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with stream.cond:
                        stream.clients -= 1

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        logger.info(f"streaming on http://<this host>:{port}")

    def push(self, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self.cond:
                self.jpg, self.seq = buf.tobytes(), self.seq + 1
                self.cond.notify_all()


class LaneVizNode(Node):
    def __init__(self):
        super().__init__("lane_viz_node")
        for k, v in {"image_topic": "/camera/color/image_raw", "da_topic": "/perception/da_mask",
                     "lookahead_topic": "/lane_follower/lookahead", "show_window": True, "snapshot_path": "",
                     "http_port": 0}.items():
            self.declare_parameter(k, v)
        get = lambda k: self.get_parameter(k).value
        self.show = get("show_window")
        self.snapshot = get("snapshot_path")
        self.frames, self.da = OrderedDict(), OrderedDict()
        self.times = []
        self.drawn = 0
        self.look, self.look_t = None, 0.0
        camera_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, get("image_topic"), lambda m: self.add(self.frames, m), camera_qos)
        self.create_subscription(Image, get("da_topic"), lambda m: self.add(self.da, m), 5)
        self.create_subscription(String, get("lookahead_topic"), self.on_look, 5)
        self.create_timer(5.0, self.report)
        self.last_draw = 0.0
        self.create_timer(1.0, self.idle)
        self.stream = Stream(get("http_port"), self.get_logger()) if get("http_port") > 0 else None
        if self.show:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info("ready: waiting for camera, da mask and /lane_follower/lookahead")

    def watching(self):
        return self.show or (self.stream and self.stream.clients > 0) or (self.snapshot and self.drawn < 30)

    def add(self, store, msg):
        store[stamp_key(msg)] = msg
        while len(store) > KEEP:
            store.popitem(last=False)
        self.draw_ready()

    def on_look(self, msg):
        try:
            self.look, self.look_t = json.loads(msg.data), time.time()
        except ValueError:
            pass

    def draw_ready(self):
        ready = [k for k in self.da if k in self.frames]
        if not ready:
            return
        key = ready[-1]
        frame_msg, da_msg = self.frames[key], self.da[key]
        for store in (self.frames, self.da):
            for k in [k for k in store if k <= key]:
                del store[k]
        if not self.watching():  # nobody looks: skip drawing
            return
        now = time.time()
        self.times = [t for t in self.times if now - t < 2.0] + [now]
        fps = (len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-3)
        look = self.look if now - self.look_t < LOOK_MAX_AGE else None
        if look is None or "bev" not in look:
            img = waiting("waiting for lane_follower (limo up) ...")
        else:
            img = render(to_bgr(frame_msg), to_mask(da_msg), look, fps)
        self.show_img(img)

    def idle(self):
        """No camera/da pair for a while: say so instead of freezing on the last frame."""
        if time.time() - self.last_draw > 1.0 and self.watching():
            self.show_img(waiting("waiting for camera and perception (limo up) ..."), counted=False)

    def show_img(self, img, counted=True):
        if counted:
            self.drawn += 1
            self.last_draw = time.time()
        if counted and self.snapshot and self.drawn == 30:
            cv2.imwrite(self.snapshot, img)
        if self.stream:
            self.stream.push(img)
        if self.show:
            cv2.imshow(WINDOW, img)
            cv2.waitKey(1)

    def report(self):
        self.get_logger().info(f"drawn frames: {self.drawn}")


def main():
    rclpy.init()
    node = LaneVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
