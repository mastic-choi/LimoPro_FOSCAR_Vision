import csv
import time
from array import array

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image

from .adapters import IN_H, IN_W, make_adapter

CHANNELS = {"rgb8": cv2.COLOR_RGB2BGR, "bgr8": None}
DA_COLOR = np.array((0, 255, 0), np.float32)
LL_COLOR = np.array((0, 0, 255), np.float32)
ALPHA = 0.45


def image_to_bgr(msg):
    """cv_bridge is not used: its boost module fails in dl_env (libgdal/libtiff conflict)."""
    if msg.encoding not in CHANNELS:
        raise ValueError(f"unsupported encoding: {msg.encoding}")
    img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
    img = img.reshape(msg.height, msg.width, 3)
    code = CHANNELS[msg.encoding]
    return cv2.cvtColor(img, code) if code is not None else img


def to_msg(img, encoding, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.step = img.shape[1] * (img.shape[2] if img.ndim == 3 else 1)
    # array('B') skips the per-element check rclpy runs for bytes, which took about 280ms per frame
    msg.data = array("B", img.tobytes())
    return msg


def draw_overlay(bgr, da, ll):
    out = bgr.astype(np.float32)
    d, l = da > 0, ll > 0
    out[d] = out[d] * (1 - ALPHA) + DA_COLOR * ALPHA
    out[l] = out[l] * (1 - ALPHA) + LL_COLOR * ALPHA
    return out.astype(np.uint8)


class LanePerceptionNode(Node):
    def __init__(self):
        super().__init__("lane_perception_node")
        p = {
            "model": "yolopv2", "backend": "trt", "weight": "", "device": "cuda",
            "input_topic": "/camera/color/image_raw", "da_thresh": 0.5, "ll_thresh": 0.5,
            "publish_overlay": False, "allow_aspect_mismatch": False, "timing_csv": "",
        }
        for k, v in p.items():
            self.declare_parameter(k, v)
        get = lambda k: self.get_parameter(k).value
        self.adapter = make_adapter(get("model"), get("backend"), get("weight"), get("device"), get("da_thresh"), get("ll_thresh"))
        self.overlay_on = get("publish_overlay")
        self.allow_mismatch = get("allow_aspect_mismatch")
        self.timing = None
        if get("timing_csv"):
            self.timing_file = open(get("timing_csv"), "w", newline="")
            self.timing = csv.writer(self.timing_file)
            self.timing.writerow(["wall", "transport", "decode", "resize", "preprocess", "infer", "postprocess", "overlay", "publish", "total"])
        out = QoSProfile(depth=1)
        self.pub_da = self.create_publisher(Image, "/perception/da_mask", out)
        self.pub_ll = self.create_publisher(Image, "/perception/ll_mask", out)
        self.pub_overlay = self.create_publisher(Image, "/perception/overlay", out)
        sub_qos = QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability, history=qos_profile_sensor_data.history)
        self.sub = self.create_subscription(Image, get("input_topic"), self.on_image, sub_qos)
        self.get_logger().info(f"ready: model={get('model')} backend={get('backend')} topic={get('input_topic')}")

    def sync(self):
        """CUDA work is asynchronous: without this, inference time would appear in a later stage."""
        if self.timing is not None and self.adapter.device.type == "cuda":
            torch.cuda.synchronize()

    def on_image(self, msg):
        t0 = time.perf_counter()
        wall = time.time()
        transport = wall - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        try:
            bgr = image_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        t1 = time.perf_counter()
        h, w = bgr.shape[:2]
        if abs(w / h - IN_W / IN_H) > 0.01:
            if not self.allow_mismatch:
                self.get_logger().error(
                    f"input {w}x{h} is not 16:9, training input is. Skipping (allow_aspect_mismatch=false).",
                    throttle_duration_sec=5.0)
                return
            self.get_logger().warn(f"input {w}x{h} squashed to {IN_W}x{IN_H}: accuracy is not valid", throttle_duration_sec=30.0)
        if (w, h) != (IN_W, IN_H):
            bgr = cv2.resize(bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        t2 = time.perf_counter()
        x = self.adapter.preprocess(bgr)
        self.sync()
        t3 = time.perf_counter()
        raw = self.adapter.infer(x)
        self.sync()
        t4 = time.perf_counter()
        da, ll = self.adapter.postprocess(raw)
        t5 = time.perf_counter()
        overlay = draw_overlay(bgr, da, ll) if self.overlay_on else None
        t6 = time.perf_counter()
        self.pub_da.publish(to_msg(da, "mono8", msg.header))
        self.pub_ll.publish(to_msg(ll, "mono8", msg.header))
        if overlay is not None:
            self.pub_overlay.publish(to_msg(overlay, "bgr8", msg.header))
        t7 = time.perf_counter()
        if self.timing is not None:
            ms = [(b - a) * 1000 for a, b in ((t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5), (t5, t6), (t6, t7), (t0, t7))]
            self.timing.writerow([f"{wall:.4f}", f"{transport * 1000:.2f}"] + [f"{v:.2f}" for v in ms])
            self.timing_file.flush()


def main():
    rclpy.init()
    node = LanePerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
