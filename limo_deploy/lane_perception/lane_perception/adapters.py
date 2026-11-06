"""Model adapters. Pre/post-processing follows training_code_v2.zip (ego-lane-v2):

- train_seg.py: EgoDataset.__getitem__, ADAPTERS, crop, IoU
- eval_ensemble.py: prep, load
- make_ego_lane_labels.py (repo): 1280x720 -> 640x360 with INTER_AREA

Input is 640x360 BGR, padded 12px top and bottom with (114,114,114) to 640x384.
Outputs are cropped back to 640x360 and thresholded with >= 0.5 in training evaluation.
"""
import importlib
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch

PAD = 12
IN_W, IN_H = 640, 360
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

HOME = os.path.expanduser("~")
WEIGHT_DIR = os.path.join(HOME, "lane_perception", "release", "ego-lane-v2")
ENGINE_DIR = os.path.join(HOME, "lane_perception", "engines")
THIRD_PARTY = os.path.join(HOME, "lane_perception", "third_party")

# name: (rgb, imagenet_norm, weight file)
SPECS = {
    "tln": (True, False, "tln_large_v2_best_da.pt"),
    "yolop": (False, True, "yolop_v2_best_da.pt"),
    "hybridnets": (True, True, "hybridnets_v2_best_da.pt"),
    "yolopv2": (True, False, "yolopv2_v2_best_da.pt"),
}


def default_weight(model, backend):
    if backend == "trt":
        return os.path.join(ENGINE_DIR, f"{model}_fp16.engine")
    return os.path.join(WEIGHT_DIR, SPECS[model][2])


def ratio_fg(t):
    return t[:, 1] / (t[:, 0] + t[:, 1] + 1e-6)


def import_from_repo(repo, module, attr):
    """Same-named modules (utils, model, lib) collide across repos, so restore sys.modules after import."""
    deps = os.path.join(THIRD_PARTY, "pydeps")
    sys.path[:0] = [repo, deps]
    try:
        obj = getattr(importlib.import_module(module), attr)
    finally:
        sys.path.remove(repo)
        sys.path.remove(deps)
        prefix = os.path.abspath(repo) + os.sep
        for name, mod in list(sys.modules.items()):
            if os.path.abspath(getattr(mod, "__file__", None) or "").startswith(prefix):
                del sys.modules[name]
    return obj


class LaneModelAdapter:
    """Shared pre/post-processing. Subclasses provide load() and infer() -> (da_prob, ll_prob)."""

    def __init__(self, model, da_thresh=0.5, ll_thresh=0.5):
        self.name = model
        self.rgb, self.norm, _ = SPECS[model]
        self.da_thresh, self.ll_thresh = da_thresh, ll_thresh
        self.device = torch.device("cpu")

    def load(self, weight_path, device):
        raise NotImplementedError

    def infer(self, tensor):
        raise NotImplementedError

    def preprocess(self, bgr_img):
        if bgr_img.shape[:2] != (IN_H, IN_W):
            raise ValueError(f"expected {IN_W}x{IN_H} BGR, got {bgr_img.shape}")
        img = cv2.copyMakeBorder(bgr_img, PAD, PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        if self.norm:
            x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(self.device)

    def postprocess(self, raw_outputs):
        da, ll = raw_outputs
        da = da[..., PAD:-PAD, :][0].float().cpu().numpy() >= self.da_thresh
        ll = ll[..., PAD:-PAD, :][0].float().cpu().numpy() >= self.ll_thresh
        return da.astype(np.uint8) * 255, ll.astype(np.uint8) * 255


class TorchAdapter(LaneModelAdapter):
    def build(self, weight_path):
        raise NotImplementedError

    def forward(self, net, x):
        raise NotImplementedError

    def load(self, weight_path, device):
        self.device = torch.device(device)
        self.net = self.build(weight_path).to(self.device).eval()

    @torch.no_grad()
    def infer(self, tensor):
        return self.forward(self.net, tensor)


class TlnAdapter(TorchAdapter):
    def build(self, weight_path):
        cls = import_from_repo(os.path.join(THIRD_PARTY, "TwinLiteNetPlus"), "model.model", "TwinLiteNetPlus")
        net = cls(Namespace(config="large"))
        net.load_state_dict(torch.load(weight_path, map_location="cpu"))
        return net

    def forward(self, net, x):
        da, ll = net(x)
        return torch.softmax(da, 1)[:, 1], torch.softmax(ll, 1)[:, 1]


class YolopAdapter(TorchAdapter):
    def build(self, weight_path):
        get_net = import_from_repo(os.path.join(THIRD_PARTY, "YOLOP"), "lib.models", "get_net")
        net = get_net(False)
        net.load_state_dict(torch.load(weight_path, map_location="cpu"))
        return net

    def forward(self, net, x):
        _, da, ll = net(x)
        return ratio_fg(da), ratio_fg(ll)


class HybridNetsAdapter(TorchAdapter):
    def build(self, weight_path):
        cls = import_from_repo(os.path.join(THIRD_PARTY, "HybridNets"), "backbone", "HybridNetsBackbone")
        net = cls(compound_coef=3, num_classes=1, ratios=[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)],
                  scales=[2**0, 2**0.70, 2**1.32], seg_classes=2, backbone_name=None, seg_mode="multiclass")
        net.load_state_dict(torch.load(weight_path, map_location="cpu"))
        return net

    def forward(self, net, x):
        p = torch.softmax(net(x)[-1], 1)
        return p[:, 1] + p[:, 2], p[:, 2]


class Yolopv2Adapter(TorchAdapter):
    def build(self, weight_path):
        # train_seg.load_yolopv2_trainable, without the ZLUDA and requires_grad parts inference does not need
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        return torch.jit.load(weight_path, map_location="cpu")

    def forward(self, net, x):
        _, seg, ll = net(x)
        return ratio_fg(seg), ll[:, 0]


class TrtAdapter(LaneModelAdapter):
    """Runs an FP16 engine whose outputs are da_prob and ll_prob at 640x384 (see export_trt.py)."""

    def load(self, weight_path, device="cuda"):
        import tensorrt as trt

        self.device = torch.device("cuda")
        torch.zeros(1, device=self.device)  # initialize the CUDA context before TensorRT uses it
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(weight_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to load engine: {weight_path}")
        self.ctx = self.engine.create_execution_context()
        dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = dtypes[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name, self.input_dtype = name, dtype
            else:
                shape = tuple(self.engine.get_tensor_shape(name))
                self.outputs[name] = torch.empty(shape, dtype=dtype, device=self.device)
                self.ctx.set_tensor_address(name, int(self.outputs[name].data_ptr()))

    def infer(self, tensor):
        x = tensor.to(self.input_dtype)
        self.ctx.set_tensor_address(self.input_name, int(x.data_ptr()))
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return self.outputs["da_prob"], self.outputs["ll_prob"]


TORCH_ADAPTERS = {"tln": TlnAdapter, "yolop": YolopAdapter, "hybridnets": HybridNetsAdapter, "yolopv2": Yolopv2Adapter}


def make_adapter(model, backend, weight="", device="cuda", da_thresh=0.5, ll_thresh=0.5):
    if model not in SPECS:
        raise ValueError(f"unknown model: {model}")
    if backend == "trt":
        adapter = TrtAdapter(model, da_thresh, ll_thresh)
    elif backend == "pt":
        adapter = TORCH_ADAPTERS[model](model, da_thresh, ll_thresh)
    else:
        raise ValueError(f"unknown backend: {backend}")
    adapter.load(weight or default_weight(model, backend), device)
    return adapter
