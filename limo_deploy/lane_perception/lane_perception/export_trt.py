"""ONNX -> TensorRT FP16. Engines output da_prob and ll_prob at 640x384; cropping stays in the adapter.

  python -m lane_perception.export_trt --model yolopv2
"""
import argparse
import os

import torch

from .adapters import ENGINE_DIR, TORCH_ADAPTERS, default_weight


class Probs(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.net = adapter.net
        self.adapter = adapter

    def forward(self, x):
        return self.adapter.forward(self.net, x)


class Yolopv2Probs(torch.nn.Module):
    """A TorchScript net cannot be traced from a Python wrapper, so the wrapper is scripted too. Same as Yolopv2Adapter.forward."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        seg, ll = out[1], out[2]
        return seg[:, 1] / (seg[:, 0] + seg[:, 1] + 1e-6), ll[:, 0]


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def replace_memory_efficient_swish(net):
    """MemoryEfficientSwish uses a custom autograd Function ONNX cannot trace. The output is the same x * sigmoid(x)."""
    n = 0
    for parent in net.modules():
        for name, child in parent.named_children():
            if type(child).__name__ == "MemoryEfficientSwish":
                setattr(parent, name, Swish())
                n += 1
    print("swish replaced:", n, flush=True)


def export_onnx(model, path):
    adapter = TORCH_ADAPTERS[model](model)
    adapter.load(default_weight(model, "pt"), "cpu")
    if model == "hybridnets":
        replace_memory_efficient_swish(adapter.net)
    wrapped = torch.jit.script(Yolopv2Probs(adapter.net)) if model == "yolopv2" else Probs(adapter)
    torch.onnx.export(wrapped.eval(), torch.zeros(1, 3, 384, 640), path, opset_version=17, dynamo=False,
                      input_names=["images"], output_names=["da_prob", "ll_prob"], do_constant_folding=True)


def build_engine(onnx_path, engine_path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("ONNX parse failed: " + "; ".join(parser.get_error(i).desc() for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    with open(engine_path, "wb") as f:
        f.write(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(TORCH_ADAPTERS))
    a = ap.parse_args()
    os.makedirs(ENGINE_DIR, exist_ok=True)
    onnx_path = os.path.join(ENGINE_DIR, f"{a.model}.onnx")
    engine_path = os.path.join(ENGINE_DIR, f"{a.model}_fp16.engine")
    export_onnx(a.model, onnx_path)
    print("onnx:", onnx_path, os.path.getsize(onnx_path), flush=True)
    build_engine(onnx_path, engine_path)
    print("engine:", engine_path, os.path.getsize(engine_path), flush=True)


if __name__ == "__main__":
    main()
