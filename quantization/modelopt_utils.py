"""ModelOpt and ONNX utilities shared by QAT/PTQ/TensorRT scripts."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

from models import DepthAwareStyleTransferNet, StyleTransferNet


class StyleTransferInferenceNet(nn.Module):
    """Forward-only two-input style-transfer module for export/deployment."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, adain: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adain = adain
        self.alpha = float(alpha)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        blended = []
        for cf, sf in zip(content_features, style_features):
            transformed = self.adain(cf, sf)
            blended.append(self.alpha * transformed + (1.0 - self.alpha) * cf)
        return self.decoder(blended)


class DepthAwareStyleTransferInferenceNet(nn.Module):
    """Exportable depth-aware wrapper with explicit depth-map inputs.

    This wrapper intentionally does not call Depth Anything. Depth is computed
    outside the TensorRT engine and passed in as Bx1xHxW tensors.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, adain: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adain = adain
        self.alpha = float(alpha)

    def forward(
        self,
        content: torch.Tensor,
        content_depth: torch.Tensor,
        style: torch.Tensor,
        style_depth: torch.Tensor,
    ) -> torch.Tensor:
        content_features = self.encoder(content, content_depth)
        style_features = self.encoder(style, style_depth)
        blended = []
        for cf, sf in zip(content_features, style_features):
            transformed = self.adain(cf, sf)
            blended.append(self.alpha * transformed + (1.0 - self.alpha) * cf)
        return self.decoder(blended)


def build_inference_model(
    model: nn.Module,
    alpha: float = 1.0,
    *,
    explicit_depth: bool = False,
) -> nn.Module:
    """Create the two-input inference wrapper used by ONNX and TensorRT."""
    if explicit_depth:
        return DepthAwareStyleTransferInferenceNet(
            model.encoder,
            model.decoder,
            model.adain,
            alpha=alpha,
        )
    return StyleTransferInferenceNet(model.encoder, model.decoder, model.adain, alpha=alpha)


def load_style_transfer_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    use_depth: bool = False,
    depth_scale: float = 2.0,
    pretrained_encoder: bool = False,
) -> nn.Module:
    """Load a training checkpoint into the repo's PyTorch model."""
    checkpoint_path = Path(checkpoint_path)
    if use_depth:
        model = DepthAwareStyleTransferNet(
            depth_scale=depth_scale,
            pretrained_encoder=pretrained_encoder,
        ).to(device)
        model.load_rgb_checkpoint(str(checkpoint_path), device)
    else:
        model = StyleTransferNet(pretrained_encoder=pretrained_encoder).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _get_modelopt_quant_config(mode: str) -> dict[str, Any]:
    import modelopt.torch.quantization as mtq

    mode = mode.lower()
    if mode == "int8":
        return copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    if mode == "int4":
        return copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    raise ValueError(f"Unsupported ModelOpt torch quantization mode: {mode}")


def prepare_modelopt_qat(
    model: nn.Module,
    forward_loop: Callable[[nn.Module], None],
    *,
    mode: str = "int8",
    algorithm: str = "max",
    disable_encoder: bool = False,
    disable_decoder: bool = False,
    disable_final_layer: bool = False,
    extra_disable_patterns: Iterable[str] | None = None,
    print_summary: bool = False,
) -> nn.Module:
    """Insert ModelOpt fake quantizers and calibrate them for QAT/PTQ-style fine-tuning."""
    import modelopt.torch.quantization as mtq

    config = _get_modelopt_quant_config(mode)
    if algorithm:
        config["algorithm"] = algorithm

    quantized = mtq.quantize(model, config, forward_loop)

    disable_patterns = []
    if disable_encoder:
        disable_patterns.append("encoder.*")
    if disable_decoder:
        disable_patterns.append("decoder.*")
    if disable_final_layer:
        disable_patterns.extend(["decoder.final.3.*", "decoder.final.3"])
    if extra_disable_patterns:
        disable_patterns.extend(extra_disable_patterns)

    for pattern in disable_patterns:
        mtq.disable_quantizer(quantized, pattern)

    if print_summary:
        mtq.print_quant_summary(quantized)
    return quantized


@contextmanager
def constant_modelopt_amax_export():
    """Materialize ModelOpt quantizer amax buffers as ONNX constants.

    ModelOpt's legacy ONNX symbolics require quantizer scales to arrive as
    constants. PyTorch otherwise traces `_amax` buffers as graph parameters,
    which prevents direct QAT Q/DQ export.
    """
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    original_get_amax = TensorQuantizer._get_amax

    def constant_get_amax(self, inputs):
        if hasattr(self, "_amax"):
            return torch.tensor(
                self._amax.detach().cpu().numpy(),
                dtype=self._amax.dtype,
                device=inputs.device,
            )
        return original_get_amax(self, inputs)

    TensorQuantizer._get_amax = constant_get_amax
    try:
        yield
    finally:
        TensorQuantizer._get_amax = original_get_amax


def load_modelopt_qat_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    use_depth: bool = False,
    depth_scale: float = 2.0,
    image_size: int = 256,
) -> nn.Module:
    """Load a checkpoint with ModelOpt fake quantizers preserved."""
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    qat_cfg = ckpt.get("qat")
    if not isinstance(qat_cfg, dict) or not qat_cfg.get("enabled"):
        raise ValueError(f"Checkpoint does not contain enabled QAT metadata: {checkpoint_path}")

    if use_depth:
        model = DepthAwareStyleTransferNet(
            depth_scale=depth_scale,
            pretrained_encoder=False,
        ).to(device)
        dummy_content = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_style = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_content_depth = torch.randn(1, 1, image_size, image_size, device=device)
        dummy_style_depth = torch.randn(1, 1, image_size, image_size, device=device)

        def forward_loop(module: nn.Module) -> None:
            content_features = module.encoder(dummy_content, dummy_content_depth)
            style_features = module.encoder(dummy_style, dummy_style_depth)
            module.decoder(module.transform_features(content_features, style_features))

    else:
        model = StyleTransferNet(pretrained_encoder=False).to(device)
        dummy_content = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_style = torch.randn(1, 3, image_size, image_size, device=device)

        def forward_loop(module: nn.Module) -> None:
            module(dummy_content, dummy_style)

    quantized = prepare_modelopt_qat(
        model,
        forward_loop,
        mode=qat_cfg.get("mode", "int8"),
        algorithm=qat_cfg.get("algorithm", "max"),
        disable_encoder=bool(qat_cfg.get("disable_encoder", False)),
        disable_decoder=bool(qat_cfg.get("disable_decoder", False)),
        disable_final_layer=bool(qat_cfg.get("disable_final_layer", False)),
        extra_disable_patterns=[
            part.strip()
            for part in str(qat_cfg.get("extra_disable_patterns", "")).split(",")
            if part.strip()
        ],
    )
    quantized.load_state_dict(ckpt["model_state_dict"], strict=True)
    quantized.eval()
    return quantized


def export_inference_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    image_size: int = 256,
    batch_size: int = 1,
    opset: int = 18,
    dynamic_axes: bool = True,
    legacy: bool = False,
    external_data: bool = True,
    quantized_export: bool = False,
    explicit_depth: bool = False,
    device: torch.device | None = None,
) -> None:
    """Export a two-input style-transfer model to ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device

    dummy_content = torch.randn(batch_size, 3, image_size, image_size, device=device)
    dummy_style = torch.randn(batch_size, 3, image_size, image_size, device=device)
    inputs: tuple[torch.Tensor, ...]
    input_names: list[str]
    if explicit_depth:
        dummy_content_depth = torch.randn(batch_size, 1, image_size, image_size, device=device)
        dummy_style_depth = torch.randn(batch_size, 1, image_size, image_size, device=device)
        inputs = (dummy_content, dummy_content_depth, dummy_style, dummy_style_depth)
        input_names = ["content", "content_depth", "style", "style_depth"]
    else:
        inputs = (dummy_content, dummy_style)
        input_names = ["content", "style"]

    axes = None
    if dynamic_axes:
        axes = {
            "content": {0: "batch", 2: "height", 3: "width"},
            "style": {0: "batch", 2: "height", 3: "width"},
            "stylized": {0: "batch", 2: "height", 3: "width"},
        }
        if explicit_depth:
            axes["content_depth"] = {0: "batch", 2: "height", 3: "width"}
            axes["style_depth"] = {0: "batch", 2: "height", 3: "width"}

    model.eval()

    export_context = nullcontext()
    if quantized_export:
        from modelopt.torch.quantization.utils import export_torch_mode

        export_context = export_torch_mode()

    constant_amax_context = constant_modelopt_amax_export() if quantized_export else nullcontext()

    with export_context, constant_amax_context:
        torch.onnx.export(
            model,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=["stylized"],
            dynamic_axes=axes,
            opset_version=opset,
            dynamo=not legacy,
            external_data=external_data,
        )

    # This validates the small model protobuf. External data is loaded lazily by ONNX.
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    onnx.checker.check_model(onnx_model)


def count_onnx_qdq_nodes(path: str | Path) -> dict[str, int]:
    """Return a small op histogram focused on Q/DQ validation."""
    model = onnx.load(str(path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {
        "QuantizeLinear": counts.get("QuantizeLinear", 0),
        "DequantizeLinear": counts.get("DequantizeLinear", 0),
        "total_nodes": sum(counts.values()),
    }
