#!/usr/bin/env python3
"""Create explicit Q/DQ ONNX models for TensorRT.

Supported flows:
  1. PTQ from an FP32 checkpoint.
  2. Direct export of a ModelOpt QAT checkpoint with learned fake-quant scales.
  3. A legacy QAT-weight export path that strips fake quantizers and recalibrates
     the plain graph with ONNX PTQ.

For QAT deployment, prefer ``--mode qat_direct_export``. It preserves the
training-time quantizer placement and produces a TensorRT-ready explicit Q/DQ
ONNX graph without running a second PTQ calibration pass.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DepthAwareStyleTransferNet, StyleTransferNet
from quantization.calibration import (
    build_image_loaders,
    collect_depth_onnx_calibration_arrays,
    collect_onnx_calibration_arrays,
)
from quantization.modelopt_utils import (
    build_inference_model,
    count_onnx_qdq_nodes,
    export_inference_onnx,
    load_modelopt_qat_model,
    load_style_transfer_model,
)


def load_qat_weights_as_fp32(
    checkpoint: str | Path,
    device: torch.device,
    *,
    use_depth: bool = False,
    depth_scale: float = 2.0,
) -> torch.nn.Module:
    """Load only architectural weights from a QAT checkpoint into the plain model."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    source = ckpt["model_state_dict"]
    filtered = {
        key: value
        for key, value in source.items()
        if "quantizer" not in key and "calibrator" not in key and "._amax" not in key
    }
    if use_depth:
        model = DepthAwareStyleTransferNet(
            pretrained_encoder=False,
            depth_scale=depth_scale,
        ).to(device)
    else:
        model = StyleTransferNet(pretrained_encoder=False).to(device)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        print(f"Unexpected non-QAT keys ignored: {unexpected}")
    if missing:
        print(f"Missing keys while loading QAT weights into FP32 graph: {missing}")
    model.eval()
    return model


def export_fp32_onnx(args, device: torch.device, output_path: Path) -> None:
    if args.mode == "qat_export":
        model = load_qat_weights_as_fp32(
            args.checkpoint,
            device,
            use_depth=args.use_depth,
            depth_scale=args.depth_scale,
        )
    else:
        model = load_style_transfer_model(
            args.checkpoint,
            device,
            use_depth=args.use_depth,
            depth_scale=args.depth_scale,
            pretrained_encoder=args.pretrained_encoder,
        )

    inference_model = build_inference_model(
        model,
        alpha=args.alpha,
        explicit_depth=args.use_depth,
    ).eval()
    export_inference_onnx(
        inference_model,
        output_path,
        image_size=args.image_size,
        opset=args.opset,
        dynamic_axes=not args.static_onnx_shapes,
        legacy=args.legacy_onnx,
        external_data=args.external_data,
        explicit_depth=args.use_depth,
        device=device,
    )


def export_direct_qat_onnx(args, output_path: Path) -> None:
    """Export QAT fake-quantized PyTorch graph directly to explicit Q/DQ ONNX."""
    device = torch.device("cpu")
    model = load_modelopt_qat_model(
        args.checkpoint,
        device,
        use_depth=args.use_depth,
        depth_scale=args.depth_scale,
        image_size=args.image_size,
    )
    inference_model = build_inference_model(
        model,
        alpha=args.alpha,
        explicit_depth=args.use_depth,
    ).eval()
    export_inference_onnx(
        inference_model,
        output_path,
        image_size=args.image_size,
        opset=args.opset,
        dynamic_axes=not args.static_onnx_shapes,
        legacy=True,
        external_data=args.external_data,
        quantized_export=True,
        explicit_depth=args.use_depth,
        device=device,
    )
    counts = count_onnx_qdq_nodes(output_path)
    print(
        f"Saved direct QAT Q/DQ ONNX: {output_path} "
        f"(Q={counts['QuantizeLinear']}, DQ={counts['DequantizeLinear']}, "
        f"nodes={counts['total_nodes']})"
    )


def quantize_onnx_ptq(args, fp32_onnx: Path, device: torch.device) -> None:
    content_loader, style_loader = build_image_loaders(
        args.content_dir,
        args.style_dir,
        image_size=args.image_size,
        batch_size=args.calib_batch_size,
        max_content_images=args.max_content_images,
        max_style_images=args.max_style_images,
        num_workers=args.num_workers,
        shuffle=False,
    )
    if args.use_depth:
        if args.mode == "qat_export":
            depth_model = load_qat_weights_as_fp32(
                args.checkpoint,
                device,
                use_depth=True,
                depth_scale=args.depth_scale,
            )
        else:
            depth_model = load_style_transfer_model(
                args.checkpoint,
                device,
                use_depth=True,
                depth_scale=args.depth_scale,
                pretrained_encoder=args.pretrained_encoder,
            )
        calibration_data = collect_depth_onnx_calibration_arrays(
            depth_model,
            content_loader,
            style_loader,
            device=device,
            max_batches=args.calib_batches,
        )
    else:
        calibration_data = collect_onnx_calibration_arrays(
            content_loader,
            style_loader,
            max_batches=args.calib_batches,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.use_depth:
        calibration_shapes = (
            f"content:{args.calib_batch_size}x3x{args.image_size}x{args.image_size},"
            f"content_depth:{args.calib_batch_size}x1x{args.image_size}x{args.image_size},"
            f"style:{args.calib_batch_size}x3x{args.image_size}x{args.image_size},"
            f"style_depth:{args.calib_batch_size}x1x{args.image_size}x{args.image_size}"
        )
    else:
        calibration_shapes = (
            f"content:{args.calib_batch_size}x3x{args.image_size}x{args.image_size},"
            f"style:{args.calib_batch_size}x3x{args.image_size}x{args.image_size}"
        )

    if args.ptq_backend in ("auto", "modelopt"):
        try:
            quantize_with_modelopt(args, fp32_onnx, output_path, calibration_data, calibration_shapes)
        except Exception as exc:
            if args.ptq_backend == "modelopt":
                raise
            print(f"ModelOpt ONNX PTQ failed, falling back to ONNX Runtime QDQ PTQ: {exc}")
            quantize_with_ort(args, fp32_onnx, output_path, calibration_data)
    else:
        quantize_with_ort(args, fp32_onnx, output_path, calibration_data)

    counts = count_onnx_qdq_nodes(output_path)
    print(
        f"Saved Q/DQ ONNX: {output_path} "
        f"(Q={counts['QuantizeLinear']}, DQ={counts['DequantizeLinear']}, "
        f"nodes={counts['total_nodes']})"
    )


def quantize_with_modelopt(
    args,
    fp32_onnx: Path,
    output_path: Path,
    calibration_data: dict[str, object],
    calibration_shapes: str,
) -> None:
    from modelopt.onnx.quantization import quantize

    quantize(
        str(fp32_onnx),
        quantize_mode=args.quantize_mode,
        calibration_data=calibration_data,
        calibration_method=args.calibration_method,
        calibration_shapes=calibration_shapes,
        calibration_eps=args.calibration_eps.split(","),
        output_path=str(output_path),
        use_external_data_format=args.external_data,
        high_precision_dtype=args.high_precision_dtype,
        op_types_to_quantize=args.op_types_to_quantize,
        op_types_to_exclude=args.op_types_to_exclude,
        nodes_to_exclude=args.nodes_to_exclude,
        simplify=args.simplify,
        opset=args.qdq_opset,
        block_size=args.block_size,
        use_zero_point=args.use_zero_point,
        log_level=args.log_level,
    )


class DictCalibrationReader:
    """ONNX Runtime calibration reader for two-input style-transfer batches."""

    def __init__(self, calibration_data: dict[str, object], batch_size: int):
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.index = 0
        self.length = next(iter(calibration_data.values())).shape[0]

    def get_next(self):
        if self.index >= self.length:
            return None
        start = self.index
        end = min(start + self.batch_size, self.length)
        self.index = end
        return {name: value[start:end] for name, value in self.calibration_data.items()}


def quantize_with_ort(
    args,
    fp32_onnx: Path,
    output_path: Path,
    calibration_data: dict[str, object],
) -> None:
    if args.quantize_mode != "int8":
        raise ValueError("ONNX Runtime fallback only supports INT8 QDQ PTQ.")

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    method_name = (args.calibration_method or "minmax").lower()
    if method_name in ("entropy", "entropycalibration"):
        method = CalibrationMethod.Entropy
    elif method_name in ("percentile",):
        method = CalibrationMethod.Percentile
    else:
        method = CalibrationMethod.MinMax

    reader = DictCalibrationReader(calibration_data, batch_size=args.calib_batch_size)
    quantize_static(
        str(fp32_onnx),
        str(output_path),
        reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=args.op_types_to_quantize,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=args.nodes_to_exclude,
        use_external_data_format=args.external_data,
        calibrate_method=method,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={
            "DedicatedQDQPair": True,
            "AddQDQPairToWeight": True,
        },
    )


def parse_csv(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize style-transfer model to Q/DQ ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content_dir", type=str, default="assets/sample_content")
    parser.add_argument("--style_dir", type=str, default="assets/sample_style")
    parser.add_argument("--output", type=str, default="exports/style_transfer_int8_qdq.onnx")
    parser.add_argument(
        "--mode",
        type=str,
        default="ptq",
        choices=["ptq", "qat_export", "qat_direct_export"],
    )

    parser.add_argument("--quantize_mode", type=str, default="int8", choices=["int8", "int4", "fp8"])
    parser.add_argument("--ptq_backend", type=str, default="auto", choices=["auto", "modelopt", "ort"])
    parser.add_argument("--calibration_method", type=str, default=None)
    parser.add_argument("--calib_batches", type=int, default=32)
    parser.add_argument("--calib_batch_size", type=int, default=1)
    parser.add_argument("--max_content_images", type=int, default=None)
    parser.add_argument("--max_style_images", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--calibration_eps", type=str, default="cpu")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--qdq_opset", type=int, default=None)
    parser.add_argument("--legacy_onnx", action="store_true")
    parser.add_argument("--static_onnx_shapes", action="store_true")
    parser.add_argument("--external_data", action="store_true", default=True)
    parser.add_argument("--no_external_data", dest="external_data", action="store_false")
    parser.add_argument("--pretrained_encoder", action="store_true", default=False)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=2.0)

    parser.add_argument("--high_precision_dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--op_types_to_quantize", type=parse_csv, default=None)
    parser.add_argument("--op_types_to_exclude", type=parse_csv, default=None)
    parser.add_argument("--nodes_to_exclude", type=parse_csv, default=None)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--use_zero_point", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--keep_fp32_onnx", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "qat_direct_export":
        export_direct_qat_onnx(args, Path(args.output))
        return

    if args.quantize_mode == "int4":
        print(
            "INT4 is exposed for experimentation, but this model is Conv-heavy and "
            "ModelOpt ONNX INT4 primarily targets MatMul/Gemm. Expect limited benefit."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.keep_fp32_onnx:
        fp32_onnx = Path(args.keep_fp32_onnx)
        export_fp32_onnx(args, device, fp32_onnx)
        quantize_onnx_ptq(args, fp32_onnx, device)
    else:
        with tempfile.TemporaryDirectory(prefix="style_ptq_") as tmp:
            fp32_onnx = Path(tmp) / "style_transfer_fp32.onnx"
            export_fp32_onnx(args, device, fp32_onnx)
            quantize_onnx_ptq(args, fp32_onnx, device)


if __name__ == "__main__":
    main()
