#!/usr/bin/env python3
"""Build TensorRT engines from ONNX with the Python API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_shape(raw: str) -> tuple[int, ...]:
    parts = tuple(int(p) for p in raw.lower().replace(",", "x").split("x") if p)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Expected NCHW shape like 1x3x256x256, got {raw}")
    return parts


def set_flag_if_available(config, trt, flag_name: str, precision: str) -> bool:
    flag = getattr(trt.BuilderFlag, flag_name, None)
    if flag is None:
        print(f"TensorRT does not expose BuilderFlag.{flag_name}; {precision} may require explicit Q/DQ ONNX.")
        return False
    config.set_flag(flag)
    return True


def get_network_input(network, name: str):
    for idx in range(network.num_inputs):
        tensor = network.get_input(idx)
        if tensor.name == name:
            return tensor
    available = [network.get_input(idx).name for idx in range(network.num_inputs)]
    raise ValueError(f"Input '{name}' not found in ONNX network. Available inputs: {available}")


def resolve_profile_shape(network_shape, requested: tuple[int, ...]) -> tuple[int, ...]:
    if len(network_shape) != len(requested):
        return requested
    return tuple(req if int(dim) < 0 else int(dim) for dim, req in zip(network_shape, requested))


def parse_args():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "int8", "fp8", "int4", "int16"],
    )
    parser.add_argument("--min_shape", type=parse_shape, default=parse_shape("1x3x256x256"))
    parser.add_argument("--opt_shape", type=parse_shape, default=parse_shape("1x3x512x512"))
    parser.add_argument("--max_shape", type=parse_shape, default=parse_shape("1x3x1024x1024"))
    parser.add_argument("--workspace_gb", type=float, default=7.5)
    parser.add_argument("--builder_optimization_level", type=int, default=5)
    parser.add_argument("--input_names", type=str, default="content,style")
    parser.add_argument("--plugins", type=str, nargs="*", default=None)
    parser.add_argument("--timing_cache", type=str, default=None)
    parser.add_argument("--save_timing_cache", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    if args.plugins:
        for plugin in args.plugins:
            if not trt.init_libnvinfer_plugins(logger, ""):
                raise RuntimeError("Failed to initialize TensorRT plugins.")
            ctypes = __import__("ctypes")
            ctypes.CDLL(plugin)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_path = Path(args.onnx)
    if not parser.parse_from_file(str(onnx_path)):
        print("TensorRT ONNX parser failed:")
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gb * (1024**3)),
    )
    if args.builder_optimization_level is not None:
        if not 0 <= args.builder_optimization_level <= 5:
            raise ValueError("--builder_optimization_level must be between 0 and 5.")
        if not hasattr(config, "builder_optimization_level"):
            raise RuntimeError("This TensorRT version does not expose builder_optimization_level.")
        config.builder_optimization_level = args.builder_optimization_level
        print(f"Builder optimization level: {args.builder_optimization_level}")

    precision = args.precision.lower()
    if precision == "fp16":
        set_flag_if_available(config, trt, "FP16", precision)
    elif precision == "bf16":
        set_flag_if_available(config, trt, "BF16", precision)
    elif precision == "int8":
        set_flag_if_available(config, trt, "INT8", precision)
        set_flag_if_available(config, trt, "FP16", "fp16 fallback")
        print("INT8 should normally be built from an explicit Q/DQ ONNX model.")
    elif precision == "fp8":
        set_flag_if_available(config, trt, "FP8", precision)
        print("FP8 requires TensorRT/GPU support and an ONNX graph with compatible quantization.")
    elif precision == "int4":
        print(
            "INT4 is only meaningful with an explicit INT4 Q/DQ ONNX graph and compatible TensorRT/GPU support. "
            "No generic builder flag is set."
        )
    elif precision == "int16":
        raise ValueError(
            "TensorRT does not provide a generic INT16 neural-network execution mode for this model. "
            "Use --precision fp16 or bf16 for 16-bit inference."
        )

    profile = builder.create_optimization_profile()
    for name in [n.strip() for n in args.input_names.split(",") if n.strip()]:
        tensor = get_network_input(network, name)
        min_shape = resolve_profile_shape(tensor.shape, args.min_shape)
        opt_shape = resolve_profile_shape(tensor.shape, args.opt_shape)
        max_shape = resolve_profile_shape(tensor.shape, args.max_shape)
        print(f"Profile {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if args.timing_cache:
        cache_path = Path(args.timing_cache)
        if cache_path.exists():
            cache = config.create_timing_cache(cache_path.read_bytes())
            config.set_timing_cache(cache, ignore_mismatch=True)

    print(f"Building TensorRT engine: precision={precision}, onnx={onnx_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed.")

    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"Saved TensorRT engine: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")

    if args.save_timing_cache:
        cache = config.get_timing_cache()
        Path(args.save_timing_cache).write_bytes(bytes(cache.serialize()))


if __name__ == "__main__":
    main()
