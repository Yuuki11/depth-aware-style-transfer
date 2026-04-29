#!/usr/bin/env python3
"""Run single-image style transfer with a TensorRT engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import load_image, save_comparison_grid, save_image


def check_cuda(result):
    err = result[0]
    if err.value != 0:
        raise RuntimeError(f"CUDA runtime error: {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT style-transfer inference")
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_trt.png")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument(
        "--depth_model_name",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
    )
    parser.add_argument("--save_grid", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine_bytes = Path(args.engine).read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {args.engine}")
    context = engine.create_execution_context()

    content_t = load_image(args.content, size=args.image_size)
    style_t = load_image(args.style, size=args.image_size)
    host_inputs = {
        "content": np.ascontiguousarray(content_t.cpu().numpy()),
        "style": np.ascontiguousarray(style_t.cpu().numpy()),
    }
    if args.use_depth:
        from models import DepthEstimator

        depth_device = "cuda" if torch.cuda.is_available() else "cpu"
        estimator = DepthEstimator(model_name=args.depth_model_name, device=depth_device)
        content_depth = estimator.from_tensor(content_t.to(depth_device))
        style_depth = estimator.from_tensor(style_t.to(depth_device))
        host_inputs["content_depth"] = np.ascontiguousarray(content_depth.cpu().numpy())
        host_inputs["style_depth"] = np.ascontiguousarray(style_depth.cpu().numpy())

    for name, array in host_inputs.items():
        context.set_input_shape(name, array.shape)

    stream = check_cuda(cudart.cudaStreamCreate())
    allocations: list[int] = []
    host_outputs: dict[str, np.ndarray] = {}

    try:
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            mode = engine.get_tensor_mode(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = tuple(context.get_tensor_shape(name))

            if mode == trt.TensorIOMode.INPUT:
                host_array = host_inputs[name].astype(dtype, copy=False)
            else:
                host_array = np.empty(shape, dtype=dtype)
                host_outputs[name] = host_array

            nbytes = host_array.nbytes
            device_ptr = check_cuda(cudart.cudaMalloc(nbytes))
            allocations.append(device_ptr)
            context.set_tensor_address(name, int(device_ptr))

            if mode == trt.TensorIOMode.INPUT:
                check_cuda(
                    cudart.cudaMemcpyAsync(
                        device_ptr,
                        host_array.ctypes.data,
                        nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    )
                )

        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT execution failed.")

        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            host_array = host_outputs[name]
            check_cuda(
                cudart.cudaMemcpyAsync(
                    host_array.ctypes.data,
                    context.get_tensor_address(name),
                    host_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )

        check_cuda(cudart.cudaStreamSynchronize(stream))
    finally:
        for ptr in allocations:
            check_cuda(cudart.cudaFree(ptr))
        check_cuda(cudart.cudaStreamDestroy(stream))

    if "stylized" not in host_outputs:
        raise RuntimeError(f"Expected output tensor 'stylized', got {list(host_outputs)}")

    output_t = torch.from_numpy(host_outputs["stylized"]).float()
    output_path = Path(args.output)
    save_image(output_t, output_path)
    print(f"Saved: {output_path}")

    if args.save_grid:
        grid_path = output_path.parent / f"{output_path.stem}_comparison.png"
        save_comparison_grid(content_t, style_t, output_t, grid_path)
        print(f"Saved comparison: {grid_path}")


if __name__ == "__main__":
    main()
