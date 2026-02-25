from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from edgebench.flops import estimate_flops_conv_gemm

import hashlib
import os
import platform
import sys

import onnx
from onnx import numpy_helper

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _shape_and_symbols(tensor_type: onnx.TypeProto.Tensor) -> Tuple[List[Optional[int]], List[Optional[str]]]:
    shape: List[Optional[int]] = []
    symbols: List[Optional[str]] = []

    if not tensor_type.HasField("shape"):
        return shape, symbols

    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            shape.append(int(d.dim_value))
            symbols.append(None)
        elif d.HasField("dim_param"):
            shape.append(None)
            symbols.append(d.dim_param)
        else:
            shape.append(None)
            symbols.append(None)

    return shape, symbols


def extract_ios(model: onnx.ModelProto) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def one_value_info(vi: onnx.ValueInfoProto) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": vi.name}

        if not vi.type.HasField("tensor_type"):
            out["type"] = "non-tensor"
            return out

        tt = vi.type.tensor_type
        elem_type = tt.elem_type
        dtype_name = onnx.TensorProto.DataType.Name(elem_type) if elem_type else "UNDEFINED"

        shape, symbols = _shape_and_symbols(tt)

        out["dtype"] = dtype_name
        out["shape"] = shape
        if any(s is not None for s in symbols):
            out["shape_symbols"] = symbols
        return out

    inputs = [one_value_info(i) for i in model.graph.input]
    outputs = [one_value_info(o) for o in model.graph.output]
    return inputs, outputs

def count_parameters(model: onnx.ModelProto) -> int:
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total += int(arr.size)
    return total

@dataclass
class AnalyzeResult:
    parameters: int
    file_size_bytes: int
    sha256: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    flops_estimate: Optional[int]

def analyze_onnx(
    model_path: str,
    compute_hash: bool = True,
    height: int | None = None,
    width: int | None = None,
):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {model_path}")

    file_size = os.path.getsize(model_path)

    model = onnx.load(model_path)

    flops_est = estimate_flops_conv_gemm(
        model,
        height=height,
        width=width,
        batch=1,
    )

    flops_total = flops_est.total if flops_est is not None else None

    onnx.checker.check_model(model)

    params = count_parameters(model)
    inputs, outputs = extract_ios(model)
    h = sha256_file(model_path) if compute_hash else ""

    return AnalyzeResult(
        parameters=params,
        file_size_bytes=file_size,
        sha256=h,
        inputs=inputs,
        outputs=outputs,
        flops_estimate=flops_total,
    )

def collect_system_info() -> Dict[str, Any]:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "machine": platform.machine(),
    }

def collect_package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}

    try:
        import onnx as _onnx

        versions["onnx"] = getattr(_onnx, "__version__", "unknown")
    except Exception:
        pass

    try:
        import onnxruntime as _ort

        versions["onnxruntime"] = getattr(_ort, "__version__", "unknown")
    except Exception:
        pass

    return versions