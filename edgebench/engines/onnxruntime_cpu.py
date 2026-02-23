from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

import onnx
import onnxruntime as ort


@dataclass
class OrtModelIO:
    name: str
    dtype: np.dtype
    shape: List[Optional[int]]


def _onnx_elemtype_to_numpy(elem_type: int) -> np.dtype:
    # minimal mapping for common types
    # ONNX TensorProto.DataType: https://onnx.ai/onnx/api/mapping.html (conceptually)
    if elem_type == onnx.TensorProto.FLOAT:
        return np.float32
    if elem_type == onnx.TensorProto.FLOAT16:
        return np.float16
    if elem_type == onnx.TensorProto.DOUBLE:
        return np.float64
    if elem_type == onnx.TensorProto.INT64:
        return np.int64
    if elem_type == onnx.TensorProto.INT32:
        return np.int32
    if elem_type == onnx.TensorProto.UINT8:
        return np.uint8
    if elem_type == onnx.TensorProto.INT8:
        return np.int8
    # default fallback
    return np.float32


def _shape_from_valueinfo(vi: onnx.ValueInfoProto) -> List[Optional[int]]:
    if not vi.type.HasField("tensor_type"):
        return []
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return []
    out: List[Optional[int]] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        else:
            out.append(None)  # symbolic or unknown
    return out


def _dtype_from_valueinfo(vi: onnx.ValueInfoProto) -> np.dtype:
    if not vi.type.HasField("tensor_type"):
        return np.float32
    elem = vi.type.tensor_type.elem_type
    return _onnx_elemtype_to_numpy(elem)


class OnnxRuntimeCpuEngine:
    name = "onnxruntime"
    device = "cpu"

    def __init__(self) -> None:
        self.sess: Optional[ort.InferenceSession] = None
        self.inputs: List[OrtModelIO] = []
        self.outputs: List[str] = []

    def load(self, model_path: str, intra_threads: int = 1, inter_threads: int = 1) -> None:
        # ORT session options can be extended later
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra_threads)
        so.inter_op_num_threads = int(inter_threads)

        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        # Load IO metadata from ONNX (for shapes/types), ORT also has metadata but ONNX is fine
        m = onnx.load(model_path)

        self.inputs = []
        for vi in m.graph.input:
            self.inputs.append(
                OrtModelIO(
                    name=vi.name,
                    dtype=_dtype_from_valueinfo(vi),
                    shape=_shape_from_valueinfo(vi),
                )
            )

        self.outputs = [o.name for o in m.graph.output]

    def make_dummy_inputs(self, batch_override: Optional[int] = None) -> Dict[str, Any]:
        """
        Create random inputs based on model input shapes.
        Unknown dims(None) will be replaced by:
          - batch dim: 1 (or batch_override)
          - others: 1
        """
        if self.sess is None:
            raise RuntimeError("Engine not loaded")

        feeds: Dict[str, Any] = {}
        for inp in self.inputs:
            shape = []
            for i, d in enumerate(inp.shape):
                if d is None:
                    if i == 0:
                        shape.append(int(batch_override) if batch_override is not None else 1)
                    else:
                        shape.append(1)
                else:
                    shape.append(int(d))
            arr = np.random.rand(*shape).astype(inp.dtype, copy=False)
            feeds[inp.name] = arr
        return feeds

    def run(self, feeds: Dict[str, Any]) -> None:
        if self.sess is None:
            raise RuntimeError("Engine not loaded")
        self.sess.run(self.outputs, feeds)