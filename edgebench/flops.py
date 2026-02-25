from dataclasses import dataclass
from typing import Dict, Optional

import onnx
from onnx import numpy_helper

@dataclass
class FlopsEstimate:
    total: int
    by_op: Dict[str, int]

def estimate_flops_conv_gemm(
    model: onnx.ModelProto,
    height: Optional[int] = None,
    width: Optional[int] = None,
    batch: int = 1,
) -> Optional[FlopsEstimate]:

    graph = model.graph
    init_map = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    total = 0
    by_op: Dict[str, int] = {}

    for node in graph.node:
        if node.op_type == "Conv":

            if len(node.input) < 2:
                continue

            weight_name = node.input[1]
            if weight_name not in init_map:
                continue

            w = init_map[weight_name] #[Cout, Cin/group, kH, kW]
            Cout, Cin_per_g, kH, kW = w.shape

            # H/W 없으면 계산 불가
            if height is None or width is None:
                return None

            # stride=2 padding=1 가정 없이
            # 단순히 출력 spatial = height/stride 정도로 근사 X
            # ToyNet은 stride=2 2번 -> 실제 계산은 정확히 하지 않고
            # Conv FLOPs는 입력 기준으로 근사

            Hout = height // 2
            Wout = width // 2

            macs = batch * Hout * Wout * Cout * (Cin_per_g * kH * kW)
            flops = 2 * macs

            total += flops
            by_op["Conv"] = by_op.get("Conv", 0) + flops

        elif node.op_type in ("Gemm", "MatMul"):

            weight_name = node.input[1]
            if weight_name not in init_map:
                continue

            w = init_map[weight_name]
            K, M = w.shape

            N = batch
            flops = 2 * N * K * M

            total += flops
            by_op["Gemm"] = by_op.get("Gemm", 0) + flops

    return FlopsEstimate(total=total, by_op=by_op)