from __future__ import annotations

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

@dataclass
class FlopsHotspot:
    name: str
    op_type: str
    flops: int


def estimate_flops_conv_gemm_detailed(
    model: onnx.ModelProto,
    *,
    height: Optional[int],
    width: Optional[int],
    batch: int = 1,
    topk: int = 10,
) -> Tuple[Optional[int], Dict[str, Optional[int]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Conv / Gemm(Linear) 중심 근사 FLOPs + breakdown + hotspots + assumptions.

    Returns:
      total_flops: Optional[int]
      breakdown: Dict[str, Optional[int]]  # conv/gemm/other/total
      hotspots: List[Dict[str, Any]]       # [{name, op_type, flops}, ...]
      assumptions: Dict[str, Any]          # {batch,height,width, note}
    """
    # height/width가 없으면(특히 dynamic H/W) total을 못 냄
    if height is None or width is None:
        assumptions = {
            "batch": batch,
            "height": height,
            "width": width,
            "note": "height/width not provided; FLOPs may be unavailable for dynamic shapes",
        }
        return None, {"conv": None, "gemm": None, "other": None, "total": None}, [], assumptions

    # NOTE: 여기서는 네가 이미 만든 estimate_flops_conv_gemm 로직을 “노드별로 쪼개서”
    # conv/gemm만 집계하도록 만든다.
    # (입력 해상도 기반으로 Conv/Gemm FLOPs를 추정하는 방식은 근사치임)

    conv_sum = 0
    gemm_sum = 0
    other_sum = 0
    hotspots: List[FlopsHotspot] = []

    # 간단한 shape 추정용: 입력(첫 input)만 H/W 지정
    # 더 정교하게 하려면 shape inference를 붙이면 됨(다음 단계 가능)
    def _get_attr_ints(node: onnx.NodeProto, key: str) -> List[int]:
        for a in node.attribute:
            if a.name == key:
                return list(a.ints)
        return []

    def _get_attr_int(node: onnx.NodeProto, key: str, default: int) -> int:
        for a in node.attribute:
            if a.name == key:
                return int(a.i)
        return default

    # 매우 러프한 Conv FLOPs:
    # FLOPs ~= 2 * Cout * Hout * Wout * (Cin/groups * Kh * Kw) * batch
    # (mul+add를 2로 봄)
    #
    # Hout/Wout은 stride/pad로 정확히 계산하려면 더 정보 필요.
    # 여기서는 "same-ish" 근사로 Hout/Wout을 height/stride, width/stride로 잡는다.
    def _conv_flops_rough(node: onnx.NodeProto) -> Optional[int]:
        # weight initializer를 찾아야 정확한 Cin/Cout/Kh/Kw를 알 수 있는데,
        # 여기서는 최소한의 근사로 "가중치 텐서 shape"를 initializer에서 찾는 방식 권장.
        # 다음 단계에서 initializer lookup을 붙일 거라, 지금은 name 기반으로 찾아본다.
        if len(node.input) < 2:
            return None
        w_name = node.input[1]
        W = None
        for init in model.graph.initializer:
            if init.name == w_name:
                W = init
                break
        if W is None:
            return None
        # ONNX Conv weight shape: [Cout, Cin/groups, Kh, Kw]
        if len(W.dims) != 4:
            return None
        cout = int(W.dims[0])
        cin_per_g = int(W.dims[1])
        kh = int(W.dims[2])
        kw = int(W.dims[3])

        strides = _get_attr_ints(node, "strides")
        sH = strides[0] if len(strides) >= 1 else 1
        sW = strides[1] if len(strides) >= 2 else 1

        groups = _get_attr_int(node, "group", 1)

        hout = max(1, height // max(1, sH))
        wout = max(1, width // max(1, sW))

        macs = batch * cout * hout * wout * (cin_per_g * kh * kw)
        flops = 2 * macs
        return int(flops)

    # GEMM FLOPs(Linear) rough:
    # FLOPs ~= 2 * M * N * K
    # Weight shape: [N, K] or [K, N] depending on transpose flags.
    def _gemm_flops_rough(node: onnx.NodeProto) -> Optional[int]:
        if len(node.input) < 2:
            return None
        b_name = node.input[1]
        B = None
        for init in model.graph.initializer:
            if init.name == b_name:
                B = init
                break
        if B is None:
            return None
        if len(B.dims) != 2:
            return None
        n = int(B.dims[0])
        k = int(B.dims[1])
        # M은 배치(또는 토큰 수)인데 여기서는 batch만으로 근사
        m = batch
        return int(2 * m * n * k)

    for idx, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{idx}"
        op = node.op_type

        f = None
        if op == "Conv":
            f = _conv_flops_rough(node)
            if f is not None:
                conv_sum += f
        elif op in ("Gemm", "MatMul"):
            f = _gemm_flops_rough(node)
            if f is not None:
                gemm_sum += f
        else:
            # other는 지금은 0으로 두거나, node count 정도만 기록할 수도 있음
            f = 0
            other_sum += 0

        if f is not None and f > 0:
            hotspots.append(FlopsHotspot(name=name, op_type=op, flops=int(f)))

    hotspots.sort(key=lambda x: x.flops, reverse=True)
    hotspots = hotspots[: max(0, int(topk))]

    total = conv_sum + gemm_sum + other_sum
    breakdown: Dict[str, Optional[int]] = {
        "conv": int(conv_sum),
        "gemm": int(gemm_sum),
        "other": int(other_sum),
        "total": int(total),
    }

    assumptions = {
        "batch": batch,
        "height": height,
        "width": width,
        "topk": topk,
        "note": "rough estimate (Conv/Gemm 중심), output size approximated via stride; for relative comparison",
    }

    hotspots_out = [{"name": h.name, "op_type": h.op_type, "flops": h.flops} for h in hotspots]
    return int(total), breakdown, hotspots_out, assumptions