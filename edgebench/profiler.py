from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

import numpy as np

from edgebench.engines.onnxruntime_cpu import OnnxRuntimeCpuEngine

@dataclass
class ProfileResult:
    engine: str
    device: str
    warmup: int
    runs: int
    latency_ms: Dict[str, float]
    extra: Dict[str, Any]

def _latency_stats_ms(samples_ms: np.ndarray) -> Dict[str, float]:
    # IMPORTANT: samples_ms must be 1D float array
    mean = float(samples_ms.mean())
    std = float(samples_ms.std(ddof=0))
    p50 = float(np.percentile(samples_ms, 50))
    p90 = float(np.percentile(samples_ms, 90))
    p99 = float(np.percentile(samples_ms, 99))
    mn = float(samples_ms.min())
    mx = float(samples_ms.max())
    return {
        "mean": mean,
        "std": std,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "min": mn,
        "max": mx,
    }

def profile_onnxruntime_cpu(
    model_path: str,
    warmup: int = 10,
    runs: int = 100,
    batch: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    intra_threads: int = 1,
    inter_threads: int = 1,
) -> ProfileResult:
    if warmup < 0 or runs <= 0:
        raise ValueError("warmup은 0 이상, runs는 1 이상이어야 합니다.")

    engine = OnnxRuntimeCpuEngine()
    engine.load(model_path, intra_threads=intra_threads, inter_threads=inter_threads)

    feeds = engine.make_dummy_inputs(
        batch_override=batch,
        height_override=height,
        width_override=width,
    )

    input_names = list(feeds.keys())

    #warmup
    for _ in range(warmup):
        engine.run(feeds)

    # timed runs
    samples = np.empty((runs,), dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        engine.run(feeds)
        t1 = time.perf_counter()
        samples[i] = (t1 - t0) * 1000.0 # ms

    stats = _latency_stats_ms(samples)

    extra = {
        "batch": batch if batch is not None else 1,
        "input_names": input_names,
        "intra_threads": intra_threads,
        "inter_threads": inter_threads,
        "height": height,
        "width": width,
    }

    return ProfileResult(
        engine=engine.name,
        device=engine.device,
        warmup=warmup,
        runs=runs,
        latency_ms=stats,
        extra=extra,
    )