from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

@dataclass
class ModelInfo:
    path: str
    file_size_bytes: int
    sha256: Optional[str] = None

@dataclass
class StaticAnalysis:
    parameters: int
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    flops_estimate: Optional[float] = None

@dataclass
class SystemInfo:
    os: str
    python: str
    packages: Dict[str, str]

@dataclass
class RuntimeProfile:
    engine: str
    device: str
    warmup: int
    runs: int
    latency_ms: Dict[str, float]
    extra: Dict[str, Any]

@dataclass
class EdgeBenchReport:
    schema_version: str
    timestamp: str
    model: "ModelInfo"
    static: "StaticAnalysis"
    system: "SystemInfo"
    meta: Dict[str, Any]
    runtime: Optional[RuntimeProfile] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write_json(self, output_path: str, indent: int = 2) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))
            f.write("\n")