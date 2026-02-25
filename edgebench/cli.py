from __future__ import annotations

import os
import typer

from datetime import datetime
from rich import print as rprint

from edgebench.analyzer import analyze_onnx, collect_package_versions, collect_system_info
from edgebench.report import EdgeBenchReport, ModelInfo, StaticAnalysis, SystemInfo, utc_now_iso, RuntimeProfile
from edgebench.profiler import profile_onnxruntime_cpu

app = typer.Typer(help="EdgeBench CLI - Edge AI Profiling Tool")

@app.command()
def analyze(
    model_path: str = typer.Argument(..., help="분석할 ONNX 모델 경로"),
    output: str = typer.Option("", "--output", "-o", help="JSON 리포트 저장 경로(미지정 시 stdout 출력)"),
    no_hash: bool = typer.Option(False, "--no-hash", help="모델 SHA256 해시 계산 비활성화(대형 모델에서 빠름)"),
):
    """
    ONNX 모델을 정적 분석하고 JSON 리포트를 출력합니다.
    """
    rprint(f"[bold]Analyzing[/bold]: {model_path}")

    result = analyze_onnx(model_path, compute_hash=(not no_hash))
    sysinfo = collect_system_info()
    pkgs = collect_package_versions()

    report = EdgeBenchReport(
        schema_version="0.1",
        timestamp=utc_now_iso(),
        model=ModelInfo(
            path=model_path,
            file_size_bytes=result.file_size_bytes, 
            sha256=(None if no_hash else result.sha256),
        ),
        static=StaticAnalysis(
            parameters=result.parameters, 
            inputs=result.inputs, 
            outputs=result.outputs,
            flops_estimate=None,
        ),
        system=SystemInfo(
            os=sysinfo["os"],
            python=sysinfo["python"],
            packages=pkgs,
        ),
        meta={
            "machine": sysinfo.get("machine"),
            "notes": "Phase 1 static analyze",
        },
    )

    if output:
        report.write_json(output)
        rprint(f"[green]Saved[/green]: {output}")
    else:
        rprint(report.to_json())

@app.command()
def profile(
    model_path: str = typer.Argument(..., help="프로파일링할 ONNX 모델 경로"),
    warmup: int = typer.Option(10, "--warmup", help="워밍업 반복 횟수"),
    runs: int = typer.Option(100, "--runs", help="측정 반복 횟수"),
    batch: int = typer.Option(1, "--batch", help="배치 크기(입력 0번째 차원 override)"),
    height: int = typer.Option(0, "--height", help="입력 height override (0이면 사용 안 함)"),
    width: int = typer.Option(0, "--width", help="입력 width override(0이면 사용 안 함)"),
    intra_threads: int = typer.Option(1, "--intra-threads", help="ONNX Runtime intra_op_num_threads"),
    inter_threads: int = typer.Option(1, "--inter-threads", help="ONNX Runtime inter_op_num_threads"),
    output: str = typer.Option("", "--output", "-o", help="JSON 리포트 저장 경로(미지정 시 stdout 출력)"),
    no_hash: bool = typer.Option(True, "--no-hash/--hash", help="profile 시 해시 계산(기본 off)"),
):
    """
    ONNX Runtime(CPU)로 실제 추론 latency를 측정하고 JSON 리포트 출력
    """
    rprint(f"[bold]Profiling[/bold]: {model_path}")

    # 1) 정적 분석도 같이 포함(리포트 일관성)
    result = analyze_onnx(model_path, compute_hash=(not no_hash))
    sysinfo = collect_system_info()
    pkgs = collect_package_versions()

    # 2) 동적 프로파일링
    prof = profile_onnxruntime_cpu(
        model_path,
        warmup=warmup,
        runs=runs,
        batch=batch,
        height=height if height > 0 else None,
        width=width if width > 0 else None,
        intra_threads=intra_threads,
        inter_threads=inter_threads,
    )

    report = EdgeBenchReport(
        schema_version="0.1",
        timestamp=utc_now_iso(),
        model=ModelInfo(
            path=model_path,
            file_size_bytes=result.file_size_bytes,
            sha256=(None if no_hash else result.sha256),
        ),
        static=StaticAnalysis(
            parameters=result.parameters,
            inputs=result.inputs,
            outputs=result.outputs,
            flops_estimate=None,
        ),
        runtime=RuntimeProfile(
            engine=prof.engine,
            device=prof.device,
            warmup=prof.warmup,
            runs=prof.runs,
            latency_ms=prof.latency_ms,
            extra=prof.extra,
        ),
        system=SystemInfo(
            os=sysinfo["os"],
            python=sysinfo["python"],
            packages=pkgs,
        ),
        meta={
            "machine": sysinfo.get("machine"),
            "notes": "Phase 1 profile",
        },
    )

    if not output:
        os.makedirs("reports", exist_ok=True)

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        auto_name = (
            f"{model_name}__"
            f"{prof.engine}_{prof.device}__"
            f"b{batch}__"
            f"h{height or 0}w{width or 0}__"
            f"r{runs}__{ts}.json"
        )

        output = os.path.join("reports", auto_name)

    report.write_json(output)
    rprint(f"[green]Saved[/green]: {output}")

if __name__ == "__main__":
    app()