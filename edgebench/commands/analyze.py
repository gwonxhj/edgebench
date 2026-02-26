from __future__ import annotations

import typer
from rich import print as rprint

from edgebench.analyzer import analyze_onnx, collect_package_versions, collect_system_info
from edgebench.report import EdgeBenchReport, ModelInfo, StaticAnalysis, SystemInfo, utc_now_iso


def analyze_cmd(
    model_path: str = typer.Argument(..., help="분석할 ONNX 모델 경로"),
    output: str = typer.Option("", "--output", "-o", help="JSON 리포트 저장 경로(미지정 시 stdout 출력)"),
    no_hash: bool = typer.Option(False, "--no-hash", help="모델 SHA256 해시 계산 비활성화(대형 모델에서 빠름)"),
    height: int = typer.Option(0, "--height", help="FLOPs 계산용 입력 height (0이면 사용 안 함)"),
    width: int = typer.Option(0, "--width", help="FLOPs 계산용 입력 width (0이면 사용 안 함)"),
):
    rprint(f"[bold]Analyzing[/bold]: {model_path}")

    result = analyze_onnx(
        model_path,
        compute_hash=(not no_hash),
        height=height if height > 0 else None,
        width=width if width > 0 else None,
    )
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
            flops_estimate=result.flops_estimate,
            flops_breakdown=result.flops_breakdown,
            flops_hotspots=result.flops_hotspots,
            flops_assumptions=result.flops_assumptions,
        ),
        system=SystemInfo(
            os=sysinfo["os"],
            python=sysinfo["python"],
            packages=pkgs,
        ),
        meta={"machine": sysinfo.get("machine"), "notes": "Phase 1 static analyze"},
        runtime=None,
    )

    if output:
        report.write_json(output)
        rprint(f"[green]Saved[/green]: {output}")
    else:
        rprint(report.to_json())