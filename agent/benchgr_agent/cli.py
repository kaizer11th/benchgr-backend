"""
BenchGR Agent CLI
-----------------
Usage:
    benchgr run                    # run all benchmarks, don't submit
    benchgr run --submit           # run + submit to leaderboard
    benchgr run --api-key KEY      # provide key inline
    benchgr config set-key KEY     # save API key to config
    benchgr info                   # show detected GPU info
"""

import click
import json
import os
import requests
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

CONFIG_PATH = Path.home() / ".benchgr" / "config.json"
API_BASE    = os.environ.get("BENCHGR_API_URL", "https://benchgr-api.up.railway.app/api")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(data: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


@click.group()
def main():
    """BenchGR — GPU Benchmark Agent"""
    pass


# ── benchgr info ──────────────────────────────────────────

@main.command()
def info():
    """Show detected GPU information."""
    from benchgr_agent.gpu_info import get_gpu_info

    console.print()
    info = get_gpu_info()

    if not info:
        console.print(Panel(
            "[red]No NVIDIA GPU detected.[/]\n"
            "Make sure NVIDIA drivers are installed and you're running on a CUDA-capable machine.",
            title="⚠ GPU Detection Failed",
            border_style="red",
        ))
        return

    t = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    t.add_column("Field", style="dim")
    t.add_column("Value", style="bold white")
    t.add_row("GPU", info["gpu_name"])
    t.add_row("Architecture", info["gpu_arch"])
    t.add_row("VRAM", f"{info['vram_gb']} GB")
    t.add_row("Driver", info["driver_version"])
    t.add_row("CUDA", info["cuda_version"])

    console.print(Panel(t, title="[blue]GPU Info[/]", border_style="blue"))


# ── benchgr config ────────────────────────────────────────

@main.group()
def config():
    """Manage BenchGR configuration."""
    pass


@config.command("set-key")
@click.argument("api_key")
def set_key(api_key):
    """Save your API key (get it from benchgr.app/dashboard)."""
    cfg = load_config()
    cfg["api_key"] = api_key
    save_config(cfg)
    console.print(f"[green]✓ API key saved to {CONFIG_PATH}[/]")


@config.command("show")
def show_config():
    """Show current config."""
    cfg = load_config()
    if not cfg:
        console.print("[dim]No config found. Run: benchgr config set-key YOUR_KEY[/]")
        return
    key = cfg.get("api_key", "")
    console.print(f"API Key: [dim]{key[:8]}...{key[-4:]}[/]")


# ── benchgr run ───────────────────────────────────────────

@main.command()
@click.option("--submit", is_flag=True, default=False, help="Submit results to leaderboard")
@click.option("--api-key", default=None, help="API key (or set with: benchgr config set-key)")
@click.option("--inference/--no-inference", default=True, help="Run AI inference suite")
@click.option("--image-gen/--no-image-gen", default=True, help="Run image generation suite")
@click.option("--cuda/--no-cuda", default=True, help="Run CUDA tensor ops suite")
@click.option("--memory/--no-memory", default=True, help="Run memory bandwidth suite")
def run(submit, api_key, inference, image_gen, cuda, memory):
    """Run GPU benchmarks."""
    from benchgr_agent.gpu_info import get_gpu_info
    from benchgr_agent.benchmarks import (
        bench_ai_inference, bench_image_generation,
        bench_cuda_tensor, bench_memory_bandwidth,
    )
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    console.print()
    console.print(Panel(
        "[bold white]BenchGR GPU Benchmark Suite[/]\n[dim]Running all 4 benchmark categories…[/]",
        border_style="blue",
    ))

    # Detect GPU
    gpu = get_gpu_info()
    if not gpu:
        console.print("[red]✗ No GPU detected. Aborting.[/]")
        return

    console.print(f"\n[dim]Detected:[/] [bold]{gpu['gpu_name']}[/] — {gpu['vram_gb']}GB VRAM\n")

    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description:<40}"),
        TimeElapsedColumn(),
    ) as p:

        if inference:
            t = p.add_task("[green]Running AI Inference…", total=None)
            results["tokens_per_sec"] = bench_ai_inference()
            p.update(t, description=f"[green]✓ AI Inference[/]  [bold]{results['tokens_per_sec']} tok/s[/]", completed=1, total=1)

        if image_gen:
            t = p.add_task("[yellow]Running Image Generation…", total=None)
            results["images_per_sec"] = bench_image_generation()
            p.update(t, description=f"[yellow]✓ Image Generation[/]  [bold]{results['images_per_sec']} img/s[/]", completed=1, total=1)

        if cuda:
            t = p.add_task("[purple]Running CUDA Tensor Ops…", total=None)
            results["tflops_fp16"] = bench_cuda_tensor()
            p.update(t, description=f"[purple]✓ CUDA TFLOPS[/]  [bold]{results['tflops_fp16']} TFLOPS[/]", completed=1, total=1)

        if memory:
            t = p.add_task("[red]Running Memory Bandwidth…", total=None)
            results["memory_bw_gbps"] = bench_memory_bandwidth()
            p.update(t, description=f"[red]✓ Memory Bandwidth[/]  [bold]{results['memory_bw_gbps']} GB/s[/]", completed=1, total=1)

    # Print results table
    console.print()
    t = Table(title="Benchmark Results", box=box.ROUNDED, border_style="blue")
    t.add_column("Suite", style="bold")
    t.add_column("Score", justify="right", style="bold white")
    t.add_column("Unit", style="dim")

    if "tokens_per_sec" in results:
        t.add_row("AI Inference", str(results["tokens_per_sec"]), "tokens/sec")
    if "images_per_sec" in results:
        t.add_row("Image Generation", str(results["images_per_sec"]), "images/sec")
    if "tflops_fp16" in results:
        t.add_row("CUDA Tensor Ops", str(results["tflops_fp16"]), "TFLOPS FP16")
    if "memory_bw_gbps" in results:
        t.add_row("Memory Bandwidth", str(results["memory_bw_gbps"]), "GB/s")

    console.print(t)

    # Submit
    if submit:
        _submit_results(api_key, gpu, results)
    else:
        console.print("\n[dim]Tip: run with --submit to post to the leaderboard.[/]")
    console.print()


def _submit_results(api_key: str, gpu: dict, results: dict):
    cfg = load_config()
    key = api_key or cfg.get("api_key")

    if not key:
        console.print("\n[red]✗ No API key found.[/]")
        console.print("  Get your key at [link=https://benchgr.app]benchgr.app[/link] → Dashboard")
        console.print("  Then run: [bold]benchgr config set-key YOUR_KEY[/bold]")
        return

    console.print("\n[dim]Submitting to leaderboard…[/]")
    payload = {**gpu, **results, "agent_version": "1.0.0"}

    try:
        resp = requests.post(
            f"{API_BASE}/results/submit",
            json=payload,
            params={"api_key": key},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            console.print(f"[green]✓ Submitted![/] Neural Score: [bold blue]{data['neural_score']}[/]")
            console.print(f"  View on leaderboard: [link=https://benchgr.app]benchgr.app[/link]")
        else:
            console.print(f"[red]✗ Submission failed:[/] {resp.text}")
    except requests.RequestException as e:
        console.print(f"[red]✗ Network error:[/] {e}")


if __name__ == "__main__":
    main()
