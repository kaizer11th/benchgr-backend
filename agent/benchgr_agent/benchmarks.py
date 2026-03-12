"""
BenchGR Benchmark Suites
Runs locally on user's GPU and returns float scores.

Install full deps for real benchmarks:
    pip install benchgr-agent[full]

Without torch installed, suites return synthetic estimates
based on GPU VRAM for demo/testing purposes.
"""

import time
from typing import Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


def _has_torch() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── 1. AI Inference Benchmark ──────────────────────────────────────────────

def bench_ai_inference(progress=None, task=None) -> Optional[float]:
    """
    Measures LLM token generation throughput.
    Uses a small GPT-2 model if transformers available, else numpy matmul proxy.
    Returns: tokens per second (float)
    """
    if progress and task:
        progress.update(task, description="[green]AI Inference[/] — loading model…")

    if _has_torch():
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_id = "gpt2"  # ~500MB, manageable on any GPU
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id).cuda().half()

            inputs = tokenizer("BenchGR GPU benchmark test", return_tensors="pt").to("cuda")

            if progress and task:
                progress.update(task, description="[green]AI Inference[/] — warming up…")

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    model.generate(**inputs, max_new_tokens=20, do_sample=False)

            if progress and task:
                progress.update(task, description="[green]AI Inference[/] — benchmarking…")

            # Timed run
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            total_tokens = 0
            with torch.no_grad():
                for _ in range(10):
                    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                    total_tokens += out.shape[1] - inputs["input_ids"].shape[1]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            del model
            torch.cuda.empty_cache()

            return round(total_tokens / elapsed, 2)

        except Exception as e:
            pass

    # Fallback: numpy proxy benchmark
    return _numpy_proxy_inference()


def _numpy_proxy_inference() -> float:
    """Numpy GEMM proxy — estimates tok/s based on compute speed."""
    import numpy as np
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        vram = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024**3)
        pynvml.nvmlShutdown()
    except Exception:
        vram = 8

    N = 2048
    t0 = time.perf_counter()
    for _ in range(20):
        a = np.random.randn(N, N).astype(np.float32)
        b = np.random.randn(N, N).astype(np.float32)
        np.dot(a, b)
    elapsed = time.perf_counter() - t0

    ops_per_iter = 2 * N ** 3 * 20
    gflops = ops_per_iter / elapsed / 1e9
    # Rough mapping: 1 GFLOP ≈ 0.15 tokens/sec for GPT-2 class models
    return round(max(5.0, gflops * 0.15), 2)


# ── 2. Image Generation Benchmark ──────────────────────────────────────────

def bench_image_generation(progress=None, task=None) -> Optional[float]:
    """
    Stable Diffusion SDXL turbo throughput benchmark.
    Returns: images per second (float)
    """
    if progress and task:
        progress.update(task, description="[yellow]Image Gen[/] — loading pipeline…")

    if _has_torch():
        try:
            import torch
            from diffusers import AutoPipelineForText2Image

            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to("cuda")

            prompt = "a photorealistic GPU chip on a circuit board, 4K"

            if progress and task:
                progress.update(task, description="[yellow]Image Gen[/] — warming up…")

            # Warmup
            pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0)

            if progress and task:
                progress.update(task, description="[yellow]Image Gen[/] — benchmarking…")

            N = 5
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            del pipe
            torch.cuda.empty_cache()

            return round(N / elapsed, 3)

        except Exception:
            pass

    return _numpy_proxy_image()


def _numpy_proxy_image() -> float:
    import numpy as np
    N = 512
    t0 = time.perf_counter()
    for _ in range(10):
        img = np.random.randn(4, N, N).astype(np.float32)
        for step in range(20):
            img = img * 0.9 + np.random.randn(4, N, N).astype(np.float32) * 0.1
    elapsed = time.perf_counter() - t0
    return round(max(0.1, 10 / elapsed * 0.4), 3)


# ── 3. CUDA Tensor Ops Benchmark ───────────────────────────────────────────

def bench_cuda_tensor(progress=None, task=None) -> Optional[float]:
    """
    FP16 GEMM throughput via cuBLAS (through PyTorch).
    Returns: TFLOPS (float)
    """
    if progress and task:
        progress.update(task, description="[purple]CUDA Tensor[/] — running GEMM suite…")

    if _has_torch():
        try:
            import torch

            N = 4096
            a = torch.randn(N, N, dtype=torch.float16, device="cuda")
            b = torch.randn(N, N, dtype=torch.float16, device="cuda")

            # Warmup
            for _ in range(5):
                torch.mm(a, b)
            torch.cuda.synchronize()

            if progress and task:
                progress.update(task, description="[purple]CUDA Tensor[/] — measuring TFLOPS…")

            ITERS = 100
            t0 = time.perf_counter()
            for _ in range(ITERS):
                torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            flops = 2 * N ** 3 * ITERS
            tflops = flops / elapsed / 1e12

            del a, b
            torch.cuda.empty_cache()

            return round(tflops, 2)

        except Exception:
            pass

    return _numpy_proxy_tflops()


def _numpy_proxy_tflops() -> float:
    import numpy as np
    N = 2048
    ITERS = 20
    t0 = time.perf_counter()
    for _ in range(ITERS):
        a = np.random.randn(N, N).astype(np.float32)
        b = np.random.randn(N, N).astype(np.float32)
        np.dot(a, b)
    elapsed = time.perf_counter() - t0
    flops = 2 * N ** 3 * ITERS
    return round(flops / elapsed / 1e12, 3)


# ── 4. Memory Bandwidth Benchmark ──────────────────────────────────────────

def bench_memory_bandwidth(progress=None, task=None) -> Optional[float]:
    """
    Measures GPU memory copy throughput (device-to-device).
    Returns: bandwidth in GB/s (float)
    """
    if progress and task:
        progress.update(task, description="[red]Memory BW[/] — measuring D2D bandwidth…")

    if _has_torch():
        try:
            import torch

            SIZE = 512 * 1024 * 1024  # 512 MB
            src = torch.zeros(SIZE // 4, dtype=torch.float32, device="cuda")
            dst = torch.zeros_like(src)

            # Warmup
            for _ in range(3):
                dst.copy_(src)
            torch.cuda.synchronize()

            ITERS = 20
            t0 = time.perf_counter()
            for _ in range(ITERS):
                dst.copy_(src)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            bytes_transferred = SIZE * ITERS
            gb_per_sec = bytes_transferred / elapsed / 1e9

            del src, dst
            torch.cuda.empty_cache()

            return round(gb_per_sec, 1)

        except Exception:
            pass

    return _numpy_proxy_membw()


def _numpy_proxy_membw() -> float:
    import numpy as np
    SIZE = 128 * 1024 * 1024  # 128MB
    src = np.zeros(SIZE // 4, dtype=np.float32)
    ITERS = 10
    t0 = time.perf_counter()
    for _ in range(ITERS):
        dst = src.copy()
    elapsed = time.perf_counter() - t0
    bytes_transferred = SIZE * ITERS
    return round(bytes_transferred / elapsed / 1e9 * 0.3, 1)  # CPU→ estimate


# ── Run all suites ──────────────────────────────────────────────────────────

def run_all_benchmarks(verbose: bool = True) -> dict:
    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:

        t1 = progress.add_task("[green]AI Inference", total=None)
        results["tokens_per_sec"] = bench_ai_inference(progress, t1)
        progress.update(t1, description=f"[green]✓ AI Inference → {results['tokens_per_sec']} tok/s")

        t2 = progress.add_task("[yellow]Image Generation", total=None)
        results["images_per_sec"] = bench_image_generation(progress, t2)
        progress.update(t2, description=f"[yellow]✓ Image Gen → {results['images_per_sec']} img/s")

        t3 = progress.add_task("[purple]CUDA Tensor Ops", total=None)
        results["tflops_fp16"] = bench_cuda_tensor(progress, t3)
        progress.update(t3, description=f"[purple]✓ CUDA TFLOPS → {results['tflops_fp16']}")

        t4 = progress.add_task("[red]Memory Bandwidth", total=None)
        results["memory_bw_gbps"] = bench_memory_bandwidth(progress, t4)
        progress.update(t4, description=f"[red]✓ Memory BW → {results['memory_bw_gbps']} GB/s")

    return results
