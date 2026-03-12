"""
GPU info detection via pynvml (no PyTorch needed for this step).
Falls back gracefully if NVIDIA driver not found.
"""
from typing import Optional, Dict


def get_gpu_info() -> Optional[Dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        name          = pynvml.nvmlDeviceGetName(handle)
        mem_info      = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver_ver    = pynvml.nvmlSystemGetDriverVersion()

        # Decode bytes if needed
        if isinstance(name, bytes):
            name = name.decode()
        if isinstance(driver_ver, bytes):
            driver_ver = driver_ver.decode()

        vram_gb = round(mem_info.total / (1024 ** 3))

        # Try to get CUDA version
        try:
            cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            major = cuda_ver // 1000
            minor = (cuda_ver % 1000) // 10
            cuda_version = f"{major}.{minor}"
        except Exception:
            cuda_version = "unknown"

        pynvml.nvmlShutdown()

        return {
            "gpu_name": name,
            "vram_gb": vram_gb,
            "driver_version": driver_ver,
            "cuda_version": cuda_version,
            "gpu_arch": _infer_arch(name),
        }

    except Exception as e:
        return None


def _infer_arch(name: str) -> str:
    name_lower = name.lower()
    if "4090" in name_lower or "4080" in name_lower or "4070" in name_lower or "4060" in name_lower:
        return "Ada Lovelace"
    if "3090" in name_lower or "3080" in name_lower or "3070" in name_lower or "3060" in name_lower:
        return "Ampere"
    if "2080" in name_lower or "2070" in name_lower or "2060" in name_lower:
        return "Turing"
    if "h100" in name_lower:
        return "Hopper"
    if "a100" in name_lower:
        return "Ampere (DC)"
    return "Unknown"
