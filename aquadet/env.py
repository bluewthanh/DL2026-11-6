"""Software and hardware environment recorded with each result."""

import platform
import subprocess
import sys


def gpu_snapshot() -> dict:
    """GPU memory already in use matters: latency is noisy on a shared GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    name, driver, used, total, util = (v.strip() for v in out.stdout.splitlines()[0].split(","))
    return {"name": name, "driver": driver, "memory_used_mib": int(used), "memory_total_mib": int(total),
            "utilization_pct": int(util)}


def environment() -> dict:
    import torch
    import ultralytics

    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "ultralytics": ultralytics.__version__, "device": "cpu"}
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["gpu"] = gpu_snapshot()
    return info
