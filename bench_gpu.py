"""Bucket 1: GPU sanity + honest CPU-vs-GPU benchmark on this machine."""
import time
import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}")

g = torch.Generator().manual_seed(42)

def timeit(fn, warmup=1, reps=3):
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

# --- elementwise: a*b+c over 100M floats ---
n = 100_000_000
for name, device in (("CPU", torch.device("cpu")), ("GPU", dev)):
    g = torch.Generator(device=device.type).manual_seed(42)  # generator must live on the same device
    a = torch.randn(n, generator=g, device=device)
    b = torch.randn(n, generator=g, device=device)
    c = torch.randn(n, generator=g, device=device)
    secs = timeit(lambda: torch.addcmul(c, a, b))
    flops = 2 * n / secs
    print(f"elementwise a*b+c (100M): {name} {secs*1000:8.1f} ms  ({flops/1e9:8.1f} GFLOP/s)")
    del a, b, c

# --- matmul: 2048^3 ---
m = 2048
flops_per = 2 * m**3
for name, device in (("CPU", torch.device("cpu")), ("GPU", dev)):
    g = torch.Generator(device=device.type).manual_seed(42)
    x = torch.randn(m, m, generator=g, device=device)
    y = torch.randn(m, m, generator=g, device=device)
    secs = timeit(lambda: x @ y)
    print(f"matmul 2048x2048:        {name} {secs*1000:8.1f} ms  ({flops_per/secs/1e9:8.1f} GFLOP/s)")
    del x, y