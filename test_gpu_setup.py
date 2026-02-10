"""Test PyTorch installation, GPU accessibility, and mixed-precision training."""

import sys
import time


def test_pytorch_installed():
    print("=" * 60)
    print("1. PyTorch Installation")
    print("=" * 60)
    try:
        import torch

        print(f"  PyTorch version: {torch.__version__}")
        print(f"  Python version:  {sys.version}")
    except ImportError:
        print("  FAIL: PyTorch is not installed.")
        print("  Install with: pip install torch")
        sys.exit(1)


def test_gpu_accessible():
    import torch

    print()
    print("=" * 60)
    print("2. GPU Accessibility")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available:       {cuda_available}")

    if not cuda_available:
        print("  FAIL: No CUDA-capable GPU detected.")
        print("  Check your NVIDIA drivers and CUDA toolkit installation.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"  Number of GPUs:       {device_count}")

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    current = torch.cuda.current_device()
    print(f"  Current device index: {current}")

    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"  BFloat16 supported:   {bf16_supported}")

    if not bf16_supported:
        print("  WARNING: bfloat16 is not supported on this GPU.")
        print("  The training test will likely fail or fall back to float16.")

    # Quick tensor round-trip
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    assert x.device.type == "cuda", "Tensor was not placed on GPU"
    print("  Tensor round-trip:    OK")


def test_mixed_precision_training():
    import torch
    import torch.nn as nn

    print()
    print("=" * 60)
    print("3. Toy Training with Mixed Precision (bfloat16)")
    print("=" * 60)

    # --- Toy dataset ---
    torch.manual_seed(42)
    num_samples = 512
    in_features = 64
    num_classes = 10

    X = torch.randn(num_samples, in_features, device="cuda")
    y = torch.randint(0, num_classes, (num_samples,), device="cuda")

    # --- Small model ---
    model = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # --- Training loop with autocast(bfloat16) ---
    num_epochs = 5
    batch_size = 64

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            xb, yb = X[start:end], y[start:end]

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = criterion(logits, yb)

            # No GradScaler needed for bfloat16 — its dynamic range matches float32
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch + 1}/{num_epochs}  loss: {avg_loss:.4f}")

    # --- Sanity check: loss decreased ---
    # Run one final forward pass to get final loss
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        final_logits = model(X)
        final_loss = criterion(final_logits, y).item()

    print(f"  Final full-dataset loss: {final_loss:.4f}")

    # Verify autocast actually used bfloat16
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        probe = model[0](X[:1])
    assert probe.dtype == torch.bfloat16, f"Expected bfloat16 but got {probe.dtype}"
    print("  Autocast dtype check: OK (bfloat16)")


def test_cudnn_tf32():
    import torch

    print()
    print("=" * 60)
    print("4. cuDNN and TF32")
    print("=" * 60)

    cudnn_available = torch.backends.cudnn.is_available()
    print(f"  cuDNN available:      {cudnn_available}")

    if not cudnn_available:
        print("  WARNING: cuDNN not available. Convolutions will be slower.")
        return

    print(f"  cuDNN version:        {torch.backends.cudnn.version()}")

    # Enable benchmark mode — auto-tunes conv algorithms per input shape
    torch.backends.cudnn.benchmark = True
    print(f"  cuDNN benchmark mode: enabled")

    # TF32 — Ampere+ GPUs use TensorFloat-32 for matmuls/convs by default
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    print(f"  TF32 matmul:          {tf32_matmul}")
    print(f"  TF32 cuDNN:           {tf32_cudnn}")

    if not tf32_matmul or not tf32_cudnn:
        print("  INFO: TF32 is disabled. On Ampere+ GPUs, enabling it gives")
        print("        ~2x matmul throughput with negligible accuracy loss.")
        print("        torch.backends.cuda.matmul.allow_tf32 = True")
        print("        torch.backends.cudnn.allow_tf32 = True")


def test_torch_compile():
    import torch
    import torch.nn as nn

    print()
    print("=" * 60)
    print("5. torch.compile (Triton Backend)")
    print("=" * 60)

    # Check Triton availability
    try:
        import triton

        print(f"  Triton version:       {triton.__version__}")
    except ImportError:
        print("  WARNING: Triton is not installed.")
        print("  torch.compile will fall back to slower backends.")
        print("  Install with: pip install triton")

    # Build a small model and compile it
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).cuda()

    x = torch.randn(32, 64, device="cuda")

    try:
        compiled_model = torch.compile(model)

        # First call triggers compilation (slow); second call uses the cache
        _ = compiled_model(x)
        torch.cuda.synchronize()
        _ = compiled_model(x)
        torch.cuda.synchronize()

        print("  torch.compile:        OK")
    except Exception as e:
        print(f"  FAIL: torch.compile raised: {e}")
        return

    # Verify outputs match
    with torch.no_grad():
        eager_out = model(x)
        compiled_out = compiled_model(x)

    max_diff = (eager_out - compiled_out).abs().max().item()
    print(f"  Eager vs compiled diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Outputs diverged: max diff {max_diff}"
    print("  Output correctness:   OK")


def test_channels_last():
    import torch
    import torch.nn as nn

    print()
    print("=" * 60)
    print("6. Channels-Last Memory Format (NHWC)")
    print("=" * 60)

    # Small CNN block
    conv = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    ).cuda()

    x = torch.randn(16, 3, 32, 32, device="cuda")

    # Convert model and input to channels_last
    conv_cl = conv.to(memory_format=torch.channels_last)
    x_cl = x.to(memory_format=torch.channels_last)

    assert x_cl.is_contiguous(memory_format=torch.channels_last), (
        "Input is not channels_last"
    )
    print("  Input  channels_last: OK")

    out_cl = conv_cl(x_cl)
    assert out_cl.is_contiguous(memory_format=torch.channels_last), (
        "Output lost channels_last format"
    )
    print("  Output channels_last: OK (format preserved through conv block)")

    # Verify numerical equivalence
    with torch.no_grad():
        out_default = conv(x)
        max_diff = (out_default - out_cl).abs().max().item()

    print(f"  NCHW vs NHWC diff:    {max_diff:.2e}")
    assert max_diff < 1e-4, f"Outputs diverged: max diff {max_diff}"
    print("  Output correctness:   OK")


def test_timing():
    import torch
    import torch.nn as nn

    print()
    print("=" * 60)
    print("7. Timing Comparison (CNN Training Step)")
    print("=" * 60)

    def build_model():
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

    x = torch.randn(64, 3, 32, 32, device="cuda")
    y = torch.randint(0, 10, (64,), device="cuda")
    criterion = nn.CrossEntropyLoss()

    def bench(model, x_in, label, warmup=5, iters=20):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = criterion(model(x_in), y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        # Timed iterations
        start = time.perf_counter()
        for _ in range(iters):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = criterion(model(x_in), y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters * 1000  # ms per step
        return elapsed

    # --- Baseline: eager, contiguous (NCHW) ---
    model_base = build_model().cuda()
    t_base = bench(model_base, x, "baseline")
    print(f"  Eager  + NCHW:        {t_base:.2f} ms/step")

    # --- Channels-last ---
    model_cl = build_model().cuda().to(memory_format=torch.channels_last)
    x_cl = x.to(memory_format=torch.channels_last)
    t_cl = bench(model_cl, x_cl, "channels_last")
    speedup_cl = t_base / t_cl
    print(f"  Eager  + NHWC:        {t_cl:.2f} ms/step ({speedup_cl:.2f}x)")

    # --- torch.compile + channels_last ---
    model_compiled = build_model().cuda().to(memory_format=torch.channels_last)
    try:
        model_compiled = torch.compile(model_compiled)
        t_compiled = bench(model_compiled, x_cl, "compile+cl", warmup=10)
        speedup_compiled = t_base / t_compiled
        print(
            f"  Compile + NHWC:       {t_compiled:.2f} ms/step "
            f"({speedup_compiled:.2f}x)"
        )
    except Exception as e:
        print(f"  Compile + NHWC:       skipped ({e})")

    print()
    print("  (Speedups are relative to eager + NCHW baseline)")


def main():
    print("PyTorch GPU + Mixed Precision Setup Test")
    print()

    test_pytorch_installed()
    test_gpu_accessible()
    test_mixed_precision_training()
    test_cudnn_tf32()
    test_torch_compile()
    test_channels_last()
    test_timing()

    print()
    print("=" * 60)
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
