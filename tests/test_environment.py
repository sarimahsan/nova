import os
import sys
import shutil
import pytest
import torch
import numpy as np

def check_version_ge(actual, target):
    # Extract only the leading numeric part (e.g. "2.0.0+cu118" -> "2.0.0")
    actual_clean = "".join([c if c.isdigit() or c == "." else "" for c in actual.split("+")[0]])
    actual_parts = [int(x) for x in actual_clean.split(".") if x.isdigit()]
    target_parts = [int(x) for x in target.split(".") if x.isdigit()]
    
    # Pad to equal length
    max_len = max(len(actual_parts), len(target_parts))
    actual_parts += [0] * (max_len - len(actual_parts))
    target_parts += [0] * (max_len - len(target_parts))
    
    return actual_parts >= target_parts

def test_python_version():
    assert sys.version_info >= (3, 9), f"Python version is too old: {sys.version}"

def test_torch_installed():
    import torch
    assert torch is not None

def test_torch_version():
    import torch
    assert check_version_ge(torch.__version__, "2.0.0"), f"PyTorch version is too old: {torch.__version__}"

def test_cuda_available():
    # If running on Kaggle, assert CUDA. Otherwise, log availability.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_DATA_PROXY_TOKEN"):
        assert torch.cuda.is_available(), "CUDA is not available on Kaggle!"
    else:
        print(f"CUDA available: {torch.cuda.is_available()}")

def test_cuda_device_count():
    count = torch.cuda.device_count()
    print(f"Detected {count} CUDA devices")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_DATA_PROXY_TOKEN"):
        assert count >= 1, "No GPUs found on Kaggle"

def test_cuda_device_name():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

def test_bf16_support():
    if torch.cuda.is_available():
        supported = torch.cuda.is_bf16_supported()
        print(f"BF16 supported on GPU: {supported}")

def test_sdpa_available():
    import torch.nn.functional as F
    assert hasattr(F, "scaled_dot_product_attention"), "SDPA function scaled_dot_product_attention not found in F"

def test_transformers_installed():
    import transformers
    assert check_version_ge(transformers.__version__, "4.30.0"), f"Transformers version too old: {transformers.__version__}"

def test_datasets_installed():
    import datasets
    assert check_version_ge(datasets.__version__, "2.14.0"), f"Datasets version too old: {datasets.__version__}"

def test_tokenizers_installed():
    import tokenizers
    assert tokenizers is not None

def test_yaml_installed():
    import yaml
    assert yaml is not None

def test_matplotlib_installed():
    import matplotlib
    assert matplotlib is not None

def test_numpy_installed():
    import numpy as np
    assert check_version_ge(np.__version__, "1.24.0"), f"Numpy version too old: {np.__version__}"

def test_disk_space():
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    assert free_gb >= 1.0, f"Low free disk space: {free_gb:.2f} GB"

def test_dataparallel_works():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        import torch.nn as nn
        model = nn.Linear(10, 10).cuda()
        dp_model = nn.DataParallel(model)
        x = torch.randn(4, 10).cuda()
        out = dp_model(x)
        assert out.shape == (4, 10)
