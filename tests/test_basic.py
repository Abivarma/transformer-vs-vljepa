"""Basic sanity tests to ensure environment is set up correctly."""

import sys

import pytest


def test_python_version():
    """Test that Python version is 3.11+."""
    assert sys.version_info >= (3, 11), "Python 3.11 or higher required"


def test_imports():
    """Test that core dependencies can be imported."""
    try:
        import numpy
        import pandas
        import torch
        import transformers

        # Verify packages are actually loaded
        assert numpy.__version__ is not None
        assert pandas.__version__ is not None
        assert torch.__version__ is not None
        assert transformers.__version__ is not None
    except ImportError as e:
        pytest.fail(f"Failed to import required package: {e}")


def test_torch_available():
    """Test that PyTorch is available."""
    import torch

    assert torch.__version__ is not None


def test_package_version():
    """Test that package version is accessible."""
    import src

    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"
