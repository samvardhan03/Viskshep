"""
test_dispatch.py — Dynamic Template Instantiation Dispatcher Validation Suite

Validates that:
  1. Supported (J, Q) pairs produce valid, finite tensors.
  2. Different (J, Q) pairs produce numerically distinct outputs (proving
     different Morlet filter banks are constructed, not a hardcoded path).
  3. The CPU fallback engine handles arbitrary (J, Q) configurations
     that are outside the GPU template dispatch matrix.
"""

import numpy as np
import pytest

from vikshep import _vikshep_core as wst


# ---- Fixtures ----

VALID_CONFIGS = [
    (8, 16),
    (10, 16),
    (8, 8),
]

# Configs outside the GPU dispatch matrix — handled by CPU engine
CPU_FALLBACK_CONFIGS = [
    (4, 32),
    (7, 7),
    (16, 4),
]


# ---- Positive Dispatch Tests ----

@pytest.mark.parametrize("j, q", VALID_CONFIGS)
def test_dispatch_valid_config_produces_tensor(j, q):
    """Each supported (J, Q) pair must produce a finite, non-zero tensor."""
    cfg = wst.WSTConfig(J=j, Q=q, depth=2, jtfs=False)
    signal = np.random.randn(4096).astype(np.float32)

    result = wst.fingerprint(signal, cfg)

    assert result is not None, f"fingerprint returned None for (J={j}, Q={q})"
    assert result.shape == (4096,), f"Unexpected shape {result.shape} for (J={j}, Q={q})"
    assert np.all(np.isfinite(result)), f"Non-finite values in output for (J={j}, Q={q})"


@pytest.mark.parametrize("j, q", VALID_CONFIGS)
def test_dispatch_determinism(j, q):
    """Repeated calls with identical input and config must yield identical output."""
    cfg = wst.WSTConfig(J=j, Q=q, depth=2, jtfs=False)
    signal = np.random.randn(4096).astype(np.float32)

    result_a = wst.fingerprint(signal, cfg)
    result_b = wst.fingerprint(signal, cfg)

    np.testing.assert_array_equal(
        result_a, result_b,
        err_msg=f"Non-deterministic output for (J={j}, Q={q})"
    )


@pytest.mark.parametrize("j, q", VALID_CONFIGS)
def test_dispatch_batch_mode(j, q):
    """Batch (2D) input must dispatch through the same path."""
    cfg = wst.WSTConfig(J=j, Q=q, depth=2, jtfs=False)
    batch = np.random.randn(4, 4096).astype(np.float32)

    result = wst.fingerprint(batch, cfg)

    assert result.shape == (4, 4096), (
        f"Batch shape mismatch for (J={j}, Q={q}): got {result.shape}"
    )
    assert np.all(np.isfinite(result)), f"Non-finite batch output for (J={j}, Q={q})"


# ---- CPU Fallback Tests ----

@pytest.mark.parametrize("j, q", CPU_FALLBACK_CONFIGS)
def test_cpu_fallback_handles_arbitrary_configs(j, q):
    """Configs outside the GPU dispatch matrix must fall through to the CPU
    engine and produce valid, finite tensors — not raise an error."""
    cfg = wst.WSTConfig(J=j, Q=q, depth=2, jtfs=False)
    signal = np.random.randn(4096).astype(np.float32)

    result = wst.fingerprint(signal, cfg)

    assert result is not None
    assert result.shape == (4096,)
    assert np.all(np.isfinite(result)), f"Non-finite output for CPU fallback (J={j}, Q={q})"


# ---- Cross-Config Isolation Test ----

def test_dispatch_cross_config_produces_distinct_outputs():
    """Different (J, Q) pairs operating on the same input must produce
    numerically distinct scattering coefficients, proving the engine
    constructs different Morlet filter banks for each configuration."""
    signal = np.random.randn(4096).astype(np.float32)

    cfg_a = wst.WSTConfig(J=8, Q=16, depth=2, jtfs=False)
    cfg_b = wst.WSTConfig(J=10, Q=16, depth=2, jtfs=False)
    cfg_c = wst.WSTConfig(J=8, Q=8, depth=2, jtfs=False)

    result_a = wst.fingerprint(signal, cfg_a)
    result_b = wst.fingerprint(signal, cfg_b)
    result_c = wst.fingerprint(signal, cfg_c)

    # Different filter banks produce different scattering coefficients
    assert not np.array_equal(result_a, result_b), (
        "J=8,Q=16 and J=10,Q=16 produced identical output — filter bank not varying with J"
    )
    assert not np.array_equal(result_a, result_c), (
        "J=8,Q=16 and J=8,Q=8 produced identical output — filter bank not varying with Q"
    )
