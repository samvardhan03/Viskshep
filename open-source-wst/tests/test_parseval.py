import numpy as np
import pytest

from vikshep import _vikshep_core as wst

@pytest.mark.skipif(not wst.cuda_available(), reason="CUDA not available")
def test_parseval_frame_energy_conservation():
    cfg = wst.WSTConfig(J=8, Q=16, depth=2, jtfs=False)
    signal_len = 4096
    n_trials = 200
    
    for _ in range(n_trials):
        x = np.random.randn(signal_len).astype(np.float32)
        
        paths = wst.scattering_paths(x, cfg)
        
        energy_in = np.linalg.norm(x)**2
        
        energy_out = 0.0
        for p in paths:
            energy_out += np.linalg.norm(p)**2

        rel_error = abs(energy_out - energy_in) / energy_in
        assert rel_error < 5e-3, f"Parseval violation! Relative error: {rel_error:.2e}"
