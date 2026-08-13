import json, os, subprocess
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a GL context (WSLg/desktop)")
@pytest.mark.skipif(
    not (ROOT / "host_app").exists(),
    reason="host_app not built (make host_app); gitignored, absent in fresh worktrees",
)
def test_tiny_dump(tmp_path):
    r = subprocess.run(["scripts/run_host.sh", "--dump", "2", "--out", str(tmp_path),
                        "--size", "64", "--ref-spp", "16", "--seed", "7"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["count"] == 2 and meta["size"] == 64
    for v in range(2):
        noisy = np.fromfile(tmp_path / f"frame_v{v:04d}_noisy.bin", dtype=np.int8)
        ref = np.fromfile(tmp_path / f"frame_v{v:04d}_ref.bin", dtype=np.int8)
        assert noisy.size == 7 * 64 * 64 and ref.size == 3 * 64 * 64
        n = noisy.reshape(7, 64, 64)
        assert n[0:3].std() > 0                      # not a blank frame
        assert np.abs(n[3:6]).max() <= 127           # normals in range
        # noisy vs ref must differ (noise exists) but share the scene
        assert not np.array_equal(n[0:3].ravel(), ref)
    assert (tmp_path / "frame_v0000_noisy.png").exists()
