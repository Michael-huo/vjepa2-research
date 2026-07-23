"""Repository-relative paths for the fixed research datasets.

Importing this module is intentionally side-effect free: it only constructs
``pathlib.Path`` values and never validates, reads, creates, or writes data.
"""

from __future__ import annotations

from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
RESEARCH_ROOT = _THIS_FILE.parents[2]
REPO_ROOT = RESEARCH_ROOT.parent
ASSETS_ROOT = RESEARCH_ROOT / "assets"
DATASETS_ROOT = ASSETS_ROOT / "datasets"
PREPARED_ROOT = ASSETS_ROOT / "prepared"

BOWLING_VIDEO_PATH = DATASETS_ROOT / "bowling" / "sample_bowling.mp4"
DAVIS_ROOT = DATASETS_ROOT / "davis2017" / "DAVIS"
DAVIS_MANIFEST_PATH = PREPARED_ROOT / "davis2017" / "manifest.json"
TUM_RGBD_ROOT = (
    DATASETS_ROOT
    / "tum_rgbd"
    / "rgbd_dataset_freiburg2_pioneer_slam"
)
TUM_RGBD_MANIFEST_PATH = (
    PREPARED_ROOT
    / "tum_rgbd"
    / "rgbd_dataset_freiburg2_pioneer_slam"
    / "manifest.json"
)
