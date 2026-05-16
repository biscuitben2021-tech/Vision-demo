"""L2CS-Net pretrained gaze model -- the `--pretrained` cursor source.

Wraps the `l2cs` package's `Pipeline` into a small step() API that
takes a BGR cv2 frame and returns `(yaw_deg, pitch_deg)` for the
first detected face -- or None when no face is found or the model
isn't loadable.

The whole module is lazy-load: torch and l2cs are imported only
when `PretrainedGaze.try_load()` is called.  This keeps the default
mouse-or-calibrated path free of the multi-second torch import cost.
A user who doesn't pass --pretrained pays zero overhead for the
opt-in pretrained model.

Weights:
    L2CS-Net's pretrained weights are hosted on Google Drive in a
    folder linked from the L2CS-Net repo README.  We auto-download
    them to `assets/L2CSNet_gaze360.pkl` on first run via the gdown
    package (also installed by --pretrained's pip line); if the
    download fails (no network, gdown not installed, file ID drifted)
    the loader returns None and the OS falls back to the calibrated
    path.

Color-space convention:
    The PretrainedGaze.step() entry point expects a BGR uint8 frame
    -- same convention as every other cv2-facing helper in the
    codebase.  l2cs.Pipeline.step internally converts to RGB for
    PyTorch consumption; that conversion is private to the wrapped
    pipeline and doesn't leak across the BGR/RGB boundary.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final, Optional

import numpy as np


# Where the L2CS gaze weights live on disk.  Relative to the project
# root (the parent of `src/`).  88MB; downloaded on first --pretrained
# run from the L2CS-Net Google Drive folder.
_WEIGHTS_FILENAME: Final[str] = "L2CSNet_gaze360.pkl"

# Google Drive file ID for L2CSNet_gaze360.pkl.  Extracted from the
# L2CS-Net README's "Download the pre-trained models" link.  If the
# file ID drifts, the download will fail with a clear message and
# the user can drop the file at assets/L2CSNet_gaze360.pkl manually.
_WEIGHTS_GDRIVE_ID: Final[str] = "18S956r4jnHtSeT8z8t3z8AoJZjVnNqQS"

# Public Google Drive folder containing all L2CS pretrained models.
# Printed in the install/download error message so the user knows
# where to source the weights manually if auto-download fails.
_WEIGHTS_PUBLIC_URL: Final[str] = (
    "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd"
)


def _project_root() -> Path:
    """Return the project root (the folder containing src/)."""
    return Path(__file__).resolve().parent.parent


def _weights_path() -> Path:
    """Canonical on-disk location for the L2CS weights."""
    return _project_root() / "assets" / _WEIGHTS_FILENAME


def _ensure_weights() -> Optional[Path]:
    """Return the weights file path, downloading from Google Drive if missing.

    Returns:
        Path to the weights file on success, None on any failure
        (network outage, missing gdown, file ID changed, user denied
        write to assets/).  In every failure path the caller falls
        back to the calibrated mode -- the OS doesn't crash.
    """
    weights = _weights_path()
    if weights.is_file():
        return weights

    try:
        import gdown
    except ImportError:
        print(
            "[pretrained] `gdown` is not installed; cannot auto-download "
            "L2CS weights.  Either run `pip install gdown` or place "
            f"L2CSNet_gaze360.pkl manually at: {weights}"
        )
        return None

    weights.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[pretrained] Downloading L2CS weights (~88MB) to {weights}...\n"
        f"             If this fails, grab L2CSNet_gaze360.pkl manually "
        f"from: {_WEIGHTS_PUBLIC_URL}"
    )
    try:
        # gdown.download by ID handles the Google Drive consent
        # interstitial and the large-file warning automatically.
        gdown.download(
            id=_WEIGHTS_GDRIVE_ID, output=str(weights), quiet=False,
        )
    except Exception as exc:                                  # noqa: BLE001
        warnings.warn(f"L2CS weight download failed: {exc}")
        return None

    return weights if weights.is_file() else None


class PretrainedGaze:
    """Thin wrapper around l2cs.Pipeline exposing a step()-based API.

    Construct via `PretrainedGaze.try_load()` -- the direct __init__
    raises on missing deps / weights and is intended only for tests
    that know the install is complete.
    """

    def __init__(self, weights_path: Path, device: str = "cpu") -> None:
        # torch and l2cs are heavy; import here so a plain
        # `import src.pretrained_gaze` doesn't pay the cost.  When
        # this method is called the user already opted in via
        # --pretrained, so the wait is expected.
        import torch
        from l2cs import Pipeline
        self._torch = torch
        # ResNet50 is the architecture L2CS-Net's published Gaze360
        # weights were trained against; the `arch` arg must match or
        # PyTorch raises a tensor-shape mismatch on the first forward
        # pass.
        self._pipeline = Pipeline(
            weights=weights_path,
            arch="ResNet50",
            device=torch.device(device),
        )

    def step(
        self, frame_bgr: np.ndarray,
    ) -> Optional[tuple[float, float]]:
        """Return (yaw_deg, pitch_deg) for the first detected face, or None.

        L2CS-Net's Pipeline runs three stages internally: a face
        detector (RetinaFace), the gaze CNN, and a post-processing
        head that emits angles in degrees.  We surface only the first
        face's angles; any exception (model error, no faces, bad
        frame) is swallowed and returns None so the caller falls
        back to the calibrated cursor.

        Coordinate convention (L2CS-Net's, NOT ours):
            yaw   -- positive when the gaze rotates RIGHT (subject's
                     right, image-mirror's left).  In degrees.
            pitch -- positive when the gaze rotates UP.  In degrees.
        """
        try:
            results = self._pipeline.step(frame_bgr)
        except Exception:                                     # noqa: BLE001
            return None
        if results is None:
            return None
        # results.yaw / results.pitch are numpy arrays with one entry
        # per detected face.  We take the first; if there are zero
        # faces the indexing raises IndexError which we treat as
        # "no usable reading".
        try:
            yaw = float(results.yaw[0])
            pitch = float(results.pitch[0])
        except (IndexError, TypeError, AttributeError):
            return None
        return yaw, pitch

    @classmethod
    def try_load(cls) -> Optional["PretrainedGaze"]:
        """Lazy-load the model.  Returns None on any setup failure.

        Order of operations:
            1. Import l2cs (verifies the pip-installable package is
               present + torch underneath is importable).
            2. Ensure weights are on disk; auto-download if missing.
            3. Construct the Pipeline.

        Each step prints a clear `[pretrained]` message on failure so
        the user knows whether the issue is install, network, or
        model load -- and the OS still boots in the calibrated path
        on a None return.
        """
        try:
            import l2cs  # noqa: F401
        except ImportError as exc:
            print(
                "[pretrained] `l2cs` is not installed -- run "
                "`pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main`."
                f"  ({exc})"
            )
            return None

        weights = _ensure_weights()
        if weights is None:
            return None

        try:
            return cls(weights)
        except Exception as exc:                              # noqa: BLE001
            print(f"[pretrained] Could not initialise L2CS Pipeline: {exc}")
            return None
