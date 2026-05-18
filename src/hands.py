"""
Hand-input module for the Vision OS demo (Phase 8).

This module is the bridge between the webcam + MediaPipe Hands and the
Compositor's drag/click surface.  Three responsibilities live here:

    1. Camera + MediaPipe lifecycle.  Open `cv2.VideoCapture`, configure
       a streaming-mode Hands model, run both off a daemon thread so the
       60 Hz OS render loop never blocks on `cap.read()`.  Without the
       thread, every frame would stall ~33ms waiting on the 30 Hz camera
       and the on-screen motion would visibly drop to 30 fps.
    2. Pinch state machine.  Adapted from the Hand-controller reference
       implementation but with ONE behavioural change for the OS:
       CLICK fires on RELEASE if the gesture never reached DRAGGING.  In
       the reference (which logs events for diagnostic purposes) the
       CLICK fires the moment a pinch is confirmed; for the OS that
       behaviour would cause a swipe-to-change-page gesture to ALSO
       open whichever app the cursor was near when the pinch started.
       Firing on release, only when the gesture stayed tap-shaped, is
       the cure.
    3. Diagnostic camera thumbnail.  A 213x120 (16:9) preview drawn on
       the BOTTOM-RIGHT of the OS canvas, with the 21-point hand mesh
       and colour-coded thumb/index dots overlaid.  Same diagnostic
       widget the reference uses; the user looks here first when a
       pinch doesn't register, because if the dots are orange (not
       green) the system isn't even seeing the pinch -- this rules out
       the OS layer in one glance.

Module color-space convention:
    The camera frame and the OS canvas are BOTH cv2 BGR buffers.  Every
    paint helper in this module accepts a BGR np.ndarray, mutates it in
    place, and returns None.  This matches the convention every other
    cv2-facing module in the codebase uses (src/icons, src/tiles,
    src/compositor).  MediaPipe wants RGB, so `_step_on_thread` does
    one BGR -> RGB convert before `hands.process(...)` -- no other
    site crosses the BGR/RGB boundary.

Threading model:
    `HandInput.__init__` spawns one daemon thread that loops on
    cap.read() -> hands.process() -> _update_state(), holding a
    `threading.Lock` only for the brief moment it copies the latest
    HandFrame candidate into a shared slot.  The OS main thread calls
    `step()` to atomically read that slot; reads never block on the
    camera.  Edge transitions (drag_just_started / drag_just_ended) are
    computed inside the lock from the prior published frame's flags so
    the OS sees each edge exactly once.

Show-day defensiveness:
    Camera open + MediaPipe load can fail (no camera, denied
    permission, mediapipe wheel mismatched).  The constructor raises
    `HandInputUnavailable` with a human-readable reason; the caller
    (phase8_hand.py) catches that exception, prints a warning, and
    proceeds with `hand_input = None`.  The OS still boots in mouse
    mode.  This is the rule the rest of the demo follows: a missing
    sensor never blocks the script.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


# ============================================================================
# Constants
# ============================================================================
#
# Every tunable is here, grouped by concern, with WHY notes.  Values
# match the Hand-controller "final" reference verbatim where the math
# is shared (PINCH_ENTER_RATIO etc.) so behaviour stays consistent with
# the standalone phase7_demo.py the user already validated.
#
# Two NEW tunables live below:
#   THUMB_W / THUMB_H / THUMB_MARGIN: the OS-side thumbnail anchor.  The
#       reference uses the same 213x120 numbers but anchors them on its
#       own black canvas; we re-import them here so the compositor can
#       overlay the thumbnail without round-tripping through the
#       Hand-controller folder.
#   CAMERA_PROBE_LIMIT: how many camera indices to try silently before
#       giving up.  The reference uses an interactive `input()` prompt;
#       the OS demo cannot block the main thread on stdin, so this
#       module probes a small range and picks the first device that
#       actually delivers a frame.

# --- Camera capture ---------------------------------------------------
CAMERA_PROBE_LIMIT: int = 6        # match reference; 0..5 covers built-in + 1-2 externals
CAMERA_W: int = 1280               # 720p; plenty of resolution for hand landmarks
CAMERA_H: int = 720
CAMERA_FPS: int = 30               # requested; the driver picks the nearest supported

# --- MediaPipe Hands model -------------------------------------------
# Two hands so a future Phase can experiment with two-hand gestures
# without re-instantiating the model; this Phase consumes only the
# first detected hand.  Confidence values straight from the reference.
MAX_HANDS: int = 2
MIN_DETECT_CONF: float = 0.7
MIN_TRACK_CONF: float = 0.5

# --- Landmark indices (MediaPipe Hands 21-point model) ---------------
# Same names the reference's gestures.py and phase7_demo.py use.
WRIST: int = 0
THUMB_TIP: int = 4
INDEX_TIP: int = 8
MID_MCP: int = 9                   # base knuckle of the middle finger -- palm-length anchor

# --- Face mesh landmark indices (MediaPipe Face Mesh 478-point model
# with refine_landmarks=True; the iris points 468..477 are ONLY present
# when refine_landmarks is enabled).  Names match the eye-tracking
# reference's phase3_features.py and the project CLAUDE.md so anyone
# moving between the two projects sees the same numbers.  Note: "LEFT"
# and "RIGHT" here are the SUBJECT's anatomical left/right; after we
# cv2.flip the camera frame they appear on the OPPOSITE side of the
# canvas.  For the gaze-baseline math that doesn't matter -- we average
# both eyes' iris-in-eye position, so the swap cancels out.
LEFT_IRIS: int = 468
RIGHT_IRIS: int = 473
LEFT_EYE_OUTER: int = 33           # nearest the ear
LEFT_EYE_INNER: int = 133          # nearest the nose
LEFT_EYE_TOP: int = 159
LEFT_EYE_BOTTOM: int = 145
RIGHT_EYE_OUTER: int = 263
RIGHT_EYE_INNER: int = 362
RIGHT_EYE_TOP: int = 386
RIGHT_EYE_BOTTOM: int = 374

# Face Mesh confidence thresholds.  Lower than the Hands defaults
# because the face is usually larger in frame than a hand and stays
# locked through the demo -- false positives are essentially zero.
FACE_MIN_DETECT_CONF: float = 0.5
FACE_MIN_TRACK_CONF: float = 0.5

# Gaze smoothing.  Raw iris position jitters ~2-5px frame-to-frame even
# when the user is staring at a fixed point; without smoothing the
# cursor twitches enough that hovering a 140x140 tile feels unreliable.
# An exponential moving average is the cheapest fix that still feels
# responsive.  alpha=0.35 is a starting point -- higher = snappier but
# noisier; lower = smoother but laggier.  Tune by feel during the demo.
GAZE_SMOOTH_ALPHA: float = 0.35

# Gaze-to-cursor gains.  When the iris moves ONE unit of normalised
# iris-in-eye position (which spans ~0..1 across the full eye width),
# the cursor moves this many canvas pixels per unit.  The defaults
# below assume a 1920-wide canvas and a typical 0.15-unit iris range
# across the comfortable gaze cone -- so a 0.15 shift produces a
# ~720px cursor movement, roughly 38% of the canvas.  Multiplied by
# the canvas dimensions at runtime to keep behaviour resolution-
# independent.  Vertical gain is higher because the iris-in-eye y
# range is much narrower than the x range (~0.05 vs ~0.15).
GAZE_GAIN_X: float = 5.0
GAZE_GAIN_Y: float = 9.0

# Pretrained-mode projection (L2CS-Net yaw/pitch -> screen pixels).
# Empirical pixels-per-degree for a user at ~50cm from a 13" MacBook:
# the horizontal half-angle is ~18 degrees and the vertical half-
# angle is ~11 degrees, so dividing canvas extent by twice those
# gives ~40 px/deg in both axes.  Used by gaze_cursor's pretrained
# branch when no per-user calibration is available.
PRETRAINED_PX_PER_DEG_X: float = 40.0
PRETRAINED_PX_PER_DEG_Y: float = 40.0
# The MacBook's camera sits above the screen, so a user fixating on
# the screen centre is gazing slightly DOWN from camera level --
# pitch_raw ~= -10 degrees in L2CS's convention (positive pitch =
# looking up).  Adding this OFFSET to the raw pitch shifts the
# zero-point to "looking at screen centre":
#     net_pitch = pitch_raw + offset
#     pitch_raw = -10 + 10 = 0  ->  sy = canvas_h/2 (centre).  GOOD.
# Sign is therefore POSITIVE.  An earlier draft had this negative,
# which sent the cursor off the bottom of the screen for a user
# looking at the centre.
PRETRAINED_PITCH_OFFSET_DEG: float = 10.0

# --- Pinch thresholds (REFERENCE VALUES, do not retune lightly) ------
# Hysteresis: easier to keep a pinch than to start one.  The reference's
# extensive tuning notes apply word-for-word, see Hand controller copy
# final/phase7_demo.py PINCH_ENTER_RATIO comment.  Lower values fire
# more easily; higher values demand a tighter fingertip touch.
PINCH_ENTER_RATIO: float = 0.35
PINCH_EXIT_RATIO: float = 0.50

# --- Click / drag rules ----------------------------------------------
# Two consecutive frames confirms a pinch (single-frame blips never
# represent intent).  After a release, a brief cooldown suppresses any
# re-trigger from finger wobble.  Reference values verbatim.
#
# Drag transitions in when wrist motion exceeds 10% of one hand-length
# between frames -- the "is the whole hand moving" check that
# distinguishes a tap from a drag.
CLICK_CONFIRM_FRAMES: int = 2
CLICK_COOLDOWN_FRAMES: int = 15
DRAG_MOVE_THRESHOLD_NPX: float = 10.0

# --- Camera thumbnail (bottom-right diagnostic widget) ---------------
# 213x120 is 16:9 at ~1/6 of 1280x720; same numbers the reference uses
# so an audience member glancing at both demos sees the same widget.
# Margin 10 from each edge matches the reference's THUMB_MARGIN; the
# 1px white border lives just OUTSIDE the image so it doesn't cover any
# preview pixels.
THUMB_W: int = 213
THUMB_H: int = 120
THUMB_MARGIN: int = 10
_THUMB_BORDER_COLOR_BGR: tuple[int, int, int] = (255, 255, 255)
_THUMB_BORDER_THICKNESS: int = 1

# Skeleton-overlay dot/line sizes on the small thumbnail.  Smaller than
# the reference's MAIN_LANDMARK_RADIUS values because we paint into
# 213x120, not a fullscreen canvas; larger dots would smother the
# underlying camera image.
_THUMB_LANDMARK_RADIUS: int = 2
_THUMB_CONNECTION_THICKNESS: int = 1
_THUMB_SKELETON_COLOR_BGR: tuple[int, int, int] = (255, 255, 255)

# Bigger, colour-coded fingertip highlights -- this is the visual
# feedback the user reads to know whether their pinch is being seen.
# Orange (not pinching) -> green (pinching) is the same colour rule the
# reference's draw_thumb_index_highlights uses.  BGR-ordered.
_TIP_HIGHLIGHT_RADIUS: int = 5
_TIP_NOT_PINCHING_COLOR_BGR: tuple[int, int, int] = (60, 130, 240)    # warm orange
_TIP_PINCHING_COLOR_BGR: tuple[int, int, int] = (90, 220, 90)         # green
_PINCH_LINE_THICKNESS: int = 2

# --- State machine labels --------------------------------------------
_STATE_IDLE: str = "idle"
_STATE_PINCH_HELD: str = "pinch_held"
_STATE_DRAGGING: str = "dragging"


# ============================================================================
# HandFrame -- the one-shot result the OS consumes per render frame
# ============================================================================


@dataclass(frozen=True)
class HandFrame:
    """One snapshot of hand state, returned by `HandInput.step`.

    The OS main loop reads this struct once per render frame and
    dispatches the actions it carries.  Every field is computed inside
    the camera thread under the publish lock; the OS thread only ever
    sees an immutable copy.

    Fields:
        present:           True iff MediaPipe detected at least one
                           hand on the latest processed camera frame.
                           Used by the thumbnail painter to decide
                           whether to overlay the skeleton at all.
        cursor_xy:         index-fingertip position in CANVAS pixels
                           (not camera pixels).  Provided for
                           diagnostic / future use; the OS does NOT
                           currently route hover off this field -- the
                           mouse remains the cursor source per the
                           Phase 8 design.
        click_now:         True for exactly one frame at the moment a
                           tap gesture (pinch -> release without drag)
                           completes.  The OS treats this as a left
                           click at the CURRENT mouse cursor position.
        drag_active:       True every frame the hand is currently in
                           the DRAGGING state.  False on the frame the
                           drag ends.
        drag_dx:           Cumulative horizontal wrist displacement
                           since the drag started, in canvas pixels.
                           Sign convention: right hand moving RIGHT on
                           screen produces POSITIVE dx (the camera
                           frame is mirrored before MediaPipe sees it,
                           so screen-right matches user-right).
        drag_dy:           Cumulative vertical displacement; kept for
                           completeness even though the OS only
                           consumes dx today.
        drag_just_started: True for exactly one frame at the moment
                           the state machine enters DRAGGING.  The OS
                           uses this to call `begin_page_drag` once
                           per gesture, not every frame.
        drag_just_ended:   True for exactly one frame at the moment
                           the drag ends (pinch released while in
                           DRAGGING).  Used to call `end_page_drag`.
        is_pinching:       True iff the latest frame's pinch ratio is
                           below threshold.  Used only by the
                           thumbnail painter to colour the fingertip
                           dots.  Distinct from click/drag because
                           we want the visual feedback to flip the
                           instant a pinch is RECOGNISED, not when
                           CONFIRMED (CLICK_CONFIRM_FRAMES later).
    """

    present: bool
    cursor_xy: tuple[int, int]
    click_now: bool
    drag_active: bool
    drag_dx: int
    drag_dy: int
    drag_just_started: bool
    drag_just_ended: bool
    is_pinching: bool


def _empty_hand_frame() -> HandFrame:
    """Return the all-zero HandFrame used when no hand is detected.

    Sentinel value: cursor_xy of (-1, -1) matches the off-canvas
    convention every other input source in this codebase uses
    (phase5_motion's _MouseContext seeds the mouse position the same
    way).  closest_tile() returns None for negative coords, so a
    cursor reading of (-1, -1) is the correct "no hover" input.
    """
    return HandFrame(
        present=False,
        cursor_xy=(-1, -1),
        click_now=False,
        drag_active=False,
        drag_dx=0,
        drag_dy=0,
        drag_just_started=False,
        drag_just_ended=False,
        is_pinching=False,
    )


# ============================================================================
# Pinch state machine -- OS-tuned variant of the reference's PinchTracker
# ============================================================================
#
# Behavioural delta from `Hand controller copy final/phase7_demo.py`:
# the reference fires CLICK the moment a pinch is confirmed.  That is
# correct for a diagnostic logger but WRONG for an OS where the same
# pinch might become a drag-to-swipe-pages gesture: we'd open whatever
# app the cursor was over when the drag started.
#
# The fix is two lines of state and one branch on release:
#   * `_became_drag: bool` flag, set when the state machine enters
#     DRAGGING for the FIRST time within a single pinch.
#   * On release, emit CLICK only if `_became_drag` stayed False.
#     Otherwise the gesture was a drag and the click is suppressed.
#
# Everything else (hysteresis, hand-size normalisation, cooldown,
# confirmation frame count) carries over from the reference verbatim
# so the feel matches what the user already validated.


@dataclass
class _PinchTracker:
    """Internal pinch state machine.  One instance per `HandInput`.

    Fields mirror the reference's PinchTracker with three additions
    that the OS needs:
        _became_drag:        True iff the current pinch has at any
                             point transitioned to DRAGGING.  Decides
                             whether a release emits CLICK.
        cumulative_dx/dy:    cumulative wrist displacement since drag
                             start, in canvas pixels.  The reference
                             measures per-frame motion only; the OS
                             needs the integrated value to drive the
                             page-swipe offset.
        wrist_at_drag_start: wrist position at the moment the state
                             machine enters DRAGGING.  cumulative_dx/dy
                             is the wrist's current position minus
                             this.  We do NOT integrate per-frame
                             deltas because integration drift would
                             accumulate across the whole gesture.
    """

    state: str = _STATE_IDLE
    held_frames: int = 0
    cooldown_frames: int = 0
    _became_drag: bool = False
    last_wrist_pos: tuple[int, int] = (0, 0)
    wrist_at_drag_start: tuple[int, int] = (0, 0)
    cumulative_dx: int = 0
    cumulative_dy: int = 0


def _pinch_ratio(points: list[tuple[int, int]]) -> float:
    """Return the unitless pinch ratio used to classify pinches.

    ratio = dist(thumb_tip, index_tip) / dist(wrist, middle_mcp).  The
    denominator is the palm length, which is a stable size reference
    even as the fingers open and close.  See the long WHY block in
    `Hand controller copy final/phase7_demo.py` for the threshold
    derivation; the values are reused verbatim above.

    Returns a large sentinel (1.0) if the palm collapsed to a single
    pixel -- a rare degenerate MediaPipe frame -- so that frame can
    never accidentally register as a pinch.
    """
    pinch_dist = math.hypot(
        points[THUMB_TIP][0] - points[INDEX_TIP][0],
        points[THUMB_TIP][1] - points[INDEX_TIP][1],
    )
    hand_size = math.hypot(
        points[WRIST][0] - points[MID_MCP][0],
        points[WRIST][1] - points[MID_MCP][1],
    )
    if hand_size < 1.0:
        return 1.0
    return pinch_dist / hand_size


def _normalised_wrist_displacement(
    wrist: tuple[int, int],
    last_wrist: tuple[int, int],
    hand_size_px: float,
) -> float:
    """Per-frame wrist motion, normalised by hand size, scaled to "npx" units.

    "npx" stands for "normalised pixels": displacement (canvas pixels)
    divided by hand size (canvas pixels), then x100.  A value of 10
    means "the wrist moved at least 10% of one hand-length between
    frames", which is camera-distance independent and matches the
    DRAG_MOVE_THRESHOLD_NPX threshold above.

    Returns 0.0 for degenerate frames (hand_size < 1px); the caller
    treats 0.0 as "do not transition to DRAGGING this frame".
    """
    if hand_size_px < 1.0:
        return 0.0
    dx = wrist[0] - last_wrist[0]
    dy = wrist[1] - last_wrist[1]
    return (math.hypot(dx, dy) / hand_size_px) * 100.0


# ============================================================================
# Geometry helpers -- convert MediaPipe landmarks to canvas pixels
# ============================================================================


def _landmarks_to_points(
    hand_landmarks: Any,
    target_w: int,
    target_h: int,
) -> list[tuple[int, int]]:
    """Convert MediaPipe's 21 normalised landmarks to integer pixel coords.

    MediaPipe returns every landmark in [0, 1] relative to the camera
    frame.  Multiplying by the target image size (which can be either
    the OS canvas or the thumbnail) gives pixel coords in that target's
    coordinate system.

    Out-of-range landmarks (MediaPipe occasionally extrapolates a
    partially occluded hand past the frame edge) are clipped to keep
    every dot drawn on the image.  Without this clipping, the
    skeleton on the thumbnail can poke off the bottom edge when the
    user's hand drops below the camera's field of view.
    """
    points: list[tuple[int, int]] = []
    for lm in hand_landmarks.landmark:
        px = int(lm.x * target_w)
        py = int(lm.y * target_h)
        px = max(0, min(target_w - 1, px))
        py = max(0, min(target_h - 1, py))
        points.append((px, py))
    return points


def _normalised_iris_in_eye(
    iris_x: float, iris_y: float,
    inner_x: float, outer_x: float,
    top_y: float, bottom_y: float,
) -> tuple[float, float]:
    """Return the iris position as a fraction of the eye-socket box.

    Returns (x_norm, y_norm) where:
        x_norm = 0  -> iris at the inner corner (nose side)
        x_norm = 1  -> iris at the outer corner (ear side)
        y_norm = 0  -> iris at the top lid
        y_norm = 1  -> iris at the bottom lid

    Forward-facing gaze sits near (0.5, 0.5).  Normalising in this
    way makes the reading invariant to where the face is in the
    frame and how big the face appears -- only the iris's position
    relative to its eye-socket survives.

    Degenerate frames (eye closed flat, lids collapse to a point)
    would divide by zero; we guard with `or 1e-9` so the math stays
    finite and the frame just produces a garbage reading that the
    EMA smooths over.  The reference uses the same guard.
    """
    eye_w = outer_x - inner_x or 1e-9
    eye_h = bottom_y - top_y or 1e-9
    return (iris_x - inner_x) / eye_w, (iris_y - top_y) / eye_h


def _compute_gaze_norm(
    face_list: list[Any], frame_w: int, frame_h: int,
) -> Optional[tuple[float, float]]:
    """Average iris-in-eye position across both eyes; None if no face.

    `face_list` is the `multi_face_landmarks` field straight off a
    MediaPipe FaceMesh result, with `refine_landmarks=True` (required
    for iris landmarks 468..477).  We average the two eyes' readings
    to halve the noise -- iris-in-eye jitter is roughly independent
    per eye, so the mean has ~sqrt(2) less variance than either eye
    alone.

    Returns None on no-face so the publisher knows to hold the last
    reading.  Frame width/height are passed in (rather than read
    from the landmarks themselves) because the landmarks are
    already-normalised; multiplying by frame dims gives pixel coords
    consistent with the camera buffer.
    """
    if not face_list:
        return None
    lms = face_list[0].landmark

    def at(idx: int) -> tuple[float, float]:
        # Pixel coords in the camera frame.  We do NOT need to clip
        # here because the reading is invariant under uniform scale.
        return lms[idx].x * frame_w, lms[idx].y * frame_h

    lix, liy = at(LEFT_IRIS)
    rix, riy = at(RIGHT_IRIS)
    l_inner = at(LEFT_EYE_INNER)
    l_outer = at(LEFT_EYE_OUTER)
    l_top = at(LEFT_EYE_TOP)
    l_bottom = at(LEFT_EYE_BOTTOM)
    r_inner = at(RIGHT_EYE_INNER)
    r_outer = at(RIGHT_EYE_OUTER)
    r_top = at(RIGHT_EYE_TOP)
    r_bottom = at(RIGHT_EYE_BOTTOM)

    lx, ly = _normalised_iris_in_eye(
        lix, liy, l_inner[0], l_outer[0], l_top[1], l_bottom[1],
    )
    rx, ry = _normalised_iris_in_eye(
        rix, riy, r_inner[0], r_outer[0], r_top[1], r_bottom[1],
    )
    return ((lx + rx) * 0.5, (ly + ry) * 0.5)


# ============================================================================
# HandInputUnavailable -- the constructor's only failure mode
# ============================================================================


class HandInputUnavailable(RuntimeError):
    """Raised when the camera or MediaPipe cannot be brought up.

    Caller (phase8_hand.py) catches this, prints a clear warning, and
    proceeds in mouse-only mode.  The exception message is the
    human-readable reason -- "No working camera was detected on
    indices 0..5" or "mediapipe failed to import" etc. -- so the
    operator at the show can spot the cause from a glance at the
    terminal.

    A subclass of RuntimeError (rather than a plain Exception) so a
    `except Exception:` upstream still catches it but `except RuntimeError:`
    catches it specifically when needed.
    """


# ============================================================================
# Public probe helper -- used by phase8_hand.py's interactive prompt
# ============================================================================
#
# The OS entry script wants the SAME terminal-prompt flow the Hand
# controller reference uses ("here are the cameras I found, type the
# index"), so we expose the probe as a module-level function callable
# BEFORE any HandInput is constructed.  Doing the probe pre-construction
# lets the user pick the right device once, in the terminal, before the
# fullscreen window grabs focus.
#
# Each probe opens a VideoCapture, reads ONE frame (the only reliable
# proof the device works -- isOpened() returns True for some stale
# device handles that never deliver a frame), records the realised
# (width, height), then releases the capture.  Releasing is critical:
# a held-open probe would block the subsequent HandInput.__init__
# call from opening the same index on macOS.


def list_available_cameras(
    max_to_probe: int = CAMERA_PROBE_LIMIT,
) -> list[tuple[int, int, int]]:
    """Probe camera indices 0..max_to_probe-1 and return the working ones.

    Each entry is `(index, width, height)`.  Width/height come from a
    real frame read (not driver-reported metadata) so the returned
    dimensions reflect what OpenCV will actually deliver to MediaPipe.

    Pure side-effect: opens and immediately releases each
    VideoCapture, so no lingering handle blocks the subsequent
    HandInput construction.  Safe to call multiple times.
    """
    found: list[tuple[int, int, int]] = []
    for idx in range(max_to_probe):
        cap = cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            found.append((idx, int(w), int(h)))
        finally:
            # Always release -- leaving a probed cam open would block
            # the real open in HandInput.__init__ below from grabbing
            # the same device on macOS.
            cap.release()
    return found


# ============================================================================
# HandInput -- the public class the OS interacts with
# ============================================================================


class HandInput:
    """Camera + MediaPipe + pinch detection, wrapped behind a non-blocking step().

    Construction:
        HandInput(camera_index=0) opens the camera at the given index
        and spins up the background thread.  Pass camera_index=None to
        probe 0..CAMERA_PROBE_LIMIT-1 and silently pick the first
        device that delivers a frame.  Raises HandInputUnavailable if
        no camera works.

    Per-frame API:
        step(canvas_w, canvas_h) -> HandFrame
            Non-blocking.  Reads the latest published HandFrame
            candidate (or _empty_hand_frame() if the thread has not
            produced one yet) under the lock and returns it.  Also
            updates the canvas-size hint the thread uses on its NEXT
            iteration so cursor_xy / drag_dx are computed against the
            OS's true canvas size.

    Thumbnail API:
        draw_thumbnail(canvas) -> None
            Paint the 213x120 camera preview + skeleton overlay into
            the bottom-right of `canvas`.  Reads the most recent
            camera frame and the most recent set of MediaPipe
            landmarks under the lock.  Safe to call from the OS thread.

    Shutdown:
        close()
            Sets the stop flag, joins the thread, releases the
            camera, closes the MediaPipe model.  Idempotent -- safe
            to call multiple times; subsequent calls are no-ops.

    Implementation notes:
        * The lock is held only for tiny copies -- the np.ndarray
          frame is published by reference (the thread allocates a
          fresh array each iteration, so the OS reader never sees a
          half-written buffer).
        * The thread reads the canvas dimensions from a small
          shared tuple each iteration; the OS writes that tuple in
          `step()`.  No need for a lock on the tuple: Python's GIL
          makes a single attribute write atomic and we tolerate the
          one-frame lag on a resize.
    """

    # ------------------------------------------------------------------
    # Construction / open
    # ------------------------------------------------------------------

    def __init__(
        self,
        camera_index: Optional[int] = None,
        pretrained_gaze: Optional["PretrainedGaze"] = None,
    ) -> None:
        """Open the camera and start the background thread.

        If `camera_index` is None, probe indices 0..CAMERA_PROBE_LIMIT-1
        silently and pick the first device that delivers a frame.  This
        avoids the reference's interactive prompt, which would block
        the OS's main thread on stdin.  The user can override the auto
        pick by passing `--camera N` on the CLI (phase8_hand.py wires
        that flag in).

        If `pretrained_gaze` is non-None (a `PretrainedGaze` instance
        loaded by the caller via `PretrainedGaze.try_load()`), the
        background thread additionally runs L2CS-Net's gaze CNN on
        each camera frame and stores `(yaw_deg, pitch_deg)` for the
        gaze_cursor projection -- `--pretrained` mode in phase 8.
        Construction does NOT load the model; pass an already-loaded
        instance.  When pretrained gaze is unavailable the thread
        skips that branch and the calibrated iris-norm path is used.

        Raises HandInputUnavailable on any open failure.  The caller is
        expected to fall back to mouse-only mode.
        """
        # Local import: keep mediapipe out of module-level imports so a
        # plain `import src.hands` does not fail just because mediapipe
        # is missing.  We let the caller decide whether to attempt
        # construction; if mediapipe is unimportable the exception
        # bubbles up here and the caller catches it.
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - defensive
            raise HandInputUnavailable(
                "mediapipe is not installed; "
                "install it or run phase7_polish.py for mouse-only mode."
            ) from exc

        self._mp = mp
        self._cap: Optional[cv2.VideoCapture] = None
        self._chosen_index: int = -1

        if camera_index is None:
            self._cap, self._chosen_index = self._auto_open_camera()
        else:
            self._cap = self._try_open_camera(camera_index)
            if self._cap is None:
                raise HandInputUnavailable(
                    f"Camera at index {camera_index} did not deliver a frame. "
                    "On macOS, check System Settings -> Privacy & Security -> "
                    "Camera and confirm your terminal / IDE has access."
                )
            self._chosen_index = camera_index

        # Configure capture geometry / fps.  cap.set() calls are
        # advisory -- the driver picks the closest supported mode.
        # We do not verify the actual values because MediaPipe handles
        # any source resolution and we'll see what we get on the
        # thumbnail.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # BUG FIX: release the camera + any already-built MediaPipe
        # model if model construction below raises.  Without this
        # cleanup a mismatched mediapipe wheel (the only realistic way
        # the Hands or FaceMesh constructor fails after the cap is
        # open) would leak the camera handle, blocking the next
        # process from opening the same device on macOS until the
        # interpreter exits.
        self._hands_model = None
        self._face_mesh = None
        try:
            # MediaPipe Hands.  static_image_mode=False enables tracking
            # between frames, which is faster than per-frame detection and
            # gives smoother landmark trajectories -- important for our
            # per-frame wrist-displacement test.
            self._hands_model = self._mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_HANDS,
                min_detection_confidence=MIN_DETECT_CONF,
                min_tracking_confidence=MIN_TRACK_CONF,
            )
            # mp_hands.HAND_CONNECTIONS is a list of (a, b) index pairs
            # describing the skeleton's bones.  Cached here so the
            # thumbnail painter does not look it up under the lock.
            self._connections = self._mp.solutions.hands.HAND_CONNECTIONS

            # MediaPipe Face Mesh -- ALSO run on each camera frame so the
            # gaze (iris-in-eye normalised position) can drive the OS
            # cursor.  Sharing the same VideoCapture + the same background
            # thread is the only way that works on macOS, where two
            # simultaneous VideoCapture instances on the same device fight
            # over the AVFoundation backing and one of them starves.
            # refine_landmarks=True is REQUIRED -- without it the iris
            # landmarks (468..477) don't exist in the output.
            self._face_mesh = self._mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=FACE_MIN_DETECT_CONF,
                min_tracking_confidence=FACE_MIN_TRACK_CONF,
            )
        except Exception as exc:                              # noqa: BLE001
            # Release everything already acquired before re-raising as
            # HandInputUnavailable.  The caller's `except
            # HandInputUnavailable` path expects a clean slate to fall
            # back to mouse-only mode.
            if self._hands_model is not None:
                try:
                    self._hands_model.close()
                except Exception:                             # noqa: BLE001
                    pass
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            raise HandInputUnavailable(
                f"MediaPipe model construction failed: {exc}"
            ) from exc

        # Shared state guarded by `self._lock`.  The thread fills
        # these; step() and draw_thumbnail() read them.
        self._lock = threading.Lock()
        self._latest_frame: HandFrame = _empty_hand_frame()
        self._latest_camera_bgr: Optional[np.ndarray] = None
        self._latest_landmarks: list[Any] = []   # list of mediapipe NormalizedLandmarkList
        self._latest_is_pinching: bool = False

        # Gaze state.  All values are normalised iris-in-eye position,
        # averaged across the two eyes:
        #     0.5 ~= iris centred in the eye-socket (forward gaze)
        #     <0.5 = iris shifted toward the nose
        #     >0.5 = iris shifted toward the ear
        # Vertical is similar (0.5 ~= middle of the lid opening).
        # `_latest_gaze_norm` is None when no face is detected.
        # `_gaze_baseline` is None until the user runs the calibration
        # screen and captures their "looking at the center" reading.
        # `_smoothed_gaze_norm` carries the EMA across frames so the
        # cursor doesn't jitter on raw landmark noise.
        self._latest_gaze_norm: Optional[tuple[float, float]] = None
        self._smoothed_gaze_norm: Optional[tuple[float, float]] = None
        self._gaze_baseline: Optional[tuple[float, float]] = None
        # 5-point calibration: list of (gaze_norm, screen_xy) samples
        # captured by `add_calibration_sample`, then a 2x3 affine
        # matrix fitted by `fit_calibration` via numpy.linalg.lstsq.
        # When the matrix is present, `gaze_cursor` uses it INSTEAD of
        # the legacy `_gaze_baseline` delta path -- the multi-point
        # affine is strictly better (corrects per-axis gain and any
        # roll between iris and screen) but the single-baseline path
        # is kept as a fallback for calibration runs that captured
        # fewer than 3 samples.
        self._calibration_samples: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []
        self._calibration_matrix: Optional[np.ndarray] = None

        # Pretrained-mode state.  The L2CS Pipeline (when provided)
        # runs on the camera thread alongside MediaPipe; the latest
        # yaw/pitch is stored here as a (yaw_deg, pitch_deg) tuple,
        # or None when no face was detected on that frame.  The
        # gaze_cursor projection consumes this when both fields are
        # populated -- pretrained takes precedence over the
        # calibrated affine.
        self._pretrained_gaze = pretrained_gaze
        self._latest_pretrained_yp: Optional[tuple[float, float]] = None
        # Face landmarks list -- one mediapipe NormalizedLandmarkList
        # when a face is detected, [] otherwise.  Used by the
        # calibration UI's overlay.
        self._latest_face_landmarks: list[Any] = []

        # Canvas-size hint (the OS canvas the cursor and drag deltas
        # are computed against).  step() updates this; the thread
        # reads it each iteration.  Default to 1920x1080 so the first
        # thread iteration -- before step() has run -- produces
        # sensible coords.
        self._canvas_w: int = 1920
        self._canvas_h: int = 1080

        # Tracker state lives on the background thread; only its
        # outputs are published via _latest_frame.
        self._tracker = _PinchTracker()
        # Edge-detection seeds.  The thread uses these to compute
        # drag_just_started / drag_just_ended -- it diffs the previous
        # iteration's drag_active flag against the current one.
        self._prev_drag_active: bool = False

        # Thread control.
        self._stop_event = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._thread_loop, name="HandInputThread", daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _try_open_camera(index: int) -> Optional[cv2.VideoCapture]:
        """Open the camera at `index` and confirm it delivers at least one frame.

        cv2.VideoCapture.isOpened() returns True for some stale device
        entries that never actually deliver a frame, which is why we
        read once -- a successful read is the only reliable proof the
        device works.

        Releases the capture and returns None on any failure.  Callers
        treat None as "skip this index".
        """
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        return cap

    def _auto_open_camera(self) -> tuple[cv2.VideoCapture, int]:
        """Probe 0..CAMERA_PROBE_LIMIT-1 and return the first working capture.

        Silent: no interactive prompt.  The OS demo can run unattended
        on a stage; an `input()` call would freeze the script.  The
        operator can pass --camera N on the CLI to override the
        auto-picked device.

        Raises HandInputUnavailable if no probed index works.
        """
        for idx in range(CAMERA_PROBE_LIMIT):
            cap = self._try_open_camera(idx)
            if cap is not None:
                return cap, idx
        raise HandInputUnavailable(
            "No working camera detected on indices 0.."
            f"{CAMERA_PROBE_LIMIT - 1}.  On macOS, check System Settings -> "
            "Privacy & Security -> Camera permissions."
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def camera_index(self) -> int:
        """The camera index this HandInput opened.  -1 if construction failed."""
        return self._chosen_index

    # ------------------------------------------------------------------
    # Gaze API -- read by the calibration screen and the OS main loop
    # ------------------------------------------------------------------

    def latest_gaze_norm(self) -> Optional[tuple[float, float]]:
        """Return the latest smoothed (x_norm, y_norm) iris-in-eye reading.

        None when no face has been detected yet.  Both axes sit near
        0.5 for a forward gaze; offsets toward 0 or 1 indicate the
        iris has shifted left/right (x) or up/down (y) within the
        eye-socket box.  The reading is averaged across both eyes and
        EMA-smoothed by GAZE_SMOOTH_ALPHA -- the same reading the
        OS cursor uses.

        Calibration UI uses this to display a live readout so the
        user can see numbers moving when they look around the screen.
        """
        with self._lock:
            return self._smoothed_gaze_norm

    def latest_face_landmarks(self) -> list[Any]:
        """Return the latest face-mesh landmark lists (one entry per face).

        Empty list when no face is detected.  The calibration UI's
        eye-overlay reads this to paint dots on the iris and eye
        corners; the OS does not consume it.
        """
        with self._lock:
            return list(self._latest_face_landmarks)

    def calibrate_gaze_center(self) -> bool:
        """Legacy single-point calibration: snapshot smoothed gaze as the centre baseline.

        Returns True on success, False if no face is currently detected.
        Kept for backward compatibility with callers that still issue
        a single SPACE press.  New code should use the 5-point flow
        (`reset_calibration` + `add_calibration_sample` * N +
        `fit_calibration`); the multi-point path produces a strictly
        more accurate cursor model.
        """
        with self._lock:
            if self._smoothed_gaze_norm is None:
                return False
            self._gaze_baseline = self._smoothed_gaze_norm
        return True

    def reset_calibration(self) -> None:
        """Drop every captured sample and the fitted affine matrix.

        Call once at the start of a 5-point calibration run so a
        previous calibration's samples don't pollute the new fit.
        Also clears the legacy single-point baseline; the two paths
        are mutually exclusive at gaze-cursor time but resetting
        both makes the calibration screen's behaviour unambiguous.
        """
        with self._lock:
            self._calibration_samples.clear()
            self._calibration_matrix = None
            self._gaze_baseline = None

    def add_calibration_sample(
        self, screen_xy: tuple[int, int],
    ) -> bool:
        """Record one (current_gaze_norm, screen_target_xy) calibration pair.

        Returns True on a successful capture, False if no face has been
        detected yet (smoothed gaze is None).  Re-tryable: the caller
        re-invokes after the user moves their head into the camera's
        view.  Does NOT fit the model -- accumulates only.  Call
        `fit_calibration` once all samples have been collected.

        Thread-safe: the gaze reading and the samples list are both
        accessed under `_lock`.  The screen_xy tuple is stored
        verbatim; callers are responsible for passing canvas-pixel
        coordinates that match what the OS will use at runtime.
        """
        with self._lock:
            current = self._smoothed_gaze_norm
            if current is None:
                return False
            self._calibration_samples.append((current, screen_xy))
        return True

    def fit_calibration(self) -> bool:
        """Fit the 2x3 affine matrix from gaze_norm to screen pixels.

        Requires at least 3 samples (the affine has 3 unknowns per
        output axis; fewer samples are underdetermined and
        np.linalg.lstsq would give junk).  With 5 samples the system
        is overdetermined and lstsq returns the least-squares
        minimum-error fit -- which is what we want against the
        noisy smoothed-gaze readings.

        Returns True on a successful fit (matrix stored on `self`),
        False when there aren't enough samples OR the captured
        samples don't span enough of the gaze-norm plane to
        determine the affine uniquely (rank-deficient design matrix
        -- e.g. the user kept their eyes still and every sample has
        the same gaze_norm).  The previous matrix is preserved on
        failure so a partial recalibration doesn't blow away the
        old model.
        """
        with self._lock:
            samples = list(self._calibration_samples)
        if len(samples) < 3:
            return False

        # Design matrix: each row is [gaze_x, gaze_y, 1] so the third
        # column captures the bias term (the (0, 0) gaze_norm maps to
        # some non-zero screen position).  lstsq solves Ax = b for each
        # output axis independently.
        a_rows = np.array(
            [[gx, gy, 1.0] for (gx, gy), _ in samples], dtype=np.float64,
        )
        # BUG FIX: reject rank-deficient designs.  np.linalg.lstsq with
        # rcond=None silently returns a minimum-norm solution on a
        # rank-deficient input -- which means an affine fit that
        # collapses the whole gaze-norm plane onto a single screen
        # point.  After such a "fit", gaze_cursor() would clamp every
        # reading to roughly the same corner regardless of where the
        # user looks.  Insisting on full rank 3 (the only rank that
        # produces a unique solution for a 3-unknown system) makes
        # fit_calibration fail loudly instead -- and the OS falls
        # back to the mouse, which is the right behaviour when the
        # captured samples don't actually describe a varying gaze.
        if np.linalg.matrix_rank(a_rows) < 3:
            return False
        y_x = np.array([sx for _, (sx, _sy) in samples], dtype=np.float64)
        y_y = np.array([sy for _, (_sx, sy) in samples], dtype=np.float64)
        try:
            p_x, *_ = np.linalg.lstsq(a_rows, y_x, rcond=None)
            p_y, *_ = np.linalg.lstsq(a_rows, y_y, rcond=None)
        except np.linalg.LinAlgError:
            # Defensive: numpy can still raise LinAlgError on some
            # pathological inputs even though we checked rank above.
            # Treat any solver failure the same as rank deficiency.
            return False

        with self._lock:
            self._calibration_matrix = np.stack([p_x, p_y], axis=0)
        return True

    def has_gaze_calibration(self) -> bool:
        """Return True if any cursor-driving gaze source is configured.

        Three sources count as 'calibrated':
            * The pretrained L2CS Pipeline (no per-user calibration
              needed; the model's yaw/pitch IS the cursor source).
            * The 5-point affine matrix (per-user calibration).
            * The legacy single-point baseline (back-compat).

        Callers use this to gate "show the gaze cursor" -- if none
        of the three exist, the OS falls back to the mouse.
        """
        with self._lock:
            return (
                self._pretrained_gaze is not None
                or self._calibration_matrix is not None
                or self._gaze_baseline is not None
            )

    def gaze_cursor(
        self, canvas_w: int, canvas_h: int,
    ) -> Optional[tuple[int, int]]:
        """Return the gaze-driven cursor position in canvas pixels.

        Resolution order (first match wins):
            1. If the pretrained L2CS Pipeline is wired in AND has
               produced a yaw/pitch for the most recent frame, project
               those angles to screen pixels:
                   screen_x = canvas_w/2 + yaw_deg * px_per_deg_x
                   screen_y = canvas_h/2 - (pitch + offset) * px_per_deg_y
               Pretrained takes precedence because the whole point
               of --pretrained is "skip the per-user calibration".
            2. Else if the 5-point affine matrix is fitted, apply it:
                   screen_x = M[0,0]*gx + M[0,1]*gy + M[0,2]
                   screen_y = M[1,0]*gx + M[1,1]*gy + M[1,2]
            3. Else if a single-point baseline exists, use the legacy
                   dx_norm = current - baseline
                   cursor = canvas_centre + dx_norm * GAZE_GAIN * canvas_size
               math.
            4. Else return None -- no calibration available, the OS
               falls back to the mouse.

        Always clamped to canvas bounds so a wild reading can't
        crash the hover hit-test with a negative index.  Returns None
        when the source is alive but no current reading exists (face
        missing for that frame), in which case the OS falls back to
        the mouse for one frame.
        """
        with self._lock:
            current        = self._smoothed_gaze_norm
            matrix         = self._calibration_matrix
            baseline       = self._gaze_baseline
            pretrained_yp  = self._latest_pretrained_yp
            has_pretrained = self._pretrained_gaze is not None

        # 1. Pretrained branch -- preferred when the model is loaded.
        if has_pretrained:
            if pretrained_yp is None:
                return None
            yaw_deg, pitch_deg = pretrained_yp
            sx = canvas_w * 0.5 + yaw_deg * PRETRAINED_PX_PER_DEG_X
            # pitch positive = looking up; screen y goes DOWN, so we
            # subtract.  PITCH_OFFSET compensates for the laptop
            # camera being above the screen (user looks at screen
            # centre with a slight downward pitch).
            sy = canvas_h * 0.5 - (
                pitch_deg + PRETRAINED_PITCH_OFFSET_DEG
            ) * PRETRAINED_PX_PER_DEG_Y
        elif current is None:
            # Iris-based paths both need a current smoothed reading.
            return None
        elif matrix is not None:
            gx, gy = current
            sx = float(matrix[0, 0] * gx + matrix[0, 1] * gy + matrix[0, 2])
            sy = float(matrix[1, 0] * gx + matrix[1, 1] * gy + matrix[1, 2])
        elif baseline is not None:
            dx_norm = current[0] - baseline[0]
            dy_norm = current[1] - baseline[1]
            sx = canvas_w * 0.5 + dx_norm * GAZE_GAIN_X * canvas_w
            sy = canvas_h * 0.5 + dy_norm * GAZE_GAIN_Y * canvas_h
        else:
            return None

        # BUG FIX: guard against NaN/Inf before int(round(...)).  L2CS
        # forward passes can emit NaN on degenerate frames, and a
        # rank-deficient (or extreme-input) affine fit can produce
        # values outside the float-to-int safe range -- in either
        # case `int(round(nan))` raises ValueError and the OS frame
        # crash-loops.  Return None on a non-finite reading so the
        # caller falls back to the mouse for one frame; the next
        # iteration's reading is usually fine.
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return None
        cursor_x = int(round(sx))
        cursor_y = int(round(sy))
        cursor_x = max(0, min(canvas_w - 1, cursor_x))
        cursor_y = max(0, min(canvas_h - 1, cursor_y))
        return cursor_x, cursor_y

    def latest_camera_frame_bgr(self) -> Optional[np.ndarray]:
        """Return the most recent (mirrored) camera BGR frame, or None.

        The calibration UI consumes this to show the live camera
        feed at calibration time.  We return a reference, not a
        copy -- the camera thread allocates a fresh ndarray every
        iteration so the OS reader never observes a half-written
        buffer.  Callers MUST treat the returned array as
        read-only; mutating it would race with the next frame.
        """
        with self._lock:
            return self._latest_camera_bgr

    def step(self, canvas_w: int, canvas_h: int) -> HandFrame:
        """Return the latest HandFrame.  Non-blocking.

        Updates the canvas-size hint the camera thread uses on its NEXT
        iteration -- the size only matters for landmark-to-canvas
        scaling and we tolerate a one-frame lag on resizes (those are
        rare in steady state).

        Acquires the lock for a single attribute read; on a 60 Hz main
        loop the worst-case contention is microseconds.

        Edge-flag consumption (BUG FIX): the camera thread runs at
        ~30 Hz while the OS main loop runs at 60 Hz, so step() can be
        called twice per camera frame.  Without explicit consumption
        the edge flags (click_now, drag_just_started, drag_just_ended)
        would be re-read on the second step() call and the OS would
        double-fire the corresponding events (two clicks per pinch,
        two begin_page_drag's per drag).  We clear the edge flags
        in-place after reading so a subsequent step() call before the
        camera thread advances sees them as False -- the camera
        thread overwrites the whole frame on its next publish, so no
        information is lost.  drag_active is a LEVEL not an edge and
        is intentionally NOT cleared here.
        """
        # Publish the latest canvas size to the thread.  Single-attribute
        # writes on Python ints are atomic under the GIL, so we don't
        # need to hold the lock for the assignment.  Reading the
        # current HandFrame DOES need the lock because the assignment
        # to self._latest_frame is on a non-atomic Python object slot
        # (well, the reference assignment IS atomic on CPython, but
        # explicitly locking documents intent and is robust to future
        # interpreter changes).
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        with self._lock:
            frame = self._latest_frame
            # BUG FIX: clear edge flags so a second step() call before
            # the camera thread advances cannot replay the same click /
            # drag-start / drag-end.  The frame we return still carries
            # the original edge values; only the stored copy is reset.
            if (frame.click_now
                    or frame.drag_just_started
                    or frame.drag_just_ended):
                self._latest_frame = HandFrame(
                    present=frame.present,
                    cursor_xy=frame.cursor_xy,
                    click_now=False,
                    drag_active=frame.drag_active,
                    drag_dx=frame.drag_dx,
                    drag_dy=frame.drag_dy,
                    drag_just_started=False,
                    drag_just_ended=False,
                    is_pinching=frame.is_pinching,
                )
            return frame

    def draw_thumbnail(self, canvas: np.ndarray) -> None:
        """Paint the 213x120 camera preview + skeleton into the bottom-right of `canvas`.

        Order:
            1. Resize the latest camera frame to (THUMB_W, THUMB_H).
            2. Overlay the 21-point skeleton plus colour-coded
               thumb/index dots.  Pinching state determines the dot
               colour (green = recognised, orange = not).
            3. Splice into `canvas` at the bottom-right corner with a
               THUMB_MARGIN gutter from both edges.
            4. Draw a 1px white border JUST OUTSIDE the thumbnail rect
               so the diagnostic widget reads as its own object
               against the dark OS wallpaper -- without the border,
               the preview's dark pixels can bleed into the OS
               background and the rectangle's edge disappears.

        Silent no-op if the thread has not produced a camera frame yet
        (the very first frame after construction).  The OS doesn't
        depend on the thumbnail; rendering nothing for one frame is
        invisible.
        """
        with self._lock:
            cam = self._latest_camera_bgr
            landmarks = list(self._latest_landmarks)  # shallow copy under the lock
            is_pinching = self._latest_is_pinching

        if cam is None:
            return

        canvas_h, canvas_w = canvas.shape[:2]
        x0 = canvas_w - THUMB_W - THUMB_MARGIN
        y0 = canvas_h - THUMB_H - THUMB_MARGIN
        # BUG FIX: defensive bounds check.  On a fullscreen MacBook Air
        # M2 canvas the thumbnail always fits, but a tiny canvas (e.g.
        # an unexpected resize event or the very first frame before
        # the cv2 window finishes resizing) could put x0 or y0 negative
        # and the np slice assignment below would fail with a shape
        # mismatch.  We silently skip the paint in that case -- the OS
        # is more important than the diagnostic widget.
        if x0 < 1 or y0 < 1 or canvas_w < THUMB_W + 2 or canvas_h < THUMB_H + 2:
            return

        thumb = cv2.resize(cam, (THUMB_W, THUMB_H))
        for hand in landmarks:
            pts = _landmarks_to_points(hand, THUMB_W, THUMB_H)
            self._draw_skeleton_on_thumb(thumb, pts, is_pinching)

        # Splice in.  Direct slice assignment is the cheapest way to
        # composite an opaque sub-image; no alpha channel involved.
        canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = thumb
        # Border one pixel outside the image patch so we don't cover
        # any preview pixels with the border stroke.
        cv2.rectangle(
            canvas,
            (x0 - 1, y0 - 1),
            (x0 + THUMB_W, y0 + THUMB_H),
            _THUMB_BORDER_COLOR_BGR,
            _THUMB_BORDER_THICKNESS,
        )

    def close(self) -> None:
        """Stop the thread, release the camera, close the MediaPipe model.

        Idempotent.  The OS calls this inside a try/finally so a crash
        mid-frame still releases the camera handle; calling close()
        twice (once via the finally, once via __del__) must not
        double-release.
        """
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        # Join with a timeout: if the camera read is genuinely hung
        # (uncommon, but possible on macOS when the user revokes
        # camera permission mid-run) we don't want close() to block
        # the OS shutdown sequence.  2 seconds is generous; the
        # thread loop polls the stop event between every read.
        self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        # MediaPipe Hands has a close() method that releases the
        # internal C++ graph; calling it here keeps the next run's
        # camera permission prompt from getting confused on macOS.
        try:
            self._hands_model.close()
        except Exception:
            # Defensive: if the model was already half-released by the
            # thread shutting down, swallow the error rather than
            # crash the OS exit path.
            pass
        # Same defensive close for FaceMesh.  Two models sharing one
        # process is unusual enough that a clean teardown matters --
        # leaving a graph open has been observed (on older mediapipe
        # builds) to delay the next process's camera open by ~1s.
        try:
            self._face_mesh.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Background thread -- one loop iteration per camera frame
    # ------------------------------------------------------------------

    def _thread_loop(self) -> None:
        """Main loop of the daemon thread.  Runs until self._stop_event is set.

        Each iteration:
            1. Read a frame from the camera.  Missed reads loop back
               immediately (the camera occasionally drops a frame
               under load; one drop is not a crash).
            2. Mirror left-right so the user's right hand appears on
               the right of the canvas (visionOS / front-camera
               convention; matches the reference).
            3. Convert to RGB and feed to MediaPipe.
            4. Advance the pinch tracker.
            5. Publish a fresh HandFrame snapshot under the lock.

        Any exception inside the loop is caught and converted to a
        no-detection frame, so a transient MediaPipe crash never kills
        the thread.  A truly dead camera shows up as a frozen
        thumbnail -- the OS still runs.
        """
        # Hot-path locals; mp.solutions.hands.HAND_CONNECTIONS lookup
        # is once per init, the tracker advance is hot, the publish is
        # at the end of every iteration.
        cap = self._cap
        if cap is None:
            return

        while not self._stop_event.is_set():
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # Camera blip.  Don't busy-spin: short event wait
                    # gives the camera time to recover and the GIL
                    # back to the main thread.  A real disconnection
                    # would just keep blipping; the OS still renders
                    # the previously published HandFrame in the
                    # meantime.
                    self._stop_event.wait(timeout=0.005)
                    continue
                # Mirror so right-hand-moving-right reads naturally.
                # This is the same flip the reference does; without
                # it, dx signs would be inverted relative to the
                # mirrored user-facing view.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Run BOTH models on the same RGB buffer.  Hands first
                # so the wrist/index data is fresh when the publisher
                # builds the HandFrame; face mesh second.  Either can
                # raise (MediaPipe occasionally throws on a degenerate
                # frame); the outer try in this loop catches that and
                # skips the publish for one frame.
                hand_results = self._hands_model.process(rgb)
                face_results = self._face_mesh.process(rgb)
                # Pretrained gaze inference runs on the camera thread
                # so the OS render loop never pays the L2CS forward-
                # pass latency synchronously.  step() takes BGR (cv2
                # native); when no face is detected it returns None,
                # which clears the published yaw/pitch and lets the
                # gaze_cursor projection fall through to the
                # calibrated path.  Caught defensively because torch
                # forward passes can raise on degenerate frames and
                # we don't want a single bad inference to kill the
                # camera loop.
                if self._pretrained_gaze is not None:
                    try:
                        yp = self._pretrained_gaze.step(frame)
                    except Exception:                         # noqa: BLE001
                        yp = None
                    with self._lock:
                        self._latest_pretrained_yp = yp
                self._publish_results(frame, hand_results, face_results)
            except Exception:
                # Defensive catch: a single bad frame should not stop
                # the loop.  We don't print here because the loop is
                # busy and a stack trace per frame would spam stderr;
                # the published HandFrame from the previous iteration
                # is retained and the OS sees one frame of staleness.
                self._stop_event.wait(timeout=0.005)
                continue

    def _publish_results(
        self,
        camera_bgr: np.ndarray,
        results: Any,
        face_results: Any,
    ) -> None:
        """Convert MediaPipe results into the latest HandFrame + thumbnail state.

        Called once per camera frame from the background thread.
        Acquires the lock once, at the end, to publish the new state.
        All the math runs OUTSIDE the lock so the OS thread is never
        blocked on tracker logic.

        `face_results` is the face-mesh output for the SAME frame.
        We use it only to update the gaze state -- the HandFrame
        itself is computed purely from `results` (hands), unchanged.
        """
        canvas_w = self._canvas_w
        canvas_h = self._canvas_h
        hands_list = results.multi_hand_landmarks or []

        # Face mesh: extract iris-in-eye normalised position.  The
        # value is None when no face is detected; the smoothed value
        # is held across short detection dropouts so the cursor
        # doesn't snap to the canvas centre every time the model
        # blips for one frame.
        face_list: list[Any] = (
            face_results.multi_face_landmarks or []
            if face_results is not None
            else []
        )
        gaze_norm = _compute_gaze_norm(face_list, camera_bgr.shape[1],
                                       camera_bgr.shape[0])
        # BUG FIX: defer publishing the gaze state until the single
        # publish-lock acquisition at the end of this method.  An
        # earlier draft wrote `self._latest_gaze_norm` /
        # `self._smoothed_gaze_norm` outside the lock; readers
        # (`latest_gaze_norm`, `gaze_cursor`, `add_calibration_sample`)
        # all acquire the lock, but the unlocked writer made that lock
        # meaningless -- and a sufficiently surprising interpreter
        # change could turn the inconsistency into observable torn
        # state.  Compute the smoothed value here, but stash it in a
        # local; the lock block at the bottom of the method publishes
        # it alongside the HandFrame and the camera buffer.
        if gaze_norm is not None:
            prev = self._smoothed_gaze_norm
            if prev is None:
                # First reading: seed the EMA with the raw value so
                # the cursor doesn't ramp from (0,0) for the first
                # GAZE_SMOOTH_ALPHA-decay window.
                smoothed_to_publish: Optional[tuple[float, float]] = gaze_norm
            else:
                smoothed_to_publish = (
                    GAZE_SMOOTH_ALPHA * gaze_norm[0]
                    + (1.0 - GAZE_SMOOTH_ALPHA) * prev[0],
                    GAZE_SMOOTH_ALPHA * gaze_norm[1]
                    + (1.0 - GAZE_SMOOTH_ALPHA) * prev[1],
                )
            latest_to_publish: Optional[tuple[float, float]] = gaze_norm
            update_gaze = True
        else:
            # When no face is detected we KEEP the previous smoothed
            # reading.  This is the "stale-on-loss" behaviour the user
            # expects: looking away from the camera for a moment
            # shouldn't warp the cursor to (0, 0); the cursor just
            # freezes where it last was until the face comes back.
            smoothed_to_publish = None
            latest_to_publish = None
            update_gaze = False

        # Reset edge flags for this iteration.  The publishable
        # HandFrame's drag_just_* fields are TRUE for exactly the
        # frame the transition happens on; on the very next iteration
        # they MUST go back to False whether or not a drag is ongoing.
        click_now = False
        drag_just_started = False
        drag_just_ended = False

        if not hands_list:
            # No hands.  Reset any in-flight gesture state.  Per the
            # prompt's state-machine spec: a drag interrupted by the
            # hand leaving the frame fires zero CLICK events (we never
            # emit click_now from this path).  We DO however emit
            # drag_just_ended if the prior state was DRAGGING -- the
            # OS layer needs that edge to rubber-band the displaced
            # page back to rest; without it, the page would be stuck
            # at the last-known drag_offset_px until the user pinches
            # again.  PINCH_HELD interruptions still emit nothing
            # because no drag was ever active for the OS.
            drag_just_ended = self._reset_tracker_if_active()
            current_drag_active = False
            cursor_xy = (-1, -1)
            is_pinching_visual = False
        else:
            # First hand only.  See the design note above.
            primary = hands_list[0]
            points = _landmarks_to_points(primary, canvas_w, canvas_h)

            # Pinch ratio uses hysteresis: while pinched we use a
            # higher exit threshold than the enter threshold.
            ratio = _pinch_ratio(points)
            was_pinching = self._tracker.state in (
                _STATE_PINCH_HELD, _STATE_DRAGGING,
            )
            threshold = PINCH_EXIT_RATIO if was_pinching else PINCH_ENTER_RATIO
            pinching_now = ratio < threshold
            # Independent of the tracker state, the visual feedback
            # flips the instant a pinch is RECOGNISED -- one frame
            # earlier than CONFIRMED.  Without this the dots stay
            # orange during the brief CLICK_CONFIRM_FRAMES window
            # and the user wonders whether the pinch was seen.
            is_pinching_visual = pinching_now

            wrist = points[WRIST]
            hand_size = math.hypot(
                points[WRIST][0] - points[MID_MCP][0],
                points[WRIST][1] - points[MID_MCP][1],
            )

            click_now, drag_just_started, drag_just_ended = (
                self._advance_tracker(pinching_now, wrist, hand_size)
            )

            cursor_xy = points[INDEX_TIP]
            current_drag_active = self._tracker.state == _STATE_DRAGGING

        new_frame = HandFrame(
            present=bool(hands_list),
            cursor_xy=cursor_xy,
            click_now=click_now,
            drag_active=current_drag_active,
            drag_dx=self._tracker.cumulative_dx if current_drag_active else 0,
            drag_dy=self._tracker.cumulative_dy if current_drag_active else 0,
            drag_just_started=drag_just_started,
            drag_just_ended=drag_just_ended,
            is_pinching=is_pinching_visual,
        )

        with self._lock:
            self._latest_frame = new_frame
            self._latest_camera_bgr = camera_bgr
            self._latest_landmarks = hands_list
            self._latest_is_pinching = is_pinching_visual
            # Publish face landmarks so the calibration UI can paint
            # the iris dots / eye outline on its preview.  We publish
            # by reference -- mediapipe returns a fresh proto object
            # per process() call, so the OS thread can't observe a
            # half-written list.
            self._latest_face_landmarks = face_list
            # BUG FIX: publish the gaze state under the same lock as
            # every other shared field.  See the comment above
            # update_gaze where these values are computed.  Skip the
            # write entirely on a no-face frame so the previous
            # smoothed reading sticks around (stale-on-loss).
            if update_gaze:
                self._latest_gaze_norm = latest_to_publish
                self._smoothed_gaze_norm = smoothed_to_publish

    def _advance_tracker(
        self,
        pinching_now: bool,
        wrist: tuple[int, int],
        hand_size_px: float,
    ) -> tuple[bool, bool, bool]:
        """Step the pinch state machine one frame.  Returns (click_now, drag_started, drag_ended).

        OS-tuned: CLICK fires on RELEASE, only if the gesture never
        entered DRAGGING.  This is the one behavioural divergence
        from the reference's PinchTracker; everything else (the
        confirm-frames count, cooldown, hand-size-normalised drag
        threshold) carries over verbatim.

        The returned tuple is consumed by `_publish_results` to fill
        the corresponding HandFrame fields.
        """
        t = self._tracker
        click_now = False
        drag_started = False
        drag_ended = False

        # Cooldown decrement happens every frame regardless of state,
        # mirroring the reference.
        if t.cooldown_frames > 0:
            t.cooldown_frames -= 1

        if t.state == _STATE_IDLE:
            if pinching_now and t.cooldown_frames == 0:
                # Begin a new pinch.  held_frames starts at 1 (this
                # frame counts as the first held frame); CLICK won't
                # actually fire until release.
                t.state = _STATE_PINCH_HELD
                t.held_frames = 1
                t._became_drag = False
                t.last_wrist_pos = wrist
                t.wrist_at_drag_start = wrist
                t.cumulative_dx = 0
                t.cumulative_dy = 0
            return click_now, drag_started, drag_ended

        if t.state == _STATE_PINCH_HELD:
            if pinching_now:
                t.held_frames += 1
                # Whole-hand motion check.  Transition to DRAGGING
                # the first frame the wrist moves more than the
                # threshold.  Once we've been in DRAGGING we set the
                # `_became_drag` sticky flag so the eventual release
                # knows not to emit a CLICK.
                displacement = _normalised_wrist_displacement(
                    wrist, t.last_wrist_pos, hand_size_px,
                )
                if (displacement > DRAG_MOVE_THRESHOLD_NPX
                        and t.held_frames >= CLICK_CONFIRM_FRAMES):
                    t.state = _STATE_DRAGGING
                    t._became_drag = True
                    drag_started = True
                    # Lock in the drag-start wrist position.  Using
                    # the CURRENT wrist (not the position the pinch
                    # started at) means the cumulative_dx the OS
                    # consumes starts at 0 the moment the drag is
                    # recognised, not after the initial "pinch
                    # confirmation" wobble.
                    t.wrist_at_drag_start = wrist
                    t.cumulative_dx = 0
                    t.cumulative_dy = 0
                t.last_wrist_pos = wrist
            else:
                # Pinch released while still in PINCH_HELD.  Emit
                # CLICK if the gesture qualified (long enough to
                # confirm, no cooldown, never became a drag).
                if (t.held_frames >= CLICK_CONFIRM_FRAMES
                        and not t._became_drag):
                    click_now = True
                    t.cooldown_frames = CLICK_COOLDOWN_FRAMES
                t.state = _STATE_IDLE
                t.held_frames = 0
                t._became_drag = False
                t.cumulative_dx = 0
                t.cumulative_dy = 0
            return click_now, drag_started, drag_ended

        if t.state == _STATE_DRAGGING:
            if pinching_now:
                # Continue the drag.  Update cumulative_dx/dy from
                # the wrist position at drag start; integrating
                # per-frame deltas would accumulate rounding drift
                # over a long swipe.
                t.cumulative_dx = wrist[0] - t.wrist_at_drag_start[0]
                t.cumulative_dy = wrist[1] - t.wrist_at_drag_start[1]
                t.last_wrist_pos = wrist
            else:
                # Pinch released while in DRAGGING.  No CLICK (that's
                # the whole reason we deferred to release-time).  Just
                # emit the drag-ended edge so the OS can run
                # end_page_drag() exactly once.
                drag_ended = True
                t.state = _STATE_IDLE
                t.held_frames = 0
                t._became_drag = False
                # Keep cumulative_dx/dy intact for the LAST frame
                # we publish drag_active=False; the OS reads dx in
                # end_page_drag's "where did we let go?" logic via
                # the prior frame's drag_dx.  Zeroing here would
                # lose that information.  However the published
                # HandFrame zeroes drag_dx when current_drag_active
                # is False (see _publish_results) so the OS only
                # ever sees the final dx on the frame the drag
                # actually ends and drag_just_ended is True.
                # Stash the final dx in cumulative so the publish
                # path can include it when drag_just_ended is True.
                # No-op here -- cumulative_dx already holds the
                # right value from the last "still pinching" frame.
            return click_now, drag_started, drag_ended

        return click_now, drag_started, drag_ended

    def _reset_tracker_if_active(self) -> bool:
        """Hand disappeared mid-gesture.  Drop state.

        Returns True iff the prior state was DRAGGING.  The caller
        propagates that as `drag_just_ended` in the next published
        HandFrame so the OS can rubber-band the page back; without
        this, a hand that exits the frame mid-swipe would leave the
        compositor's drag state stuck at the last drag offset.

        PINCH_HELD interruptions return False -- no drag was ever
        active for the OS, so no edge to emit.  CLICK events are
        never synthesised from this path: the user might have just
        flicked their hand out of frame, and inferring intent there
        would be a misinterpretation.
        """
        t = self._tracker
        if t.state == _STATE_IDLE:
            return False
        was_dragging = (t.state == _STATE_DRAGGING)
        t.state = _STATE_IDLE
        t.held_frames = 0
        t._became_drag = False
        t.cumulative_dx = 0
        t.cumulative_dy = 0
        return was_dragging

    # ------------------------------------------------------------------
    # Thumbnail painting helpers
    # ------------------------------------------------------------------

    def _draw_skeleton_on_thumb(
        self,
        thumb: np.ndarray,
        points: list[tuple[int, int]],
        is_pinching: bool,
    ) -> None:
        """Paint the 21-point hand skeleton + thumb/index highlights onto `thumb`.

        Two layers:
            1. The full 21-point mesh: white lines for the bones,
               white dots for the joints.  Bones first, dots on top so
               the joints visually anchor the bones at junctions.
            2. The thumb-tip + index-tip highlights: bigger filled
               dots and a connecting line, coloured by `is_pinching`
               (green if so, orange if not).  This is the visual
               feedback the user reads to know whether a pinch is
               recognised.

        `points` must have at least 21 entries (the standard
        MediaPipe Hands landmark count).  Out-of-range indices would
        raise IndexError; we treat that as a bug rather than catch
        it, matching the rest of the codebase's loud-failure rule.
        """
        # Bones first.  Iterating HAND_CONNECTIONS (a frozenset of
        # (start, end) tuples) preserves a deterministic draw order
        # under CPython, but the order doesn't matter visually --
        # every line is the same colour and width.
        for start_idx, end_idx in self._connections:
            cv2.line(
                thumb,
                points[start_idx],
                points[end_idx],
                _THUMB_SKELETON_COLOR_BGR,
                _THUMB_CONNECTION_THICKNESS,
            )
        # Joints over the bones.
        for pt in points:
            cv2.circle(
                thumb, pt, _THUMB_LANDMARK_RADIUS,
                _THUMB_SKELETON_COLOR_BGR, -1,
            )
        # Thumb-index highlights on top of the basic skeleton.
        color = (
            _TIP_PINCHING_COLOR_BGR if is_pinching
            else _TIP_NOT_PINCHING_COLOR_BGR
        )
        cv2.line(
            thumb,
            points[THUMB_TIP],
            points[INDEX_TIP],
            color,
            _PINCH_LINE_THICKNESS,
        )
        cv2.circle(thumb, points[THUMB_TIP], _TIP_HIGHLIGHT_RADIUS, color, -1)
        cv2.circle(thumb, points[INDEX_TIP], _TIP_HIGHLIGHT_RADIUS, color, -1)


# ============================================================================
# Public surface
# ============================================================================
#
# Only three names from this module are meant to be imported by phase
# scripts:
#
#     HandInput              -- the camera + MediaPipe wrapper class.
#     HandFrame              -- the per-step result dataclass.
#     HandInputUnavailable   -- the constructor's failure exception.
#
# Everything else (_PinchTracker, the threshold constants, the
# drawing helpers, _empty_hand_frame) is module-private and subject to
# change without notice.  THUMB_W and THUMB_H are also re-exported
# so the compositor can reserve their bottom-right rect if a future
# Phase wants to keep notifications from overlapping the thumbnail;
# nothing in Phase 8 needs them externally yet.
# ============================================================================
