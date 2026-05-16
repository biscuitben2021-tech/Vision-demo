"""
Phase 8 -- Hand tracking on top of the Phase 7 polish layer.

Goals (delta from Phase 7):
    * Pinch click.  MediaPipe Hands on the built-in webcam recognises
      a thumb-to-index pinch; a tap (pinch released without dragging)
      fires a left click at the CURRENT mouse cursor position.  The
      mouse remains the cursor source -- the hand never moves it.
    * Pinch-and-drag swipes between home pages.  Holding the pinch
      and moving the wrist horizontally drags the home pages with the
      hand; releasing the pinch snaps the page to the closer
      neighbour (or rubber-bands back).  Behaviour mirrors iPadOS /
      Vision Pro: hand goes LEFT -> tiles go LEFT -> next page is
      revealed from the right.
    * Camera thumbnail.  A 213x120 16:9 preview sits in the canvas's
      bottom-right corner with the detected hand skeleton and
      colour-coded thumb/index dots overlaid.  This is the
      diagnostic widget the user reads to confirm the system is
      seeing their hand at all -- before assuming the OS interaction
      is wrong.
    * Everything else from Phase 7 (live clock, scheduled
      notifications, cross-fade transitions, FPS counter, ESC/Q
      quit) carries straight over.

Show-day defensiveness:
    The HandInput constructor can fail (no camera, no permission,
    mediapipe not installed).  This script catches that, prints a
    clear warning, and proceeds in mouse-only mode.  The result is
    indistinguishable from Phase 7 except for a missing thumbnail.
    The OS ALWAYS boots; tracking is best-effort.  ESC / Q remain
    the global quit even when the camera thread is mid-frame.

Module color-space convention:
    BGR for the cv2 pixel buffer; constants imported with `_BGR` go
    straight to cv2 calls, `_RGB` cross into PIL via `draw_text`.
    Same convention as src/design, src/tiles, src/icons, phase1-7.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Final, Optional

import cv2
import numpy as np

from src.compositor import Compositor
from src.design import draw_fps_hud, draw_text, load_font
from phase6_app_window import _close_button_rect


# How big the gaze-snap zone around the close button is.  Anything in
# the top-left _CLOSE_SNAP_PX-by-_CLOSE_SNAP_PX square gets snapped
# onto the close-button centre so the user can fire a close gesture
# by glancing roughly at the corner instead of needing pixel-perfect
# aim on a 32x32 target.  Tuned to be a quarter of a 1080p canvas:
# big enough to land for a casual top-left glance, small enough that
# a glance at the top centre still uses the raw gaze coordinate.
_CLOSE_SNAP_PX: Final[int] = 280
from phase1_canvas import (
    FPS_EMA_ALPHA,
    QUIT_KEY_ESC,
    QUIT_KEY_Q_L,
    QUIT_KEY_Q_U,
    WINDOW_NAME,
    make_fullscreen_window,
    screen_size,
)
from phase5_motion import (
    now_ms_relative,
    reduced_motion_requested_via_cli,
    reduced_motion_requested_via_keypoll,
)


# ----------------------------------------------------------------------------
# Notification schedule
# ----------------------------------------------------------------------------
#
# Same fixed schedule Phase 7 uses.  Keeping the same two notifications
# means a side-by-side comparison of Phase 7 and Phase 8 looks
# identical for the OS chrome; only the input layer changes.  If a
# future Phase wants a different schedule it can override this list.

NOTIF_SCHEDULE: Final[list[tuple[int, str, str]]] = [
    (10_000, "Apple Music", "Now playing: Bloom - Radiohead"),
    (30_000, "Messages",   "Mom: see you tonight"),
]


# ----------------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------------
#
# Two flags this phase needs:
#   --reduced-motion  -- forwarded to the Compositor and the motion
#                        state.  Phase 5/6/7 also accept this flag
#                        via the same module helper; we just include
#                        it in argparse here so it shows up in --help.
#   --camera N        -- override the auto-picked camera index.  The
#                        HandInput silently probes 0..5 by default;
#                        pass --camera 2 (say) if the user wants the
#                        external USB cam instead of the built-in.
#
# argparse rather than the phase5 sys.argv-string-search idiom:  this
# phase introduces a flag with a VALUE (--camera N), which the
# string-search approach cannot handle cleanly.  argparse also
# auto-generates --help, which is friendly for the operator at the
# show.

_CLI_FLAG_REDUCED_MOTION: Final[str] = "--reduced-motion"
_CLI_FLAG_CAMERA: Final[str] = "--camera"
_CLI_FLAG_PRETRAINED: Final[str] = "--pretrained"


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    """Parse the Phase 8 CLI flags.

    Returns an argparse Namespace with .reduced_motion (bool) and
    .camera (Optional[int]).  Pulled out as a helper to keep main()
    short.  argv is passed in (not read via sys.argv directly) so
    tests can drive it.
    """
    parser = argparse.ArgumentParser(
        prog="phase8_hand",
        description=(
            "Vision OS demo with hand-pinch click + drag-to-swipe.  "
            "Press ESC or Q to quit."
        ),
    )
    parser.add_argument(
        _CLI_FLAG_REDUCED_MOTION,
        action="store_true",
        help="Disable all animations (fade-up, hover scale, transitions).",
    )
    parser.add_argument(
        _CLI_FLAG_CAMERA,
        type=int,
        default=None,
        metavar="INDEX",
        help=(
            "Override the auto-picked camera index.  Without this flag, "
            "the script probes 0..5 and uses the first working device."
        ),
    )
    parser.add_argument(
        _CLI_FLAG_PRETRAINED,
        action="store_true",
        help=(
            "Use L2CS-Net pretrained gaze instead of the 5-point per-user "
            "calibration.  Skips the calibration screen; the model's "
            "yaw/pitch output is projected to screen pixels directly.  "
            "Requires the l2cs pip package and the L2CSNet_gaze360.pkl "
            "weights at assets/ (auto-downloaded on first use via gdown)."
        ),
    )
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------
# Main-loop context -- shared by the mouse callback and the per-frame loop
# ----------------------------------------------------------------------------
#
# Same dataclass pattern Phase 7 uses, but with two additional fields
# Phase 8's hand input doesn't replace:
#   - The mouse callback is still the SOLE cursor source.  The hand
#     never moves the cursor.
#   - The pending_click flag fuses two sources: a left-mouse-click
#     AND a hand-tap-release.  Both arrive as one-shot events and
#     fire a click at the current cursor position; the main loop
#     OR's them together when calling compose_frame.
#
# Why the hand doesn't get its own pending field: from the
# Compositor's perspective there's no distinction between a mouse
# click and a hand tap -- both are "left click at mouse_xy, this
# frame".  Collapsing them into one flag keeps the main loop simple.


@dataclass
class _MainLoopContext:
    """Mutable bundle shared between the cv2 mouse callback and main().

    Fields:
        mouse_xy:        last known cursor position in canvas pixels.
                         Initialised to (-1, -1) so any pre-first-event
                         frame treats the cursor as off-canvas.
        pending_click:   True if a LBUTTONDOWN has been observed since
                         the last consume.  Cleared by main() each
                         frame.  Hand-tap clicks DO NOT go through
                         this field; main() OR's them at the
                         compose_frame call site.
        notif_fired:     boolean flag per scheduled notification, see
                         Phase 7 for the rationale.
    """

    mouse_xy: tuple[int, int] = (-1, -1)
    pending_click: bool = False
    notif_fired: list[bool] = field(
        default_factory=lambda: [False] * len(NOTIF_SCHEDULE),
    )


def _mouse_callback(
    event: int, x: int, y: int, flags: int, param: object,
) -> None:
    """Record mouse position + click events into the shared context.

    Identical to Phase 7's callback.  The hand input does not flow
    through this callback at all -- it lives in src/hands.py on its
    own daemon thread and is polled from the main loop directly.
    """
    assert isinstance(param, _MainLoopContext)
    if event == cv2.EVENT_MOUSEMOVE:
        param.mouse_xy = (x, y)
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        param.mouse_xy = (x, y)
        param.pending_click = True


def _nearest_tile_index(
    xy: tuple[int, int],
    tile_rects: list[tuple[int, int, int, int]],
) -> Optional[int]:
    """Return the index of the tile whose centre is closest to `xy`.

    Unlike phase5_motion.closest_tile (which only returns a tile when
    the cursor is INSIDE one), this helper returns the nearest tile
    regardless of whether the cursor's actually on it -- which is the
    behaviour the gaze-lock wants.  Even when the user's iris drift
    lands the gaze a few pixels into the gutter, the lock should still
    pick "the tile they were obviously looking at".

    Returns None only when `tile_rects` is empty -- a defensive case
    that protects against geometry-not-yet-built states.

    Distance is squared-Euclidean against each tile's centre point;
    no square root is taken because argmin doesn't need it (monotone
    under sqrt).
    """
    if not tile_rects:
        return None
    x, y = xy
    best_idx = 0
    best_d2 = None
    for i, (tx, ty, tw, th) in enumerate(tile_rects):
        cx = tx + tw // 2
        cy = ty + th // 2
        d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        if best_d2 is None or d2 < best_d2:
            best_idx = i
            best_d2 = d2
    return best_idx


def _fire_due_notifications(
    compositor: Compositor,
    ctx: _MainLoopContext,
    now_ms: int,
) -> None:
    """Enqueue any scheduled notifications whose delay has elapsed.

    Same Phase 7 logic, lifted unchanged.  Lives here (not on the
    Compositor) because the schedule is a presentation-layer concern
    that varies between demos.
    """
    for i, (delay_ms, title, body) in enumerate(NOTIF_SCHEDULE):
        if not ctx.notif_fired[i] and now_ms >= delay_ms:
            compositor.enqueue_notification(title, body, now_ms)
            ctx.notif_fired[i] = True


def _update_fps(prev_t: float, prev_ema: float) -> tuple[float, float, float]:
    """Update the FPS EMA and return (now, dt, new_ema).

    Same helper Phase 7 uses; reproduced here so this script can run
    standalone without importing phase7_polish (which would create a
    circular dependency at first import).
    """
    now = time.perf_counter()
    dt = now - prev_t
    if dt <= 0.0:
        return now, dt, prev_ema
    instant_fps = 1.0 / dt
    if prev_ema == 0.0:
        new_ema = instant_fps
    else:
        new_ema = FPS_EMA_ALPHA * instant_fps + (1.0 - FPS_EMA_ALPHA) * prev_ema
    return now, dt, new_ema


# ----------------------------------------------------------------------------
# Hand-input bring-up -- defensive construction
# ----------------------------------------------------------------------------
#
# The HandInput constructor can fail in three identifiable ways:
#   1. mediapipe is not installed (HandInputUnavailable with a
#      mediapipe-specific message).
#   2. No camera was detected on indices 0..5 (HandInputUnavailable
#      with a permission hint).
#   3. The named --camera index is not connected (same exception).
#
# We catch all three the same way: print a clear, single-line
# warning to stderr, then return None.  main() proceeds in
# mouse-only mode.  The OS NEVER fails to boot because of a
# tracking problem; that is the show-day rule.


def _resolve_camera_index(cli_camera: Optional[int]) -> Optional[int]:
    """Pick the camera index, prompting the user when not pre-specified.

    Behaviour mirrors the Hand-controller reference's `phase7_demo.py`:
      * If `--camera N` was passed, use that index without probing
        (the operator already knows which one they want).
      * Otherwise probe 0..CAMERA_PROBE_LIMIT-1, print the working
        devices, and read the operator's pick from stdin.
      * If the probe finds nothing, return None and let the rest of
        the script fall back to mouse-only mode -- the OS still boots.
      * If exactly one camera is found, skip the prompt (no choice to
        make) and use it directly.

    Why this lives in phase8_hand.py rather than in src.hands: a
    stdin prompt is a TERMINAL concern, not a HAND-INPUT concern.
    src.hands is also used (in spirit) by future Phases that may
    drive the camera index from a config file or a UI picker; keeping
    the prompt out of the library means those callers don't inherit
    a hidden stdin read.

    Return value:
        int -- a usable camera index, ready to pass into HandInput.
        None -- no working camera was found OR the user is willing to
                run mouse-only (they typed a non-numeric input at the
                prompt to bail).
    """
    if cli_camera is not None:
        # Operator-supplied index.  Trust the operator -- if the index
        # is bad, HandInput's own probe will raise HandInputUnavailable
        # which the caller's _try_construct_hand_input absorbs.
        return cli_camera

    # Lazy import so a broken hands.py (e.g. mediapipe missing) doesn't
    # kill the OS at startup; the script can still run mouse-only.
    try:
        from src.hands import list_available_cameras, CAMERA_PROBE_LIMIT
    except ImportError as exc:
        print(
            f"[phase8] Could not import src.hands: {exc}.  "
            "Skipping camera probe; mouse-only mode.",
            file=sys.stderr,
        )
        return None

    print("[phase8] Probing for cameras...")
    cameras = list_available_cameras(CAMERA_PROBE_LIMIT)
    if not cameras:
        print(
            "[phase8] No working cameras detected.  "
            "On macOS, check System Settings -> Privacy & Security -> "
            "Camera permissions for your terminal/IDE.  "
            "Continuing in mouse-only mode.",
            file=sys.stderr,
        )
        return None

    if len(cameras) == 1:
        idx, w, h = cameras[0]
        print(f"[phase8] Only one camera available: [{idx}] {w}x{h} -- using it.")
        return idx

    print("Available cameras:")
    for idx, w, h in cameras:
        print(f"  [{idx}] {w}x{h}")
    default = cameras[0][0]
    valid_indices = {c[0] for c in cameras}

    # EOFError = stdin closed (piped run): use the default silently.
    # KeyboardInterrupt = user hit Ctrl+C during the prompt: re-raise
    # so they actually quit (don't silently fall back to mouse-only,
    # they clearly want out).
    try:
        raw = input(f"Pick a camera by index [{default}]: ").strip()
    except EOFError:
        raw = ""

    if not raw:
        return default
    try:
        chosen = int(raw)
    except ValueError:
        print(
            f"[phase8] '{raw}' is not a number; using [{default}].",
            file=sys.stderr,
        )
        return default
    if chosen not in valid_indices:
        print(
            f"[phase8] Index {chosen} not in the list; using [{default}].",
            file=sys.stderr,
        )
        return default
    return chosen


def _try_construct_hand_input(
    camera_index: Optional[int],
    pretrained_gaze=None,
):
    """Try to bring up the HandInput.  Return it on success, None on failure.

    src.hands itself is pure-Python at module scope (mediapipe is
    imported lazily inside HandInput.__init__), so `from src.hands
    import HandInput` is safe in any environment where cv2 and numpy
    work.  If that import nonetheless fails (e.g. a corrupted
    install), we still want the OS to boot in mouse-only mode -- so
    we wrap even the import in a try/except.

    `pretrained_gaze` is forwarded straight to HandInput.__init__;
    pass None for the calibrated path, or a PretrainedGaze instance
    for --pretrained mode.
    """
    try:
        from src.hands import HandInput, HandInputUnavailable
    except ImportError as exc:
        print(
            f"[phase8] Could not import src.hands: {exc}.  "
            "Hand tracking disabled; falling back to mouse-only mode.",
            file=sys.stderr,
        )
        return None
    try:
        hand_input = HandInput(
            camera_index=camera_index,
            pretrained_gaze=pretrained_gaze,
        )
    except HandInputUnavailable as exc:
        print(
            f"[phase8] Hand tracking unavailable: {exc}  "
            "Falling back to mouse-only mode.",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        # Catch-all for unexpected errors (e.g. a future MediaPipe
        # version changes its API).  The OS should NOT die because
        # the tracker did; print the error and continue.
        print(
            f"[phase8] Unexpected hand-tracking failure: {exc!r}.  "
            "Falling back to mouse-only mode.",
            file=sys.stderr,
        )
        return None
    if hand_input.camera_index >= 0:
        print(
            f"[phase8] Hand tracking online on camera index "
            f"{hand_input.camera_index}.",
        )
    return hand_input


# ----------------------------------------------------------------------------
# Eye-tracking calibration screen
# ----------------------------------------------------------------------------
#
# Runs BEFORE the OS opens.  Shows a fullscreen window with:
#     * the live camera feed in the top-left of the canvas, with iris
#       dots and eye outlines drawn on top so the user can see what
#       the system is reading;
#     * a large crosshair / target dot in the centre of the canvas;
#     * a short instruction line ("Look at the centre dot, then
#       press SPACE").
#
# When the user presses SPACE we ask HandInput to snapshot the
# current smoothed gaze as the "looking-at-centre" baseline; in the
# OS that baseline is the origin point for cursor placement.  ESC
# bails out without setting a baseline -- the OS still runs, just
# with the mouse driving the cursor (gaze-cursor returns None until
# a baseline exists).
#
# Why a separate window (not the OS fullscreen window): the OS
# window is created with the WND_PROP_FULLSCREEN flag which on macOS
# claims the entire display and locks us out of resizing it.  The
# calibration screen is a one-shot UI before the OS boots, so it
# uses a plain WINDOW_NORMAL window we can dismiss cleanly with
# destroyWindow -- once it's gone, make_fullscreen_window can claim
# the screen for the OS.
#
# This screen is COSMETIC -- nothing OS-critical depends on it.  If
# the user dismisses with ESC or the face isn't detected, the OS
# boots regardless and the cursor falls back to the mouse.

_CAL_WINDOW_NAME: Final[str] = "Vision OS -- Eye Calibration"
_CAL_TARGET_OUTER_R: Final[int] = 26    # the visible "look here" ring
_CAL_TARGET_INNER_R: Final[int] = 6     # tight dot inside the ring
_CAL_TARGET_COLOR_BGR: Final[tuple[int, int, int]] = (255, 255, 255)
_CAL_PREVIEW_W: Final[int] = 480        # camera preview width on the cal canvas
_CAL_PREVIEW_H: Final[int] = 270        # 16:9 to match the camera aspect
_CAL_PREVIEW_MARGIN: Final[int] = 32
_CAL_TEXT_COLOR_BGR: Final[tuple[int, int, int]] = (240, 240, 240)
_CAL_TEXT_DIM_BGR: Final[tuple[int, int, int]] = (140, 140, 140)
_CAL_SPACE_KEY: Final[int] = 32
_CAL_ESC_KEY: Final[int] = 27

# Face-mesh landmark indices we draw on the calibration preview.
# Mirrors src.hands -- duplicated here so this helper does not need
# to reach into the hands module for constants.  Iris dots + the
# four eye-corner / lid points each.
_CAL_EYE_DRAW_INDICES: Final[tuple[int, ...]] = (
    468, 473,                      # iris centres (left, right)
    33, 133, 159, 145,             # left eye outer/inner/top/bottom
    263, 362, 386, 374,            # right eye outer/inner/top/bottom
)


def _paint_calibration_preview(
    canvas: np.ndarray,
    camera_bgr,
    face_landmarks_list,
    preview_x: int,
    preview_y: int,
) -> None:
    """Paint the camera-feed preview + iris overlay into a region of the canvas.

    `camera_bgr` is the most recent mirrored frame from the
    HandInput; `face_landmarks_list` is the corresponding face-mesh
    output (one entry per detected face).  We resize the camera
    frame to (preview_w, preview_h) then walk the eye landmarks and
    drop a coloured dot at each one's pixel position.  When no face
    is detected we paint the camera frame alone -- the user can see
    themselves moving, just no overlay, which is the diagnostic
    cue they need to reposition.
    """
    if camera_bgr is None:
        # First frame before the camera thread has published anything.
        # Paint a placeholder rect so the layout still reads.
        cv2.rectangle(
            canvas,
            (preview_x, preview_y),
            (preview_x + _CAL_PREVIEW_W, preview_y + _CAL_PREVIEW_H),
            (40, 40, 40), -1,
        )
        return

    src_h, src_w = camera_bgr.shape[:2]
    preview = cv2.resize(
        camera_bgr, (_CAL_PREVIEW_W, _CAL_PREVIEW_H),
        interpolation=cv2.INTER_AREA,
    )

    # Iris + eye-corner dots, scaled from camera-frame normalised
    # coords to preview-frame pixels.  Two colours so the iris reads
    # distinctly from the eye-corner markers.
    iris_color = (90, 220, 90)        # green -- iris centre
    corner_color = (60, 130, 240)     # orange -- eye corners / lids
    if face_landmarks_list:
        lms = face_landmarks_list[0].landmark
        for idx in _CAL_EYE_DRAW_INDICES:
            x_norm = lms[idx].x
            y_norm = lms[idx].y
            px = int(x_norm * _CAL_PREVIEW_W)
            py = int(y_norm * _CAL_PREVIEW_H)
            # Clip to preview rect -- a head turn can put a landmark
            # off the camera frame and we don't want cv2.circle to
            # crash on a negative index.
            if 0 <= px < _CAL_PREVIEW_W and 0 <= py < _CAL_PREVIEW_H:
                colour = iris_color if idx in (468, 473) else corner_color
                cv2.circle(preview, (px, py), 3, colour, -1)

    # Paste the preview into the calibration canvas.
    canvas[
        preview_y:preview_y + _CAL_PREVIEW_H,
        preview_x:preview_x + _CAL_PREVIEW_W,
    ] = preview
    # 1px border around the preview.  Same diagnostic visual the
    # OS-time thumbnail uses, repeated here for consistency.
    cv2.rectangle(
        canvas,
        (preview_x - 1, preview_y - 1),
        (preview_x + _CAL_PREVIEW_W, preview_y + _CAL_PREVIEW_H),
        (255, 255, 255), 1,
    )


def _paint_calibration_target(
    canvas: np.ndarray, cx: int, cy: int,
) -> None:
    """Draw a centred crosshair + ring at (cx, cy) on the calibration canvas.

    Two concentric circles: a large hollow ring at _CAL_TARGET_OUTER_R
    and a solid dot at _CAL_TARGET_INNER_R.  The ring gives the eye
    something to fixate on; the inner dot is the actual aim point.
    Plus a 1px crosshair so a user with shaky tracking can verify
    they're centred on the dot rather than just "near the ring".
    """
    cv2.circle(canvas, (cx, cy), _CAL_TARGET_OUTER_R,
               _CAL_TARGET_COLOR_BGR, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), _CAL_TARGET_INNER_R,
               _CAL_TARGET_COLOR_BGR, -1, cv2.LINE_AA)
    # Crosshair lines extending slightly past the ring.
    span = _CAL_TARGET_OUTER_R + 18
    cv2.line(canvas, (cx - span, cy), (cx + span, cy),
             (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - span), (cx, cy + span),
             (90, 90, 90), 1, cv2.LINE_AA)


def _paint_calibration_text(
    canvas: np.ndarray, has_face: bool, gaze_norm,
) -> None:
    """Two-line instruction at the bottom of the calibration canvas.

    Line 1: the main instruction ("Look at the centre dot, then
    press SPACE").  White, large.
    Line 2: a live readout that flips between two states:
        * "Searching for your face..."  when no face is detected
        * "Gaze x=0.50 y=0.50"           when a face is being tracked
    The readout gives the user instant confirmation the camera is
    seeing them; if they press SPACE before the readout shows
    numbers, the calibration silently fails and the OS boots without
    a baseline.
    """
    canvas_h, canvas_w = canvas.shape[:2]
    title = "Look at the centre dot, then press SPACE."
    subtitle_y = canvas_h - 50
    title_y = canvas_h - 80
    (tw, th), _ = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1,
    )
    cv2.putText(
        canvas, title,
        ((canvas_w - tw) // 2, title_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, _CAL_TEXT_COLOR_BGR, 1, cv2.LINE_AA,
    )

    if has_face and gaze_norm is not None:
        sub = f"Gaze  x={gaze_norm[0]:.2f}  y={gaze_norm[1]:.2f}"
        sub_color = _CAL_TEXT_DIM_BGR
    elif has_face:
        sub = "Face detected -- hold still"
        sub_color = _CAL_TEXT_DIM_BGR
    else:
        sub = "Searching for your face..."
        sub_color = _CAL_TEXT_DIM_BGR
    (sw, sh), _ = cv2.getTextSize(
        sub, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1,
    )
    cv2.putText(
        canvas, sub,
        ((canvas_w - sw) // 2, subtitle_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, sub_color, 1, cv2.LINE_AA,
    )

    hint = "ESC to skip (mouse-driven cursor)"
    (hw, _hh), _ = cv2.getTextSize(
        hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
    )
    cv2.putText(
        canvas, hint,
        ((canvas_w - hw) // 2, canvas_h - 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA,
    )


# 5-point calibration sequence.  Each entry is (label, fx, fy) where
# fx/fy are fractions of the screen extent (e.g. 0.1 = 10% from the
# left edge).  We inset the corners 10% / 15% so the user doesn't
# need to crane to look at the literal corner pixel -- that's
# uncomfortable and the iris range is more linear on the inset
# region.  Clockwise order from top-left so the eye's saccade
# pattern is predictable: TL -> TR -> BR -> BL -> Centre.
_CAL_POINTS: Final[tuple[tuple[str, float, float], ...]] = (
    ("top-left",     0.10, 0.15),
    ("top-right",    0.90, 0.15),
    ("bottom-right", 0.90, 0.85),
    ("bottom-left",  0.10, 0.85),
    ("centre",       0.50, 0.50),
)


def _run_eye_calibration(
    hand_input, screen_w: int, screen_h: int,
) -> bool:
    """5-point fullscreen calibration: TL -> TR -> BR -> BL -> Centre.

    For each fixation point, the user is shown a target dot in the
    corresponding screen position with instructions to "Look at the
    dot, press SPACE".  When SPACE is pressed, the current smoothed
    iris-norm is recorded as a sample paired with the screen target.
    After all five fixations are captured, `HandInput.fit_calibration`
    fits a 2x3 affine transform from iris-norm -> screen-pixel space
    via least-squares; that transform then drives `gaze_cursor()`
    during the OS main loop.

    The calibration screen reuses the already-fullscreen OS window
    (`WINDOW_NAME`) so the user's physical eye movements during cal
    match the geometry the OS will be running at.  Calibrating in a
    smaller window would produce a model that overshoots when the OS
    goes fullscreen: corners at the OS-extent edge would land beyond
    the trained iris range.

    Returns:
        True  -- all 5 samples captured AND the affine fit succeeded.
        False -- user pressed ESC, closed the window, or only
                 partial samples were captured.  When 3+ samples
                 were captured before bailing we still attempt a fit
                 (a 3-point fit is degraded but better than mouse).

    Args:
        hand_input: live HandInput instance.  Its background thread
                    must already be producing smoothed gaze readings,
                    so the caller should sleep briefly after construct
                    to let the camera warm up.
        screen_w, screen_h: fullscreen canvas dimensions in pixels.
                            The fixation positions are computed from
                            these via `_CAL_POINTS`.
    """
    hand_input.reset_calibration()

    instr_font  = load_font("text", 21)
    label_font  = load_font("text", 15)

    print(
        f"[phase8] 5-point calibration -- look at each dot and press "
        f"SPACE.  ESC bails (fit attempted with whatever samples "
        f"were captured)."
    )

    captured = 0
    for i, (label, fx, fy) in enumerate(_CAL_POINTS):
        tx = int(screen_w * fx)
        ty = int(screen_h * fy)

        # Hold-this-dot loop: re-render every ~15ms with the latest
        # face-detection state so the operator gets live feedback.
        # SPACE advances; ESC bails the whole calibration.
        while True:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            _paint_calibration_target(canvas, tx, ty)

            # Instructions kept on the opposite half of the screen
            # from the fixation dot so the text doesn't compete with
            # the target for the user's eye.  If the dot is in the
            # top half, text goes below the screen centre; vice
            # versa.
            instr_y = (
                screen_h * 5 // 8 if fy < 0.5 else screen_h * 3 // 8
            )
            draw_text(
                canvas,
                "Look at the dot, then press SPACE",
                x=screen_w // 2, y=instr_y,
                color_rgb=(245, 245, 250), font=instr_font,
                align="center",
            )
            draw_text(
                canvas,
                f"{i + 1} of {len(_CAL_POINTS)}  -  {label}",
                x=screen_w // 2, y=instr_y + 32,
                color_rgb=(170, 170, 185), font=label_font,
                align="center",
            )

            gaze_norm = hand_input.latest_gaze_norm()
            if gaze_norm is None:
                draw_text(
                    canvas,
                    "no face detected -- center yourself in the camera",
                    x=screen_w // 2, y=screen_h - 60,
                    color_rgb=(210, 190, 90), font=label_font,
                    align="center",
                )
            else:
                draw_text(
                    canvas,
                    f"gaze  x={gaze_norm[0]:.2f}  y={gaze_norm[1]:.2f}",
                    x=screen_w // 2, y=screen_h - 60,
                    color_rgb=(120, 120, 130), font=label_font,
                    align="center",
                )
            draw_text(
                canvas,
                "ESC to skip",
                x=screen_w // 2, y=screen_h - 32,
                color_rgb=(110, 110, 120), font=label_font,
                align="center",
            )

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(15) & 0xFF

            if key == _CAL_SPACE_KEY:
                if hand_input.add_calibration_sample((tx, ty)):
                    captured += 1
                    break
                # SPACE with no face -- print a hint, stay on this
                # fixation point until the user moves into view.
                print(
                    "[phase8] No face detected -- center yourself in "
                    "the camera and press SPACE again."
                )
                continue

            if key in (_CAL_ESC_KEY, ord("q"), ord("Q")):
                fitted = hand_input.fit_calibration()
                if captured >= 3:
                    print(
                        f"[phase8] Skipped early at {captured}/"
                        f"{len(_CAL_POINTS)} samples; "
                        f"fit {'succeeded' if fitted else 'failed'}."
                    )
                else:
                    print(
                        f"[phase8] Skipped early at {captured} samples; "
                        f"need 3+ for a fit -- cursor will follow the mouse."
                    )
                return fitted

            try:
                visible = cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE,
                )
            except cv2.error:
                visible = 0.0
            if visible < 1.0:
                # Window closed -- bail with whatever we have.
                fitted = hand_input.fit_calibration()
                return fitted

    fitted = hand_input.fit_calibration()
    if fitted:
        print(f"[phase8] Calibration complete ({captured} samples).")
    else:
        print(f"[phase8] Calibration fit FAILED at {captured} samples.")
    return fitted


def main() -> None:
    """Run the Phase 8 fullscreen loop until ESC, Q, or window close.

    Loop structure:
        1. Compute now_ms against the t0 baseline.
        2. Service scheduled notifications.
        3. Poll the cv2 mouse callback's context (cursor + click).
        4. Poll the HandInput for the latest HandFrame.
        5. Forward click + drag events into the Compositor.
        6. Compose the frame, layer FPS, layer hand thumbnail, show.
        7. Check for quit keys / window-close.

    The hand thumbnail is drawn AFTER both the compositor output AND
    the FPS HUD because the thumbnail is a diagnostic surface that
    must remain readable.  In practice the FPS HUD lives top-right
    and the thumbnail bottom-right, so they don't overlap; we paint
    the thumbnail last so a future relocation of either widget
    cannot accidentally obscure the more-important diagnostic.

    Defensive shutdown: hand_input.close() runs in a finally block
    so the camera handle is released even if compose_frame raises
    or the cv2 window is closed mid-frame.
    """
    args = _parse_cli(sys.argv[1:])

    # Camera selection happens BEFORE the fullscreen window is created
    # so the stdin prompt sits cleanly in the terminal -- if we built
    # the cv2 window first, the OS would grab the screen and the
    # operator couldn't see what they were typing.  Pass-through is
    # silent when --camera N is on the CLI; otherwise this prints the
    # available cameras and reads the operator's pick.
    chosen_camera = _resolve_camera_index(args.camera)

    # Optional --pretrained model load.  Done BEFORE HandInput so
    # the camera thread can receive the model and start running
    # inference from its very first frame.  try_load() returns None
    # on any failure (deps missing, weights missing, model load
    # failure); we print the reason and fall through to the
    # calibrated path -- the OS still boots either way.
    pretrained_gaze = None
    if args.pretrained:
        from src.pretrained_gaze import PretrainedGaze
        print(
            "[phase8] --pretrained set: loading L2CS-Net "
            "(skips the 5-point calibration screen)."
        )
        pretrained_gaze = PretrainedGaze.try_load()
        if pretrained_gaze is None:
            print(
                "[phase8] L2CS could not be loaded -- "
                "falling back to the calibrated path."
            )

    # Bring up the HandInput BEFORE the OS window so the background
    # camera thread is producing readings by the time we draw the
    # first calibration dot.  Order: camera up -> fullscreen window
    # claimed -> calibration (in that fullscreen window) -> OS main
    # loop (same window).
    if chosen_camera is None:
        hand_input = None
    else:
        hand_input = _try_construct_hand_input(
            chosen_camera, pretrained_gaze=pretrained_gaze,
        )

    # Open the fullscreen window FIRST so calibration shows at the
    # same geometry the OS will run at.  Calibrating in a smaller
    # window then handing off to a bigger fullscreen would shift the
    # corner targets outside the trained iris-norm range and the
    # cursor would land off-screen at the extremes.
    make_fullscreen_window(WINDOW_NAME)
    screen_w, screen_h = screen_size(WINDOW_NAME)

    # 5-point calibration.  Only shown when hand input is online
    # AND we're NOT in pretrained mode (pretrained skips per-user
    # calibration entirely).  Returns False on ESC / quit / window-
    # close / fewer than 3 samples; the OS still boots in mouse-
    # only mode in that case.  The 0.4s sleep gives the camera +
    # MediaPipe graph time to publish their first reading so the
    # calibration screen's "face detected" indicator is accurate
    # from frame 1.
    if hand_input is not None and pretrained_gaze is None:
        time.sleep(0.4)
        _run_eye_calibration(hand_input, screen_w, screen_h)
    elif hand_input is not None and pretrained_gaze is not None:
        # Give the camera thread a moment to publish its first
        # L2CS inference before the OS starts hit-testing against
        # the gaze cursor.  Without this brief pause the first
        # ~10 frames render with the mouse as cursor source, which
        # reads as a hiccup.
        time.sleep(0.4)

    # Reduced-motion detection.  Combines the argparse-parsed flag
    # (which is itself parsed against `_CLI_FLAG_REDUCED_MOTION`) with
    # the legacy sys.argv check and the cv2 R-key poll, so a user who
    # is still in the habit of typing `--reduced-motion` directly gets
    # the same effect.  reduced_motion_requested_via_cli is a no-op if
    # the flag isn't present, so OR-chaining with args.reduced_motion
    # is safe.
    reduced_motion = (
        args.reduced_motion
        or reduced_motion_requested_via_cli()
        or reduced_motion_requested_via_keypoll()
    )

    compositor = Compositor(reduced_motion=reduced_motion)
    ctx = _MainLoopContext()
    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, ctx)

    # Time baseline -- cv2 tick clock, the same one phase5's
    # now_ms_relative reads.  Compositor expects now_ms relative to
    # this baseline.
    t0_ticks = cv2.getTickCount()

    last_t = time.perf_counter() - (1.0 / 60.0)
    fps_ema = 0.0

    try:
        while True:
            canvas_w, canvas_h = screen_size(WINDOW_NAME)
            last_t, _dt, fps_ema = _update_fps(last_t, fps_ema)
            now_ms = now_ms_relative(t0_ticks)

            _fire_due_notifications(compositor, ctx, now_ms)

            # Consume the mouse click and the cached cursor xy in the
            # same step Phase 7 uses.
            mouse_pressed = ctx.pending_click
            ctx.pending_click = False
            mouse_xy = ctx.mouse_xy

            # Poll the hand input for the latest HandFrame.  Non-
            # blocking; returns _empty_hand_frame() if the thread has
            # not produced one yet (only on the very first frame).
            #
            # The hand input writes the most recent canvas size into
            # its shared state via step()'s parameters; the camera
            # thread reads that on its next iteration to scale
            # cursor / dx into canvas pixels.  If hand_input is None
            # (bring-up failed) we synthesise an empty HandFrame
            # locally so the downstream click/drag dispatch is
            # uniform.
            if hand_input is not None:
                hand_frame = hand_input.step(canvas_w, canvas_h)
            else:
                hand_frame = _no_hand_frame()

            # Cursor source: gaze when the calibration captured a
            # baseline AND the face is currently being tracked;
            # otherwise mouse.  hand_input.gaze_cursor() encapsulates
            # both gates -- it returns None when either condition
            # fails -- so the choice here is a single conditional.
            # This is the load-bearing eye-control behaviour: the
            # user's gaze IS the cursor, and the mouse becomes the
            # fallback for when the camera loses the face.
            #
            # gaze_xy is kept in scope (not None only when the gaze
            # pipeline is live) so the post-compose pass can compute
            # the locked tile AT the same coordinates the hover
            # hit-test just consumed.
            gaze_xy: Optional[tuple[int, int]] = None
            if hand_input is not None:
                gaze_xy = hand_input.gaze_cursor(canvas_w, canvas_h)
                if gaze_xy is not None:
                    mouse_xy = gaze_xy

            # Gaze lock: snap to the NEAREST home-tile so the user
            # gets unambiguous focus feedback even when the raw gaze
            # lands between icons.  Only meaningful on the home
            # screen; compositor's _paint_gaze_lock_chip skips the
            # paint during drags/snaps and when the OS is in app
            # state, so passing the latest snap every frame is safe.
            #
            # CRITICAL: when a lock is active we ALSO override
            # mouse_xy with the locked tile's centre.  Reason: the
            # compositor's click hit-test (and hover hit-test) reads
            # the raw mouse_xy, so without this override a hand
            # pinch fires "click at gaze_xy" which is usually a few
            # pixels into the gutter between tiles -- the click
            # then misses every tile rect and nothing happens.
            # Snapping mouse_xy to the locked tile's centre means
            # "what the chip shows" and "what the click hits" are
            # the same point.  Only override in HOME state; in APP
            # state the gaze should drive the close-button hit-test
            # at raw coordinates (the chip isn't shown there).
            if gaze_xy is not None and compositor.state == "home" and compositor.geometry is not None:
                locked = _nearest_tile_index(
                    gaze_xy, compositor.geometry.tile_rects,
                )
                compositor.set_gaze_lock(locked)
                compositor.set_close_button_focused(False)
                if locked is not None:
                    tx, ty, tw, th = compositor.geometry.tile_rects[locked]
                    mouse_xy = (tx + tw // 2, ty + th // 2)
            elif gaze_xy is not None and compositor.state == "app":
                # Close-button snap.  The close button is 32x32 in
                # the top-left corner; pixel-perfect gaze on a
                # target that small is unrealistic.  When the gaze
                # lands anywhere in the top-left ~ _CLOSE_SNAP_PX x
                # _CLOSE_SNAP_PX region, snap mouse_xy to the close
                # button's centre AND tell the compositor to paint
                # the focus ring -- the user gets a clear "pinch
                # closes the app" signal AND the pinch actually
                # registers on the button rect.
                compositor.set_gaze_lock(None)
                in_snap_zone = (
                    gaze_xy[0] < _CLOSE_SNAP_PX
                    and gaze_xy[1] < _CLOSE_SNAP_PX
                )
                compositor.set_close_button_focused(in_snap_zone)
                if in_snap_zone:
                    cb_x, cb_y, cb_w, cb_h = _close_button_rect()
                    mouse_xy = (cb_x + cb_w // 2, cb_y + cb_h // 2)
            else:
                compositor.set_gaze_lock(None)
                compositor.set_close_button_focused(False)

            # Page-drag dispatch.  Only meaningful on the home screen
            # -- drags that start inside an app are silently absorbed
            # by the Compositor's internal guards.  We still call
            # update_page_drag during drag_active to keep the
            # offset's cumulative value in sync; Compositor's
            # update_page_drag is a no-op when _drag_active is False
            # (i.e. when we never called begin_page_drag for this
            # gesture), so the calls below are safe in any state.
            if hand_frame.drag_just_started and compositor.state == "home":
                compositor.begin_page_drag(now_ms)
            if hand_frame.drag_active:
                compositor.update_page_drag(hand_frame.drag_dx)
            if hand_frame.drag_just_ended:
                compositor.end_page_drag(now_ms)

            # OR the mouse click and the hand tap.  Both fire at the
            # current cursor position (the mouse).  A simultaneous
            # mouse + hand tap collapses to a single click, which is
            # the correct behaviour: we don't want a stray hand
            # gesture to double-trigger an app launch.
            click_this_frame = mouse_pressed or hand_frame.click_now

            frame = compositor.compose_frame(
                now_ms=now_ms,
                canvas_w=canvas_w, canvas_h=canvas_h,
                mouse_xy=mouse_xy,
                mouse_pressed=click_this_frame,
            )

            # FPS HUD on top of the OS chrome.  Same anchor as every
            # other phase.  Drawn before the thumbnail so the
            # thumbnail can sit "over" the FPS HUD in the (rare)
            # case the two layouts overlap; in practice the FPS HUD
            # is top-right and the thumbnail bottom-right so there's
            # no overlap, but the order is defensive.
            draw_fps_hud(frame, fps_ema)

            # Hand thumbnail in the bottom-right corner.  Painted
            # AFTER the FPS HUD because the diagnostic value of the
            # thumbnail is higher: if the audience can see the
            # thumbnail, the operator can read the pinch indicator
            # and prove the system is seeing their hand.  No-op when
            # hand_input is None.
            if hand_input is not None:
                hand_input.draw_thumbnail(frame)

            # Note: the soft glowing gaze ball used to live here.  It
            # was replaced by the gaze-lock chip rendered inside the
            # compositor's _render_home -- the chip wraps the closest
            # tile to where the user is looking, giving an
            # unambiguous "you are focused on Music" signal even when
            # the underlying iris-delta gaze drifts a few px between
            # tiles.  See compositor.set_gaze_lock above.

            # Plain cv2.imshow -- same call Phase 7 uses.  AppKit
            # handles the Retina scaling on the display side; passing
            # a pre-upscaled buffer through `hidpi_imshow` on a
            # fullscreen window can land as a black frame on some
            # macOS Sequoia builds (the upscaled buffer doesn't match
            # the fullscreen backing and the window draws empty).
            # Trading the ~15-25 ms upscale for a reliable display is
            # the right call on a stage demo.
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (QUIT_KEY_ESC, QUIT_KEY_Q_L, QUIT_KEY_Q_U):
                break

            # Bail gracefully if the user closed the window some
            # other way (cmd-W, in the rare case the cv2 window
            # falls out of fullscreen).
            if cv2.getWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_VISIBLE,
            ) < 1.0:
                break
    finally:
        # Camera handle release.  Runs even if compose_frame raises
        # mid-loop.  hand_input.close() is idempotent so a double
        # call (this finally + an explicit close earlier) is safe.
        if hand_input is not None:
            try:
                hand_input.close()
            except Exception as exc:
                # We're already on the exit path; suppress secondary
                # exceptions but emit a warning so a future developer
                # spotting them can investigate.
                warnings.warn(
                    f"hand_input.close() failed during shutdown: {exc!r}",
                    stacklevel=1,
                )
        cv2.destroyAllWindows()


def _no_hand_frame():
    """Return a synthesised "no hand" HandFrame for the fallback path.

    The hand_input is None when bring-up failed; main() still wants
    something HandFrame-shaped to dispatch off, so we return one
    here with every flag at its "nothing happening" value.  This
    keeps the main loop free of "if hand_input is None" branches
    around every gesture check.

    Imported lazily inside the function so a hypothetical
    src.hands ImportError (cv2 / numpy missing -- very unusual but
    possible) doesn't crash this fallback path too.  In practice the
    import always succeeds since src.hands has no mediapipe import
    at module scope; the try/except is the belt-and-braces version
    of the same logic.
    """
    try:
        from src.hands import HandFrame
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
    except ImportError:
        # Fall back to a duck-typed namespace.  argparse's Namespace
        # supports attribute access just like a dataclass would; the
        # main loop only reads named fields, so this works as a
        # stand-in even though it isn't a HandFrame instance.
        return argparse.Namespace(
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


if __name__ == "__main__":
    main()
