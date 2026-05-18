"""
phase7_demo.py -- Standalone hand-tracking demo with calibrated click.

A self-contained MediaPipe Hands visualiser that runs the
pinch / dead-zone / open-hand interaction model from the demo spec.

Interaction model:
    * Hand in a PINCH  -> cursor is active and follows the index
                           fingertip; nothing else fires.
    * Hand fully OPEN  -> a CLICK fires once at the current cursor
                           position.  User must return to a pinch
                           before another click can register.
    * In between       -> DEAD ZONE.  Nothing fires; prevents
                           accidental clicks on a half-formed
                           gesture.

Calibration (manual, SPACE-driven, at startup):
    Step 1: Prompt asks for a tight pinch.  The live camera
            thumbnail is shown so the user can confirm the hand is
            visible and tracked.  SPACE captures the current
            openness sample into `closed_openness` and advances.
    Step 2: Same pattern for a wide-open hand; SPACE captures into
            `open_openness`.  A sanity check rejects the sample if
            `open_openness <= closed_openness + epsilon`.
    Step 3: "Calibration complete. Press SPACE to begin." -- on
            SPACE the calibration UI exits and the main loop runs.

    Manual capture was chosen over a timed countdown because the
    user knows when their hand is correctly posed; a timer either
    rushes them or wastes seconds, and a noisy frame at the moment
    the timer hits zero can pollute the calibration.

Thresholds derived from calibration:
    click_threshold     = closed + CLICK_TRIGGER_RATIO * (open - closed)
    dead_zone_threshold = open  * DEAD_ZONE_OPEN_RATIO

State machine per frame (using `openness`, defined below):
    openness <  click_threshold       -> PINCHING
    click_threshold <= openness < dead_zone_threshold -> DEAD ZONE
    openness >= dead_zone_threshold    -> OPEN  (eligible for click)

Openness metric:
    Mean Euclidean distance from the wrist landmark to all five
    fingertip landmarks, normalised by the wrist-to-middle-MCP
    distance.  Wrist-to-fingertip-avg is more stable than thumb-to-
    index distance because it captures the FULL hand state -- a
    fast index-only flick wouldn't trigger a false click.  The
    normalisation removes camera-distance bias: hold the hand
    nearer or further from the lens and the ratio is unchanged.

Module color-space convention:
    Every cv2 buffer in this module is BGR.  MediaPipe wants RGB so
    the camera frame is converted once per loop iteration before
    `hands.process(...)`; no other site crosses the byte-order
    boundary.

Press ESC or Q to quit at any point.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Final, Optional

import cv2
import mediapipe as mp
import numpy as np


# ============================================================================
# Tuning constants -- every threshold the demo cares about
# ============================================================================
#
# Changing these is the operator's primary way to adapt the demo to a new
# user's hand or a new lighting condition.  Each constant carries a
# WHY-it-affects-what comment so a future tuner doesn't have to read the
# state machine to understand which way to push the knob.

# How far between calibrated_closed and calibrated_open to place the
# PINCHING / DEAD-ZONE boundary.  Below the boundary the hand reads as
# "pinched and tracking"; above it the hand starts to feel ambiguous.
# Lower = easier to leave PINCH (more accidental dead-zone hits);
# Higher = pinch state holds longer (cursor stays active even as the
# hand starts opening, which can mask early-open jitter).
CLICK_TRIGGER_RATIO: Final[float] = 0.70

# Fraction of calibrated_open the hand must reach to count as OPEN
# (eligible for a click).  Acts as the DEAD ZONE / OPEN boundary.
# 0.90 means "only a very deliberate full-open hand fires a click".
# Lower = easier clicks but accidental fires on half-open gestures.
DEAD_ZONE_OPEN_RATIO: Final[float] = 0.90

# Minimum consecutive frames the hand must stay in the OPEN state
# before a click commits.  Filters out flicker clicks from a single
# noisy frame where the openness spiked past the threshold.
# Higher = more robust to noise, but slower-feeling clicks.
MIN_OPEN_FRAMES: Final[int] = 3

# Cooldown after a successful click before another click can fire.
# Doubles as a guard against the same gesture firing twice while
# the hand is still re-pinching; combined with the "must return to
# PINCH before next click" rule below this prevents double-fire on
# every plausible motion.
CLICK_COOLDOWN_FRAMES: Final[int] = 20

# Length of the click-ripple animation in frames.  The ripple
# expands from a small dot to ~3x its initial radius and fades to
# zero alpha over this many frames.  Shorter = subtler feedback;
# longer = more obvious but the rings start stacking on rapid
# clicks.
RIPPLE_FRAMES: Final[int] = 20

# Minimum gap between open and closed openness for a calibration
# pair to be considered usable.  Below this we reject and ask for a
# wider open hand; the FSM math (threshold = closed + ratio*(open -
# closed)) collapses if the two means are essentially identical, so
# this guards against the user calibrating two near-identical poses
# (either a too-loose "pinch" or a too-cramped "open").
CALIBRATION_MIN_GAP: Final[float] = 0.05

# How many frames a transient calibration hint ("No hand visible",
# "Open value not high enough") stays on screen after SPACE was
# pressed in an invalid state.  At ~30fps this is ~2 seconds --
# long enough to read, short enough not to feel like the UI froze.
FLASH_FRAMES: Final[int] = 60

# Event log: how many recent events to display in the bottom-left.
EVENT_LOG_MAX: Final[int] = 6


# ============================================================================
# Layout constants
# ============================================================================

# Demo canvas.  Black 1280x720 -- standard 16:9 at a comfortable
# laptop-screen size.  Not promoted to fullscreen by default; the
# demo is meant to run windowed so the operator can see the
# terminal during calibration debugging.
CANVAS_W: Final[int] = 1280
CANVAS_H: Final[int] = 720

# Webcam capture size.  cv2.VideoCapture.set is advisory -- the
# driver picks the closest supported mode -- but asking for 720p
# at the camera matches the canvas's aspect ratio so the
# thumbnail doesn't squash.
CAMERA_W: Final[int] = 1280
CAMERA_H: Final[int] = 720

# Camera thumbnail in the bottom-right corner.  320x180 = 16:9 at
# 1/4 canvas width.  Big enough to see the hand mesh on; small
# enough that it doesn't dominate the demo.
THUMB_W:      Final[int] = 320
THUMB_H:      Final[int] = 180
THUMB_MARGIN: Final[int] = 16

# Window name shown in the cv2 title bar.  Cosmetic only.
WINDOW_NAME: Final[str] = "Hand Demo (phase7)"

# Quit keys.  ESC + lowercase / uppercase Q so muscle memory from
# every other phase script keeps working here.
QUIT_KEYS: Final[tuple[int, ...]] = (27, ord("q"), ord("Q"))

# Advance / confirm key during calibration.  ASCII 32 is the
# SPACE bar -- the universal "I'm ready, take the sample" gesture
# from every dual-action UI on the planet.
KEY_SPACE: Final[int] = 32


# ============================================================================
# Colours (all BGR -- this file never goes through PIL)
# ============================================================================

COLOR_BLACK:     Final[tuple[int, int, int]] = (0, 0, 0)
COLOR_WHITE:     Final[tuple[int, int, int]] = (255, 255, 255)
# Dimmed white for the dead-zone cursor + event log timestamps.
# Picked so it reads as "intentionally muted" rather than "the
# system is broken and barely lit".
COLOR_DIM:       Final[tuple[int, int, int]] = (140, 140, 140)
# Red border for "no hand detected" warning on the thumbnail.
# Saturated enough to grab the eye at a glance.
COLOR_RED:       Final[tuple[int, int, int]] = (60, 60, 235)


# ============================================================================
# Hand-landmark indices (MediaPipe Hands convention)
# ============================================================================
#
# MediaPipe publishes 21 landmarks per hand.  Indices we care about:
#   0  = WRIST            (origin for the openness metric)
#   4  = THUMB_TIP
#   8  = INDEX_TIP        (drives the cursor position)
#   9  = MIDDLE_MCP       (knuckle of the middle finger -- the
#                          stable normalisation anchor; doesn't move
#                          much regardless of finger pose)
#   12 = MIDDLE_TIP
#   16 = RING_TIP
#   20 = PINKY_TIP
#
# Full MediaPipe diagram:
#   https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

WRIST_IDX:       Final[int]               = 0
INDEX_TIP_IDX:   Final[int]               = 8
MIDDLE_MCP_IDX:  Final[int]               = 9
FINGERTIP_IDXS:  Final[tuple[int, ...]]   = (4, 8, 12, 16, 20)

# Connection pairs for drawing the hand mesh.  Pulled from
# MediaPipe so we don't have to hand-roll the 20-line graph.
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


# ============================================================================
# Small data types
# ============================================================================

@dataclass
class CalibrationData:
    """Captured calibration values + derived per-frame thresholds.

    `closed` and `open` are the raw means of the openness metric
    over the calibration steps; `click_threshold` and
    `dead_zone_threshold` are the per-frame decision boundaries
    derived from them.  Keeping derived values on the object means
    the main loop never re-derives them every frame.
    """
    closed: float
    open: float
    click_threshold: float
    dead_zone_threshold: float

    @classmethod
    def from_raw(cls, closed: float, open_: float) -> "CalibrationData":
        click = closed + CLICK_TRIGGER_RATIO * (open_ - closed)
        dead  = open_ * DEAD_ZONE_OPEN_RATIO
        return cls(
            closed=closed, open=open_,
            click_threshold=click, dead_zone_threshold=dead,
        )


@dataclass
class Ripple:
    """A single in-flight click ripple.

    The animation is purely a function of (frame_idx - frame_started),
    so we don't need to store any per-frame state -- the renderer
    derives radius and alpha from the age each frame.
    """
    x: int
    y: int
    frame_started: int


# ============================================================================
# Openness metric -- the load-bearing measurement
# ============================================================================

def compute_openness(landmarks) -> float:
    """Return the wrist-to-fingertip mean distance, normalised by hand size.

    Why the mean-of-five rather than thumb-to-index distance:
    thumb-to-index moves first when a hand starts to open, so a
    fast tap could spike the metric briefly even though the rest of
    the fingers haven't unfurled.  Averaging across all five
    fingertips makes the metric track WHOLE-HAND state, which is
    the actual signal we want for "fully open vs pinched".

    Normalised by the wrist-to-middle-MCP distance.  That segment
    barely changes regardless of finger pose, so it's the natural
    "hand size" scalar.  The resulting ratio is camera-distance
    invariant: a hand 30cm from the lens and the same hand 70cm
    away produce the same openness number for the same pose.

    Returns 0.0 if the hand size is degenerate (rare; happens on
    the very first frame when the tracker hasn't locked yet) --
    that value sits comfortably in the DEAD ZONE so it doesn't
    accidentally fire a click during a tracker hiccup.
    """
    wrist  = _landmark_xyz(landmarks[WRIST_IDX])
    middle = _landmark_xyz(landmarks[MIDDLE_MCP_IDX])
    hand_size = float(np.linalg.norm(middle - wrist))
    if hand_size < 1e-6:
        return 0.0
    tips = [_landmark_xyz(landmarks[i]) for i in FINGERTIP_IDXS]
    mean_spread = float(np.mean([np.linalg.norm(t - wrist) for t in tips]))
    return mean_spread / hand_size


def _landmark_xyz(lm) -> np.ndarray:
    """Pluck (x, y, z) out of a MediaPipe NormalizedLandmark proto."""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


# ============================================================================
# Camera open + MediaPipe Hands construction
# ============================================================================

def verify_camera_at_startup() -> cv2.VideoCapture:
    """Pre-flight the camera before any UI shows; exit cleanly on failure.

    Opens cv2.VideoCapture(0), confirms both `isOpened()` and a
    successful read of a real frame, then returns the live handle.
    On any failure prints the human-readable diagnostics block
    described in the demo spec (camera-in-use / permission /
    not-present) and exits with code 1 -- no traceback, no half-
    shown window.  Returning a working VideoCapture lets the caller
    skip a second open in `main()`.

    Why a dedicated pre-flight rather than a defensive try/except
    inside the main loop: the failure modes here (camera taken by
    another app, permission denied) need a friendly message to
    stdout, not a stack trace from somewhere mid-calibration.  By
    the time the user is staring at the prompt, the camera is
    proven; everything downstream can assume `cap.read()` mostly
    works.
    """
    cap = cv2.VideoCapture(0)
    ok_open = cap.isOpened()
    ok_read = False
    if ok_open:
        ok_read, frame = cap.read()
        ok_read = bool(ok_read) and frame is not None
    if not (ok_open and ok_read):
        cap.release()
        cv2.destroyAllWindows()
        print("Camera unavailable: cv2.VideoCapture(0) could not be opened.")
        print(
            "If the camera is in use by another app (Zoom, FaceTime, "
            "Photo Booth), close it and try again."
        )
        print(
            "If you have not granted camera permission, open System "
            "Settings -> Privacy & Security -> Camera and enable it "
            "for your terminal, then re-run."
        )
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    return cap


def build_hands_model() -> mp.solutions.hands.Hands:
    """Construct the MediaPipe Hands tracker with single-hand demo settings."""
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


# ============================================================================
# Calibration screens (manual, SPACE-driven)
# ============================================================================
#
# Three screens form the calibration UI: PINCH, OPEN, READY.  Each
# renders the same chrome -- prompt centred above the midline, a
# hint line below it (live openness readout / "No hand visible" /
# transient error flash), and the camera thumbnail in the bottom-
# right so the user can see what the tracker sees before they hit
# SPACE.  The user controls the pace: a sample is only taken on a
# deliberate SPACE press while a hand is in view.

def render_prompt(canvas: np.ndarray, prompt: str) -> None:
    """Paint the primary instruction prompt onto a black calibration canvas.

    Centred horizontally, sitting above the midline so the hint
    line below has room.  Using cv2.putText (not PIL) intentionally
    -- this is a diagnostic surface, not OS chrome, so the font
    fidelity bar is lower and we get to skip the PIL/numpy alpha-
    composite cost.
    """
    h, w = canvas.shape[:2]
    (pw, _), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(
        canvas, prompt, ((w - pw) // 2, h // 2 - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2, cv2.LINE_AA,
    )


def render_hint(canvas: np.ndarray, hint: str, color: tuple[int, int, int]) -> None:
    """Paint the secondary hint line (live readout or transient error flash).

    Smaller than the prompt and rendered in DIM by default, RED
    when used for an error flash (e.g. "No hand visible") so the
    user sees the problem immediately.
    """
    if not hint:
        return
    h, w = canvas.shape[:2]
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        canvas, hint, ((w - hw) // 2, h // 2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )


def _process_camera_frame(
    cap: cv2.VideoCapture,
    hands: mp.solutions.hands.Hands,
) -> tuple[Optional[np.ndarray], object, bool, Optional[float]]:
    """Read+mirror one camera frame and run MediaPipe.

    Returns (mirrored_bgr_frame, results, has_hand, openness).
    The frame is None if the read failed; callers should skip
    rendering the thumbnail when that happens.
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None, False, None
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    has_hand = bool(results.multi_hand_landmarks)
    openness = (
        compute_openness(results.multi_hand_landmarks[0].landmark)
        if has_hand else None
    )
    return frame, results, has_hand, openness


def _render_calibration_frame(
    canvas: np.ndarray,
    prompt: str,
    hint: str,
    hint_color: tuple[int, int, int],
    camera_frame: Optional[np.ndarray],
    results,
    has_hand: bool,
) -> None:
    """Paint one full calibration frame: black bg + prompt + hint + thumbnail.

    Factored out because all three calibration screens (PINCH,
    OPEN, READY) share the exact same chrome -- only the prompt
    and hint change between them.  The thumbnail is always shown
    when a camera frame is available, so the user can see what the
    tracker sees during every step (Bug 3 fix).
    """
    canvas[:] = COLOR_BLACK
    render_prompt(canvas, prompt)
    render_hint(canvas, hint, hint_color)
    if camera_frame is not None:
        multi = results.multi_hand_landmarks if results is not None else None
        draw_camera_thumbnail(canvas, camera_frame, multi, has_hand)


def _build_step_hint(
    has_hand: bool, openness: Optional[float], flash: str,
) -> tuple[str, tuple[int, int, int]]:
    """Choose the hint line + colour for one calibration frame.

    Priority: an active flash message (SPACE was pressed in an
    invalid state) always wins -- it's the most actionable signal
    the user just produced.  Otherwise: "No hand visible." in RED
    if the tracker can't see one, or a live openness readout in
    DIM if it can.  The readout is the user's "is my pose close
    enough" guide before they commit.
    """
    if flash:
        return flash, COLOR_RED
    if not has_hand:
        return "No hand visible. Show your hand to the camera.", COLOR_RED
    assert openness is not None  # has_hand implies a sample was taken
    return f"Openness: {openness:.3f}  --  press SPACE to capture.", COLOR_DIM


def run_calibration_step(
    cap: cv2.VideoCapture,
    hands: mp.solutions.hands.Hands,
    prompt: str,
    reject_predicate: Optional[Callable[[float], bool]] = None,
    reject_message: str = "",
) -> Optional[float]:
    """Run one manual SPACE-driven openness capture.

    Returns the captured openness when SPACE is pressed while a
    hand is visible and (optionally) the sample passes the
    `reject_predicate(sample) -> bool` test.  Returns None if the
    user quits (ESC/Q) during the step.

    `reject_predicate` is used by the OPEN step to enforce the
    "open value must clearly exceed closed value" sanity check
    without leaking that domain logic into this function.
    """
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    flash_text: str = ""
    flash_remaining: int = 0
    while True:
        camera_frame, results, has_hand, openness = _process_camera_frame(
            cap, hands,
        )
        if flash_remaining > 0:
            flash_remaining -= 1
        else:
            flash_text = ""
        hint, hint_color = _build_step_hint(
            has_hand, openness, flash_text,
        )
        _render_calibration_frame(
            canvas, prompt, hint, hint_color, camera_frame, results, has_hand,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            return None
        if key == KEY_SPACE:
            if not has_hand or openness is None:
                flash_text = "No hand visible -- pinch in view, then press SPACE again."
                flash_remaining = FLASH_FRAMES
                continue
            if reject_predicate is not None and reject_predicate(openness):
                flash_text = reject_message
                flash_remaining = FLASH_FRAMES
                continue
            return openness


def run_ready_screen(
    cap: cv2.VideoCapture,
    hands: mp.solutions.hands.Hands,
) -> bool:
    """Show 'Calibration complete' until SPACE is pressed.

    Returns True if the user pressed SPACE to begin, False if they
    quit (ESC/Q).  Keeps showing the thumbnail so the user can
    sanity-check tracking one last time before going live.
    """
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    prompt = "Calibration complete. Press SPACE to begin."
    while True:
        camera_frame, results, has_hand, _ = _process_camera_frame(
            cap, hands,
        )
        _render_calibration_frame(
            canvas, prompt, "", COLOR_DIM, camera_frame, results, has_hand,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            return False
        if key == KEY_SPACE:
            return True


# ============================================================================
# Renderers
# ============================================================================

def draw_hand_landmarks(
    canvas: np.ndarray, landmarks, frame_w: int, frame_h: int,
) -> None:
    """Draw the 21 hand landmarks + their connecting bones on the canvas.

    Landmarks are in normalised [0..1] coords; we scale to canvas
    pixels.  Bones first (so the joint dots sit on top of the
    lines), then dots, then a slightly bigger highlight on the
    index-tip so the cursor-driving landmark is visually distinct
    from the other twenty.
    """
    pts = [
        (int(lm.x * frame_w), int(lm.y * frame_h))
        for lm in landmarks
    ]
    # Bones first.
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], COLOR_WHITE, 2, cv2.LINE_AA)
    # Joint dots.
    for p in pts:
        cv2.circle(canvas, p, 4, COLOR_WHITE, -1, cv2.LINE_AA)


def draw_fingertip_cursor(
    canvas: np.ndarray, x: int, y: int, state: str,
) -> None:
    """Draw the index-fingertip cursor dot, dimmed in the dead zone.

    The dim/bright contrast IS the visual feedback for the dead
    zone: a user whose hand drifts into the ambiguous range sees
    the cursor grey-out and immediately knows the system won't
    register their click.  Without this cue they'd be guessing
    whether the next pinch-open will fire.
    """
    fill = COLOR_DIM if state == "DEAD ZONE" else COLOR_WHITE
    cv2.circle(canvas, (x, y), 10, fill, -1, cv2.LINE_AA)
    # White outline on top so the cursor still reads on a busy
    # landmark mesh; outline is full-white regardless of state so
    # the cursor never DISAPPEARS in the dead zone.
    cv2.circle(canvas, (x, y), 11, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_ripples(
    canvas: np.ndarray, ripples: list[Ripple], frame_idx: int,
) -> None:
    """Render every in-flight click ripple.

    Ripple geometry: 20px starting radius growing to 80px over
    RIPPLE_FRAMES, with the stroke colour faded linearly to black
    so the ring dissolves cleanly.  Caller is responsible for
    pruning expired ripples (we don't mutate the list here).
    """
    for r in ripples:
        age = frame_idx - r.frame_started
        if age < 0 or age >= RIPPLE_FRAMES:
            continue
        t = age / RIPPLE_FRAMES                  # 0..1
        radius = int(20 + t * 60)                # 20 -> 80 px
        fade = 1.0 - t                           # 1.0 -> 0.0
        color = tuple(int(c * fade) for c in COLOR_WHITE)
        cv2.circle(canvas, (r.x, r.y), radius, color, 2, cv2.LINE_AA)


def draw_state_label(canvas: np.ndarray, state: str) -> None:
    """Big top-left state read-out ('PINCHING' / 'OPEN' / 'DEAD ZONE' / 'NO HAND').

    Anchored at (24, 60) so the baseline sits well clear of the
    top edge at large font sizes.  Larger than the event log so
    the operator can read it at a glance during the demo -- the
    state label is the single most useful debugging signal here.
    """
    cv2.putText(
        canvas, state, (24, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLOR_WHITE, 3, cv2.LINE_AA,
    )


def draw_event_log(canvas: np.ndarray, events: list[str]) -> None:
    """Render the last EVENT_LOG_MAX events stacked bottom-left.

    Newest event is at the top of the visible stack (closest to
    the canvas's vertical middle); oldest is at the bottom.  This
    matches the convention of every shell tail -- newest at the
    end -- read upward.
    """
    line_h = 22
    base_y = CANVAS_H - 24
    # Iterate newest-first; place each line ascending up the canvas.
    for i, line in enumerate(reversed(events)):
        y = base_y - i * line_h
        cv2.putText(
            canvas, line, (24, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_DIM, 1, cv2.LINE_AA,
        )


def draw_camera_thumbnail(
    canvas: np.ndarray, camera_frame: np.ndarray,
    multi_hand_landmarks, has_hand: bool,
) -> None:
    """Paste the live camera feed + hand mesh into the bottom-right corner.

    Border colour codes detection: WHITE when a hand is being
    tracked, RED when no hand is found.  The border is the
    operator's at-a-glance camera health check -- a red border in
    the middle of a demo means "fix your camera before continuing".
    """
    if camera_frame is None or camera_frame.size == 0:
        return
    thumb = cv2.resize(
        camera_frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA,
    )
    if multi_hand_landmarks:
        # Re-draw the mesh on the thumbnail at THUMB resolution.
        # Re-running the projection here (rather than scaling the
        # main-canvas pts) keeps the thumbnail's mesh crisp at any
        # canvas size.
        _draw_landmarks_on_image(thumb, multi_hand_landmarks[0].landmark)
    x0 = CANVAS_W - THUMB_W - THUMB_MARGIN
    y0 = CANVAS_H - THUMB_H - THUMB_MARGIN
    canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = thumb
    border = COLOR_WHITE if has_hand else COLOR_RED
    cv2.rectangle(
        canvas, (x0 - 1, y0 - 1), (x0 + THUMB_W, y0 + THUMB_H),
        border, 2, cv2.LINE_AA,
    )


def _draw_landmarks_on_image(img: np.ndarray, landmarks) -> None:
    """Draw 21 landmarks + connections directly onto a BGR image, in place.

    Used by the thumbnail painter; identical geometry to
    `draw_hand_landmarks` but scaled to the image's own size so
    callers don't have to thread a (frame_w, frame_h) tuple
    through.  Smaller dots / thinner lines so the mesh doesn't
    obscure the camera feed at thumbnail resolution.
    """
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], COLOR_WHITE, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 2, COLOR_WHITE, -1, cv2.LINE_AA)


# ============================================================================
# Click state machine
# ============================================================================

@dataclass
class ClickFsm:
    """Tracks open-frame count + cooldown for click commit decisions.

    A separate object so the per-frame state machine logic is in
    one place rather than spread across loop-local ints.  Mutates
    in place via `tick(state)` and exposes `should_fire()` for the
    caller to consult before drawing a ripple.
    """
    open_frames: int = 0
    cooldown:    int = 0
    armed:       bool = True   # True iff a pinch has been observed since last click

    def tick(self, state: str) -> bool:
        """Advance the FSM by one frame.  Returns True iff a click commits."""
        if self.cooldown > 0:
            self.cooldown -= 1
        # The "must re-pinch" rule: once we've fired a click, the
        # FSM disarms; a transition through PINCHING re-arms it.
        if state == "PINCHING":
            self.armed = True
        if state == "OPEN":
            self.open_frames += 1
        else:
            self.open_frames = 0
        if (
            self.armed
            and self.open_frames >= MIN_OPEN_FRAMES
            and self.cooldown == 0
        ):
            self.cooldown = CLICK_COOLDOWN_FRAMES
            self.armed = False
            self.open_frames = 0
            return True
        return False


# ============================================================================
# State classification
# ============================================================================

def classify_state(
    openness: float, cal: CalibrationData, has_hand: bool,
) -> str:
    """Map openness + face-presence to a single state string for the FSM.

    Returns one of "NO HAND" / "PINCHING" / "DEAD ZONE" / "OPEN".
    Pulled out so the main loop has one obvious line for the
    state decision and the FSM has one input to consume.
    """
    if not has_hand:
        return "NO HAND"
    if openness < cal.click_threshold:
        return "PINCHING"
    if openness < cal.dead_zone_threshold:
        return "DEAD ZONE"
    return "OPEN"


# ============================================================================
# Main loop
# ============================================================================

def main_loop(
    cap: cv2.VideoCapture,
    hands: mp.solutions.hands.Hands,
    cal: CalibrationData,
) -> None:
    """Run the demo until ESC / Q.  Reads camera, runs FSM, paints canvas."""
    events: deque[str] = deque(maxlen=EVENT_LOG_MAX)
    ripples: list[Ripple] = []
    fsm = ClickFsm()
    last_logged_state: str = ""
    frame_idx = 0

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    while True:
        ok, camera_frame = cap.read()
        if not ok or camera_frame is None:
            # Camera blip; show the previous frame, briefly wait.
            cv2.waitKey(5)
            continue
        camera_frame = cv2.flip(camera_frame, 1)
        results = hands.process(
            cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB),
        )
        has_hand = bool(results.multi_hand_landmarks)
        landmarks = (
            results.multi_hand_landmarks[0].landmark if has_hand else None
        )

        openness = compute_openness(landmarks) if landmarks else 0.0
        state = classify_state(openness, cal, has_hand)

        # FSM + click commit (writes to event log + spawns a ripple).
        if fsm.tick(state) and landmarks is not None:
            tip_x = int(landmarks[INDEX_TIP_IDX].x * CANVAS_W)
            tip_y = int(landmarks[INDEX_TIP_IDX].y * CANVAS_H)
            events.append(
                f"{time.strftime('%H:%M:%S')} -- CLICK at "
                f"({tip_x}, {tip_y})"
            )
            ripples.append(
                Ripple(x=tip_x, y=tip_y, frame_started=frame_idx),
            )

        # Log meaningful state transitions (skip DEAD ZONE / NO HAND
        # so the event log stays focused on actionable signal).
        if state != last_logged_state and state in ("PINCHING", "OPEN"):
            events.append(f"{time.strftime('%H:%M:%S')} -- {state}")
        last_logged_state = state

        # ----- paint -----
        canvas[:] = COLOR_BLACK
        if landmarks is not None:
            draw_hand_landmarks(canvas, landmarks, CANVAS_W, CANVAS_H)
            tip_x = int(landmarks[INDEX_TIP_IDX].x * CANVAS_W)
            tip_y = int(landmarks[INDEX_TIP_IDX].y * CANVAS_H)
            draw_fingertip_cursor(canvas, tip_x, tip_y, state)
        # Prune expired ripples in place, then draw the survivors.
        ripples = [
            r for r in ripples
            if frame_idx - r.frame_started < RIPPLE_FRAMES
        ]
        draw_ripples(canvas, ripples, frame_idx)
        draw_state_label(canvas, state)
        draw_event_log(canvas, list(events))
        draw_camera_thumbnail(
            canvas, camera_frame,
            results.multi_hand_landmarks, has_hand,
        )

        cv2.imshow(WINDOW_NAME, canvas)
        frame_idx += 1
        if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
            break


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """Verify camera, run manual calibration, hand off to the main loop."""
    # Bug 1 fix: pre-flight the camera *before* any window opens.
    # On failure, `verify_camera_at_startup` prints diagnostics and
    # calls sys.exit(1); we never reach the line below.
    cap = verify_camera_at_startup()
    hands = build_hands_model()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

    try:
        # Step 1 -- PINCH.  No reject predicate: any visible hand
        # produces a usable closed sample (we sanity-check the
        # *pair* in step 2 instead).
        closed = run_calibration_step(
            cap, hands,
            "Pinch your fingers together, then press SPACE to capture.",
        )
        if closed is None:
            return
        # Step 2 -- OPEN.  Reject if the new sample isn't clearly
        # wider than the closed one; without this guard the FSM
        # thresholds collapse and every frame reads as DEAD ZONE.
        open_ = run_calibration_step(
            cap, hands,
            "Open your hand wide, then press SPACE to capture.",
            reject_predicate=lambda s: s <= closed + CALIBRATION_MIN_GAP,
            reject_message=(
                "Open value not high enough -- try again with a wider open hand."
            ),
        )
        if open_ is None:
            return
        cal = CalibrationData.from_raw(closed, open_)
        print(
            f"[phase7_demo] Calibrated: closed={closed:.3f}, "
            f"open={open_:.3f}, click_threshold="
            f"{cal.click_threshold:.3f}, dead_zone_threshold="
            f"{cal.dead_zone_threshold:.3f}"
        )
        # Step 3 -- READY.  SPACE to begin; ESC/Q still bails.
        if not run_ready_screen(cap, hands):
            return
        main_loop(cap, hands, cal)
    finally:
        # Defensive shutdown: release the camera + close any cv2
        # window even if the loop raises.  Hands has no explicit
        # close() in older mediapipe versions; .close() lands in
        # 0.10.20+, so we only call it if available.
        cap.release()
        cv2.destroyAllWindows()
        close = getattr(hands, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
