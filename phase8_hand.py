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

from src.compositor import Compositor
from src.design import draw_fps_hud
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


def _try_construct_hand_input(camera_index: Optional[int]):
    """Try to bring up the HandInput.  Return it on success, None on failure.

    src.hands itself is pure-Python at module scope (mediapipe is
    imported lazily inside HandInput.__init__), so `from src.hands
    import HandInput` is safe in any environment where cv2 and numpy
    work.  If that import nonetheless fails (e.g. a corrupted
    install), we still want the OS to boot in mouse-only mode -- so
    we wrap even the import in a try/except.
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
        hand_input = HandInput(camera_index=camera_index)
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

    make_fullscreen_window(WINDOW_NAME)

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

    # Hand input bring-up.  None on failure; the rest of the loop
    # treats hand_input is None as "no thumbnail, no pinch click,
    # no page swipe" and the OS still runs in mouse-only mode.
    # `chosen_camera` is either the operator's pick (from --camera or
    # the interactive prompt) or None if the probe found nothing -- in
    # which case _try_construct_hand_input will also return None and
    # the OS boots mouse-only.
    if chosen_camera is None:
        hand_input = None
    else:
        hand_input = _try_construct_hand_input(chosen_camera)

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
            # same step Phase 7 uses.  The cursor reading stays the
            # mouse position -- Phase 8 deliberately does NOT route
            # the hand cursor into here.
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
