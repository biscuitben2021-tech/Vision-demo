"""
Phase 7 -- Polish layer.  END OF PART 1.

Goals (delta from Phase 6):
    * Live clock in the status bar.  The hard-coded "10:30 AM" from
      phase4_home_screen.py is replaced by `datetime.now().strftime(...)`;
      the time you see on the bar is the time on the wall.  Implemented
      inside src/compositor.py's _render_status_bar.
    * Notifications.  Two fake notifications (Apple Music, Messages)
      slide in from the right edge of the screen at 10s and 30s after
      process start.  Each is a 240x64 glass card with RADIUS_TILE_SMALL
      corners, holding for ~4s before sliding back off.
    * Smooth 300ms cross-fade between HOME and APP states.  The hard
      cut from Phase 6 is replaced by an alpha-blended transition: both
      scenes are composed into separate buffers and lerped per
      ease_emphasized(t) for the duration.
    * Everything from Phase 6 (HOME/APP state machine, fade-up entry,
      hover scale, close button, reduced-motion, FPS counter, ESC/Q
      quit) carries over -- but the state machine now lives inside the
      Compositor instead of as a free-standing AppState at module scope.

Why this phase exists:
    Phases 1-6 build up to a fake OS that already feels coherent.
    Phase 7 is the END of Part 1 -- the polish layer that takes "looks
    like a fake OS" to "could be shipped as a stage demo on its own".
    The brief is explicit: if Parts 2 (eye tracking) and 3 (hand
    tracking) collapse, Phase 7's script is what the user runs at the
    show.  Three features land here precisely because they are the
    three details that separate a tech demo from a working operating
    system in the eye of a non-technical audience.

Architecture choice:
    The phase-script-owns-state pattern from Phase 6 stops scaling at
    Phase 7's polish features.  A live clock + notifications + a cross-
    fade transition + the existing HOME/APP machinery is too many
    mutating-state-per-frame concerns for one main loop to coordinate
    while staying under the 30-line / no-globals rules.  Phase 7
    factors all of that into src/compositor.Compositor, which becomes
    the single per-frame entry point.  This main script is now a
    thin wrapper: open window, build compositor, pump frames, forward
    mouse events, render FPS, quit on ESC.

Module color-space convention:
    BGR for the cv2 pixel buffer; constants imported with `_BGR` go
    straight to cv2 calls, `_RGB` cross into PIL via `draw_text`.
    Same convention as src/design, src/tiles, src/icons, phase1-6.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final

import cv2

from src.compositor import Compositor
from src.design import load_font
from phase1_canvas import (
    FPS_EMA_ALPHA,
    FPS_FONT_SIZE,
    QUIT_KEY_ESC,
    QUIT_KEY_Q_L,
    QUIT_KEY_Q_U,
    WINDOW_NAME,
    make_fullscreen_window,
    screen_size,
)
from phase4_home_screen import render_fps_below_bar
from phase5_motion import (
    now_ms_relative,
    reduced_motion_requested_via_cli,
    reduced_motion_requested_via_keypoll,
)


# ----------------------------------------------------------------------------
# Notification schedule
# ----------------------------------------------------------------------------
#
# Two scheduled notifications, both content directly from the Phase 7
# prompt.  Encoded as a list of (delay_ms, title, body) tuples; the
# main loop consumes each tuple once when the wall clock crosses its
# delay.  Storing them in a list (rather than as two if-branches in
# the main loop) means adding a third notification is a one-line tuple
# append rather than another timed branch.
#
# `delay_ms` is measured against the process-start tick baseline, the
# same now_ms baseline the compositor uses.  Notifications spawn AT
# their delay; the slide-in animation takes the first 300ms of their
# 4-second lifetime, so the audience sees the card peeking in around
# (delay + 50ms) and reading it cleanly by (delay + 300ms).

NOTIF_SCHEDULE: Final[list[tuple[int, str, str]]] = [
    (10_000, "Apple Music", "Now playing: Bloom - Radiohead"),
    (30_000, "Messages",   "Mom: see you tonight"),
]


# ----------------------------------------------------------------------------
# Main-loop context -- shared by the mouse callback and the per-frame loop
# ----------------------------------------------------------------------------
#
# The cv2 mouse callback fires on cv2's event-pump thread (waitKey ticks
# it) and writes into this object; the main loop reads from it each
# frame and clears the consumed flag.  We do NOT call into Compositor
# from the callback -- the callback's contract is "drop the event into
# a queue, the main loop will service it".  This keeps state mutations
# linearised against the per-frame cadence rather than racing with it.
#
# Why a dataclass instead of two parallel module-level variables:
# cv2.setMouseCallback's `param` argument is the standard hand-off
# point for non-global mouse state in this codebase, and we want a
# single object reference to thread through it.  Same pattern phase5
# and phase6 use for their _MouseContext bundles; here the bundle is
# even simpler because the Compositor consumes click and position
# events at frame granularity, not on callback entry.


@dataclass
class _MainLoopContext:
    """Mutable bundle shared between the cv2 mouse callback and main().

    Fields:
        mouse_xy:        last known cursor position in canvas pixels.
                         Initialised to (-1, -1) so any pre-first-event
                         frame treats the cursor as off-canvas
                         (closest_tile returns None for negative coords,
                         which is the correct "no hover" behaviour).
        pending_click:   True if a LBUTTONDOWN has been observed since
                         the last consume.  The main loop checks this
                         each frame, passes it to compose_frame, then
                         clears it.  A single bool (not a counter) is
                         intentional: rapid double-clicks intra-frame
                         collapse to a single trigger, which matches
                         Phase 6's hard-cut behaviour.
        notif_fired:     boolean flag per scheduled notification.  Once
                         the main loop has fired one, the corresponding
                         slot flips to True so it never re-fires.
                         Sized to len(NOTIF_SCHEDULE) at construction
                         time so the index lookups stay branchless.
    """

    mouse_xy: tuple[int, int] = (-1, -1)
    pending_click: bool = False
    notif_fired: list[bool] = field(default_factory=lambda: [False] * len(NOTIF_SCHEDULE))


def _mouse_callback(
    event: int, x: int, y: int, flags: int, param: object,
) -> None:
    """Record mouse position + click events into the shared context.

    Two events of interest:
        EVENT_MOUSEMOVE   -- update the cached cursor position.
        EVENT_LBUTTONDOWN -- set pending_click; also update position so
                             a click that happens BEFORE any move (rare
                             but possible if the cursor was already
                             over the window) still lands at the right
                             coords on the next compose_frame.

    Other events (right-click, scroll, drag) are ignored: Phase 7's
    interaction surface is left-click only, matching Phase 6.

    This callback INTENTIONALLY does not call into the Compositor.
    Doing so would risk a partial-state-update mid-render if cv2's
    event thread fires concurrently with the main loop; queuing the
    event for the next frame is the cleaner data flow.
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

    Iterates NOTIF_SCHEDULE in order; for each slot whose `notif_fired`
    is still False AND whose `delay_ms` is <= the current `now_ms`,
    we call `compositor.enqueue_notification` and mark the slot as
    fired.  Once-only firing is a simple bool flip per slot -- no
    persistent storage needed because the demo's runtime is bounded
    to one stage appearance.

    Why this isn't on the Compositor: the schedule is a
    presentation-layer concern, not an OS-state concern.  The compositor
    knows how to render and reap a notification queue; the schedule of
    which notifications to spawn for THIS particular demo lives in the
    phase script next to the rest of the demo's hard-coded content.
    """
    for i, (delay_ms, title, body) in enumerate(NOTIF_SCHEDULE):
        if not ctx.notif_fired[i] and now_ms >= delay_ms:
            compositor.enqueue_notification(title, body, now_ms)
            ctx.notif_fired[i] = True


def _update_fps(prev_t: float, prev_ema: float) -> tuple[float, float, float]:
    """Update the FPS EMA and return (now, dt, new_ema).

    Same FPS update phase1-6 use, hoisted into a helper so main()
    stays focused on the per-frame data flow rather than carrying
    its own copy of the EMA seed.  Returns the wall-clock `now` so
    the caller can store it as next frame's `prev_t` without re-reading
    perf_counter (one read per frame, not two).

    Edge case: dt <= 0 (rare, but possible if perf_counter ticks
    backwards on a context switch).  We return the previous EMA
    unchanged in that case rather than dividing by zero or polluting
    the average.
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


def main() -> None:
    """Run the Phase 7 fullscreen loop until ESC, Q, or window close.

    Loop structure:
        1. Compute `now_ms` against the t0 baseline.
        2. Service the scheduled notifications.
        3. Read mouse position + pending click from the cv2 callback's
           shared context, consuming the click flag.
        4. Ask the compositor to compose a frame.
        5. Layer the FPS counter on top (kept out of the compositor
           because it is a diagnostic surface, not part of the OS).
        6. Show the frame; check for quit keys / window-close.

    `if __name__ == "__main__":` at the bottom of the file lets the
    script run standalone (`python phase7_polish.py`) and also
    importable for the post-loop's `from phase7_polish import main; main()`.
    """
    make_fullscreen_window(WINDOW_NAME)

    # Reduced-motion detection.  Same two-way check phase5/6 use --
    # CLI flag is the reliable mechanism, R-key poll is the
    # best-effort fallback.  In reduced-motion mode the Compositor's
    # cross-fade duration collapses to zero (the cut is effectively
    # hard, matching the rest of the reduced-motion vocabulary).
    reduced_motion = (
        reduced_motion_requested_via_cli()
        or reduced_motion_requested_via_keypoll()
    )

    fps_font = load_font(role="text", size=FPS_FONT_SIZE)
    compositor = Compositor(reduced_motion=reduced_motion)
    ctx = _MainLoopContext()

    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, ctx)

    # Time baseline.  cv2.getTickCount + getTickFrequency is the same
    # clock the compositor's internal motion timers use (via
    # phase5's now_ms_relative), so we share that domain rather than
    # picking up time.perf_counter for the now_ms math.
    t0_ticks = cv2.getTickCount()

    last_t = time.perf_counter() - (1.0 / 60.0)
    fps_ema = 0.0

    while True:
        canvas_w, canvas_h = screen_size(WINDOW_NAME)
        last_t, _dt, fps_ema = _update_fps(last_t, fps_ema)
        now_ms = now_ms_relative(t0_ticks)

        # Service scheduled notifications BEFORE composing the frame so
        # a notification due "right now" appears on this frame's
        # render rather than the next one.  Saves a frame of latency
        # in the audience's eye.
        _fire_due_notifications(compositor, ctx, now_ms)

        # Consume the click flag.  Even if the user clicks during the
        # compose call (cv2 might service waitKey on another thread),
        # we want the click to land on the NEXT compose_frame call,
        # not this one mid-render.  Reading then clearing here gives
        # us that one-shot semantics.
        mouse_pressed = ctx.pending_click
        ctx.pending_click = False
        mouse_xy = ctx.mouse_xy

        frame = compositor.compose_frame(
            now_ms=now_ms,
            canvas_w=canvas_w, canvas_h=canvas_h,
            mouse_xy=mouse_xy,
            mouse_pressed=mouse_pressed,
        )

        # FPS counter sits OUTSIDE the compositor: it's a diagnostic
        # surface, not part of the OS chrome the audience should see.
        # `render_fps_below_bar` anchors it below the status bar in
        # the top-right -- same placement phase4/5/6 use.
        render_fps_below_bar(frame, f"{fps_ema:5.1f} fps", fps_font, canvas_w)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (QUIT_KEY_ESC, QUIT_KEY_Q_L, QUIT_KEY_Q_U):
            break

        # Bail gracefully if the user closed the window some other way.
        if cv2.getWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_VISIBLE,
        ) < 1.0:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
