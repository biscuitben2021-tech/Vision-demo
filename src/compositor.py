"""
Master frame compositor for the Vision OS demo (Phase 7).

The Compositor class owns ALL operating-system UI state and is the single
per-frame entry point for rendering.  Phases 1-6 distributed responsibility
across phase scripts (each phase script owned its own state machine, mouse
callback, and paint dispatch); Phase 7 collapses that down to one object
so polish features -- the cross-fade between HOME and APP states, the
notification queue, the live clock -- can reason over a coherent state
graph instead of patching three phases' main loops.

State machine (extended from Phase 6):

    +----------+   mouse click  +---------------+ blend done  +-------+
    |   HOME   | -------------> | TRANSITIONING | -----------> |  APP  |
    |          |                |  home -> app  |              |       |
    +----------+                +---------------+              +-------+
         ^                              ^                          |
         |                              |                          |
         |          +---------------+   |                          |
         +----------| TRANSITIONING |<--+                          |
                    |  app -> home  |<-------------- close button -+
                    +---------------+

The two transition arrows are NEW in Phase 7 -- Phase 6 hard-cut between
the two states.  A TransitionState records the from/to screens and the
start_ms; while `state == "transitioning"`, each frame composes BOTH
screens into separate buffers and alpha-blends them.  When the eased
progress hits 1.0, we snap to the destination state.

Three new polish layers also live here:

    1. Live clock -- the status bar's clock placeholder is replaced by
       `datetime.datetime.now().strftime("%-I:%M %p")` per frame.  No
       extra timer thread; just a per-frame strftime call (microsecond
       cost) so we always read the wall clock.
    2. Notifications -- a glass-card slide-in queue rendered on top of
       whichever scene is current (or the transition blend).  Each
       notification owns its spawn_ms + lifetime; the compositor reaps
       expired entries and computes per-card x-offsets from the eased
       slide curve.
    3. Cross-fade transitions -- the "hard cut" from Phase 6 is replaced
       by a 300ms eased alpha blend.  See `_compose_transition` for the
       blend math.

Module color-space convention:
    The cv2 pixel buffer is BGR throughout this module.  Every numpy
    canvas is allocated as (h, w, 3) uint8 BGR.  Constants from
    src.design imported with `_BGR` suffix go straight to cv2 calls;
    `_RGB` constants are forwarded to draw_text which handles the
    PIL boundary.  No raw colour tuples are introduced here.

Public surface (everything below the dataclasses is module-internal):
    Notification         -- queued notification record
    TransitionState      -- in-flight home/app cross-fade record
    Compositor           -- the OS state container + frame painter
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
from PIL import ImageFont

from src.apps import RENDERERS
from src.design import (
    GAP_VIEWPORT,
    NAV_HEIGHT,
    RADIUS_TILE_SMALL,
    TEXT_MUTED_RGB,
    TEXT_ON_DARK_RGB,
    draw_text,
    load_font,
)
from src.icons import draw_glass_panel, paint_warm_aurora
from src.motion import ease_emphasized

# Phase imports.  Same pattern Phase 6 uses to lean on Phase 5 / Phase 4
# helpers: don't reinvent geometry / painters that already exist, just
# pull them in.  These are public-by-convention symbols even though
# they don't have a top-level __all__ declaration.
from phase4_home_screen import (
    APPS,
    GRID_COLS,
    STATUS_BAR_PAD_X,
    STATUS_WORDMARK,
    TILE_H,
    TILE_W,
    _get_label_font,
    _get_status_bar_fonts,
    make_dark_canvas,
)
from phase5_motion import (
    GridGeometry,
    MotionState,
    _render_tile_with_motion,
    build_grid_geometry,
    build_motion_state,
    closest_tile,
    paint_grid_with_motion,
)
from phase6_app_window import (
    _close_button_rect,
    _draw_close_button,
    _point_in_rect,
)
# FadeUpState is the per-tile entry-animation record.  Page 2+ uses
# PRE-COMPLETED FadeUpStates (start_ms negative enough that .value()
# immediately returns 1.0 opacity / 0.0 y-offset) so the swipe itself
# IS the visual entry -- a second fade-up would compete with the user's
# horizontal drag and look broken.
from src.motion import FadeUpState as _FadeUpState


# ============================================================================
# Constants -- transition + notification timing and geometry
# ============================================================================
#
# Pulled out as module-level Finals so the WHY for each value lives next
# to its declaration and a future tweak (longer transition, taller
# notification card) only touches one spot.  These are NOT in src/design
# because they are Phase-7-specific surface decisions, not site-wide
# design tokens; src/design holds the cross-phase tokens that EVERY
# phase imports.

# Cross-fade between HOME and APP scenes.  300ms is the prompt-mandated
# value; it's short enough that a user clicking around feels snappy and
# long enough that the eye reads the transition as an animation rather
# than a flicker.  We use the same ease_emphasized curve as the rest of
# the demo to keep the motion vocabulary uniform.
TRANSITION_DURATION_MS: int = 300

# Notification card dimensions.  240x64 reads as "small enough to ignore,
# large enough to read at a glance" -- the same proportions iPadOS
# notifications use.  Anchored to the right edge with GAP_VIEWPORT
# inset so the card sits in the same visual column as the FPS counter.
NOTIFICATION_W: int = 240
NOTIFICATION_H: int = 64

# Per-card animation timing.  300ms slide-in / 300ms slide-out brackets
# the lifetime; the hold phase is the remaining middle section.  Pull
# the slide and total lifetime apart so the lifetime can be tweaked
# without redoing the in/out math.
NOTIFICATION_SLIDE_MS: int = 300
NOTIFICATION_DEFAULT_LIFETIME_MS: int = 4000

# Vertical stacking gap between adjacent notification cards.  12px is
# tighter than GAP_TILE (16px) on purpose -- the cards belong to one
# stack and read as a column, not as independent floating tiles.
NOTIFICATION_STACK_GAP: int = 12

# Top edge of the notification stack.  The prompt fixes this at
# `NAV_HEIGHT + 16` so the topmost card sits 16px below the status bar.
NOTIFICATION_Y_BASE: int = NAV_HEIGHT + 16

# Right gutter for the notification stack.  16px (GAP_VIEWPORT) matches
# the canvas-edge inset every other UI element uses.
NOTIFICATION_RIGHT_GUTTER: int = GAP_VIEWPORT

# Inner padding for the notification card's text.  16px horizontal puts
# the title's leading edge a clean inset from the rounded corner; the
# vertical offsets below (12 for title, 36 for body) split the 64px
# height into a 24-baseline title row and a 24-baseline body row.
NOTIFICATION_PAD_X: int = 16
NOTIFICATION_TITLE_Y: int = 12
NOTIFICATION_BODY_Y: int = 36

# Notification typography.  Apple's actual notification banner uses SF
# Pro Text Semibold at 14px for the title; load_font does not expose
# Text Semibold, so we use Text Regular here.  The body falls one
# point smaller at 13 to establish hierarchy without a weight change.
# Documented as a KNOWN COMPROMISE alongside the existing instances in
# src/tiles.py and phase4_home_screen.py.
NOTIFICATION_TITLE_FONT_SIZE: int = 14
NOTIFICATION_BODY_FONT_SIZE: int = 13


# ============================================================================
# Multi-page home -- Phase 8 paging constants + roster for page 2+
# ============================================================================
#
# Phase 8 adds a swipe-able home screen.  The hand drags horizontally;
# the page contents follow the hand; on release the page snaps to the
# nearest neighbour if the drag passed the threshold or back to the
# current page otherwise.
#
# Visual model mirrors iPad / Vision Pro: hand moves LEFT -> tiles move
# LEFT (toward off-screen-left) -> the RIGHT neighbour page slides in
# from the right edge.  The threshold below is "if the cumulative
# horizontal drag exceeds 25% of canvas_w, commit to the neighbour;
# otherwise rubber-band back".  iPadOS uses ~25-30% for the same
# decision; 25% feels right at this canvas size and matches the user's
# stated brief.
#
# Snap animation duration is 250ms eased through ease_emphasized, the
# same easing curve the rest of the demo uses for in-flight motion.
# 250ms is short enough that the audience reads the page change as
# direct manipulation and long enough that the new page reads as a
# distinct surface rather than as a hard cut.

PAGE_SNAP_DURATION_MS: int = 250
PAGE_SWIPE_THRESHOLD_FRACTION: float = 0.25   # 25% of canvas_w; see block above

# APPS_PAGE_2 is the roster for the SECOND home page.  Same eight
# app_ids as phase4_home_screen.APPS but in a visibly different order
# -- this is the simplest "page 2 looks different" rule that does not
# require new icon glyphs / renderers.  Introducing fresh app_ids on
# page 2 would have no entry in `src/apps.RENDERERS` and the open-app
# path would crash; reusing the existing eight keeps every transition
# valid.
#
# The order is APPS reversed: Demo first, Safari last.  Any arbitrary
# permutation works; reversal is the most visually obvious
# "different page" rearrangement at a glance.

APPS_PAGE_2: list[tuple[str, str]] = list(reversed(APPS))


# ============================================================================
# Dataclasses -- queued notification + in-flight transition
# ============================================================================


@dataclass
class Notification:
    """One slide-in glass-card notification, queued for rendering.

    Fields:
        title:        bold-ish first line shown on the card.
        body:         secondary line, truncated with "..." if it exceeds
                      the card's inner content width.
        spawn_ms:     wall-clock millisecond at which the card was
                      enqueued.  Drives the slide-in start and the
                      reap-after-lifetime check.
        lifetime_ms:  total time on screen, including the 300ms
                      slide-in and 300ms slide-out.  At
                      `now - spawn_ms >= lifetime_ms` the card is
                      dropped from the queue.  Defaults to 4000ms --
                      enough to read the body once at a glance.

    Lifecycle (with default lifetime 4000ms):
        [   0,  300] ms  -- sliding in from the right.
        [ 300, 3700] ms  -- holding at final x.
        [3700, 4000] ms  -- sliding back out to the right.
        [4000,  inf] ms  -- dropped from the queue on the next iteration.
    """

    title: str
    body: str
    spawn_ms: int
    lifetime_ms: int = NOTIFICATION_DEFAULT_LIFETIME_MS


@dataclass
class TransitionState:
    """Cross-fade in flight from one screen to another.

    Fields:
        from_screen:  the screen we were on before the click.  The
                      compositor renders this into a "from" buffer
                      each frame during the transition.
        to_screen:    the screen we are heading to.  Rendered into a
                      separate "to" buffer; alpha-blended over the
                      from buffer per `ease_emphasized(t)`.
        to_app_id:    set when going home -> app; the app id to
                      render in the destination scene.  None when
                      going app -> home.
        from_app_id:  set when going app -> home; the app id to keep
                      rendering in the FROM scene as it fades out.
                      None when going home -> app (FROM scene is the
                      home grid, which doesn't need an app id).
        start_ms:     wall-clock millisecond at which the transition
                      began.  Drives the eased progress.
        duration_ms:  total transition duration.  Defaults to 300ms.

    Snap condition:  when `(now_ms - start_ms) >= duration_ms`, the
    Compositor clears this object, sets `state` to `to_screen`, and
    if `to_screen == "app"` records `current_app_id = to_app_id`.
    That snap is done in `_advance_transition_if_done` -- centralising
    the math in one helper so the rules about which fields move where
    only live in one place.
    """

    from_screen: Literal["home", "app"]
    to_screen: Literal["home", "app"]
    to_app_id: Optional[str]
    from_app_id: Optional[str]
    start_ms: int
    duration_ms: int = TRANSITION_DURATION_MS


# ============================================================================
# Compositor -- the OS state container + per-frame painter
# ============================================================================
#
# The Compositor is intentionally large by this codebase's standards
# (~10 public + private methods) because it represents the central
# dispatch for the entire OS UI.  Splitting it into many small classes
# would just shuffle the same fields around; keeping the state in one
# object makes the per-frame data flow obvious: state in -> compose
# frame -> state advanced for next frame.
#
# Why a class instead of free functions: the OS state graph IS state.
# The from/to-app ids, the transition record, the notification queue,
# the cached geometry, the motion state -- they all evolve together
# in response to a single per-frame call.  Threading 10 mutable
# variables through helper functions would be ugly; bundling them in
# `self` is the structured alternative CLAUDE.md's "no global mutable
# state" rule explicitly allows.


class Compositor:
    """The Vision OS frame compositor and state owner.

    Owns the screen state machine, the in-flight transition (if any),
    the notification queue, hover/fade-up motion state, and the cached
    grid geometry.  Exposes one per-frame method (`compose_frame`) plus
    a notification-enqueue helper.

    Construction-time inputs:
        reduced_motion: True if --reduced-motion was passed or R was
                        held at startup.  Stored on the instance; the
                        motion state is built with the same flag and
                        the cross-fade duration collapses to a hard
                        cut when this is True (an animation flag set
                        once at startup, not per-call).

    Public methods:
        compose_frame(now_ms, canvas_w, canvas_h, mouse_xy,
                      mouse_pressed) -> np.ndarray
            The single per-frame call.  Returns the next BGR frame.
        enqueue_notification(title, body, now_ms) -> None
            Append a fresh Notification record to the queue.

    Per-frame data flow inside compose_frame:

          mouse_pressed -> _handle_click          # may start a transition
          state == "transitioning" ?
              advance, possibly snap to dest      # _advance_transition_if_done
          mouse_xy -> hover update (only on HOME) # set HoverState
          render the appropriate scene(s)
          render the notification overlay
          return the frame to the caller (FPS / etc. layered by caller)

    The state machine never observes mouse drag or scroll; the only
    interactive event Phase 7 cares about is a left-click (delivered
    via `mouse_pressed=True` exactly once per click).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, *, reduced_motion: bool) -> None:
        """Initialise the OS state at process start (t=0).

        We don't take a canvas size at construction -- compose_frame's
        first call rebuilds the geometry the first time it sees one.
        Deferring it means the caller doesn't have to thread a
        screen-size lookup into Compositor's constructor before the
        cv2 window is even created.

        `t0_ticks` is the wall-clock baseline that every now_ms passed
        into compose_frame is measured against externally; we store it
        only for use by sites that need cv2's tick clock directly
        (currently none -- the caller does all wall-clock math).
        Keeping it on the instance for symmetry with phase6's AppState
        and to allow a future Phase to query it without re-plumbing.
        """
        self.state: Literal["home", "app", "transitioning"] = "home"
        self.current_app_id: Optional[str] = None
        self.transition: Optional[TransitionState] = None
        self.notifications: list[Notification] = []
        self.reduced_motion: bool = reduced_motion

        # Motion + geometry are rebuilt lazily on the first compose_frame
        # call (when we know the canvas dimensions); seed with None so
        # the lazy-init guard reads cleanly.
        self.geometry: Optional[GridGeometry] = None
        self.motion: MotionState = build_motion_state(reduced_motion=reduced_motion)

        # Font cache for the notification card text.  Loading PIL
        # truetype fonts is not free; we lazy-load on first use the
        # same way src/tiles.py does, so importing this module stays
        # side-effect free.
        self._notif_title_font: Optional[ImageFont.FreeTypeFont] = None
        self._notif_body_font: Optional[ImageFont.FreeTypeFont] = None

        # Cached for compose_frame's status bar and FPS-counter call
        # sites.  These are the same fonts phase4/5/6 use; reusing
        # them keeps the typography identical to the earlier phases.
        self._wordmark_font, self._clock_font = _get_status_bar_fonts()
        self._label_font = _get_label_font()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def enqueue_notification(self, title: str, body: str, now_ms: int) -> None:
        """Append a fresh Notification record to the queue.

        The card immediately enters its slide-in phase on the next
        compose_frame call.  `spawn_ms` is the caller-supplied wall
        clock so the queue's animation timing stays in lock step with
        the rest of the compositor's now_ms parameter -- not with
        whichever clock the caller might be using internally.
        """
        self.notifications.append(Notification(
            title=title, body=body, spawn_ms=now_ms,
        ))

    def compose_frame(
        self,
        now_ms: int,
        canvas_w: int,
        canvas_h: int,
        mouse_xy: tuple[int, int],
        mouse_pressed: bool,
    ) -> np.ndarray:
        """Return the next BGR frame to display; drives all state transitions.

        Always returns a fresh BGR canvas of shape (canvas_h, canvas_w, 3);
        callers should pass it straight to cv2.imshow without further
        compositing except for the FPS counter, which is the caller's
        responsibility (kept out of the compositor to preserve the
        "diagnostics live outside the OS" boundary phase1-6 established).
        """
        self._ensure_geometry(canvas_w, canvas_h)
        self._advance_transition_if_done(now_ms)

        # Click handling happens BEFORE the hover update so that a
        # click-on-tile starts the transition with the correct app id
        # before any state mutation could mask it.  During a transition
        # we suppress all clicks (see _handle_click).
        if mouse_pressed:
            self._handle_click(now_ms, mouse_xy)

        # Hover update is gated on the current state -- hover scale
        # should only apply on the HOME scene (not on APP, not during
        # transitions).  Restricting the set_hover call rather than
        # letting it record decay/rise transitions during APP keeps
        # the post-close return-to-home visual clean: no spurious
        # hover bumps from cursor positions we accumulated while the
        # cursor was over an app.
        if self.state == "home":
            assert self.geometry is not None
            mx, my = mouse_xy
            tile_id = closest_tile(mx, my, self.geometry.tile_rects)
            self.motion.hover_state.set_hover(tile_id, now_ms)

        # Compose the actual frame.  Three top-level cases driven by
        # the current state; each delegates to a sibling helper.
        frame = make_dark_canvas(canvas_w, canvas_h)
        if self.state == "transitioning":
            self._compose_transition(frame, canvas_w, canvas_h, now_ms, mouse_xy)
        else:
            self._compose_screen(self.state, frame, canvas_w, canvas_h,
                                 now_ms, mouse_xy)

        # Notifications sit OVER the composed frame -- they need to
        # remain legible during a transition rather than fading
        # mid-slide-in.  The status bar is NOT redrawn here: the
        # static-path call in _compose_screen already painted it for
        # the home case, the static APP case intentionally omits the
        # bar (immersive app), and the transition path painted the
        # bar inside the home sub-frame so it fades naturally with the
        # blend.  A redraw here would override that fade.
        self._render_notifications(frame, canvas_w, canvas_h, now_ms)
        return frame

    # ------------------------------------------------------------------
    # Lazy initialisers
    # ------------------------------------------------------------------

    def _ensure_geometry(self, canvas_w: int, canvas_h: int) -> None:
        """(Re)build the grid geometry cache if the canvas size has changed.

        The first call after construction has `self.geometry is None`
        and unconditionally builds the cache.  Subsequent calls only
        rebuild when the canvas dimensions differ from the cached
        ones -- the same resize-detection pattern phase4/5/6's main
        loops use, hoisted into the compositor so the caller doesn't
        have to know about geometry at all.
        """
        if (
            self.geometry is None
            or self.geometry.canvas_w != canvas_w
            or self.geometry.canvas_h != canvas_h
        ):
            self.geometry = build_grid_geometry(canvas_w, canvas_h)

    # ------------------------------------------------------------------
    # State machine -- transition advancement + click handling
    # ------------------------------------------------------------------

    def _advance_transition_if_done(self, now_ms: int) -> None:
        """If an in-flight transition has reached t=1, snap to the dest state.

        Called at the top of every compose_frame.  Two outcomes:
            1. No transition in flight, or transition still has time
               left -- this is a no-op.
            2. Transition's `elapsed >= duration_ms` -- clear the
               transition record and adopt `to_screen` as the new
               `state`.  If `to_screen == "app"`, also record
               `current_app_id` from the transition's to_app_id.
        Centralising the snap conditions here means future Phases that
        want a different transition type only touch this one helper.
        """
        if self.state != "transitioning" or self.transition is None:
            return
        elapsed = now_ms - self.transition.start_ms
        if elapsed < self.transition.duration_ms:
            return

        # Snap.  Whichever screen we were heading to is now the resting
        # state; the from_screen's app id is no longer relevant.  Clear
        # current_app_id when landing on home -- it stays set when
        # landing on app so subsequent frames can dispatch to the
        # renderer.
        target = self.transition.to_screen
        if target == "app":
            self.current_app_id = self.transition.to_app_id
        else:
            self.current_app_id = None
        self.state = target
        self.transition = None

    def _handle_click(self, now_ms: int, mouse_xy: tuple[int, int]) -> None:
        """Translate a one-shot click into either a transition start or a no-op.

        Three cases:
            1. state == "home" and click is on a tile rect -- start a
               home -> app transition aimed at that tile's app id.
            2. state == "app" and click is on the close-button rect --
               start an app -> home transition.
            3. state == "transitioning" -- absorb the click silently.
               Queuing clicks during transitions is the kind of "buffer
               your intent" UX feature that helps prosumer apps but
               distracts in a stage demo, where unintended re-opens
               break the choreography.
        """
        if self.state == "transitioning":
            return

        mx, my = mouse_xy

        if self.state == "home":
            assert self.geometry is not None
            tile_id = closest_tile(mx, my, self.geometry.tile_rects)
            if tile_id is None:
                return
            app_id, _display_name = APPS[tile_id]
            self._start_transition(
                from_screen="home", to_screen="app",
                to_app_id=app_id, from_app_id=None, now_ms=now_ms,
            )
            return

        # state == "app"
        if _point_in_rect(mx, my, _close_button_rect()):
            self._start_transition(
                from_screen="app", to_screen="home",
                to_app_id=None, from_app_id=self.current_app_id, now_ms=now_ms,
            )

    def _start_transition(
        self, *,
        from_screen: Literal["home", "app"],
        to_screen: Literal["home", "app"],
        to_app_id: Optional[str],
        from_app_id: Optional[str],
        now_ms: int,
    ) -> None:
        """Record a fresh TransitionState and flip the compositor state.

        Centralised so the three call sites (the two _handle_click
        branches today, plus any future programmatic transition
        triggers in Phase 8+) share one entry point.  In reduced-motion
        mode the transition duration collapses to a single frame
        (`duration_ms = 0`) so the cross-fade is effectively a hard
        cut -- which is exactly what the accessibility setting asks
        for, and matches the rest of the reduced-motion behaviour
        established in Phase 5.
        """
        duration = 0 if self.reduced_motion else TRANSITION_DURATION_MS
        self.transition = TransitionState(
            from_screen=from_screen, to_screen=to_screen,
            to_app_id=to_app_id, from_app_id=from_app_id,
            start_ms=now_ms, duration_ms=duration,
        )
        self.state = "transitioning"

    # ------------------------------------------------------------------
    # Scene composition -- home, app, transition
    # ------------------------------------------------------------------

    def _compose_screen(
        self,
        target: Literal["home", "app"],
        frame: np.ndarray,
        w: int,
        h: int,
        now_ms: int,
        mouse_xy: tuple[int, int],
    ) -> None:
        """Render one of HOME or APP into `frame` for the static (non-transition) path.

        Used only when `state == "home"` or `state == "app"`.  Includes
        the home-side status bar because in the static case there is
        no separate "draw the bar after the blend" pass.  The
        transition path renders its two sub-scenes directly via
        `_render_for_transition_side` (which deliberately skips the
        status bar in both halves so a single full-opacity bar can be
        drawn over the blended result at the end of compose_frame).
        """
        if target == "home":
            self._render_home(frame, w, h, now_ms)
            self._render_status_bar(frame, w)
            return
        # target == "app"
        self._render_app(frame, w, h, now_ms)
        # No status bar on APP -- visionOS apps are immersive; the
        # system bar collapses when an app is foregrounded.

    def _render_home(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        now_ms: int,
    ) -> None:
        """Paint the 4x2 grid + label band into `frame`, no chrome.

        Hover scale is applied via the compositor's MotionState exactly
        as Phase 5 does -- but only when the live state is "home", which
        the caller (compose_frame) gates by NOT calling set_hover during
        transitions.  During a transition the renderer reads frozen
        MotionState (no fresh set_hover calls have happened on this
        frame), so each tile's hover_state lerp decays toward 1.0
        naturally over HOVER_SCALE_DURATION_MS.  Visually this is correct:
        the cursor moves over the half-faded home tile during the
        transition and the hover bumps gently decay away rather than
        snapping.

        Note that `w` and `h` are part of the signature for symmetry
        with `_render_app` even though `paint_grid_with_motion` reads
        the geometry off `self.geometry` rather than the function args.
        Keeping the signature uniform is a small price for callers
        that dispatch by string ("render this side") rather than by
        type.
        """
        assert self.geometry is not None
        # Paint the warm-aurora wallpaper as the first layer so the
        # glass panels above it have varied content to refract through.
        # The aurora is cached per resolution -- per-frame cost is just
        # a single np.copyto.  Without this layer the underlying
        # wallpaper is BG_DARK (#000) and the Liquid Glass effect
        # becomes nearly invisible: blurring black yields black, and
        # the frost tint reads as a flat darker rectangle.
        paint_warm_aurora(frame, w, h)
        paint_grid_with_motion(
            frame, self.geometry, self.motion, self._label_font, now_ms,
        )

    def _render_app(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        now_ms: int,
    ) -> None:
        """Paint the current app's content + close button into `frame`.

        The app id is either `current_app_id` (static APP state) or the
        transition's from/to app id (transition halves); the resolution
        of which id to use is done by the caller.  We default to
        `current_app_id` when none of the transition fields name an
        app -- but that path only fires on the static-APP frame; during
        a transition the caller renders this twice, each with a
        different id, via `_render_app_for_id`.

        Falls back to HOME if `current_app_id is None` -- this is the
        same defensive branch Phase 6 has at its dispatch site;
        absorbing the inconsistency instead of crashing matches the
        "loud failure for bugs, silent recovery for races" rule the
        codebase follows.
        """
        if self.current_app_id is None:
            self._render_home(frame, w, h, now_ms)
            return
        self._render_app_for_id(frame, self.current_app_id)

    def _render_app_for_id(self, frame: np.ndarray, app_id: str) -> None:
        """Paint app `app_id` fullscreen + the close button on top.

        Pulled out as a small helper so the transition path can render
        the FROM app (in the home->app case, the from_app_id) and the
        TO app independently into separate buffers without each having
        to know about the current_app_id field.  RENDERERS is the same
        dispatch table Phase 6 uses; we just look up the right entry.
        """
        h, w = frame.shape[:2]
        renderer = RENDERERS[app_id]
        renderer(frame, w, h)  # type: ignore[operator]
        _draw_close_button(frame, app_id)

    def _compose_transition(
        self,
        canvas: np.ndarray,
        w: int,
        h: int,
        now_ms: int,
        mouse_xy: tuple[int, int],
    ) -> None:
        """Alpha-blend the FROM and TO scenes into `canvas` per eased t.

        Algorithm:
            1. Resolve the eased progress t in [0, 1].  Clamp at the
               endpoints (Compositor only reaches this method when
               state == "transitioning", which already implies the
               transition is in-flight, but defensive clamping costs
               nothing and prevents float-rounding > 1.0 artefacts).
            2. Allocate two fresh BG_DARK canvases the same size as
               `canvas` and call _render_for_transition_side for each.
               Separate buffers (rather than rendering both into the
               same canvas) is mandatory: the TO scene must NOT see
               any FROM pixels under it, or its glass panels would
               brighten the wrong wallpaper.
            3. Blend per-pixel with addWeighted (cv2's hand-tuned C++
               implementation is faster than a Python lerp on (h, w, 3)
               buffers).  `out = (1-t)*from + t*to`.
            4. Splice the blended result into `canvas`.  The notification
               overlay paints on top after this returns; the status bar
               is NOT redrawn over the blend (the from-side rendered it
               at full opacity and the blend fades it naturally with the
               home scene, which is the desired visionOS behaviour).

        Why we don't pre-allocate the two sub-canvases on `self`:
        their lifetime is bounded to one compose_frame call; allocating
        each frame is one numpy.empty per buffer, which is microseconds
        at typical screen sizes.  Keeping the lifetime short means a
        canvas-size change is handled implicitly (next frame allocates
        new buffers of the new size) without an explicit resize path.
        """
        assert self.transition is not None
        elapsed = now_ms - self.transition.start_ms
        # duration_ms can be 0 in reduced-motion mode; guard with a max
        # to avoid a divide-by-zero on the t = elapsed / duration line.
        # When duration is 0 we clamp t to 1.0 (the transition is
        # already done as far as the eye is concerned), and the snap
        # at the top of compose_frame will land on the destination
        # state on the very next frame.
        if self.transition.duration_ms <= 0:
            t = 1.0
        else:
            raw = elapsed / float(self.transition.duration_ms)
            t = max(0.0, min(1.0, raw))
        eased = ease_emphasized(t)

        from_buf = make_dark_canvas(w, h)
        to_buf = make_dark_canvas(w, h)
        self._render_for_transition_side(from_buf, w, h, now_ms, side="from")
        self._render_for_transition_side(to_buf, w, h, now_ms, side="to")

        # addWeighted: dst = src1 * a + src2 * b + gamma.  gamma=0 here.
        # Float intermediate isn't needed -- cv2 handles uint8 saturation
        # for us, which gives bit-exact results for an alpha blend of two
        # uint8 buffers and is significantly faster than the float path.
        blended = cv2.addWeighted(from_buf, 1.0 - eased, to_buf, eased, 0.0)
        canvas[:, :] = blended

    def _render_for_transition_side(
        self,
        buf: np.ndarray,
        w: int,
        h: int,
        now_ms: int,
        side: Literal["from", "to"],
    ) -> None:
        """Render one side of the in-flight transition into `buf`.

        `side` selects whether to draw the FROM scene or the TO scene
        on this buffer.  The screen kind ("home" / "app") comes from
        the corresponding field of the transition record.  This is a
        thin dispatcher so the blend math in _compose_transition only
        has to know "give me the two halves" without re-implementing
        the home/app branch logic.
        """
        assert self.transition is not None
        if side == "from":
            screen = self.transition.from_screen
            app_id = self.transition.from_app_id
        else:
            screen = self.transition.to_screen
            app_id = self.transition.to_app_id

        if screen == "home":
            # The home sub-frame includes the status bar.  When the
            # transition blend runs, the bar appears at (1 - eased_t)
            # opacity, naturally fading out as we cross home -> app
            # (and fading in on the reverse).  This matches visionOS
            # behaviour: the bar is part of the home environment, not
            # a free-floating overlay -- entering an immersive app
            # collapses the bar along with the rest of the home scene.
            self._render_home(buf, w, h, now_ms)
            self._render_status_bar(buf, w)
        else:
            # APP side of the transition: render the named app into
            # this buffer.  None app_id would be a bug (we'd have to
            # choose what to draw without context); assert it explicitly
            # so a future Phase that wires up a new transition path
            # without filling in the app id field fails loudly here.
            assert app_id is not None, (
                f"Transition side {side!r} targets app screen but its "
                f"app id is None.  Caller of _start_transition must "
                f"supply {('from_app_id' if side == 'from' else 'to_app_id')}."
            )
            self._render_app_for_id(buf, app_id)

    # ------------------------------------------------------------------
    # Status bar + clock
    # ------------------------------------------------------------------

    def _render_status_bar(self, frame: np.ndarray, canvas_w: int) -> None:
        """Draw the translucent 44px-tall status bar with the live clock.

        Mirrors phase4's `paint_status_bar` but reads the wall clock
        each frame instead of using the hard-coded "10:30 AM"
        placeholder.  We use `%-I:%M %p` -- the GNU-style padding
        suppressor that strips the leading zero from the hour ("9:30
        AM" not "09:30 AM").  macOS supports this directive; on
        platforms that don't, the surrounding try/except falls back
        to `%I:%M %p` followed by a manual `lstrip("0")` so we never
        ship "09:30 AM" -- that leading zero is the kind of detail
        that breaks the visionOS feel.
        """
        # Glass surface across the full top bar.  Same draw_glass_panel
        # call phase4 makes -- we duplicate it here so the compositor
        # owns ALL frame composition end-to-end and doesn't reach back
        # into the phase4 painter.  The 0 corner radius keeps the bar
        # flush edge-to-edge.
        draw_glass_panel(frame, x=0, y=0, w=canvas_w, h=NAV_HEIGHT, radius=0)

        # Live clock string.  strftime is microsecond-cheap; we do it
        # per frame so the displayed time always matches the wall clock
        # rather than drifting against a captured value.  The leading-
        # zero strip is the only place we deviate from a single
        # strftime call.
        clock_text = self._format_clock(datetime.datetime.now())

        # Wordmark + clock placement matches phase4's paint_status_bar:
        # both use the same font, vertically centred via the font's
        # bbox.  Pulled directly from that helper's math so any future
        # tweak to status-bar metrics happens in one place.
        _, top_bbox, _, bot_bbox = self._wordmark_font.getbbox(STATUS_WORDMARK)
        text_h = bot_bbox - top_bbox
        text_y = (NAV_HEIGHT - text_h) // 2

        draw_text(frame, STATUS_WORDMARK,
                  x=STATUS_BAR_PAD_X, y=text_y,
                  color_rgb=TEXT_ON_DARK_RGB,
                  font=self._wordmark_font, align="left")

        draw_text(frame, clock_text,
                  x=canvas_w - STATUS_BAR_PAD_X, y=text_y,
                  color_rgb=TEXT_ON_DARK_RGB,
                  font=self._clock_font, align="right")

    def _format_clock(self, now: datetime.datetime) -> str:
        """Return a "1:23 PM"-style clock string for the wall clock `now`.

        macOS / glibc supports `%-I` (no-pad hour).  Windows does not;
        on those platforms strftime raises ValueError on the `%-` and
        we fall back to `%I` followed by a manual leading-zero strip.
        Either path produces the same visual result; the try/except
        is the portability seam.
        """
        try:
            return now.strftime("%-I:%M %p")
        except ValueError:
            return now.strftime("%I:%M %p").lstrip("0")

    # ------------------------------------------------------------------
    # Notification overlay
    # ------------------------------------------------------------------

    def _render_notifications(
        self,
        frame: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        now_ms: int,
    ) -> None:
        """Iterate the queue, reap expired entries, render each remaining card.

        Reap-then-render pass keeps the queue size bounded: a card
        whose age exceeds its lifetime is dropped *before* we walk the
        list, so we never spend even one render cycle on a card that
        wouldn't be visible anyway.

        Stacking order: the prompt asks "Most recent on top" -- i.e.
        the newest notification sits at the highest position (lowest y)
        on screen.  We iterate the queue in REVERSE so the newest
        entries get the first stack slots; older entries push down.
        """
        # Reap.  Build a fresh list of survivors; the cost is one
        # allocation per frame regardless of queue size, which beats
        # in-place index deletion when we want a stable iteration
        # below.
        survivors: list[Notification] = []
        for n in self.notifications:
            if now_ms - n.spawn_ms < n.lifetime_ms:
                survivors.append(n)
        self.notifications = survivors

        # Render the survivors with newest-first stacking.  Each card
        # gets a slot index that maps to a y offset; we share that
        # math with _notification_card_y so the layout is described
        # once.
        if not self.notifications:
            return
        self._lazy_load_notif_fonts()
        for slot_idx, notif in enumerate(reversed(self.notifications)):
            self._render_notification_card(frame, notif, slot_idx,
                                           canvas_w, now_ms)

    def _lazy_load_notif_fonts(self) -> None:
        """Populate the title/body font cache on first use.

        We split this out (instead of putting it in __init__) so the
        Compositor can be constructed before any cv2 window exists --
        font loading hits the filesystem and shouldn't run at import.
        Once both fonts are loaded, subsequent calls are no-ops.
        """
        if self._notif_title_font is None:
            self._notif_title_font = load_font(
                role="text", size=NOTIFICATION_TITLE_FONT_SIZE,
            )
        if self._notif_body_font is None:
            self._notif_body_font = load_font(
                role="text", size=NOTIFICATION_BODY_FONT_SIZE,
            )

    def _notification_card_y(self, slot_idx: int) -> int:
        """Return the top-y of the card at vertical stack slot `slot_idx`.

        Slot 0 is the topmost (most recent); each subsequent slot adds
        (card height + stack gap) to the y.  This is the math the
        prompt fixes: "12px gap between cards", "most recent on top".

        Encapsulating it here keeps the stack-direction decision in
        ONE place -- if future Phase 8 wants the stack to grow upward
        instead, only this function changes.
        """
        return (
            NOTIFICATION_Y_BASE
            + slot_idx * (NOTIFICATION_H + NOTIFICATION_STACK_GAP)
        )

    def _notification_card_x(self, notif: Notification, canvas_w: int,
                             now_ms: int) -> int:
        """Return the current x of `notif`'s left edge, given its lifecycle.

        Three phases keyed on age:
            age in [0, 300)             -- sliding in.  x lerps from
                                           canvas_w (off-right) to the
                                           final resting x, eased.
            age in [300, lifetime-300)  -- holding.  x = final resting.
            age in [lifetime-300, life) -- sliding out.  x lerps back
                                           from resting to canvas_w,
                                           eased.

        Lerp ranges are computed against ease_emphasized(t) so the
        slide reads as the same Apple-emphasized motion the rest of
        the demo uses.  Clamping the ages at the phase boundaries
        prevents off-by-one float discrepancies near the transition
        instants where one frame might sit on the boundary.
        """
        age = now_ms - notif.spawn_ms
        final_x = canvas_w - NOTIFICATION_W - NOTIFICATION_RIGHT_GUTTER
        start_x = canvas_w  # off the right edge, fully hidden

        slide_in_end = NOTIFICATION_SLIDE_MS
        slide_out_start = notif.lifetime_ms - NOTIFICATION_SLIDE_MS

        if age < slide_in_end:
            t = age / float(NOTIFICATION_SLIDE_MS)
            eased = ease_emphasized(max(0.0, min(1.0, t)))
            return int(round(start_x + (final_x - start_x) * eased))
        if age < slide_out_start:
            return final_x
        # slide-out phase: t = 0 at slide_out_start, 1.0 at lifetime_ms
        t = (age - slide_out_start) / float(NOTIFICATION_SLIDE_MS)
        eased = ease_emphasized(max(0.0, min(1.0, t)))
        return int(round(final_x + (start_x - final_x) * eased))

    def _render_notification_card(
        self,
        frame: np.ndarray,
        notif: Notification,
        slot_idx: int,
        canvas_w: int,
        now_ms: int,
    ) -> None:
        """Paint one notification card at its current animated position.

        Order:
            1. Glass panel at (x, y), 240x64 with RADIUS_TILE_SMALL.
            2. Title at (x+16, y+12), 14px Text Regular, on-dark colour.
            3. Body  at (x+16, y+36), 13px Text Regular, muted colour;
               truncated with "..." if too wide.
        """
        x = self._notification_card_x(notif, canvas_w, now_ms)
        y = self._notification_card_y(slot_idx)

        draw_glass_panel(frame, x=x, y=y,
                         w=NOTIFICATION_W, h=NOTIFICATION_H,
                         radius=RADIUS_TILE_SMALL)

        # Inner content width for truncation: width minus the two
        # 16px inner pads.  We pre-truncate before drawing so glyphs
        # never spill past the rounded corner.
        inner_w = NOTIFICATION_W - 2 * NOTIFICATION_PAD_X

        assert self._notif_title_font is not None
        assert self._notif_body_font is not None
        title_text = _truncate_to_width(
            notif.title, self._notif_title_font, inner_w,
        )
        body_text = _truncate_to_width(
            notif.body, self._notif_body_font, inner_w,
        )

        draw_text(frame, title_text,
                  x=x + NOTIFICATION_PAD_X, y=y + NOTIFICATION_TITLE_Y,
                  color_rgb=TEXT_ON_DARK_RGB,
                  font=self._notif_title_font, align="left")
        draw_text(frame, body_text,
                  x=x + NOTIFICATION_PAD_X, y=y + NOTIFICATION_BODY_Y,
                  color_rgb=TEXT_MUTED_RGB,
                  font=self._notif_body_font, align="left")


# ============================================================================
# Internal helpers (free functions, not Compositor methods)
# ============================================================================
#
# Two reasons these aren't methods:
#   1. They're pure -- no `self` state needed.
#   2. They could be unit-tested independently if a future Phase wants
#      to.  Methods would require constructing a full Compositor for
#      every test.


def _truncate_to_width(
    text: str, font: ImageFont.FreeTypeFont, max_w: int,
) -> str:
    """Return `text` shortened with a trailing "..." if it overflows `max_w`.

    Walks the string left to right looking for the longest prefix that
    fits AT MOST `max_w` pixels when followed by an ellipsis.  Returns
    the original string unchanged if no truncation is needed.

    We compare against `font.getlength`, which gives the typographic
    advance width -- the right number for "does this string fit in
    that many pixels?" -- rather than `font.getbbox`, which would
    over-include side bearings and cut the string a few characters
    early.  Same idiom `_wrap_two_lines` in src/tiles uses.

    Edge cases:
        - Empty input returns "".
        - Input that fits exactly returns unchanged.
        - Input whose first character + "..." already overflows
          returns just "..." -- not great, but the alternative is to
          return an empty string and lose the indication that data
          was elided.  Notification bodies in the demo are short
          enough that this branch is unreachable in practice.
    """
    if not text:
        return ""
    if int(font.getlength(text)) <= max_w:
        return text

    ellipsis = "..."
    ellipsis_w = int(font.getlength(ellipsis))
    # Binary-search-like prefix shrink: walk down from the full length
    # until the prefix + ellipsis fits.  Linear-time but on short
    # strings (notification bodies) the constant factor is negligible
    # and the code stays obvious.
    for i in range(len(text), 0, -1):
        prefix = text[:i].rstrip()
        if int(font.getlength(prefix)) + ellipsis_w <= max_w:
            return prefix + ellipsis
    return ellipsis
