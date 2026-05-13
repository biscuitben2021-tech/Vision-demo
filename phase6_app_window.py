"""
Phase 6 -- Click a tile, open a fake app window.

Goals (delta from Phase 5):
    * State machine: HOME (the eight-tile grid) <-> APP_OPEN (one app
      drawn fullscreen).  No animated transitions yet -- those land in
      Phase 7.  Phase 6 is HARD CUTS only: the moment the user clicks a
      tile we replace the canvas; the moment they click the close
      button we go back.
    * Click handling: the mouse callback now consumes EVENT_LBUTTONDOWN
      as well as EVENT_MOUSEMOVE.  On HOME, a click is hit-tested
      against the eight tile rects; on APP_OPEN it's hit-tested against
      the 32x32 close button at top-left.
    * Fake app rendering: each of the eight home apps has its own
      content function over in src/apps.py.  Phase 6 dispatches by
      app_id through `RENDERERS[app_id]` and lets that function fill
      the whole canvas with the app's pixels.
    * Close button: a 32x32 glass disc at (GAP_VIEWPORT + 12, GAP_VIEWPORT
      + 12) with an X glyph in the middle.  The glyph colour flips to
      light only for `music` (which has a black wallpaper); every other
      app gets the dark X on its light wallpaper.  WHY documented at
      the call site.
    * Everything else from Phase 5 (fade-up on home entry, hover scale,
      reduced-motion toggle, FPS counter, ESC/Q quit) carries straight
      over.  In APP_OPEN the home tiles aren't visible, so hover scale
      effectively does nothing -- but we keep updating it so a returning
      user resumes with no visible reset.

Why this phase exists:
    Phases 1-5 prove the home screen reads and animates.  Phase 6 is
    where the demo gets a tap target: stage volunteers can now click
    an app, see something different, click again to return.  By the
    end of Phase 6 the demo is internally consistent enough that the
    Phase 7 polish layer (live clock, notification slides, smooth
    open/close transitions) sits on top of a working state machine
    rather than introducing one.

State machine, in one diagram:

    +---------+   tile click   +-----------+
    |  HOME   | -------------> | APP_OPEN  |
    |         | <------------- |  app_id=X |
    +---------+ close button   +-----------+

    HOME paints the eight-tile grid (Phase 5's paint_grid_with_motion).
    APP_OPEN paints RENDERERS[open_app_id] then the close button on top.
    Transitions are instantaneous (HARD CUTS); animated transitions
    are Phase 7's job.

Module color-space convention:
    BGR for the cv2 pixel buffer; constants imported with `_BGR` go
    straight to cv2 calls, `_RGB` cross into PIL via `draw_text`.
    Same convention as src/design, src/tiles, src/icons, phase1-5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final, Literal

import cv2
import numpy as np

from src.apps import RENDERERS
from src.design import (
    BG_DARK_BGR,
    GAP_VIEWPORT,
    TEXT_ON_DARK_RGB,
    TEXT_ON_LIGHT_RGB,
    draw_fps_hud,
)
from src.icons import draw_glass_panel
from phase1_canvas import (
    FPS_EMA_ALPHA,
    QUIT_KEY_ESC,
    QUIT_KEY_Q_L,
    QUIT_KEY_Q_U,
    WINDOW_NAME,
    make_fullscreen_window,
    screen_size,
)
from phase4_home_screen import (
    APPS,
    _get_label_font,
    _get_status_bar_fonts,
    make_dark_canvas,
    paint_status_bar,
)
# Note: previous versions also imported render_fps_below_bar from
# phase4_home_screen; that helper was deleted when the FPS counter was
# centralised in src.design.draw_fps_hud (top-right, dim grey, anchored
# 20px from the corner).  The same draw_fps_hud is now called from
# main() AFTER the per-screen paint, so _paint_home and _paint_app no
# longer take fps_font / fps_text arguments.
from phase5_motion import (
    GridGeometry,
    MotionState,
    build_grid_geometry,
    build_motion_state,
    closest_tile,
    now_ms_relative,
    paint_grid_with_motion,
    reduced_motion_requested_via_cli,
    reduced_motion_requested_via_keypoll,
)


# ----------------------------------------------------------------------------
# Close button geometry
# ----------------------------------------------------------------------------
#
# 32x32 glass disc anchored at (GAP_VIEWPORT + 12, GAP_VIEWPORT + 12).
# The "+12" inset pulls the button slightly INSIDE the comfortable
# 16px viewport gutter -- the button still reads as flush to the
# top-left corner but the X never touches the canvas edge.
#
# We deliberately reuse `draw_glass_panel` (the same primitive that
# composes home-screen tiles) so the close button feels like a piece
# of the system chrome rather than an OpenCV widget glued on top.  The
# 16px corner radius is half the button side, which produces an almost-
# circular pill that reads as a disc on cursory inspection but degrades
# gracefully to a rounded square if the canvas is rendered at a tiny
# resolution where the glass-panel mask softens the corners further.

CLOSE_BTN_SIZE:    Final[int] = 32
CLOSE_BTN_INSET:   Final[int] = 12
CLOSE_BTN_RADIUS:  Final[int] = 16

# X glyph extent.  Inset from the button edge by ~8px so the X reads as
# centred glyph rather than corner-touching cross.  cv2.line with
# thickness=2 gives a clean, AA-smoothed line at this scale.
_X_INSET:          Final[int] = 9
_X_THICKNESS:      Final[int] = 2


# ----------------------------------------------------------------------------
# State machine types
# ----------------------------------------------------------------------------
#
# Two screens: HOME (the grid) and APP_OPEN (one app fullscreen).  We
# use a Literal type for the state so static type checkers catch typos
# at the call site rather than failing silently at render time.  The
# open app id is None on HOME and a string on APP_OPEN; that's the
# invariant the renderer relies on.

ScreenState = Literal["home", "app"]


@dataclass
class AppState:
    """Phase 6's full UI state in one mutable container.

    Fields:
        state:        current screen -- "home" or "app".  Initialised
                      to "home" so the user sees the grid first.
        open_app_id:  if state == "app", the id of the app currently
                      open (one of the keys of `RENDERERS`).  None on
                      "home".  Maintaining the invariant
                      "(state == 'home') iff (open_app_id is None)" is
                      the entire job of the two transition helpers below.

    We deliberately keep this state OUT of the motion state -- they
    have different lifetimes and access patterns.  Motion state is
    consulted every frame by the painter; AppState is read by the
    painter (to dispatch home vs app) and mutated by the mouse
    callback.  Separating them makes the data flow obvious.
    """

    state:        ScreenState = "home"
    open_app_id:  str | None  = None

    def open_app(self, app_id: str) -> None:
        """Transition HOME -> APP_OPEN with the given app id.

        Caller (the mouse callback) is responsible for verifying the
        click was actually inside a tile rect before invoking this;
        we don't re-check the geometry here so the function stays a
        pure state mutation.

        Asserts the precondition `app_id in RENDERERS` -- a typo at
        the call site fails loudly at click time instead of silently
        opening to a black screen.  That matches the loud-failure
        philosophy in src/icons.draw_app_icon and elsewhere.
        """
        assert app_id in RENDERERS, (
            f"Unknown app_id {app_id!r}; expected one of "
            f"{sorted(RENDERERS.keys())}."
        )
        self.state = "app"
        self.open_app_id = app_id

    def close_app(self) -> None:
        """Transition APP_OPEN -> HOME, clearing the open app id.

        Idempotent -- calling close_app() while already on HOME is a
        no-op rather than an error.  The mouse callback could in
        principle deliver a stray click during a transition flicker,
        and we'd rather absorb that than crash.
        """
        self.state = "home"
        self.open_app_id = None


# ----------------------------------------------------------------------------
# Close button rendering + hit-testing
# ----------------------------------------------------------------------------

def _close_button_rect() -> tuple[int, int, int, int]:
    """Return the (x, y, w, h) of the close button on the current frame.

    Pulled out as a helper so the renderer AND the click hit-test
    share a single source of truth for the button's geometry.  If
    Phase 7 wants to move the button (animated nav-bar collapse, say),
    only this function needs updating.
    """
    return (
        GAP_VIEWPORT + CLOSE_BTN_INSET,
        GAP_VIEWPORT + CLOSE_BTN_INSET,
        CLOSE_BTN_SIZE,
        CLOSE_BTN_SIZE,
    )


def _draw_close_button(frame: np.ndarray, open_app_id: str) -> None:
    """Paint the 32x32 glass close button + X glyph in the top-left.

    The glyph colour decision is the meaningful WHY here:

        On a light wallpaper (every app except Music) the X should be
        DARK so it reads against the bright surface.  Drawing a light
        X over light glass produces a "pale ghost" effect; the user
        can find the button by motion but the X itself disappears
        unless they look carefully.

        On Music's pure-black wallpaper the opposite is true: a dark
        X over the glass-tinted-toward-black surface becomes invisible
        -- the glass brightens the wallpaper slightly but not enough
        to make a 31-grey X stand out.  A light X over the same
        surface reads cleanly.

    The simplest rule that captures this is "Music gets the light X,
    everything else gets the dark X" -- which is exactly what the
    Phase 6 prompt prescribes.  We could in principle measure the
    underlying pixel luminance and pick adaptively, but with a fixed
    eight-app roster the lookup is one branch and the rule is
    self-documenting.
    """
    x, y, w, h = _close_button_rect()
    draw_glass_panel(frame, x=x, y=y, w=w, h=h, radius=CLOSE_BTN_RADIUS)

    # X glyph in the disc.  Two diagonal lines, both with LINE_AA so
    # the corners read as crisp strokes rather than as 8-bit stair
    # steps -- the same rationale that drove every other AA call in
    # this codebase.
    if open_app_id == "music":
        x_color_rgb = TEXT_ON_DARK_RGB    # light glyph on dark surface
    else:
        x_color_rgb = TEXT_ON_LIGHT_RGB   # dark glyph on light surface
    # PIL RGB -> cv2 BGR for the cv2.line call.  We don't import a
    # _BGR variant here because the choice is dynamic; the conversion
    # is one tuple flip and keeping the design.py constants RGB-keyed
    # avoids defining `TEXT_ON_*_BGR` duplicates just for this site.
    x_color_bgr = (x_color_rgb[2], x_color_rgb[1], x_color_rgb[0])

    p1 = (x + _X_INSET,           y + _X_INSET)
    p2 = (x + w - _X_INSET - 1,   y + h - _X_INSET - 1)
    p3 = (x + w - _X_INSET - 1,   y + _X_INSET)
    p4 = (x + _X_INSET,           y + h - _X_INSET - 1)
    cv2.line(frame, p1, p2, x_color_bgr,
             thickness=_X_THICKNESS, lineType=cv2.LINE_AA)
    cv2.line(frame, p3, p4, x_color_bgr,
             thickness=_X_THICKNESS, lineType=cv2.LINE_AA)


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    """Return True if (x, y) lies inside `rect` = (rx, ry, rw, rh).

    Standard half-open containment -- rx <= x < rx + rw and so on.
    Pulled out as a one-liner so the mouse callback reads as
    "is the click inside the close button rect?" rather than re-doing
    the four-comparison expansion at each site.
    """
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


# ----------------------------------------------------------------------------
# Mouse callback context
# ----------------------------------------------------------------------------
#
# The Phase 5 callback only needed geometry + motion + t0; Phase 6 adds
# the AppState reference.  Same dataclass pattern (kept private to this
# module by the underscore prefix) so the callback closure captures one
# explicit context object instead of four bare locals.

@dataclass
class _MouseContext:
    """Mutable bundle of references the cv2 mouse callback reads on each event.

    Updating `geometry` here (from the main loop, on resize) keeps the
    callback's hit-test in sync with the current canvas size without
    re-registering the callback.  AppState is mutated by the callback
    itself when the user clicks a tile or the close button.
    """

    geometry:  GridGeometry
    motion:    MotionState
    app_state: AppState
    t0_ticks:  int


def _mouse_callback(
    event: int, x: int, y: int, flags: int, param: object,
) -> None:
    """Route mouse events to either the hover update or the state machine.

    Three event types matter here:

        EVENT_MOUSEMOVE   -- update HoverState.  Phase 5 already does
                             this; we keep the same code path so a
                             returning user (back on HOME after closing
                             an app) sees their cursor pick up the
                             nearest tile immediately.
        EVENT_LBUTTONDOWN -- check the current screen:
                                 - on HOME: which tile rect (if any)
                                   contains the click?  Open that app.
                                 - on APP_OPEN: is the click inside the
                                   close button rect?  Close the app.
        Anything else     -- ignored.  Click drag, right click, scroll
                             are not part of Phase 6's surface.

    `param` is the _MouseContext bundle; we assert the type on entry
    so a misregistered callback fails loudly instead of producing
    silent NoneType errors deep inside the hit-test.
    """
    assert isinstance(param, _MouseContext)
    ctx = param

    if event == cv2.EVENT_MOUSEMOVE:
        # Mouse-move handling unchanged from Phase 5: figure out the
        # tile under the cursor (or None) and stamp it onto the
        # hover state with the current wall-clock time.
        tile_id = closest_tile(x, y, ctx.geometry.tile_rects)
        now_ms = now_ms_relative(ctx.t0_ticks)
        ctx.motion.hover_state.set_hover(tile_id, now_ms)
        return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Click handling.  Dispatch on current screen state.  Calling
    # AppState methods here keeps the callback small and means the
    # state transitions are testable in isolation (give an AppState,
    # call open_app/close_app, assert).
    if ctx.app_state.state == "home":
        tile_id = closest_tile(x, y, ctx.geometry.tile_rects)
        if tile_id is not None:
            # APPS[i] = (app_id, display_name) -- we only need the id.
            app_id, _ = APPS[tile_id]
            ctx.app_state.open_app(app_id)
        return

    # APP_OPEN: only the close button is interactive.  Anything else is
    # absorbed silently so a user clicking around inside an app doesn't
    # accidentally trigger a side effect.
    if ctx.app_state.state == "app":
        if _point_in_rect(x, y, _close_button_rect()):
            ctx.app_state.close_app()


# ----------------------------------------------------------------------------
# Render dispatch
# ----------------------------------------------------------------------------
#
# Per-frame, the main loop calls one of two paint routines based on the
# AppState.  We keep both as standalone functions (rather than a single
# if/else inline in main) so each path's intent is self-evident and so
# Phase 7's animated transition can interpolate between them without
# threading new flags through main's render code.


def _paint_home(
    canvas: np.ndarray,
    width: int,
    geometry: GridGeometry,
    motion: MotionState,
    label_font,
    wordmark_font,
    clock_font,
    now_ms: int,
) -> None:
    """Paint the HOME screen exactly as Phase 5 does: grid + status.

    Order matters back-to-front:
        1. The black wallpaper is already in the canvas from main's
           background fill.
        2. The grid of glass tiles sits on top of the wallpaper.
        3. The status bar covers the top 44px of any bleeding tile
           pixels (the layout has no overlap by design, but the order
           keeps the occlusion correct if a future tile bleeds up).

    The FPS counter is NOT painted here -- main() draws it last via
    draw_fps_hud so it stays in the top-right corner of every phase
    and never gets occluded by the status bar.
    """
    paint_grid_with_motion(canvas, geometry, motion, label_font, now_ms)
    paint_status_bar(canvas, width, wordmark_font, clock_font)


def _paint_app(
    canvas: np.ndarray,
    open_app_id: str,
    width: int,
) -> None:
    """Paint the APP_OPEN screen: app contents fullscreen, then chrome.

    Order matters back-to-front:
        1. Dispatch to the app's renderer; it owns the entire canvas
           and fills its own background (Music goes dark, everything
           else light).  We do NOT pre-fill before calling it -- the
           renderer's _fill is the first call inside.
        2. Close button sits on top of the app content at the top-left.

    No status bar on APP_OPEN.  visionOS apps are immersive; the
    system bar collapses when an app is foregrounded.

    FPS counter is drawn by main() AFTER this returns, via
    draw_fps_hud -- same anchor as HOME so a developer's eye learns
    one screen position regardless of which state is active.
    """
    renderer = RENDERERS[open_app_id]
    renderer(canvas, width, canvas.shape[0])   # type: ignore[operator]

    _draw_close_button(canvas, open_app_id)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def _initial_state(width: int, height: int) -> tuple[
    GridGeometry, MotionState, AppState, int,
]:
    """Build the four state objects the main loop owns at startup.

    Pulled out as a helper to keep main() under the 30-line cap.  The
    return order matches the order they're used in main().
    """
    reduced_motion = (
        reduced_motion_requested_via_cli()
        or reduced_motion_requested_via_keypoll()
    )
    geometry = build_grid_geometry(width, height)
    motion   = build_motion_state(reduced_motion=reduced_motion)
    app      = AppState()                       # state="home", open_app_id=None
    t0_ticks = cv2.getTickCount()
    return geometry, motion, app, t0_ticks


def main() -> None:
    """Run the Phase 6 fullscreen loop until ESC or Q is pressed."""
    make_fullscreen_window(WINDOW_NAME)

    # FPS HUD's font is cached internally inside draw_fps_hud; only the
    # status-bar fonts need preloading at this scope.
    wordmark_font, clock_font = _get_status_bar_fonts()
    label_font = _get_label_font()

    width, height = screen_size(WINDOW_NAME)
    canvas = make_dark_canvas(width, height)
    geometry, motion, app_state, t0_ticks = _initial_state(width, height)

    mouse_ctx = _MouseContext(
        geometry=geometry, motion=motion,
        app_state=app_state, t0_ticks=t0_ticks,
    )
    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, mouse_ctx)

    last_t = time.perf_counter() - (1.0 / 60.0)
    fps_ema = 0.0

    while True:
        cur_w, cur_h = screen_size(WINDOW_NAME)
        if (cur_w, cur_h) != (width, height):
            width, height = cur_w, cur_h
            canvas = make_dark_canvas(width, height)
            geometry = build_grid_geometry(width, height)
            mouse_ctx.geometry = geometry
        else:
            # Wipe the previous frame.  Each renderer fills the canvas
            # with its own background colour, so the BG_DARK pre-fill
            # is only strictly needed on HOME; we do it unconditionally
            # to keep the frame loop a single flat block.
            canvas[:, :] = BG_DARK_BGR

        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0.0:
            instant_fps = 1.0 / dt
            fps_ema = (
                instant_fps
                if fps_ema == 0.0
                else FPS_EMA_ALPHA * instant_fps
                     + (1.0 - FPS_EMA_ALPHA) * fps_ema
            )

        now_ms = now_ms_relative(t0_ticks)

        # Dispatch on the current screen state.  Keeping the dispatch
        # at this single point (rather than scattering app-vs-home
        # branches through paint helpers) means Phase 7's animated
        # transition only needs to wrap this branch.
        if app_state.state == "home":
            _paint_home(
                canvas, width, geometry, motion,
                label_font, wordmark_font, clock_font,
                now_ms,
            )
        else:
            # Defensive: open_app_id is the source of truth for which
            # app to draw, but if we somehow got into "app" state with
            # a None id we fall back to HOME rather than crash.
            if app_state.open_app_id is None:
                app_state.state = "home"
                continue
            _paint_app(canvas, app_state.open_app_id, width)

        # FPS counter is the ABSOLUTE last paint -- top-right corner,
        # dim grey, never occluded by the status bar or any chrome.
        draw_fps_hud(canvas, fps_ema)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (QUIT_KEY_ESC, QUIT_KEY_Q_L, QUIT_KEY_Q_U):
            break

        if cv2.getWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_VISIBLE,
        ) < 1.0:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
