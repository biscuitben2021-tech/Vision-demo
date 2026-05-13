"""
Phase 5 -- Vision OS home screen with entry + hover motion.

Goals (delta from Phase 4):
    * Fade-up on entry: each of the eight home tiles starts invisible
      and 24px below its resting position, then rises and fades in over
      800ms.  Tiles stagger 50ms apart -- the first tile starts
      animating immediately, the eighth starts 350ms in.  The whole
      grid is in place by ~1.15s after process start.
    * Cursor-driven hover scale: when the mouse cursor enters a tile,
      that tile eases from 1.0x to 1.02x over 200ms.  On leave it eases
      back.  Crossing tiles fast (the cursor sweeps the grid in a
      single motion) is handled without snap-to-target kicks -- see
      `HoverState` in src/motion.py for the discontinuity-prevention
      math.
    * Reduced-motion mode:  passing `--reduced-motion` on the command
      line OR holding `R` at startup disables every animation -- tiles
      appear instantly at full opacity, hover scaling is bypassed.
      Useful for low-end machines and accessibility.
    * Everything else (wallpaper, status bar, grid layout, FPS counter,
      ESC/Q quit) carries straight over from Phase 4.

Why this phase exists:
    Phases 1-4 prove a static visionOS home screen reads as Apple-grade.
    Phase 5 is where it starts to *feel* alive.  The two motion primitives
    here (fade-up and hover scale) are the same two that Phases 6+ will
    reuse for app-window open/close transitions and notification slides.
    Locking them down here -- in particular the eased curves coming
    through `src/motion.py` -- means every later phase inherits a
    consistent animation vocabulary instead of inventing its own.

Reduced-motion limitations:
    The `R` keypoll is single-shot at startup, AFTER the window has been
    created and BEFORE the main loop begins.  We poll with
    `cv2.waitKey(1)`, which returns the keycode of any pending keypress
    -- a best-effort "held R" detection rather than a true keydown-state
    query (OpenCV does not expose modifier / held-key state on macOS).
    If the user presses R during the brief window between
    `cv2.namedWindow` returning and the poll fire, we catch it;
    otherwise the CLI flag is the reliable mechanism.  Documenting the
    limitation here so future-me does not waste an afternoon trying to
    fix the unfixable.

Module color-space convention:
    BGR for the cv2 pixel buffer; constants imported with `_BGR` go
    straight to cv2 calls, `_RGB` cross into PIL via `draw_text`.
    Same convention as src/design, src/tiles, src/icons, phase1-4.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from src.design import (
    BG_DARK_BGR,
    FADE_UP_DURATION_MS,
    NAV_HEIGHT,
    RADIUS_APP_ICON,
    TEXT_ON_DARK_RGB,
    draw_fps_hud,
    draw_text,
)
from src.icons import draw_app_icon, draw_glass_panel
from src.motion import FadeUpState, HoverState
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
    GRID_COLS,
    GRID_TOP_GUTTER,
    ICON_SIZE,
    LABEL_BAND_HEIGHT,
    LABEL_GAP_FROM_TILE,
    TILE_H,
    TILE_W,
    _get_label_font,
    _get_status_bar_fonts,
    grid_size,
    make_dark_canvas,
    paint_status_bar,
    tile_origin,
)


# ----------------------------------------------------------------------------
# Motion configuration
# ----------------------------------------------------------------------------
#
# Single source of truth for the per-tile entry stagger.  CLAUDE.md
# specifies a 50ms gap between adjacent tiles' fade-up starts; encoding
# that as a constant here keeps phase 6 / 7 from inventing their own
# values for the same intent.

STAGGER_MS: Final[int] = 50

# Reduced-motion key.  We accept either case so callers don't have to
# care about CapsLock state.  This is the same convention QUIT_KEY_Q_L /
# QUIT_KEY_Q_U use over in phase1_canvas.
REDUCED_MOTION_KEY_L: Final[int] = ord("r")
REDUCED_MOTION_KEY_U: Final[int] = ord("R")

# CLI flag string.  Tested against sys.argv exactly; no argparse here --
# this is a single-flag binary toggle, and adding argparse would mean
# every other phase script needs the same boilerplate for symmetry.
CLI_FLAG_REDUCED_MOTION: Final[str] = "--reduced-motion"


# ----------------------------------------------------------------------------
# Geometry cache -- shared by the renderer AND the mouse callback
# ----------------------------------------------------------------------------
#
# The mouse callback runs OUTSIDE the main loop (cv2 fires it on its own
# thread on macOS in some configurations), so it cannot recompute grid
# geometry from a fresh `grid_size()` call each event -- nothing
# guarantees the screen dimensions are stable at that moment.  We
# pre-compute the eight tile rects once per resize and store them in a
# dataclass that the callback closure captures by reference.
#
# Storing the canvas dimensions alongside the rects means a resize is
# detectable by comparing (cur_w, cur_h) against the cached values; if
# they differ, we rebuild the cache and continue.


@dataclass
class GridGeometry:
    """Cached grid layout for one canvas size.

    Rebuilt only when the canvas dimensions change (rare in steady
    state, but possible if a projector gets plugged in mid-demo).  The
    `tile_rects` list is in roster order -- index i corresponds to
    APPS[i], matching the iteration order in `paint_grid`.

    Fields:
        canvas_w, canvas_h: the dimensions this cache was built for.
                            A mismatch signals a rebuild is needed.
        tile_rects:         eight (x, y, w, h) tuples, one per tile.
                            Hit-tested by the mouse callback to figure
                            out which tile (if any) the cursor sits
                            over.  Drawn by the renderer in the same
                            order.
    """

    canvas_w: int
    canvas_h: int
    tile_rects: list[tuple[int, int, int, int]]


def build_grid_geometry(canvas_w: int, canvas_h: int) -> GridGeometry:
    """Compute the eight tile rects for a canvas of (canvas_w, canvas_h).

    Layout is identical to Phase 4's `paint_grid`: 4x2 grid, fixed-pitch
    tiles, centred horizontally below the status bar.  We hand-roll the
    iteration here (rather than importing `paint_grid` and intercepting
    it) because `paint_grid` paints and we just want the rects.  Phase 4
    already exposes `tile_origin` and `grid_size`, so this is six lines
    of glue.
    """
    grid_w, _ = grid_size()
    grid_x = (canvas_w - grid_w) // 2
    grid_y = NAV_HEIGHT + GRID_TOP_GUTTER

    rects: list[tuple[int, int, int, int]] = []
    for i in range(len(APPS)):
        col = i % GRID_COLS
        row = i // GRID_COLS
        tile_x, tile_y = tile_origin(col, row, grid_x, grid_y)
        rects.append((tile_x, tile_y, TILE_W, TILE_H))
    return GridGeometry(canvas_w=canvas_w, canvas_h=canvas_h, tile_rects=rects)


def closest_tile(
    x: int, y: int, tile_rects: list[tuple[int, int, int, int]],
) -> int | None:
    """Return the index of the tile under (x, y), or None if no hit.

    This is a strict hit-test, not a nearest-tile lookup -- the function
    name is mildly misleading and that's deliberate.  "closest" is what
    Phase 8's gaze cursor will want (a noisy gaze estimate rarely lands
    EXACTLY inside a tile, so we'll switch to nearest-within-radius
    there); for the mouse in Phase 5 strict containment is what feels
    right.  Keeping the name `closest_tile` now means Phase 8 can
    extend the signature without churning the call sites.

    Returns the FIRST matching index since the eight tiles never overlap
    in the grid layout -- there can be at most one hit.
    """
    for i, (tx, ty, tw, th) in enumerate(tile_rects):
        if tx <= x < tx + tw and ty <= y < ty + th:
            return i
    return None


# ----------------------------------------------------------------------------
# Reduced-motion detection
# ----------------------------------------------------------------------------
#
# Two ways to ask for reduced motion:
#     1. Pass `--reduced-motion` on the command line.  Reliable.
#     2. Hold `R` while the window opens.  Best-effort; we poll cv2's
#        keyboard once after window creation.  See the file-level
#        docstring's "Reduced-motion limitations" section.

def reduced_motion_requested_via_cli() -> bool:
    """Return True if `--reduced-motion` was passed on the command line."""
    return CLI_FLAG_REDUCED_MOTION in sys.argv


def reduced_motion_requested_via_keypoll() -> bool:
    """Return True if R is in cv2's keyboard queue at this moment.

    Single waitKey(1) poll -- this is intentionally not a tight loop.
    The function is called once at startup, after the window has been
    created and is visible, but before the main render loop begins.
    If the user presses (or is holding) R at that instant, we catch
    it; otherwise we don't.  This is the "best effort" path; the CLI
    flag is the reliable mechanism.

    The 1ms timeout gives the macOS event loop just enough time to
    deliver any already-queued keypress without delaying startup
    perceptibly.
    """
    key = cv2.waitKey(1) & 0xFF
    return key in (REDUCED_MOTION_KEY_L, REDUCED_MOTION_KEY_U)


# ----------------------------------------------------------------------------
# Time source
# ----------------------------------------------------------------------------
#
# We use cv2.getTickCount / cv2.getTickFrequency rather than time.perf_counter
# specifically because the prompt asks for it; in practice on Apple Silicon
# both are wall-clock-microsecond-accurate and the choice is a wash.  Using
# the cv2 timer keeps us in the same time domain as cv2's own profiling
# helpers, which is convenient if Phase 13's MediaPipe integration ever
# needs to correlate frames across the two clocks.
#
# We zero the clock at startup so all FadeUpState start_ms values are
# small integers (e.g. 0, 50, 100, ...) instead of carrying around large
# tick-count offsets.

def now_ms_relative(t0_ticks: int) -> int:
    """Return the milliseconds elapsed since the tick count `t0_ticks`.

    Uses cv2's tick clock divided by tick frequency.  Returning an int
    (not a float) means FadeUpState comparisons against integer
    start_ms values are exact -- floating point milliseconds would
    open the door to tiny ordering glitches near the 50ms staggered
    boundaries.
    """
    ticks = cv2.getTickCount() - t0_ticks
    seconds = ticks / cv2.getTickFrequency()
    return int(seconds * 1000.0)


# ----------------------------------------------------------------------------
# Per-tile renderer with motion
# ----------------------------------------------------------------------------
#
# Phase 4 paints each tile directly into the canvas.  Phase 5 cannot do
# that:  the tile needs to be scaled (hover state) and alpha-blended
# (fade-up) against whatever is already in the canvas at its target
# position, which means we must render the tile INTO a temporary
# sub-image first, then composite it.  This is the same alpha-over
# pipeline `draw_text` uses internally; we're just doing it at tile
# scale instead of glyph scale.
#
# Pipeline, per visible tile, per frame:
#
#     1. Build a TILE_W x TILE_H BGR sub-image filled with BG_DARK
#        (the wallpaper underneath -- the glass panel reads its
#        underlying pixels for its brighten-and-tint pass, so we
#        must seed the sub-image with the same colour as the canvas
#        background).
#     2. Draw the glass panel + app icon into that sub-image, just as
#        paint_tile does in phase 4.
#     3. If hover scale != 1.0, resize the sub-image by that factor
#        with cv2.INTER_LINEAR.  We deliberately don't use INTER_CUBIC
#        here: at 1.02 the size change is too small for cubic to be
#        worth the extra cycles, and INTER_LINEAR's slight smoothing
#        actually helps the eye not pick up the resize as a discrete
#        step.
#     4. Compute the paste origin.  The tile's resting position is
#        (tile_x, tile_y); add y_offset for the fade-up rise, and
#        adjust x / y by half the scale delta so the tile grows from
#        its centre rather than the top-left corner.
#     5. Alpha-blend the scaled sub-image into the canvas at the
#        paste origin, using `opacity` as the blend factor.
#     6. Draw the tile's label below, at (label_x, label_y + y_offset)
#        with the same opacity applied via a sub-image alpha-blend.


def _build_tile_subimage(app_id: str) -> np.ndarray:
    """Return a fresh TILE_W x TILE_H BGR buffer with one tile rendered into it.

    Pre-filled with BG_DARK_BGR so the glass panel's brighten-and-tint
    pass has the correct underlying colour to read from.  If we left
    the buffer uninitialised, the glass surface would read random
    bytes as "wallpaper" and the tile would flicker every frame.

    The caller scales and composites this buffer; we do not do any
    cropping or scaling here.  This keeps the render-vs-composite
    split clean and lets _render_tile_with_motion focus on geometry.
    """
    sub = np.empty((TILE_H, TILE_W, 3), dtype=np.uint8)
    sub[:, :] = BG_DARK_BGR

    # Glass panel: same parameters as phase 4's paint_tile, but at the
    # sub-image's origin (0, 0) rather than (tile_x, tile_y).
    draw_glass_panel(sub, x=0, y=0,
                     w=TILE_W, h=TILE_H, radius=RADIUS_APP_ICON)

    # Icon centred inside the sub-image.
    icon_cx = TILE_W // 2
    icon_cy = TILE_H // 2
    draw_app_icon(sub, cx=icon_cx, cy=icon_cy,
                  size=ICON_SIZE, app_id=app_id)
    return sub


def _alpha_blend_subimage(
    canvas: np.ndarray,
    sub: np.ndarray,
    paste_x: int,
    paste_y: int,
    opacity: float,
) -> None:
    """Alpha-blend `sub` into `canvas` at (paste_x, paste_y) with `opacity`.

    Standard "over" composite with a constant alpha across the whole
    sub-image:  out = sub*opacity + canvas*(1-opacity).  Clipped to the
    canvas extent -- off-screen pastes are a silent no-op rather than
    an exception, matching draw_glass_panel / draw_text's contracts.

    `sub` is read-only; `canvas` is mutated in place.
    """
    canvas_h, canvas_w = canvas.shape[:2]
    sub_h, sub_w = sub.shape[:2]

    # Clip the destination rect to canvas bounds.  Negative origins
    # shift the source patch by the same offset so the visible portion
    # composites correctly.
    x0 = max(0, paste_x)
    y0 = max(0, paste_y)
    x1 = min(canvas_w, paste_x + sub_w)
    y1 = min(canvas_h, paste_y + sub_h)
    if x1 <= x0 or y1 <= y0:
        return

    sx0 = x0 - paste_x
    sy0 = y0 - paste_y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    src = sub[sy0:sy1, sx0:sx1].astype(np.float32)
    dst = canvas[y0:y1, x0:x1].astype(np.float32)
    out = src * opacity + dst * (1.0 - opacity)
    np.clip(out, 0.0, 255.0, out=out)
    canvas[y0:y1, x0:x1] = out.astype(np.uint8)


def _render_label_with_motion(
    canvas: np.ndarray,
    tile_x: int,
    tile_y: int,
    display_name: str,
    label_font,
    y_offset: float,
    opacity: float,
) -> None:
    """Draw the tile's label below it, with the fade-up offset/opacity applied.

    We render the label into a tiny BG_DARK sub-image (the wallpaper
    colour, so opacity blending against the wallpaper is a visual
    no-op outside the glyphs themselves) and then alpha-blend that
    sub-image into the canvas.  This is the cheapest way to fade text
    without modifying draw_text's signature for every other phase.

    The sub-image is sized to the full grid-cell label band so the
    glyphs always fit even on long labels ("Calendar", "Settings"),
    and aligned so the label's horizontal centre sits over the tile's
    horizontal centre -- matching Phase 4's `paint_tile` placement.
    """
    # Label band: matches Phase 4's exactly -- a horizontal slab the
    # width of the tile, LABEL_BAND_HEIGHT tall, sitting
    # LABEL_GAP_FROM_TILE below the tile's resting bottom.
    band_x = tile_x
    band_y = tile_y + TILE_H + LABEL_GAP_FROM_TILE
    band_w = TILE_W
    band_h = LABEL_BAND_HEIGHT

    sub = np.empty((band_h, band_w, 3), dtype=np.uint8)
    sub[:, :] = BG_DARK_BGR

    # draw_text's `x` is the centre when align="center"; placing it at
    # band_w//2 inside the sub-image gives a label centred horizontally
    # within the tile's column.  `y=0` anchors the glyphs to the top of
    # the sub-image.
    draw_text(sub, display_name,
              x=band_w // 2, y=0,
              color_rgb=TEXT_ON_DARK_RGB, font=label_font, align="center")

    # Composite with the fade-up offset and opacity applied.  Sub-pixel
    # y-offsets are rounded to the nearest int (cv2 paste origin is
    # integer-only) -- matching the rounding _render_tile_with_motion
    # uses so the tile and its label stay vertically aligned during the
    # fade-up.  At the 24px max offset on the eased curve, the rounding
    # is imperceptible.
    _alpha_blend_subimage(
        canvas, sub,
        paste_x=band_x,
        paste_y=band_y + int(round(y_offset)),
        opacity=opacity,
    )


def _render_tile_with_motion(
    canvas: np.ndarray,
    tile_x: int,
    tile_y: int,
    app_id: str,
    display_name: str,
    label_font,
    opacity: float,
    y_offset: float,
    scale: float,
) -> None:
    """Render one tile + label with fade-up + hover transformations applied.

    Steps mirror the pipeline laid out in the module's per-tile
    renderer block above.  Scaling is around the tile's centre rather
    than its top-left, so 1.02 reads as the tile lifting toward the
    viewer rather than slumping toward the bottom-right.

    Off-canvas pastes are silently clipped by _alpha_blend_subimage.
    """
    # 1-2. Build the tile + icon sub-image (glass panel, app icon).
    sub = _build_tile_subimage(app_id)

    # 3. Apply the hover scale.  cv2.resize takes (w, h), not (h, w).
    if scale != 1.0:
        new_w = max(1, int(round(TILE_W * scale)))
        new_h = max(1, int(round(TILE_H * scale)))
        sub = cv2.resize(sub, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        new_w, new_h = TILE_W, TILE_H

    # 4. Centre-scaled paste origin.  Half the scale delta on each
    #    axis moves the paste origin up and left, keeping the tile's
    #    visual centre at (tile_x + TILE_W/2, tile_y + TILE_H/2 + y_offset).
    dx = (new_w - TILE_W) // 2
    dy = (new_h - TILE_H) // 2
    paste_x = tile_x - dx
    paste_y = tile_y - dy + int(round(y_offset))

    # 5. Alpha-blend the scaled tile into the canvas with `opacity`.
    _alpha_blend_subimage(canvas, sub, paste_x, paste_y, opacity)

    # 6. Label below the tile, with the same fade-up offset/opacity
    #    but no scaling -- a 1.02 scale on the 13px label moves
    #    pixels by sub-pixel amounts and produces no visible
    #    difference, while keeping the label sharp avoids re-rastering
    #    glyphs every frame on hover.
    _render_label_with_motion(
        canvas, tile_x, tile_y, display_name, label_font,
        y_offset=y_offset, opacity=opacity,
    )


# ----------------------------------------------------------------------------
# Animation state container
# ----------------------------------------------------------------------------
#
# Bundling the eight FadeUpStates and the single HoverState into one
# object means the main loop and the mouse callback share a single
# reference, and the painter takes one parameter instead of two
# parallel lists.  The dataclass is mutable in only one way -- the
# HoverState's set_hover -- which keeps the "no global mutable state"
# rule satisfied: state lives in an explicit object owned by main(),
# not at module scope.


@dataclass
class MotionState:
    """All per-frame animation state for the home screen.

    Fields:
        fade_states:    one FadeUpState per tile, in roster order.  In
                        reduced-motion mode these are pre-armed with
                        start_ms = -duration so .value() returns the
                        completed (1.0, 0.0) for every call.
        hover_state:    a single HoverState tracking which tile (if
                        any) the cursor is over.  Always present even
                        in reduced-motion -- the mouse callback still
                        runs, the renderer just ignores the scale.
        reduced_motion: True if --reduced-motion was passed or R was
                        held at startup.  The renderer reads this
                        directly to bypass scale lookups.
    """

    fade_states: list[FadeUpState]
    hover_state: HoverState
    reduced_motion: bool


def build_motion_state(reduced_motion: bool) -> MotionState:
    """Initialise per-tile motion state at process start (t=0).

    In reduced-motion mode every FadeUpState is constructed with
    start_ms = -FADE_UP_DURATION_MS so that .value(now_ms) returns
    (1.0, 0.0) on the very first frame -- tiles appear instantly at
    full opacity.  This is cheaper than threading a `reduced_motion`
    flag through every value() call.
    """
    if reduced_motion:
        # Backdate each tile's start so .value() reads it as already
        # complete on the first frame.  Stagger is irrelevant in this
        # mode but we keep it consistent for any future code that
        # might iterate fade_states expecting a deterministic order.
        fade_states = [
            FadeUpState(start_ms=-FADE_UP_DURATION_MS - 1)
            for _ in range(len(APPS))
        ]
    else:
        # Tile i starts its fade-up i * STAGGER_MS into the animation.
        # First tile starts immediately (start_ms = 0), eighth tile
        # starts at 7 * 50 = 350ms.  The grid is fully animated in by
        # ~1.15s (350 + 800).
        fade_states = [
            FadeUpState(start_ms=i * STAGGER_MS) for i in range(len(APPS))
        ]
    return MotionState(
        fade_states=fade_states,
        hover_state=HoverState(),
        reduced_motion=reduced_motion,
    )


# ----------------------------------------------------------------------------
# Mouse callback
# ----------------------------------------------------------------------------
#
# cv2.setMouseCallback fires for every mouse event in the window.  We
# only care about MOUSEMOVE -- click handling lands in Phase 6, scroll
# in Phase 7.  The callback is passed a closure that holds references to
# the geometry cache, the motion state, and the t0_ticks baseline, so
# it can look up the current tile and stamp it onto hover_state without
# touching any module-global mutable state.

@dataclass
class _MouseContext:
    """Mutable container passed by reference into the mouse callback.

    We pass this through cv2.setMouseCallback's `param` argument, so the
    callback closure doesn't capture local variables that get rebound
    on canvas resize.  Updating `geometry` here is the mechanism the
    main loop uses to keep the callback's hit-test in sync with the
    current canvas size.
    """

    geometry: GridGeometry
    motion: MotionState
    t0_ticks: int


def _mouse_callback(
    event: int, x: int, y: int, flags: int, param: object,
) -> None:
    """Update HoverState when the cursor moves.

    Called by cv2 on every mouse event.  We discard everything except
    MOUSEMOVE; click / scroll are handled in later phases.

    `param` is the `_MouseContext` instance passed to
    cv2.setMouseCallback in `main()`.  Type-hinted as `object` because
    cv2's stub annotates the param as Any; we cast on the assignment
    line below.
    """
    if event != cv2.EVENT_MOUSEMOVE:
        return
    assert isinstance(param, _MouseContext)
    # The cv2 mouse callback delivers coordinates in canvas space, which
    # is exactly what closest_tile expects -- no transform needed.
    tile_id = closest_tile(x, y, param.geometry.tile_rects)
    now_ms = now_ms_relative(param.t0_ticks)
    param.motion.hover_state.set_hover(tile_id, now_ms)


# ----------------------------------------------------------------------------
# Grid painter with motion
# ----------------------------------------------------------------------------

def paint_grid_with_motion(
    canvas: np.ndarray,
    geometry: GridGeometry,
    motion: MotionState,
    label_font,
    now_ms: int,
) -> None:
    """Paint the 4x2 home-screen grid with per-tile motion applied.

    Iteration order is roster order (same as Phase 4's paint_grid), so
    tile index matches APPS and matches geometry.tile_rects.  Reduced-
    motion mode short-circuits the scale lookup -- saving a function
    call per tile per frame, which matters at 60Hz with eight tiles.
    """
    for i, (app_id, display_name) in enumerate(APPS):
        tile_x, tile_y, _, _ = geometry.tile_rects[i]
        opacity, y_offset = motion.fade_states[i].value(now_ms)

        # Reduced motion: hover scale is always 1.0.  Doing the lookup
        # would still return 1.0 (no transitions ever recorded), but
        # skipping the dict access is cheaper -- and "reduced motion"
        # means "no motion of any kind", which the explicit branch
        # makes literal.
        if motion.reduced_motion:
            scale = 1.0
        else:
            scale = motion.hover_state.scale_for(i, now_ms)

        _render_tile_with_motion(
            canvas, tile_x, tile_y,
            app_id, display_name, label_font,
            opacity=opacity, y_offset=y_offset, scale=scale,
        )


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main() -> None:
    """Run the Phase 5 fullscreen loop until ESC or Q is pressed."""
    make_fullscreen_window(WINDOW_NAME)

    # Reduced-motion detection.  CLI flag is checked first (cheap and
    # always reliable); the keypoll is the fallback for users who don't
    # want to deal with the terminal at showtime.
    reduced_motion = (
        reduced_motion_requested_via_cli()
        or reduced_motion_requested_via_keypoll()
    )

    # Font loads, mirroring Phase 4's pattern.  Reused per frame via
    # the module-level cache in phase4_home_screen / src.tiles.  The
    # FPS HUD's font is owned by `draw_fps_hud` (cached internally),
    # so it doesn't appear in this preload list.
    wordmark_font, clock_font = _get_status_bar_fonts()
    label_font = _get_label_font()

    # Canvas / geometry / motion state.  Each rebuilt on resize.
    width, height = screen_size(WINDOW_NAME)
    canvas = make_dark_canvas(width, height)
    geometry = build_grid_geometry(width, height)
    motion = build_motion_state(reduced_motion=reduced_motion)

    # Time baseline -- everything downstream is "ms since now".
    t0_ticks = cv2.getTickCount()

    # Wire up the mouse callback.  The _MouseContext is the bridge
    # between the callback (which fires on cv2's thread / when waitKey
    # pumps) and the main loop's per-frame state.  Mutating its
    # `geometry` field on resize is how we keep the hit-test in sync
    # without re-registering the callback.
    mouse_ctx = _MouseContext(geometry=geometry, motion=motion, t0_ticks=t0_ticks)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, mouse_ctx)

    last_t = time.perf_counter() - (1.0 / 60.0)
    fps_ema = 0.0

    while True:
        # Adapt to a changed display rect.  Same pattern as Phase 4 but
        # ALSO rebuild the geometry cache + sync the mouse context.
        cur_w, cur_h = screen_size(WINDOW_NAME)
        if (cur_w, cur_h) != (width, height):
            width, height = cur_w, cur_h
            canvas = make_dark_canvas(width, height)
            geometry = build_grid_geometry(width, height)
            mouse_ctx.geometry = geometry
        else:
            canvas[:, :] = BG_DARK_BGR

        # FPS (identical to phase 4 / phase 1).
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

        # Paint order (back to front):
        #   1. Grid of tiles with motion (fade-up + hover scale)
        #   2. Status bar over the wallpaper / any bleeding tile pixels
        #   3. FPS counter in the top-right corner -- ABSOLUTE LAST
        #      paint so neither the status bar nor any future overlay
        #      can occlude it.
        paint_grid_with_motion(canvas, geometry, motion, label_font, now_ms)
        paint_status_bar(canvas, width, wordmark_font, clock_font)
        draw_fps_hud(canvas, fps_ema)

        cv2.imshow(WINDOW_NAME, canvas)

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
