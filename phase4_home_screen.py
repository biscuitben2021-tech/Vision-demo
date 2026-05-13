"""
Phase 4 -- Vision OS home screen.

Goals:
    * Fill the canvas with BG_DARK (#000000) -- the wallpaper of the
      fake OS.  This is the one place in the demo where pure black is
      the correct call: visionOS's environment fades out into actual
      darkness around the user, and the home tiles need to read as
      floating glass against that absence.
    * Paint a 44px-tall translucent status bar across the top.  Left:
      "vision" wordmark; right: a "10:30 AM" clock placeholder
      (Phase 7 wires a live clock in -- here we hard-code so the bar
      can be visually validated independently).
    * Paint a 4-column x 2-row grid of 8 floating glass app tiles.
      Each tile holds one rounded coloured app icon centred inside it,
      with the app's display name printed below the tile in muted
      white.  Tiles use the `draw_glass_panel` primitive from
      `src/icons.py`; icons use `draw_app_icon` from the same module.
    * Preserve the FPS counter from Phase 1 (top-right corner) so
      every phase share a common diagnostic surface.
    * ESC or Q quits.

Why this phase exists:  Phase 3 proved a single rounded *opaque* tile
reads as Apple-grade.  Phase 4 adds the two visual jumps that turn
that into a Vision OS look -- the *glass* compositing pass, and the
procedural app-icon glyphs.  If those land here, Phase 5 (motion) and
Phase 6 (fake app windows) get an established surface vocabulary to
build on rather than each inventing their own.

Layout math, in one place:

    canvas_w x canvas_h     = the realised fullscreen rect
    status_bar              = (0, 0, canvas_w, NAV_HEIGHT)            -- 44px tall

    tile_w = tile_h         = 140                                     -- square tile, fixed
    icon_size               = 100                                     -- icon inside the tile
    grid_cols               = 4
    grid_rows               = 2
    grid_w = 4*tile_w + 3*GRID_GAP
    grid_h = 2*tile_h + 1*GRID_GAP + 2*LABEL_BAND  -- LABEL_BAND is the room below each tile for its label
    grid_x = (canvas_w - grid_w) // 2
    grid_y = NAV_HEIGHT + GRID_TOP_GUTTER

The grid is fixed-size and horizontally centred -- it does NOT stretch
to fill the viewport like Phase 3's marketing tile grid did.  visionOS
home tiles are a fixed pitch regardless of resolution; the wallpaper
takes the slack.

Module color-space convention:
    The cv2 pixel buffer is BGR.  Constants with `_BGR` suffix go to
    cv2 calls; constants with `_RGB` cross into PIL via `draw_text`.
    Same convention as src/design, src/tiles, phase1-3.
"""

from __future__ import annotations

import time
from typing import Final

import cv2
import numpy as np

from src.design import (
    BG_DARK_BGR,
    NAV_HEIGHT,
    RADIUS_APP_ICON,
    TEXT_ON_DARK_RGB,
    draw_fps_hud,
    draw_text,
    load_font,
    make_canvas as _design_make_canvas,
)
from src.icons import draw_app_icon, draw_glass_panel
from phase1_canvas import (
    FPS_EMA_ALPHA,
    QUIT_KEY_ESC,
    QUIT_KEY_Q_L,
    QUIT_KEY_Q_U,
    WINDOW_NAME,
    make_fullscreen_window,
    screen_size,
)


# ----------------------------------------------------------------------------
# Status bar constants
# ----------------------------------------------------------------------------
#
# 44px is the height Apple uses for the visionOS / iPadOS top bar -- big
# enough to read at glance distance, small enough that it doesn't crowd
# the wallpaper.  We use the existing NAV_HEIGHT token from src/design so
# the value lives in one place.
#
# KNOWN COMPROMISE: the spec calls for "SF Pro Text 14px".  load_font
# only exposes ("display" -> Display Semibold) and ("text" -> Text
# Regular).  We use Text Regular here.  visionOS uses an even lighter
# weight (closer to "Light") for status-bar glyphs; Text Regular reads
# slightly heavier than ideal but is the closest weight we have.  If a
# Phase 7 polish pass adds load_font(role="text-semibold") we can
# revisit.

STATUS_BAR_FONT_SIZE: Final[int] = 14
STATUS_BAR_PAD_X:     Final[int] = 16     # horizontal inset for left / right text
STATUS_BAR_RADIUS:    Final[int] = 0      # status bar runs edge-to-edge; no corners

# Hard-coded clock for Phase 4.  Phase 7 replaces this with
# `time.strftime("%-I:%M %p")` so the demo looks live during the show.
# We deliberately freeze it here so the home-screen layout can be
# eyeballed without watching the seconds tick.
STATUS_CLOCK_PLACEHOLDER: Final[str] = "10:30 AM"

# Left wordmark.  Lowercase to match Apple's existing wordmark style
# ("watchOS", "visionOS") -- a single lowercase word reads as a system
# brand rather than as an app title.
STATUS_WORDMARK: Final[str] = "vision"


# ----------------------------------------------------------------------------
# Home-screen grid constants
# ----------------------------------------------------------------------------
#
# Tile pitch is fixed regardless of resolution -- a hallmark of
# visionOS / iPadOS / iOS, where icons live on a fixed-size grid and
# the wallpaper takes the slack at the margins.  We do not stretch
# tiles to fill the viewport like phase3_tile_grid does.

TILE_W:      Final[int] = 140    # square home-screen tile
TILE_H:      Final[int] = 140
ICON_SIZE:   Final[int] = 100    # icon inside a 140px tile leaves 20px breathing room each side
GRID_GAP:    Final[int] = 24     # gap between adjacent tiles (both axes)
GRID_COLS:   Final[int] = 4
GRID_ROWS:   Final[int] = 2

# Space *below* a tile reserved for the wordmark label ("Safari",
# "Photos").  Label sits 10px below the tile bottom and is 18px tall;
# 28 is rounded up so the next row of tiles never crowds an ascender.
LABEL_GAP_FROM_TILE: Final[int] = 10
LABEL_BAND_HEIGHT:   Final[int] = 28
LABEL_FONT_SIZE:     Final[int] = 13

# Distance from the status bar's bottom to the top of the first tile
# row.  ~80px is the "breathing room" the prompt calls for; same value
# used for tile interior padding elsewhere, intentionally to give the
# home screen the same visual rhythm as a marketing page.
GRID_TOP_GUTTER: Final[int] = 80


# ----------------------------------------------------------------------------
# App roster -- order is left-to-right, top-to-bottom in the grid
# ----------------------------------------------------------------------------
#
# The home screen's left-to-right top-to-bottom order matters: a
# returning user scans the same positions every time, so the order
# becomes part of the muscle memory of the demo.  Storing the roster
# as a list of (app_id, display_name) tuples keeps the iteration loop
# in `paint_grid` a tight three-liner.
#
# `app_id` keys here MUST match the keys in `_ICON_TINTS` /
# `_ICON_DRAWERS` over in src/icons.py.  A typo raises ValueError at
# draw time -- which is the loud-failure mode we want over a silent
# blank tile.

APPS: Final[list[tuple[str, str]]] = [
    ("safari",   "Safari"),
    ("photos",   "Photos"),
    ("music",    "Music"),
    ("notes",    "Notes"),
    ("mail",     "Mail"),
    ("calendar", "Calendar"),
    ("settings", "Settings"),
    ("demo",     "Demo"),
]


# ----------------------------------------------------------------------------
# Layout math -- pure functions, no I/O
# ----------------------------------------------------------------------------

def grid_size() -> tuple[int, int]:
    """Return the (width, height) the entire 4x2 grid occupies in pixels.

    Width  = 4 tiles + 3 gaps between them.
    Height = 2 tile rows + 1 inter-row gap + 2 label bands (one per
             row, sitting below each row's tiles).

    Pure function -- depends only on module constants, so the result
    is the same every call.  Kept as a function rather than a constant
    so future phases can swap TILE_W / GRID_COLS at runtime if needed.
    """
    width  = GRID_COLS * TILE_W + (GRID_COLS - 1) * GRID_GAP
    height = (
        GRID_ROWS * TILE_H
        + (GRID_ROWS - 1) * GRID_GAP
        + GRID_ROWS * (LABEL_GAP_FROM_TILE + LABEL_BAND_HEIGHT)
    )
    return width, height


def tile_origin(
    col: int, row: int, grid_x: int, grid_y: int,
) -> tuple[int, int]:
    """Return the top-left (x, y) of the tile at grid cell (col, row).

    col / row are 0-indexed; (0, 0) is the top-left tile.  `grid_x` and
    `grid_y` are the top-left of the whole grid (passed in so the
    main loop can centre the grid horizontally and anchor it under the
    status bar without this helper having to know about the canvas).
    """
    # Each row's vertical offset includes the previous row's tile +
    # its label band + the inter-row gap.  Encoding that in arithmetic
    # rather than a loop keeps this a single expression.
    row_pitch = TILE_H + LABEL_GAP_FROM_TILE + LABEL_BAND_HEIGHT + GRID_GAP
    x = grid_x + col * (TILE_W + GRID_GAP)
    y = grid_y + row * row_pitch
    return x, y


# ----------------------------------------------------------------------------
# Frame painters
# ----------------------------------------------------------------------------

def paint_status_bar(
    canvas: np.ndarray,
    canvas_w: int,
    wordmark_font,
    clock_font,
) -> None:
    """Draw the translucent 44px-tall status bar across the top of the canvas.

    Order matters: glass surface goes down FIRST, then the wordmark and
    clock are composited on top.  Drawing the text first would have it
    immediately overwritten by the glass pass.

    Both text anchors sit 16px (STATUS_BAR_PAD_X) from their respective
    edges and are vertically centred within the bar.  Vertical centring
    uses the font's bbox so descenders don't kick the baseline off.
    """
    # Glass surface across the full width.  STATUS_BAR_RADIUS=0 makes
    # this a sharp-cornered rect -- the bar runs edge-to-edge in
    # visionOS, no rounding required.
    draw_glass_panel(canvas, x=0, y=0,
                     w=canvas_w, h=NAV_HEIGHT,
                     radius=STATUS_BAR_RADIUS)

    # Vertical centring: getbbox returns (left, top, right, bottom).
    # The visible glyph height is bottom - top.  `draw_text` anchors at
    # the top of the rendered patch (already corrected for the font's
    # internal `top` offset inside the helper), so the right number to
    # subtract from NAV_HEIGHT is simply `text_h` -- there is no
    # additional `top_bbox` correction needed here.  Subtracting it
    # would push the baseline above the bar by the font's ascent.
    _, top_bbox, _, bot_bbox = wordmark_font.getbbox(STATUS_WORDMARK)
    text_h = bot_bbox - top_bbox
    text_y = (NAV_HEIGHT - text_h) // 2

    # Left: "vision" wordmark.  Anchored at STATUS_BAR_PAD_X from the
    # left edge, vertically centred.
    draw_text(canvas, STATUS_WORDMARK,
              x=STATUS_BAR_PAD_X, y=text_y,
              color_rgb=TEXT_ON_DARK_RGB, font=wordmark_font, align="left")

    # Right: "10:30 AM" clock.  align="right" puts the right edge of
    # the rendered string at the given x; we offset by STATUS_BAR_PAD_X
    # from the canvas right edge.  Vertical anchor reuses text_y (both
    # strings are the same point size, identical baseline).
    draw_text(canvas, STATUS_CLOCK_PLACEHOLDER,
              x=canvas_w - STATUS_BAR_PAD_X, y=text_y,
              color_rgb=TEXT_ON_DARK_RGB, font=clock_font, align="right")


def paint_tile(
    canvas: np.ndarray,
    tile_x: int,
    tile_y: int,
    app_id: str,
    display_name: str,
    label_font,
) -> None:
    """Paint one glass tile + its icon + its label below.

    Layered top-to-bottom:
        1. Glass panel at (tile_x, tile_y), TILE_W x TILE_H,
           RADIUS_APP_ICON corners.
        2. App icon (rounded coloured square + glyph) centred inside
           that panel.  ICON_SIZE leaves a uniform 20px inset.
        3. Wordmark label below the tile, centred horizontally on the
           tile's centre x.
    """
    # 1. Glass tile.
    draw_glass_panel(canvas, x=tile_x, y=tile_y,
                     w=TILE_W, h=TILE_H,
                     radius=RADIUS_APP_ICON)

    # 2. Icon centred inside the tile.
    icon_cx = tile_x + TILE_W // 2
    icon_cy = tile_y + TILE_H // 2
    draw_app_icon(canvas, cx=icon_cx, cy=icon_cy,
                  size=ICON_SIZE, app_id=app_id)

    # 3. Label, anchored to the centre x of the tile, sitting
    #    LABEL_GAP_FROM_TILE pixels below the tile's bottom edge.
    #    draw_text's `y` is the top of the bounding box, so this is a
    #    straight pixel offset -- no centring math.
    label_y = tile_y + TILE_H + LABEL_GAP_FROM_TILE
    draw_text(canvas, display_name,
              x=icon_cx, y=label_y,
              color_rgb=TEXT_ON_DARK_RGB, font=label_font, align="center")


def paint_grid(canvas: np.ndarray, canvas_w: int) -> None:
    """Paint the centred 4x2 grid of home-screen tiles."""
    grid_w, _ = grid_size()
    grid_x = (canvas_w - grid_w) // 2
    grid_y = NAV_HEIGHT + GRID_TOP_GUTTER

    label_font = _get_label_font()

    for i, (app_id, display_name) in enumerate(APPS):
        col = i % GRID_COLS
        row = i // GRID_COLS
        tile_x, tile_y = tile_origin(col, row, grid_x, grid_y)
        paint_tile(canvas, tile_x, tile_y, app_id, display_name, label_font)


# NOTE: An earlier version of this file defined `render_fps_below_bar`
# which anchored the FPS counter at (canvas_w - 16, NAV_HEIGHT + 16) in
# TEXT_ON_DARK_RGB (near-white).  That helper was the source of the
# "FPS in the wrong place, wrong colour" bug: the counter sat directly
# under the status bar, painted bright on dark, which both clashed
# with the clock placement and stayed bright when an app's light
# wallpaper paged in.  The canonical FPS HUD now lives in
# `src.design.draw_fps_hud` (top-right, 20px inset, TEXT_TERTIARY_RGB
# dim grey, drawn as the LAST paint each frame so nothing occludes it).
# Every phase that previously called `render_fps_below_bar` has been
# routed through `draw_fps_hud`; the old helper has been removed.


# ----------------------------------------------------------------------------
# Font cache
# ----------------------------------------------------------------------------
#
# Loading a PIL truetype font costs a few ms; doing it per frame at
# 60Hz shows up as a measurable hit on the M2.  We memoise the three
# fonts this phase uses (status wordmark, status clock, tile label)
# on each helper's function object.  Same pattern as src/tiles._get_tile_fonts.
#
# CLAUDE.md's "no global mutable state" rule targets behavioural
# globals; a pure cache that returns the same object on the same key
# is the safe exception every codebase makes.

def _get_status_bar_fonts() -> tuple:
    """Return the (wordmark_font, clock_font) for the status bar."""
    cache = getattr(_get_status_bar_fonts, "_cache", None)
    if cache is None:
        wm = load_font(role="text", size=STATUS_BAR_FONT_SIZE)
        ck = load_font(role="text", size=STATUS_BAR_FONT_SIZE)
        cache = (wm, ck)
        _get_status_bar_fonts._cache = cache  # type: ignore[attr-defined]
    return cache


def _get_label_font():
    """Return the SF Pro Text Regular 13px font used under each tile."""
    cache = getattr(_get_label_font, "_cache", None)
    if cache is None:
        cache = load_font(role="text", size=LABEL_FONT_SIZE)
        _get_label_font._cache = cache  # type: ignore[attr-defined]
    return cache


def make_dark_canvas(width: int, height: int) -> np.ndarray:
    """Return a fresh BGR canvas painted with BG_DARK (pure black).

    Phase 1's `make_canvas` defaults to BG_LIGHT, which is correct for
    marketing pages but wrong for the visionOS home screen -- the
    home wallpaper is the one place in this demo where pure #000 is
    the designed colour, per CLAUDE.md.  This helper now delegates to
    `src.design.make_canvas` with `color=BG_DARK_BGR` so the actual
    allocation logic lives in one place; only the choice of
    wallpaper colour stays as a per-phase decision.

    Crucial property: the ENTIRE canvas comes back pre-filled with
    BG_DARK_BGR.  No subsequent paint in `_render_home` /
    `_compose_screen` may leave any pixel in the wrong wallpaper colour;
    that is the contract this factory upholds and the reason every
    home-screen rendering path routes through it (including the
    transition-side sub-buffers in `src.compositor`).
    """
    return _design_make_canvas(width, height, BG_DARK_BGR)


# ----------------------------------------------------------------------------
# Main loop -- same structure as Phase 1 / 2 / 3
# ----------------------------------------------------------------------------

def main() -> None:
    """Run the Phase 4 fullscreen loop until ESC or Q is pressed."""
    make_fullscreen_window(WINDOW_NAME)

    # FPS HUD owns its own font cache (see src.design.draw_fps_hud);
    # only the status-bar fonts need preloading at this scope.
    wordmark_font, clock_font = _get_status_bar_fonts()

    width, height = screen_size(WINDOW_NAME)
    canvas = make_dark_canvas(width, height)

    # Same FPS seeding scheme as Phase 1: start last_t one frame in the
    # past so dt is non-zero on frame 1 and the EMA doesn't have to
    # absorb a 1e6-fps spike.
    last_t:  float = time.perf_counter() - (1.0 / 60.0)
    fps_ema: float = 0.0

    while True:
        # Adapt to a changed display rect.  Rare in steady state but
        # possible if a projector gets plugged in mid-demo.
        cur_w, cur_h = screen_size(WINDOW_NAME)
        if (cur_w, cur_h) != (width, height):
            width, height = cur_w, cur_h
            canvas = make_dark_canvas(width, height)
        else:
            # Repaint BG_DARK to wipe the previous frame.  At 1920x1080
            # this is a ~6MB memcpy, trivial next to imshow's compositor
            # handoff.  No need for scissored rects yet.
            canvas[:, :] = BG_DARK_BGR

        # FPS measurement -- identical to Phase 1.
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

        # Paint order (back to front):
        #   1. Grid of tiles (sits over the wallpaper)
        #   2. Status bar (sits over the wallpaper AND its own slice
        #      of any tiles that happen to extend into the top 44px --
        #      they don't, by layout, but the order makes the
        #      occlusion correct if a future tile bleeds up).
        #   3. FPS counter -- ABSOLUTE LAST paint, anchored in the
        #      top-right corner via draw_fps_hud.  Drawing it last is
        #      what guarantees the status bar (which sits in the same
        #      top edge of the canvas) never occludes the counter.
        paint_grid(canvas, width)
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
