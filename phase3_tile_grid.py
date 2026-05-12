"""
Phase 3 -- Static 2x2 grid of Apple-style tiles.

Goals:
    * Lay out four tiles in a 2-column x 2-row grid that fills the
      viewport, with a 16px gutter around the group and a 16px gap
      between adjacent tiles.
    * Alternate light / dark tile themes diagonally so each row has
      one of each colour, matching the half-and-half rhythm of
      apple.com's product pages.
    * Each tile contains the eyebrow + headline + subhead + chevron
      CTA pair pattern from apple_SKILL.md, rendered by `draw_tile` in
      src/tiles.py.
    * Keep the FPS counter from Phase 1 in the top-right corner.
    * ESC or Q quits.

Why this phase exists:  if the rounded_rect primitive and the eyebrow /
headline / subhead / CTA stack don't look unmistakably Apple here, every
later phase that composes tiles -- the Vision OS home screen in Phase 4,
the fake app windows in Phase 6 -- inherits the same flaw and gets
harder to fix in place.  Phase 3 is the load-bearing visual unit; lock
it down before moving on.

Layout math, in one place:

    canvas_w x canvas_h   = the realised fullscreen rect
    tile_w  = (canvas_w - 2*GAP_VIEWPORT - GAP_TILE) // 2
    tile_h  = (canvas_h - 2*GAP_VIEWPORT - GAP_TILE) // 2

    top-left     = (GAP_VIEWPORT,                       GAP_VIEWPORT)
    top-right    = (GAP_VIEWPORT + tile_w + GAP_TILE,   GAP_VIEWPORT)
    bottom-left  = (GAP_VIEWPORT,                       GAP_VIEWPORT + tile_h + GAP_TILE)
    bottom-right = (GAP_VIEWPORT + tile_w + GAP_TILE,   GAP_VIEWPORT + tile_h + GAP_TILE)

The two integer divisions can leave 1px of "slack" on the bottom and
right edges when (canvas - 2*gutter - gap) is odd.  We do NOT try to
absorb that into the last tile -- a 1px asymmetry is invisible at the
distances this demo gets viewed from, and keeping all four tiles the
same exact size makes the layout read as a true grid rather than a
"three tiles + one slightly different tile" arrangement.

Page background choice:  BG_LIGHT (#fbfbfd), the same warm near-white
Phase 1 establishes.  The 16px gaps between tiles let that background
show through as the visual separator -- exactly the "no border, no
drop shadow, separation = gap" rule from CLAUDE.md.  We chose
BG_LIGHT over BG_NEUTRAL (#f5f5f7) here because the dark tiles in the
grid are pure black; against #fbfbfd the contrast reads tighter and
more graphic than against the slightly darker neutral.

Module color-space convention:
    The cv2 pixel buffer is BGR.  This file only ever touches `_BGR`
    constants (for the cv2 background fill).  All `_RGB` work happens
    inside `draw_tile` via `draw_text`; the suffixes are the contract.
"""

from __future__ import annotations

import time
from typing import Final

import cv2
import numpy as np

from src.design import (
    BG_LIGHT_BGR,
    GAP_TILE,
    GAP_VIEWPORT,
    load_font,
)
from src.tiles import draw_tile
from phase1_canvas import (
    FPS_EMA_ALPHA,
    FPS_FONT_SIZE,
    QUIT_KEY_ESC,
    QUIT_KEY_Q_L,
    QUIT_KEY_Q_U,
    WINDOW_NAME,
    make_canvas,
    make_fullscreen_window,
    render_fps,
    screen_size,
)


# ----------------------------------------------------------------------------
# Tile content -- the four marketing cards the grid displays
# ----------------------------------------------------------------------------
#
# Each entry is the exact argument bundle `draw_tile` expects, minus the
# x/y/w/h that are computed per-frame from the canvas size.  Storing the
# content as a list of dicts (rather than four parallel constants) keeps
# the iteration loop in main() a tight three lines.
#
# The light/dark assignment is diagonal:  top-left light, top-right dark,
# bottom-left dark, bottom-right light.  This matches the "half and half"
# rhythm Apple's pages use to keep the eye moving down the column rather
# than allowing one theme to dominate a row.  Strict checkerboarding
# (alternate every cell in both directions) is what produces it on a 2x2.
#
# Copy follows the voice rules in CLAUDE.md and apple_SKILL.md: short,
# confident, terse.  No exclamation marks.  No emoji.  The eyebrow is
# always a single word naming the product; the headline is the
# aspirational claim; the subhead is the practical reassurance.

_GridEntry = dict[str, object]

TILES: Final[list[_GridEntry]] = [
    {
        "theme":    "light",
        "eyebrow":  "Photos",
        "headline": "Every memory, instantly.",
        "subhead":  "Your library, at a glance.",
        "cta_pair": ("Learn more", "Open"),
    },
    {
        "theme":    "dark",
        "eyebrow":  "Music",
        "headline": "Hello, soundtrack.",
        "subhead":  "Pick up where you left off.",
        "cta_pair": ("Learn more", "Open"),
    },
    {
        "theme":    "dark",
        "eyebrow":  "Notes",
        "headline": "Think it. Keep it.",
        "subhead":  "Everywhere you go.",
        "cta_pair": ("Learn more", "Open"),
    },
    {
        "theme":    "light",
        "eyebrow":  "Safari",
        "headline": "The web, in a window.",
        "subhead":  "Browse beautifully.",
        "cta_pair": ("Learn more", "Open"),
    },
]


# ----------------------------------------------------------------------------
# Layout math -- pure functions, no I/O
# ----------------------------------------------------------------------------

def tile_size(canvas_w: int, canvas_h: int) -> tuple[int, int]:
    """Return the (tile_w, tile_h) for a 2x2 grid filling the canvas.

    Pure function so the math is testable and obvious -- the main loop
    just calls this every frame and threads the result into the four
    `draw_tile` calls.  Integer division here may drop up to 1px on the
    bottom/right edges; see this module's docstring for why we don't
    absorb the slack into the last tile.
    """
    tile_w = (canvas_w - 2 * GAP_VIEWPORT - GAP_TILE) // 2
    tile_h = (canvas_h - 2 * GAP_VIEWPORT - GAP_TILE) // 2
    return tile_w, tile_h


def tile_origin(col: int, row: int, tile_w: int, tile_h: int) -> tuple[int, int]:
    """Return the top-left (x, y) for the tile at grid cell (col, row).

    col / row are 0-indexed; (0, 0) is top-left.  The math is the same
    pattern repeated four times, so this helper exists purely to keep
    the call site readable -- compare
        tile_origin(0, 1, tile_w, tile_h)
    against the inline
        (GAP_VIEWPORT, GAP_VIEWPORT + tile_h + GAP_TILE).
    """
    x = GAP_VIEWPORT + col * (tile_w + GAP_TILE)
    y = GAP_VIEWPORT + row * (tile_h + GAP_TILE)
    return x, y


# ----------------------------------------------------------------------------
# Frame painter
# ----------------------------------------------------------------------------

def paint_grid(canvas: np.ndarray, canvas_w: int, canvas_h: int) -> None:
    """Paint all four tiles into `canvas` for the current frame.

    Order of the tiles list is row-major: index 0 is (col=0, row=0),
    1 is (col=1, row=0), 2 is (col=0, row=1), 3 is (col=1, row=1).
    We compute size + origin once per call (cheap arithmetic, no
    allocations) and delegate everything else to `draw_tile`.
    """
    tile_w, tile_h = tile_size(canvas_w, canvas_h)

    for i, entry in enumerate(TILES):
        col = i % 2
        row = i // 2
        x, y = tile_origin(col, row, tile_w, tile_h)
        # `**entry` unpacks the dict directly into draw_tile's keyword
        # arguments.  The dict keys are chosen to exactly match
        # draw_tile's parameter names; if they ever drift apart, this
        # raises TypeError on the next frame, which is the loud failure
        # we want rather than a silent rename mismatch.
        draw_tile(canvas, x, y, tile_w, tile_h, **entry)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Main loop -- identical structure to Phase 1 / Phase 2
# ----------------------------------------------------------------------------

def main() -> None:
    """Run the Phase 3 fullscreen loop until ESC or Q is pressed."""
    make_fullscreen_window(WINDOW_NAME)

    # FPS font is the only one this module loads directly; the tile
    # renderer caches its own four fonts internally on first draw.
    fps_font = load_font(role="text", size=FPS_FONT_SIZE)

    width, height = screen_size(WINDOW_NAME)
    canvas = make_canvas(width, height)

    # Same FPS seeding scheme as Phase 1: start last_t one frame in the
    # past so dt is non-zero and we don't get a 1e6-fps spike on frame 1
    # that takes the EMA dozens of frames to bleed off.
    last_t:  float = time.perf_counter() - (1.0 / 60.0)
    fps_ema: float = 0.0

    while True:
        # Adapt to a changed display rect (rare in steady state but
        # possible if a projector gets plugged in mid-demo).
        cur_w, cur_h = screen_size(WINDOW_NAME)
        if (cur_w, cur_h) != (width, height):
            width, height = cur_w, cur_h
            canvas = make_canvas(width, height)
        else:
            # Repaint the BG_LIGHT background.  This wipes the previous
            # frame's tiles + FPS string in one fill rather than
            # scissoring around each rect.  At 1920x1080 it's a ~6MB
            # memcpy -- negligible next to imshow's compositor handoff.
            canvas[:, :] = BG_LIGHT_BGR

        # FPS measurement -- identical pattern to Phase 1.
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

        # 4 tiles + top-right FPS counter.  The order matters only in
        # that paint_grid must run before render_fps so the FPS string
        # is never accidentally drawn under a tile that overlaps the
        # top-right corner -- it won't on a 2x2, but Phase 4's home
        # screen has more cells.  Establishing the order here keeps it
        # stable across the phase progression.
        paint_grid(canvas, width, height)
        render_fps(canvas, f"{fps_ema:5.1f} fps", fps_font, width)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (QUIT_KEY_ESC, QUIT_KEY_Q_L, QUIT_KEY_Q_U):
            break

        # Bail out gracefully if the user closed the window some other
        # way (cmd-W in a rare non-fullscreen state) instead of spinning
        # forever on a window that no longer exists.
        if cv2.getWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_VISIBLE,
        ) < 1.0:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
