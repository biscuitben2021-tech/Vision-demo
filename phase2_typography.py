"""
Phase 2 -- Apple-grade typography on the off-white canvas.

Goals:
    * Render an H1 hero ("Hello, vision.") in SF Pro Display Semibold at
      80px, centered horizontally, sitting at roughly the 30% line of the
      screen -- which is where Apple anchors hero headlines on apple.com
      (high enough that the eye reads it first, low enough that it has
      breathing room above it).
    * Render a muted subhead 12px below it, also centered.
    * Render two "Learn more >"-pattern CTAs side by side, 24px apart,
      24px below the subhead, the pair horizontally centered as a group.
    * Keep the FPS counter from Phase 1 in the top-right corner.

Why this phase exists, in a sentence:  if SF Pro Display rendered through
PIL does not look unmistakably Apple at this size on this background, no
amount of tile-grid polish in Phase 3+ will save the design.  The
typography is the load-bearing element of the whole aesthetic; this
phase proves it works before anything else gets layered on top.

apple_SKILL.md notes that "Anything coming out of cv2.putText will
betray the entire design" -- that is why every glyph on this page goes
through PIL via the `draw_text` helper in `src.design`.  We do not call
`cv2.putText` once.

Centering is done by passing `align="center"` to `draw_text`, rather
than computing `(canvas_w - text_w) // 2` at each call site.  Doing the
math in the helper means every caller -- this phase, Phase 3's tile
headlines, Phase 6's app titles -- shares the same anchor semantics
instead of each re-deriving them and drifting a couple of pixels apart.

Module color-space convention:
    The cv2 pixel buffer is BGR.
    Every color constant we pass into `draw_text` is named `_RGB` because
    PIL renders it.  The cv2-side fill uses `_BGR`.  These suffixes are
    the same convention `src.design` documents.

Reduced motion / quit:
    No animations in Phase 2; nothing to skip.  ESC or Q quits the loop.
"""

from __future__ import annotations

import time
from typing import Final

import cv2
import numpy as np

from src.design import (
    ACCENT_LIGHT_RGB,
    BG_LIGHT_BGR,
    CTA_GAP,
    TEXT_MUTED_RGB,
    TEXT_ON_LIGHT_RGB,
    draw_fps_hud,
    draw_text,
    load_font,
)
from phase1_canvas import (
    FPS_EMA_ALPHA,
    QUIT_KEY_ESC,
    QUIT_KEY_Q_L,
    QUIT_KEY_Q_U,
    WINDOW_NAME,
    make_canvas,
    make_fullscreen_window,
    screen_size,
)


# ----------------------------------------------------------------------------
# Type sizes and copy
# ----------------------------------------------------------------------------
#
# These values come straight from the typography table in apple_SKILL.md
# and CLAUDE.md.  They are not knobs -- treat them as constants of the
# universe.  If a render looks "off" do not adjust here; check first that
# the font is actually SF Pro Display (and not the Helvetica Neue
# fallback, which is half a percent narrower) and that the background is
# BG_LIGHT (#fbfbfd, not pure white).

# H1 hero: SF Pro Display Semibold, 80px on desktop.  The 12-character
# headline below ("Hello, vision.") fits in one line at this size on any
# screen wider than ~860 logical px, so we do not need to wrap.
H1_SIZE:    Final[int] = 80
H1_COPY:    Final[str] = "Hello, vision."

# Subhead: SF Pro Text Regular, 21px.  Slightly bolder than the marketing
# spec's stock 17px because at 80px H1 a 17px subhead looks abandoned --
# 21px lets it read as a single typographic unit with the headline.  This
# matches the body/subhead row of the typography table verbatim.
SUBHEAD_SIZE: Final[int] = 21
SUBHEAD_COPY: Final[str] = "Spatial computing, reimagined."

# CTAs: SF Pro Text Regular 17px.  The trailing character is the actual
# Unicode chevron U+203A (`›`), NOT the ASCII greater-than `>`.  apple.com
# uses the typographic single-right-pointing angle quotation mark for
# this exact pattern, and the rendered shape is noticeably more graceful.
# Forgetting this and shipping a `>` is the kind of off-by-one detail
# that makes a render look "almost right" without being able to say why.
CTA_SIZE:           Final[int] = 17
CTA_LEFT_COPY:      Final[str] = "Learn more ›"
CTA_RIGHT_COPY:     Final[str] = "Try the demo ›"

# Vertical layout anchors.
#
# Apple sits hero headlines around the 30% mark of the viewport so the
# eye lands on them first.  `draw_text` uses the top of the bbox as its
# y-anchor, so to put the *baseline* near 30% we offset upward by the
# font's ascent.  Without that subtraction the headline would sit a
# baseline-height too low, which on an 80px font is roughly 60-65px of
# visible drift.
HERO_ANCHOR_FRAC:   Final[float] = 0.30

# Vertical gaps in the headline stack.  These are the same spirit as
# CLAUDE.md's spacing tokens but tighter than tile-internal padding
# because the hero block is one typographic unit, not separate tiles.
GAP_H1_TO_SUBHEAD:  Final[int] = 12
GAP_SUBHEAD_TO_CTA: Final[int] = 24


# ----------------------------------------------------------------------------
# CTA row layout helper
# ----------------------------------------------------------------------------

def draw_centered_cta_row(
    canvas: np.ndarray,
    canvas_w: int,
    y_top: int,
    left_text: str,
    right_text: str,
    font,  # PIL ImageFont; signature kept loose to mirror render_fps
) -> None:
    """Draw two CTAs side by side, centered as a group, at `y_top`.

    Layout math, in one place so the calling main loop stays readable:

        group_w = advance(left) + CTA_GAP + advance(right)
        start_x = (canvas_w - group_w) // 2
        left  drawn at start_x
        right drawn at start_x + advance(left) + CTA_GAP

    We use `font.getlength` (typographic advance), not `font.getbbox`
    (visible glyph extents), for exactly the same reason `render_fps` in
    Phase 1 does -- it is what a text shaper would use to step between
    glyphs and gives the CTA pair the spacing the eye expects.

    `align="left"` is correct here because we have already done the
    centering math at the group level; each individual CTA is anchored
    to its own start x.  If we used align="center" per CTA we would
    overlap them.
    """
    left_advance  = int(round(font.getlength(left_text)))
    right_advance = int(round(font.getlength(right_text)))
    group_w = left_advance + CTA_GAP + right_advance

    start_x = (canvas_w - group_w) // 2

    draw_text(
        canvas, left_text,
        x=start_x, y=y_top,
        color_rgb=ACCENT_LIGHT_RGB, font=font, align="left",
    )
    draw_text(
        canvas, right_text,
        x=start_x + left_advance + CTA_GAP, y=y_top,
        color_rgb=ACCENT_LIGHT_RGB, font=font, align="left",
    )


# ----------------------------------------------------------------------------
# Headline stack composition
# ----------------------------------------------------------------------------

def compose_hero(
    canvas: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    h1_font,
    subhead_font,
    cta_font,
) -> None:
    """Paint H1 + subhead + CTA row onto `canvas`, centered horizontally.

    Vertical placement is anchored to the H1 baseline at
    HERO_ANCHOR_FRAC of the viewport height.  Everything below stacks
    relative to the H1's actual rendered height -- not to a fixed
    px-from-top -- so the block remains visually balanced if a future
    phase swaps the H1 copy for something taller.
    """
    # PIL.getmetrics returns (ascent, descent).  Ascent = pixels from
    # baseline up to the top of the tallest glyph; descent = pixels from
    # baseline down to the bottom of the deepest descender.  draw_text
    # places the text's top at the y we pass it, so to anchor the baseline
    # to HERO_ANCHOR_FRAC we shift up by `ascent`.
    h1_ascent, _h1_descent = h1_font.getmetrics()
    baseline_y = int(canvas_h * HERO_ANCHOR_FRAC)
    h1_top_y   = baseline_y - h1_ascent

    # H1 hero.  align="center" means `x` is the horizontal midpoint of
    # the rendered text; passing canvas_w // 2 horizontally centers it
    # without us ever having to measure the glyph run at the call site.
    draw_text(
        canvas, H1_COPY,
        x=canvas_w // 2, y=h1_top_y,
        color_rgb=TEXT_ON_LIGHT_RGB, font=h1_font, align="center",
    )

    # Subhead position: GAP_H1_TO_SUBHEAD below the H1's bbox bottom.
    # The H1 bbox height (ascent + descent) is the right thing to add
    # here -- not just the font size -- because the descender of an `,`
    # or `g` in a future headline would otherwise crash into the subhead.
    _, _, _, h1_bbox_bottom = h1_font.getbbox(H1_COPY)
    subhead_top_y = h1_top_y + h1_bbox_bottom + GAP_H1_TO_SUBHEAD

    draw_text(
        canvas, SUBHEAD_COPY,
        x=canvas_w // 2, y=subhead_top_y,
        color_rgb=TEXT_MUTED_RGB, font=subhead_font, align="center",
    )

    # CTA row position: GAP_SUBHEAD_TO_CTA below the subhead's bbox
    # bottom, computed the same way to stay stable against copy changes.
    _, _, _, subhead_bbox_bottom = subhead_font.getbbox(SUBHEAD_COPY)
    cta_top_y = subhead_top_y + subhead_bbox_bottom + GAP_SUBHEAD_TO_CTA

    draw_centered_cta_row(
        canvas, canvas_w, cta_top_y,
        CTA_LEFT_COPY, CTA_RIGHT_COPY, cta_font,
    )


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main() -> None:
    """Run the Phase 2 fullscreen loop until ESC or Q is pressed."""
    make_fullscreen_window(WINDOW_NAME)

    # Load every font ONCE up front.  PIL truetype loads are not free
    # (the OS opens the file, parses the table directory, etc.), and
    # doing it once per frame would show up as a noticeable FPS drop
    # at 60Hz.  The FPS HUD's font is now owned by `draw_fps_hud` (it
    # caches its own font internally), so this list shrinks by one.
    h1_font      = load_font(role="display", size=H1_SIZE)
    subhead_font = load_font(role="text",    size=SUBHEAD_SIZE)
    cta_font     = load_font(role="text",    size=CTA_SIZE)

    width, height = screen_size(WINDOW_NAME)
    canvas = make_canvas(width, height)

    # Same FPS seeding scheme as Phase 1.  Identical reasoning: avoid a
    # 1e6-fps spike on the first frame that takes the EMA seconds to
    # bleed off.
    last_t:  float = time.perf_counter() - (1.0 / 60.0)
    fps_ema: float = 0.0

    while True:
        # Adapt to a changed display rect (rare, but the demo may run on
        # an external projector).  In steady state we just wipe and reuse.
        cur_w, cur_h = screen_size(WINDOW_NAME)
        if (cur_w, cur_h) != (width, height):
            width, height = cur_w, cur_h
            canvas = make_canvas(width, height)
        else:
            # Repaint background to wipe the previous frame's glyphs.
            # The text from frame N must not bleed into frame N+1; doing
            # it as a single BGR fill is faster than scissoring around
            # each text rect.
            canvas[:, :] = BG_LIGHT_BGR

        # FPS measurement, identical pattern to Phase 1.
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

        # Hero block (H1, subhead, CTA row) plus the top-right FPS counter.
        # draw_fps_hud is the last paint so the counter never gets occluded
        # by a status bar or notification.
        compose_hero(canvas, width, height, h1_font, subhead_font, cta_font)
        draw_fps_hud(canvas, fps_ema)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (QUIT_KEY_ESC, QUIT_KEY_Q_L, QUIT_KEY_Q_U):
            break

        # Bail out gracefully if the user closed the window some other
        # way (e.g. cmd-W in a rare non-fullscreen state) so we never
        # spin forever on a window that no longer exists.
        if cv2.getWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_VISIBLE,
        ) < 1.0:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
