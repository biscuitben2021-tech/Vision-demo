"""
Rounded tile primitive + Apple-style tile renderer.

Two responsibilities live in this module:

    1. `rounded_rect` -- the geometric primitive.  OpenCV has no native
       rounded rectangle; we synthesise one from two crossing filled rects
       plus four antialiased corner circles.  Every visible "tile" in this
       demo (and most of the chrome in later phases) is built on this
       primitive.
    2. `draw_tile` -- the composite that lays an Apple marketing tile
       on top of a `rounded_rect` background: eyebrow + headline (wrapping
       to 2 lines max) + muted subhead + a row of "Learn more >"-pattern
       chevron CTAs.

The two are split deliberately.  `rounded_rect` is reused as-is for app
icon backgrounds in Phase 4 and the fake app windows in Phase 6.  Future
phases that need rounded glass without the marketing-copy stack should
call `rounded_rect` directly and not be forced to provide eyebrow/headline
strings just to get a shape.

Module color-space convention:
    The cv2 pixel buffer is BGR.  Constants imported from src.design with
    a `_BGR` suffix are passed straight to cv2 calls; constants with a
    `_RGB` suffix cross the boundary into PIL via `draw_text`.  No raw
    tuples are introduced in this file -- every colour comes from
    src.design with its byte-order baked into the name.

Vertical-rhythm convention:
    All vertical stacking inside a tile uses the real PIL font metrics
    -- `font.getbbox(text)` for visible glyph extents -- rather than
    magic pixel offsets.  This is the same reason `compose_hero` in
    phase2_typography.py reads the H1 bbox bottom before placing the
    subhead: copy changes don't drift the layout because the next
    element anchors to the previous element's actual rendered height.
"""

from __future__ import annotations

from typing import Final, Literal

import cv2
import numpy as np
from PIL import ImageFont

from src.design import (
    ACCENT_DARK_RGB,
    ACCENT_LIGHT_RGB,
    BG_DARK_BGR,
    BG_LIGHT_BGR,
    CTA_GAP,
    PAD_TILE_TOP,
    PAD_TILE_X,
    RADIUS_TILE_LARGE,
    TEXT_MUTED_RGB,
    TEXT_ON_DARK_RGB,
    TEXT_ON_LIGHT_RGB,
    draw_text,
    load_font,
)


# ============================================================================
# Type-size constants for a single Apple-style tile
# ============================================================================
#
# Pulled straight from the typography table in CLAUDE.md / apple_SKILL.md.
# These are not knobs.  If a render looks "off" check the font role and
# the background colour before nudging these by +/-2px.

# H3 eyebrow.  Spec calls for SF Pro Text Semibold at 21px.  KNOWN COMPROMISE:
# `load_font` in src/design.py only exposes ("display" -> SF Pro Display
# Semibold) and ("text" -> SF Pro Text Regular); there is no Text Semibold
# axis exposed today.  We use `text` (Regular) here and accept the trade.
# The typographic hierarchy still reads: the eyebrow is smaller than the
# H2 below it, and on light tiles its strong near-black colour against
# the muted subhead is the actual hierarchy cue.  Phase 7 polish can
# extend load_font with a weight parameter and revisit.
EYEBROW_SIZE: Final[int] = 21

# H2 tile headline.  SF Pro Display Semibold, 48px.  Wraps to 2 lines when
# its advance exceeds the interior content width; never shrinks.  The
# 12-character cap in apple_SKILL.md is a *style guideline* for copy
# authors, not a render-time constraint -- we wrap on word boundaries
# rather than rejecting strings longer than that.
HEADLINE_SIZE: Final[int] = 48

# Body / subhead.  SF Pro Text Regular, 21px.  Same size as eyebrow on
# purpose: their visual difference comes from colour (near-black vs
# muted) and from sitting on either side of a 48px headline, not from a
# point-size shift.
SUBHEAD_SIZE: Final[int] = 21

# CTA row.  SF Pro Text Regular, 17px.  Same as the Phase 2 CTAs.
CTA_SIZE: Final[int] = 17


# ----------------------------------------------------------------------------
# Vertical rhythm gaps inside a tile (px)
# ----------------------------------------------------------------------------
#
# These describe the *vertical breathing room* between the four typographic
# blocks in a tile.  They are intentionally smaller than the tile padding
# (PAD_TILE_TOP = 80, PAD_TILE_X = 40) because they live *inside* a
# single tile, not between tiles.  CLAUDE.md fixes these as 12 / 16 / 24;
# we treat them as constants of the universe rather than knobs.

GAP_EYEBROW_TO_HEADLINE: Final[int] = 12   # eyebrow bbox bottom -> headline top
GAP_HEADLINE_TO_SUBHEAD: Final[int] = 16   # headline last line's bbox bottom -> subhead top
GAP_SUBHEAD_TO_CTA:      Final[int] = 24   # subhead bbox bottom -> CTA row top

# Headline line-height multiplier.  apple_SKILL.md specifies 1.05 for H1
# and we extend the same value to H2 here; at 48px that's ~50px between
# baselines on a two-line wrap.  Using a multiplier of font size rather
# than getbbox-height keeps lines visually even when one line happens
# to lack descenders (e.g. "Hello, soundtrack." vs "Music")
HEADLINE_LINE_HEIGHT_MULT: Final[float] = 1.05

# CTA chevron.  Unicode U+203A (single right-pointing angle quotation
# mark), NOT the ASCII greater-than.  Apple's marketing site uses the
# real typographic chevron; rendering a plain `>` is the kind of detail
# that makes a tile look "almost right" without anyone being able to
# articulate why.  Phase 2 already established this convention.
_CHEVRON: Final[str] = " ›"


# ============================================================================
# Theming helpers
# ============================================================================
#
# A tile's theme picks four coupled colours at once: the fill that draws
# its background, the strong "ink" colour for eyebrow + headline, the
# muted colour for the subhead, and the accent for CTAs.  Threading all
# four through every helper signature is ugly; we resolve them once into
# a single typed dict and pass that down.

Theme = Literal["light", "dark"]


def _theme_colors(theme: Theme) -> dict[str, tuple[int, int, int]]:
    """Return the colour palette for a tile of the given theme.

    Keys:
        fill_bgr     -- cv2 fill for `rounded_rect` (BGR).
        ink_rgb      -- eyebrow + headline text colour (RGB, for PIL).
        muted_rgb    -- subhead text colour (RGB).  KNOWN COMPROMISE: we
                        reuse TEXT_MUTED_RGB on both themes.  Apple's dark
                        tiles use a slightly desaturated brighter muted
                        ("dark muted") that we have not yet tokenised.  On
                        pure black the existing #86868b still reads cleanly
                        -- it is just a hair dimmer than ideal.
        accent_rgb   -- CTA link colour (RGB).  Brighter on dark, so it
                        reads against #000.

    Why a dict instead of a NamedTuple: this is a private throwaway shape,
    passed only between sibling helpers in this module.  A dataclass would
    add ceremony without buying type safety we don't already get from the
    `Theme` Literal upstream.
    """
    if theme == "light":
        return {
            "fill_bgr":   BG_LIGHT_BGR,
            "ink_rgb":    TEXT_ON_LIGHT_RGB,
            "muted_rgb":  TEXT_MUTED_RGB,
            "accent_rgb": ACCENT_LIGHT_RGB,
        }
    if theme == "dark":
        return {
            "fill_bgr":   BG_DARK_BGR,
            "ink_rgb":    TEXT_ON_DARK_RGB,
            "muted_rgb":  TEXT_MUTED_RGB,
            "accent_rgb": ACCENT_DARK_RGB,
        }
    raise ValueError(f"Unknown theme {theme!r}; expected 'light' or 'dark'.")


# ============================================================================
# rounded_rect -- the geometric primitive
# ============================================================================

def rounded_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
    color_bgr: tuple[int, int, int],
) -> None:
    """Draw a filled rounded rectangle into `frame`, mutating it in place.

    Geometry, the standard "cross + four corner circles" idiom:

        The tile occupies pixel range  [x, x+w-1] x [y, y+h-1].
        Split it into a vertical bar (full width, inset on Y by `radius`)
        and a horizontal bar (full height, inset on X by `radius`).
        Together they cover everything except the four corner squares.
        Fill each corner square with an antialiased circle whose center
        sits at  (x+radius,     y+radius)            -- top-left
                 (x+w-1-radius, y+radius)            -- top-right
                 (x+radius,     y+h-1-radius)        -- bottom-left
                 (x+w-1-radius, y+h-1-radius)        -- bottom-right
        and whose radius is exactly `radius`.  The `-1` offsets account
        for cv2.rectangle's inclusive bottom-right corner: a rect drawn
        from (x, y) to (x+w-1, y+h-1) fills exactly w*h pixels.

    The two straight bars do NOT need cv2.LINE_AA -- their edges are
    axis-aligned and AA would just blur them.  The four circles MUST
    use LINE_AA; without it the corners read as jagged 8-bit pixel
    steps and the whole tile looks unfinished.

    Args:
        frame:     BGR uint8 image to draw into.  Mutated in place.
        x, y:      top-left pixel of the bounding rect.
        w, h:      width and height in pixels.  Must satisfy
                   w >= 2*radius and h >= 2*radius; if either is smaller
                   the function silently clamps `radius` so the geometry
                   degrades gracefully rather than raising.
        radius:    corner radius in pixels.  0 produces a plain rect.
        color_bgr: BGR tuple in 0..255 -- pass a `_BGR` constant from
                   src.design, never a hand-rolled tuple.
    """
    # Clamp radius so a too-small tile degrades into a regular rect or
    # an oval rather than producing a malformed shape with negative-sized
    # bars.  Callers should not rely on this in production layouts, but
    # at small canvas sizes during the brief "screen size not yet known"
    # window in main() it prevents crashes.
    r = max(0, min(radius, w // 2, h // 2))

    # Vertical bar: full height, inset on X by r.  Covers the entire
    # center column.  Pass thickness=-1 to get a filled rect.
    cv2.rectangle(
        frame, (x + r, y), (x + w - 1 - r, y + h - 1),
        color_bgr, thickness=-1,
    )
    # Horizontal bar: full width, inset on Y by r.  Covers the entire
    # center row.  Together with the vertical bar this fills everything
    # except the four corner squares.
    cv2.rectangle(
        frame, (x, y + r), (x + w - 1, y + h - 1 - r),
        color_bgr, thickness=-1,
    )

    # Corner circles.  thickness=-1 for filled; LINE_AA for smooth edges
    # -- this is the single biggest visual-quality decision in the entire
    # tile renderer.  Drop it and every tile looks like a 1990s GUI.
    cv2.circle(frame, (x + r,         y + r),         r, color_bgr,
               thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x + w - 1 - r, y + r),         r, color_bgr,
               thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x + r,         y + h - 1 - r), r, color_bgr,
               thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x + w - 1 - r, y + h - 1 - r), r, color_bgr,
               thickness=-1, lineType=cv2.LINE_AA)


# ============================================================================
# Word-wrap helper for the headline
# ============================================================================

def _wrap_two_lines(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_w: int,
) -> list[str]:
    """Greedy word-wrap `text` into at most two lines that each fit `max_w`.

    Greedy is the right algorithm here: marketing headlines are short and
    we never wrap to more than two lines -- a Knuth/Plass paragraph
    layout would be wildly overkill and produce identical results.

    Strategy:
        Walk the words left to right.  Append each word to the current
        line; if adding it would push the advance past `max_w`, push
        the current line into `lines` and start a new one.  Once we
        already have ONE line accumulated, we never push again -- the
        second line is allowed to overflow horizontally with the rest
        of the words concatenated onto it.  Overflow is the
        loud-failure mode the codebase prefers over silent truncation;
        marketing copy that hits this branch wasn't written to the
        12-character-per-line guideline and we want the author to see
        the problem at render time.

    Returns a list of 1 or 2 strings.  Empty input returns [""] so the
    caller can iterate without a length branch.
    """
    if not text:
        return [""]

    words = text.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else current + " " + word
        # getlength returns typographic advance -- the right number to
        # compare against `max_w`; getbbox includes side bearings that
        # would cause us to wrap a few pixels too early.
        fits = int(font.getlength(candidate)) <= max_w

        if fits or not current:
            # Always accept the first word of a line even if it overflows
            # on its own; otherwise a single very long word would loop
            # forever with `current` never advancing.
            current = candidate
            continue

        if len(lines) >= 1:
            # We are already on line 2 and the next word does not fit.
            # Append it anyway -- overflow is the visible failure mode.
            current = candidate
            continue

        # Standard wrap: push the filled line, start the next one with
        # this word as the seed.
        lines.append(current)
        current = word

    if current:
        lines.append(current)

    return lines


# ============================================================================
# Sub-renderers for the typographic stack
# ============================================================================
#
# Each helper draws one block and returns the y-coordinate where the
# NEXT block should start.  The caller threads that y forward, which is
# the same flow `compose_hero` uses in phase2_typography.py.

def _draw_eyebrow(
    frame: np.ndarray,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    color_rgb: tuple[int, int, int],
) -> int:
    """Draw the eyebrow at (x, y), return the y where the next block starts.

    The eyebrow is anchored to (x, y) at the top of its bounding box.
    The returned `next_y` is `y + bbox.bottom + GAP_EYEBROW_TO_HEADLINE`
    so the caller doesn't have to know about the gap constant.
    """
    draw_text(frame, text, x=x, y=y, color_rgb=color_rgb, font=font,
              align="left")
    _, _, _, bbox_bottom = font.getbbox(text)
    return y + bbox_bottom + GAP_EYEBROW_TO_HEADLINE


def _draw_headline(
    frame: np.ndarray,
    x: int,
    y: int,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    color_rgb: tuple[int, int, int],
) -> int:
    """Draw a 1- or 2-line headline at (x, y); return the next-block y.

    Lines are placed at fixed offsets driven by HEADLINE_LINE_HEIGHT_MULT
    times the font size, NOT by the getbbox of each individual line.
    Using a fixed line-height keeps the spacing between baselines visually
    even regardless of whether a particular line happens to contain
    descenders -- mixing getbbox-driven spacing with text like
    "Hello, soundtrack." (descenders) and "Music" (none) would produce
    a stack that looks subtly tilted to the eye.
    """
    line_height = int(round(font.size * HEADLINE_LINE_HEIGHT_MULT))

    for i, line in enumerate(lines):
        draw_text(frame, line, x=x, y=y + i * line_height,
                  color_rgb=color_rgb, font=font, align="left")

    # Anchor the next block to the LAST line's actual bbox bottom rather
    # than to a synthetic (y + n*line_height) -- that way the subhead's
    # gap is measured from real visible glyphs, not from where the next
    # invisible line would have started.
    last_y = y + (len(lines) - 1) * line_height
    _, _, _, last_bbox_bottom = font.getbbox(lines[-1])
    return last_y + last_bbox_bottom + GAP_HEADLINE_TO_SUBHEAD


def _draw_subhead(
    frame: np.ndarray,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    color_rgb: tuple[int, int, int],
) -> int:
    """Draw the subhead at (x, y); return the y where the CTA row starts."""
    draw_text(frame, text, x=x, y=y, color_rgb=color_rgb, font=font,
              align="left")
    _, _, _, bbox_bottom = font.getbbox(text)
    return y + bbox_bottom + GAP_SUBHEAD_TO_CTA


def _draw_cta_row(
    frame: np.ndarray,
    content_x: int,
    content_w: int,
    y: int,
    left_label: str,
    right_label: str,
    font: ImageFont.FreeTypeFont,
    color_rgb: tuple[int, int, int],
) -> None:
    """Draw two CTAs side-by-side, centered as a group within the content column.

    `content_x` / `content_w` bound the tile's interior text column
    (i.e. tile_x + PAD_TILE_X and tile_w - 2*PAD_TILE_X respectively).
    The pair is centered as a unit within that column with CTA_GAP
    between the two labels.

    align="left" is correct here: we have already done the group-level
    centering, so each individual CTA is anchored to its own start x.
    Using align="center" per CTA would overlap them.  This mirrors
    `draw_centered_cta_row` in phase2_typography.py exactly.

    The caller passes BARE labels ("Learn more", "Open"); the chevron
    is appended here so the tile API stays clean and no caller can
    accidentally ship a `>` instead of `›`.
    """
    left_text  = left_label  + _CHEVRON
    right_text = right_label + _CHEVRON

    left_advance  = int(round(font.getlength(left_text)))
    right_advance = int(round(font.getlength(right_text)))
    group_w = left_advance + CTA_GAP + right_advance

    start_x = content_x + (content_w - group_w) // 2

    draw_text(frame, left_text,  x=start_x, y=y,
              color_rgb=color_rgb, font=font, align="left")
    draw_text(frame, right_text,
              x=start_x + left_advance + CTA_GAP, y=y,
              color_rgb=color_rgb, font=font, align="left")


# ============================================================================
# Font cache (per-process; not module-global mutable state)
# ============================================================================
#
# PIL truetype loads are not free: opening the .ttf and parsing its
# table directory shows up as a measurable cost when done per frame at
# 60 Hz on the M2.  We memoise the four fonts a tile uses behind a
# tiny helper.  This is module-private state, lives only on the
# function object, and is purely a performance optimisation -- it has
# no effect on rendered output.
#
# CLAUDE.md's "no global mutable state" rule targets *behavioural*
# globals (cursors, mode toggles, layout state).  A pure cache that
# returns identical objects on identical keys is a safe exception
# every Python codebase makes; the alternative is passing five font
# handles through every draw_tile call.

def _get_tile_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    """Return the four fonts used by `draw_tile`, loaded once per process."""
    cache = getattr(_get_tile_fonts, "_cache", None)
    if cache is None:
        cache = {
            "eyebrow":  load_font(role="text",    size=EYEBROW_SIZE),
            "headline": load_font(role="display", size=HEADLINE_SIZE),
            "subhead":  load_font(role="text",    size=SUBHEAD_SIZE),
            "cta":      load_font(role="text",    size=CTA_SIZE),
        }
        _get_tile_fonts._cache = cache  # type: ignore[attr-defined]
    return cache


# ============================================================================
# draw_tile -- compose a full Apple marketing tile
# ============================================================================

def draw_tile(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    theme: Theme,
    eyebrow: str,
    headline: str,
    subhead: str,
    cta_pair: tuple[str, str],
) -> None:
    """Render one Apple-style tile into `frame` at (x, y), size w by h.

    Layout, top to bottom inside the tile:

        +-------------------------------------------------+
        |                                                 |  <- PAD_TILE_TOP (80px)
        |   Eyebrow                                       |
        |   <12px>                                        |
        |   Headline first line                           |
        |   Headline second line (if wrapped)             |
        |   <16px>                                        |
        |   Subhead                                       |
        |   <24px>                                        |
        |           Learn more >    Open >                |  <- CTA row centered
        |                                                 |     in content column
        +-------------------------------------------------+

    Eyebrow / headline / subhead are LEFT-aligned to the interior content
    column (tile_x + PAD_TILE_X).  That column reads top-down-left like
    a print ad headline group -- the eye scans the eyebrow first, then
    the headline, then the subhead, all on a single vertical axis.

    The CTA row is centered as a group within the interior content
    column.  Centering it (rather than left-aligning) matches the apple.com
    pattern where the two-link cluster reads as a single horizontal unit
    pinned below the typographic stack rather than continuing the left
    column.

    Headline wraps to at most 2 lines on word boundaries; never shrinks.
    The font is constant at 48px regardless of how long the headline is,
    matching apple_SKILL.md's "wrap, never shrink" rule.

    The caller passes bare CTA labels ("Learn more", "Open"); the chevron
    is appended internally so no caller can accidentally ship a `>` in
    place of U+203A.

    Args:
        frame:    BGR uint8 frame to draw into.  Mutated in place.
        x, y:     tile top-left in frame pixels.
        w, h:     tile width / height.
        theme:    "light" or "dark" -- selects background + text + accent.
        eyebrow:  H3 string, e.g. "Photos".
        headline: H2 string, e.g. "Every memory, instantly."  Wraps to
                  two lines if it exceeds the interior content width.
        subhead:  Muted body string, e.g. "Your library, at a glance."
        cta_pair: (left_label, right_label) without chevrons, e.g.
                  ("Learn more", "Open").
    """
    palette = _theme_colors(theme)
    fonts   = _get_tile_fonts()

    # 1. Background.  Apple's tiles have NO border, NO drop shadow, NO
    #    outline -- visual separation comes from the 16px GAP_TILE
    #    between tiles plus the rounded corners alone.  Resist the urge
    #    to add a 1px stroke "to make it pop"; the flatness IS the look.
    rounded_rect(frame, x, y, w, h,
                 radius=RADIUS_TILE_LARGE, color_bgr=palette["fill_bgr"])

    # 2. Compute the interior content column.  Text never extends into
    #    PAD_TILE_X on either side.
    content_x = x + PAD_TILE_X
    content_w = w - 2 * PAD_TILE_X
    cursor_y  = y + PAD_TILE_TOP

    # 3. Wrap the headline NOW (before drawing the eyebrow) so we know
    #    its final line count -- the caller could in principle pass a
    #    headline that wraps to 2 lines, and we need to size the gap to
    #    the subhead off the LAST line's bbox bottom, not the first's.
    headline_lines = _wrap_two_lines(headline, fonts["headline"], content_w)

    # 4. Draw each block in order, threading `cursor_y` forward.
    cursor_y = _draw_eyebrow(
        frame, content_x, cursor_y, eyebrow,
        fonts["eyebrow"], palette["ink_rgb"],
    )
    cursor_y = _draw_headline(
        frame, content_x, cursor_y, headline_lines,
        fonts["headline"], palette["ink_rgb"],
    )
    cursor_y = _draw_subhead(
        frame, content_x, cursor_y, subhead,
        fonts["subhead"], palette["muted_rgb"],
    )
    _draw_cta_row(
        frame, content_x, content_w, cursor_y,
        cta_pair[0], cta_pair[1],
        fonts["cta"], palette["accent_rgb"],
    )
