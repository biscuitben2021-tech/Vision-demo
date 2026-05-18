"""
Design tokens for the Vision OS demo.

All colors in this module exist in BOTH RGB (for PIL) and BGR (for OpenCV).
Use the suffix that matches the library you are calling.  Mixing them
silently produces washed-out blue tiles instead of warm whites -- a class
of bug that is visually subtle and exhausting to track down later.

    PIL takes RGB.   OpenCV takes BGR.   Same hex, different 3-tuple.

We pre-compute both forms so no module ever has to remember which way the
byte order goes.  Pick the constant by suffix, never by guessing.

Module color-space convention:
    `_RGB` -> pass to PIL (`PIL.ImageDraw.Draw.text`, etc.)
    `_BGR` -> pass to OpenCV (`cv2.rectangle`, `cv2.circle`, etc.)

Reading order:  palette  ->  spacing  ->  motion  ->  typography  ->  helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# Palette
# ============================================================================
#
# Each color is stored as (R, G, B) and (B, G, R) -- a tuple of three uint8
# values, in 0..255.  The hex code in each comment is the canonical Apple
# source value; the rationale below it is the WHY.  These values come
# straight from apple_SKILL.md and were chosen by Apple, not by us.
# Resist the urge to "tweak" them.


# ----------------------------------------------------------------------------
# Page surfaces
# ----------------------------------------------------------------------------

# #fbfbfd -- Apple's restraint: never pure white.  Pure #ffffff reads as a
# fluorescent bulb and crushes the highlights on rounded glass tiles --
# every surface ends up looking overexposed and the page feels cheap.
# This warm near-white sits as paper.  If a render starts to look wrong
# and you cannot tell why, check this first: someone almost certainly
# replaced BG_LIGHT with (255, 255, 255).
BG_LIGHT_RGB:   Final[tuple[int, int, int]] = (251, 251, 253)
BG_LIGHT_BGR:   Final[tuple[int, int, int]] = (253, 251, 251)

# #000000 -- The one place we use pure black, and only for tile rows that
# alternate against BG_LIGHT.  Apple's marketing site is composed in
# roughly half-and-half light/dark tile rows; the contrast IS the layout.
BG_DARK_RGB:    Final[tuple[int, int, int]] = (0, 0, 0)
BG_DARK_BGR:    Final[tuple[int, int, int]] = (0, 0, 0)

# #f5f5f7 -- Secondary surface for footers, secondary tiles, search inputs.
# One small step darker than BG_LIGHT.  Notice this is the same value as
# TEXT_ON_DARK below -- that is intentional: Apple reuses the same warm
# neutral on both ends of the contrast scale.
BG_NEUTRAL_RGB: Final[tuple[int, int, int]] = (245, 245, 247)
BG_NEUTRAL_BGR: Final[tuple[int, int, int]] = (247, 245, 245)


# ----------------------------------------------------------------------------
# Text colors
# ----------------------------------------------------------------------------

# #1d1d1f -- "Pure black on near-black would be invisible; pure black on
# near-white is too aggressive."  This near-black has just enough warmth
# to avoid the printing-on-receipt-paper effect.  Use for every primary
# headline that sits on a light background.
TEXT_ON_LIGHT_RGB:   Final[tuple[int, int, int]] = (29, 29, 31)
TEXT_ON_LIGHT_BGR:   Final[tuple[int, int, int]] = (31, 29, 29)

# #f5f5f7 -- Same numeric value as BG_NEUTRAL.  On a pure-black tile, pure
# #ffffff text is searing; this warm neutral is the perceptual inverse of
# TEXT_ON_LIGHT.  The repeated value is deliberate.
TEXT_ON_DARK_RGB:    Final[tuple[int, int, int]] = (245, 245, 247)
TEXT_ON_DARK_BGR:    Final[tuple[int, int, int]] = (247, 245, 245)

# #86868b -- Muted body / subhead color.  Lighter than TEXT_ON_LIGHT by
# enough that subheads recede behind headlines without disappearing.  This
# is the canonical "muted gray" you see under every H2 on apple.com.
TEXT_MUTED_RGB:      Final[tuple[int, int, int]] = (134, 134, 139)
TEXT_MUTED_BGR:      Final[tuple[int, int, int]] = (139, 134, 134)

# #6e6e73 -- Tertiary: disclaimers, footnotes, the FPS counter, anything
# that should be legible but ignorable.  Sits one step darker than muted
# -- it can stand alone in a quiet corner without competing for the eye.
TEXT_TERTIARY_RGB:   Final[tuple[int, int, int]] = (110, 110, 115)
TEXT_TERTIARY_BGR:   Final[tuple[int, int, int]] = (115, 110, 110)


# ----------------------------------------------------------------------------
# Accents (the "Learn more >" link blue)
# ----------------------------------------------------------------------------

# #0066cc -- The literal "Learn more >" link blue on light backgrounds.
# Reserved for CTAs and inline links; never for body text.
ACCENT_LIGHT_RGB: Final[tuple[int, int, int]] = (0, 102, 204)
ACCENT_LIGHT_BGR: Final[tuple[int, int, int]] = (204, 102, 0)

# #2997ff -- Same link, brighter so it has the contrast to read on a
# pure-black tile.  Do not use this color on a light background; it's
# tuned for #000.
ACCENT_DARK_RGB:  Final[tuple[int, int, int]] = (41, 151, 255)
ACCENT_DARK_BGR:  Final[tuple[int, int, int]] = (255, 151, 41)

# #0077ed -- Hover state for ACCENT_LIGHT.  Subtle shift -- if it's too
# bright the link "pops" out of the layout, which is the opposite of
# Apple.  A 17-point hue nudge is the entire animation budget.
ACCENT_HOVER_RGB: Final[tuple[int, int, int]] = (0, 119, 237)
ACCENT_HOVER_BGR: Final[tuple[int, int, int]] = (237, 119, 0)


# ----------------------------------------------------------------------------
# Hairlines / dividers
# ----------------------------------------------------------------------------

# #d2d2d7 -- Only ever used in the footer dividers and form input borders.
# If you reach for this on a tile, stop -- tile separation comes from the
# 16px GAP_TILE alone, not from a stroke.  No borders on tiles, ever.
HAIRLINE_RGB: Final[tuple[int, int, int]] = (210, 210, 215)
HAIRLINE_BGR: Final[tuple[int, int, int]] = (215, 210, 210)


# ============================================================================
# Spacing tokens (pixels, desktop scale)
# ============================================================================
#
# The whole layout breathes off these values.  Resist the urge to fudge
# them by +/-2px per tile.  The 16px tile gap and the 24px tile radius
# are the load-bearing values -- they're what makes the grid read as
# "Apple" without any chrome.  Borders, drop shadows, dividers: all
# absent.  Visual separation = gap + radius.

GAP_TILE:          Final[int] = 16    # gap between adjacent tiles
GAP_VIEWPORT:      Final[int] = 16    # gutter from tile group to viewport edge
PAD_TILE_TOP:      Final[int] = 80    # top padding inside a tile
PAD_TILE_X:        Final[int] = 40    # horizontal padding inside a tile
RADIUS_TILE_LARGE: Final[int] = 24    # corner radius for big tiles
RADIUS_TILE_SMALL: Final[int] = 18    # corner radius for small/mobile tiles
RADIUS_APP_ICON:   Final[int] = 28    # corner radius for Vision OS app icons
MAX_CONTENT_WIDTH: Final[int] = 980   # inner content column max width
CTA_GAP:           Final[int] = 24    # gap between "Learn more >  Buy >"
NAV_HEIGHT:        Final[int] = 44    # the blurred top nav bar


# ============================================================================
# Motion tokens (milliseconds + cubic bezier control points)
# ============================================================================
#
# Wired up in src/motion.py during Phase 5.  Listed here as constants so
# every phase imports motion timing from the same authoritative place --
# no module gets to invent its own 250ms.

FADE_UP_DURATION_MS:     Final[int] = 800
# Apple's "ease-emphasized" curve: slow start, fast middle, gentle stop.
# Pulled straight from apple_SKILL.md.  Implemented in motion.ease().
FADE_UP_EASING:          Final[tuple[float, float, float, float]] = (
    0.28, 0.11, 0.32, 1.0,
)
HOVER_SCALE_DURATION_MS: Final[int] = 200
CTA_COLOR_DURATION_MS:   Final[int] = 200
NAV_BLUR_DURATION_MS:    Final[int] = 250


# ============================================================================
# Typography -- font discovery + sized font loading
# ============================================================================
#
# On modern macOS (Big Sur and later) SF Pro is shipped as a single
# variable font at /System/Library/Fonts/SFNS.ttf -- one file with
# weight, width, and optical-size axes baked in.  Older Apple docs
# reference per-weight `.otf` files like `SF-Pro-Display-Semibold.otf`;
# those do not exist on a fresh install of macOS Sequoia or later.
#
# PIL >= 9.5 supports variable fonts via `font.set_variation_by_name(...)`,
# which is how we pick "Semibold" or "Regular" at load time.  The
# optical-size axis is keyed to the font size we ask for: at >= 20px the
# variable font auto-swaps to Display glyph shapes; below that, Text
# shapes.  So we never specify Display vs Text directly -- the role-based
# load_font helper just chooses a weight and lets the size do the rest.
#
# Helvetica Neue is the universal fallback.  It ships in HelveticaNeue.ttc
# (a TrueType Collection of weights at known indices) on every Mac going
# back a decade.  When SFNS is not readable, we hop to that.
#
# We deliberately do NOT use PIL's bundled default bitmap font as a final
# fallback.  It's pixel-fitted and ruins the entire aesthetic; a missing
# system font should fail loudly, not silently render an ugly demo.

_SFNS_PATH:           Final[Path] = Path("/System/Library/Fonts/SFNS.ttf")
_HELVETICA_NEUE_PATH: Final[Path] = Path("/System/Library/Fonts/HelveticaNeue.ttc")

# Helvetica Neue collection face indices.  These have been stable across
# every macOS release since Big Sur; do not re-derive them at runtime.
_HELVETICA_NEUE_REGULAR_INDEX: Final[int] = 0
_HELVETICA_NEUE_BOLD_INDEX:    Final[int] = 2

# Named instances we ask SFNS to switch to.  Variable fonts expose a list
# of named weight presets ("Regular", "Medium", "Semibold", "Bold", ...);
# Phase 1 only ever asks for these two.  Later phases may need more.
_SFNS_SEMIBOLD: Final[str] = "Semibold"
_SFNS_REGULAR:  Final[str] = "Regular"


# Text supersampling factor.  draw_text renders glyphs at this multiple
# of the requested size into an oversized RGBA patch, then INTER_AREA-
# downsamples back to the native bbox.  This is MSAA for text: the
# downsample integrates more glyph detail per output pixel, producing
# noticeably cleaner anti-aliased edges than PIL's direct render at the
# native size.  2 is the sweet spot -- 3 or 4 give diminishing returns
# while costing 9x / 16x more PIL render time.
_TEXT_SUPERSAMPLE: Final[int] = 2

# Cache of (font_id, supersample_factor) -> supersampled FreeTypeFont.
# Loading a TrueType at a new size involves re-parsing the font file's
# table directory, which is measurable at 60Hz across multiple draw_text
# calls per frame.  We key by (id(font), supersample) because two fonts
# loaded from the same path with the same size are functionally
# identical for our pipeline; the id() lookup is O(1) and avoids us
# having to track the variation name through every call.
_SUPERSAMPLED_FONT_CACHE: dict[tuple[int, int], ImageFont.FreeTypeFont] = {}


def _get_supersampled_font(
    font: ImageFont.FreeTypeFont, supersample: int,
) -> ImageFont.FreeTypeFont:
    """Return a supersampled copy of `font` (sized `supersample`x).

    The returned font is structurally identical to the input except for
    its render size.  Variation (Semibold/Regular for SFNS) is preserved
    by reading the active variation name from the original font and
    re-applying it on the copy.  If PIL doesn't expose
    `get_variation_by_axes()` (older versions), the copy renders at the
    default weight -- ugly but functional.
    """
    key = (id(font), supersample)
    cached = _SUPERSAMPLED_FONT_CACHE.get(key)
    if cached is not None:
        return cached

    # BUG FIX: also preserve the TrueType collection face `index`.  When
    # the SFNS variable font is missing and we fall back to Helvetica
    # Neue, `load_font` loads from HelveticaNeue.ttc with
    # `index=_HELVETICA_NEUE_BOLD_INDEX` (2) for "display" role.  Without
    # forwarding `font.index` here, the supersampled copy defaults to
    # index 0 (Regular), so display text would render at Regular weight
    # in the oversized patch and downsample to a too-thin glyph.  PIL's
    # FreeTypeFont exposes `.index` since 9.x; older versions return 0
    # via getattr default which matches the truetype default.
    big = ImageFont.truetype(
        font.path,
        size=font.size * supersample,
        index=getattr(font, "index", 0),
    )
    # Carry over the variation name if the source font has one.  PIL
    # >= 9.5 exposes `get_variation_by_axes` which we can probe; older
    # versions just fall through to the default weight.
    variation_name = getattr(font, "_vd_variation_name", None)
    if variation_name is not None:
        try:
            big.set_variation_by_name(variation_name)
        except (OSError, AttributeError):
            pass
    _SUPERSAMPLED_FONT_CACHE[key] = big
    return big


def load_font(role: str, size: int) -> ImageFont.FreeTypeFont:
    """Return a PIL TrueType font for the given role and size.

    role:
        "display" -> SF Pro Display Semibold.  Used for H1 / H2 / app titles.
        "text"    -> SF Pro Text Regular.       Used for body, subheads,
                                                FPS counter, footnotes.

    size:  pixel height to render at.

    SF Pro's variable font automatically swaps the optical-size variant
    (Display vs Text glyph shapes) based on size, so we don't pass that
    axis ourselves -- larger sizes get Display shapes, smaller sizes get
    Text shapes.  We only pin the weight.

    Raises:
        ValueError        -- unknown role.
        FileNotFoundError -- neither SFNS nor Helvetica Neue is installed.
    """
    if role not in ("display", "text"):
        raise ValueError(
            f"Unknown font role {role!r}; expected 'display' or 'text'."
        )

    # Preferred: SF Pro variable font.
    if _SFNS_PATH.is_file():
        font = ImageFont.truetype(str(_SFNS_PATH), size=size)
        variation = _SFNS_SEMIBOLD if role == "display" else _SFNS_REGULAR
        # set_variation_by_name is a no-op on PIL < 9.5 and raises OSError
        # if the variation name doesn't exist in this exact .ttf release.
        # Either way the font still works -- it just renders at the
        # default weight (Regular), which is a graceful degradation.
        try:
            font.set_variation_by_name(variation)
        except (OSError, AttributeError):
            pass
        # Stash the variation name on the font object so the text
        # supersampler can reapply it when it loads a 2x-sized copy of
        # the same font.  Custom attributes on PIL fonts are persisted
        # for the lifetime of the object; nothing else reads this
        # field except `_get_supersampled_font`.
        font._vd_variation_name = variation  # type: ignore[attr-defined]
        return font

    # Fallback: Helvetica Neue collection.
    if _HELVETICA_NEUE_PATH.is_file():
        index = (
            _HELVETICA_NEUE_BOLD_INDEX
            if role == "display"
            else _HELVETICA_NEUE_REGULAR_INDEX
        )
        return ImageFont.truetype(
            str(_HELVETICA_NEUE_PATH), size=size, index=index
        )

    raise FileNotFoundError(
        f"Neither {_SFNS_PATH} nor {_HELVETICA_NEUE_PATH} could be found. "
        "This demo expects SF Pro or Helvetica Neue installed in their "
        "default macOS locations."
    )


# ============================================================================
# Text drawing helper -- the PIL/cv2 boundary lives here
# ============================================================================
#
# OpenCV ships a built-in font (`cv2.putText`) but it is a 1980s blitter
# face: no antialiasing worth the name, no kerning, no subpixel placement.
# Apple-grade typography at the sizes we need (12px footnotes, 80px
# headlines) is simply not possible with it.
#
# So every piece of text in this demo is rendered via PIL onto a small
# RGBA bitmap, then alpha-composited into the cv2 frame.  The conversion
# is local to a tight bounding box around the glyphs -- pasting a 80x16
# pixel patch is essentially free, where converting the whole 2560x1664
# frame to PIL and back would dominate the per-frame budget.
#
# This function is the ONLY place where the PIL/cv2 byte-order dance is
# allowed to live.  Everywhere else in the codebase, the suffix-typed
# constants from this module are passed straight to whichever library
# needs them.

def _align_x_offset(
    text: str,
    font: ImageFont.FreeTypeFont,
    align: str,
) -> int:
    """Return the integer pixel offset to subtract from the caller's `x`.

    Why a separate helper:  alignment is conceptually a horizontal anchor
    decision -- "is the given x the left, center, or right of this run?"
    -- and it has nothing to do with the rasterise/composite pipeline that
    `draw_text` does after it.  Splitting it out keeps `draw_text` under
    30 lines and lets callers reason about anchor math without scrolling
    through alpha blending code.

    We use `font.getlength(text)` rather than `font.getbbox(text)` because
    getlength returns the typographic advance width -- exactly what you
    want for horizontal anchoring (it is what a text shaper would place
    the next character at).  getbbox over-includes side bearings, which
    causes center- and right-aligned headings to drift a couple of pixels
    relative to the visible glyph extents.  This is exactly the same
    reason `render_fps` in phase1_canvas.py uses getlength.

    Anchor semantics:
        "left"   ->  x is the left edge of the glyph run; offset = 0.
        "center" ->  x is the horizontal midpoint;      offset = advance / 2.
        "right"  ->  x is the right edge of the glyph run; offset = advance.
    """
    if align == "left":
        # Fast path; preserves the pre-existing API exactly so every
        # caller written against the old signature renders identically.
        return 0
    advance = int(round(font.getlength(text)))
    if align == "center":
        return advance // 2
    if align == "right":
        return advance
    raise ValueError(
        f"Unknown align {align!r}; expected 'left', 'center', or 'right'."
    )


def draw_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color_rgb: tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
    align: str = "left",
) -> None:
    """Composite `text` onto a BGR cv2 `frame`, mutating it in place.

    The text is rendered into a transparent PIL image sized to the glyph
    bounding box, then alpha-blended into the frame.  Using a tightly
    cropped patch (rather than wrapping the whole frame in PIL) keeps
    the per-frame cost flat regardless of frame resolution.

    Arguments:
        frame:     BGR uint8 image, shape (H, W, 3).  Mutated in place.
        text:      string to render.  Empty / whitespace-only strings are
                   a no-op.
        x, y:      anchor of the text's bounding box, in frame pixels.
                   The role of `x` depends on `align` (see below); `y` is
                   always the top of the bounding box.  Off-frame
                   coordinates are clipped.
        color_rgb: (R, G, B) tuple in 0..255.  Note RGB, not BGR -- PIL
                   draws it.  The conversion to BGR happens inside, in the
                   `rgba[..., 2::-1]` slice that reverses channel order
                   without copying the array.
        font:      a PIL ImageFont.FreeTypeFont as returned by load_font.
        align:     "left" (default), "center", or "right".  Selects which
                   horizontal point of the rendered text sits at `x`.
                   Centering math is done here, not at the call site --
                   that way every screen ("center the H1", "center the
                   subhead", "right-align the FPS") uses the same anchor
                   semantics instead of each caller re-deriving them.
    """
    if not text:
        return

    # Apply the alignment anchor BEFORE the existing pipeline runs.  This
    # is the entire point of the split: from here on, the function does
    # exactly what it did before align was added, so align="left" is a
    # bit-for-bit no-op relative to the previous behaviour.
    x = x - _align_x_offset(text, font, align)

    # Measure the glyphs.  getbbox returns (left, top, right, bottom) in
    # font space.  For some fonts `left` is negative (left side-bearing of
    # the first glyph extends past x=0); we offset the actual draw call by
    # -left so the glyph pulls back into the canvas.
    left, top, right, bottom = font.getbbox(text)
    width  = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return

    # Supersample the text: render PIL at 2x the target size into an
    # oversized RGBA patch, then cv2.INTER_AREA downsample back to the
    # native bbox before compositing.  AREA-downsampling produces much
    # cleaner anti-aliased glyph edges than PIL's direct render at the
    # native size; the cost is a 4x bigger PIL render (small in absolute
    # terms because text bboxes are tiny), traded for crisper text in
    # the output buffer.  This is the rendering-equivalent of MSAA: we
    # cannot fix the macOS HiDPI upscale that happens after imshow, but
    # we CAN feed it a higher-quality source.
    ss = _TEXT_SUPERSAMPLE
    big_font = _get_supersampled_font(font, ss)
    big_left, big_top, big_right, big_bottom = big_font.getbbox(text)
    big_w = big_right - big_left
    big_h = big_bottom - big_top
    if big_w <= 0 or big_h <= 0:
        return

    # Render onto a transparent RGBA patch at supersampled size.  Alpha 0
    # background means only the glyph itself contributes to the composite.
    patch = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text(
        (-big_left, -big_top), text, fill=color_rgb + (255,), font=big_font,
    )

    # PIL -> numpy at supersampled resolution, then INTER_AREA downsample
    # to the target bbox.  INTER_AREA is the canonical "downsample with
    # area averaging" filter -- mathematically equivalent to integrating
    # the source pixels under each destination pixel, which is what you
    # want for clean anti-aliased downscaling.
    rgba_big = np.array(patch)
    rgba = cv2.resize(rgba_big, (width, height), interpolation=cv2.INTER_AREA)
    bgr  = rgba[..., 2::-1]                              # R,G,B -> B,G,R
    alpha = rgba[..., 3:4].astype(np.float32) * (1.0 / 255.0)

    # Clip the destination rectangle to the frame.  Off-screen text is a
    # silent no-op rather than an exception -- callers regularly compute
    # positions that may overflow during transitions.
    frame_h, frame_w = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + width)
    y1 = min(frame_h, y + height)
    if x1 <= x0 or y1 <= y0:
        return

    # Map the clipped destination rect back into source coordinates.
    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    dst = frame[y0:y1, x0:x1].astype(np.float32)
    src = bgr[sy0:sy1, sx0:sx1].astype(np.float32)
    a   = alpha[sy0:sy1, sx0:sx1]

    # Standard "over" compositing: out = src*a + dst*(1-a).  In place.
    frame[y0:y1, x0:x1] = (a * src + (1.0 - a) * dst).astype(np.uint8)


# ============================================================================
# Canvas factory + FPS HUD
# ============================================================================
#
# Two utilities every phase needs and which were previously duplicated
# (phase1's `make_canvas` painted BG_LIGHT; phase4's `make_dark_canvas`
# painted BG_DARK; each phase rolled its own FPS-rendering helper at a
# different y-anchor and a different colour).  Centralising them here
# means:
#   * Every phase pulls its background-fill from one place, so a future
#     wallpaper swap touches one constant rather than seven phase scripts.
#   * The FPS counter sits at exactly the same screen coordinate in every
#     phase, in the same dim tertiary colour, so a developer's eye learns
#     one anchor regardless of which phase is being run for development.
#
# Both helpers live in this module because they are pure design-token
# applications: a colour and a font, rendered into the BGR frame.  Putting
# them in src/icons or src/tiles would couple them to the glass/tile
# vocabulary they are deliberately separate from.


def make_canvas(
    width: int,
    height: int,
    color: tuple[int, int, int] = BG_LIGHT_BGR,
) -> np.ndarray:
    """Return a fresh BGR canvas painted with `color`.

    Default is BG_LIGHT_BGR -- #fbfbfd, the warm near-white the marketing
    phases (Phases 1-3) are built on.  The home-screen / app-window phases
    (Phases 4+) pass BG_DARK_BGR to start the canvas pure black, which is
    the visionOS wallpaper.  Pure white (255, 255, 255) is NEVER the right
    answer here -- Apple's restraint is to never use pure white as a page
    surface; the warm near-white is what makes a rendered tile read as
    paper rather than as a fluorescent bulb.

    Why this lives in src/design rather than per-phase: a previous version
    of the demo had each phase paint its own background fill at the top of
    main(), which led to a class of bug where the home-screen path would
    end up with a near-white strip at the top of the canvas before the
    BG_DARK fill landed.  Routing every phase through a single factory
    closes that gap by construction -- you cannot forget to fill the
    background if creating the canvas IS the fill.

    Args:
        width, height: pixel extents.  Allocated as shape (height, width, 3)
                       to match cv2's row-major buffer layout.
        color:         BGR tuple in 0..255.  Defaults to BG_LIGHT_BGR so
                       existing callers (phase1/2/3) that don't pass a
                       colour keep their marketing-page background.

    Returns:
        A fresh uint8 BGR ndarray of shape (height, width, 3) painted
        uniformly with `color`.
    """
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    # Slice assignment to the BGR triple broadcasts across every pixel in
    # one numpy pass -- significantly faster than np.full for the same
    # result, and clearer than np.empty + cv2.rectangle.
    canvas[:, :] = color
    return canvas


# FPS HUD anchor.  The prompt fixes this at (frame_w - 20, 20) so the
# counter sits 20px from the top-right corner -- close enough to be
# unobtrusive but far enough from the edge to read at glance distance.
# Used by `draw_fps_hud` below; pulled out as constants so future tweaks
# touch one site rather than the call body.
_FPS_HUD_MARGIN: Final[int] = 20

# Footnote size -- matches the "Small / footnote" row of the typography
# table in apple_SKILL.md.  The FPS counter is a diagnostic surface, not
# part of the OS chrome, so it pulls the smallest type token.
_FPS_HUD_FONT_SIZE: Final[int] = 12


def _get_fps_hud_font() -> ImageFont.FreeTypeFont:
    """Return the cached SF Pro Text Regular 12px font for the FPS counter.

    PIL truetype loads are not free -- opening the font file and parsing
    its table directory is measurable at 60Hz on the M2.  Same caching
    idiom `src/tiles._get_tile_fonts` uses: state lives on the function
    object, not at module scope, so importing this module is side-effect
    free.

    The font's role and size are fixed (the FPS counter is the only call
    site) so we don't expose them as parameters -- the function returns
    the one font this HUD needs, full stop.
    """
    cached = getattr(_get_fps_hud_font, "_cache", None)
    if cached is None:
        cached = load_font(role="text", size=_FPS_HUD_FONT_SIZE)
        _get_fps_hud_font._cache = cached  # type: ignore[attr-defined]
    return cached


def draw_fps_hud(frame: np.ndarray, fps: float) -> None:
    """Render the FPS counter into the absolute top-right of `frame`.

    Anchored at (frame_w - 20, 20) with align="right" and color
    TEXT_TERTIARY_RGB so it can sit unobtrusively on top of any
    background -- light page, dark home screen, mid-transition blend.

    Must be the LAST thing drawn each frame; otherwise the status bar
    or a sliding notification will paint over it.

    The colour deliberately is TEXT_TERTIARY_RGB (#6e6e73) -- a dim grey
    that reads against both light and dark surfaces without competing
    with foreground content.  Earlier phases painted the FPS counter in
    TEXT_ON_DARK_RGB (#f5f5f7, near-white), which read fine on the dark
    home screen but became a bright blob against light-app wallpapers
    (Photos, Notes, etc.).  Tertiary is the colour the design system
    reserves for "legible but ignorable" diagnostics.

    Args:
        frame: BGR uint8 ndarray, mutated in place.  Must be the
               renderer's final output -- this helper is the last paint
               in the pipeline.
        fps:   the current frames-per-second value to display.  Formatted
               internally as `f"{fps:5.1f} fps"` so every phase shows the
               exact same string format.

    The colour-space convention: this helper accepts a BGR `frame`
    (cv2's native format) and forwards an RGB colour to `draw_text`,
    which handles the cv2/PIL byte-order translation internally.  No
    raw BGR/RGB tuples are introduced here.
    """
    h, w = frame.shape[:2]
    font = _get_fps_hud_font()
    # f"{fps:5.1f} fps" keeps the digit count stable (e.g. "22.9 fps",
    # " 9.7 fps") so the right-aligned anchor doesn't visually jitter
    # by one glyph width as the EMA crosses 10/100 fps boundaries.
    text = f"{fps:5.1f} fps"
    draw_text(
        frame, text,
        x=w - _FPS_HUD_MARGIN,
        y=_FPS_HUD_MARGIN,
        color_rgb=TEXT_TERTIARY_RGB,
        font=font,
        align="right",
    )


# ============================================================================
# Gaze cursor -- a soft glowing dot that tracks where the user is looking
# ============================================================================
#
# When the gaze pipeline is driving the cursor (rather than the mouse),
# there is no built-in indicator showing WHERE that gaze cursor sits --
# the system mouse pointer is no longer the source of truth.  Without a
# visible marker the audience cannot tell whether the OS thinks the
# user is looking at the Music tile or the Notes tile, and a hover
# highlight alone is too quiet to read at a distance.
#
# The gaze ball is three concentric circles painted via cv2.circle with
# LINE_AA, all alpha-blended onto the canvas through a small RGBA patch:
#
#     1. Outer halo -- 28px radius, near-white, ~12% opacity.  Soft
#        ambient glow around the eye position.
#     2. Mid ring   -- 14px radius, near-white, ~45% opacity, 2px stroke
#        (drawn unfilled).  The visible "ring" of the gaze marker.
#     3. Inner dot  -- 4px radius, near-white, 90% opacity, filled.
#        The crisp centre point that tracks pixel-exact gaze.
#
# Total visual extent ~56px square -- comfortably within a single
# RADIUS_APP_ICON tile but big enough to read from across the room.

# Gaze ball geometry.  All radii are in pixels.  The patch is sized so
# the outer halo plus a 4px breathing margin fits inside.
_GAZE_PATCH_RADIUS:    Final[int] = 32     # half-width of the RGBA patch
_GAZE_HALO_RADIUS:     Final[int] = 28
_GAZE_RING_RADIUS:     Final[int] = 14
_GAZE_RING_STROKE:     Final[int] = 2
_GAZE_DOT_RADIUS:      Final[int] = 4

# Alpha values in 0..255 (PIL's native alpha range).  Tuned so the
# whole marker reads as a soft glow rather than a hard CAD cursor.
_GAZE_HALO_ALPHA:      Final[int] = 30      # 12%
_GAZE_RING_ALPHA:      Final[int] = 115     # 45%
_GAZE_DOT_ALPHA:       Final[int] = 230     # 90%

# Cursor colour.  Near-white, very faintly cool (255, 250, 245 BGR =
# 245, 250, 255 RGB).  Pure (255, 255, 255) reads as a fluorescent
# spot; the slight blue lift in RGB matches the rim-highlight colour
# the Liquid Glass panels use, so the marker visually belongs to the
# same chrome family.
_GAZE_COLOR_BGR:       Final[tuple[int, int, int]] = (255, 250, 245)


def _build_gaze_patch() -> np.ndarray:
    """Build the cached RGBA gaze-marker patch, one row of (h, w, 4) uint8.

    Pure geometry -- depends only on the constants above -- so it's
    safe to memoise and reuse for every call to `draw_gaze_cursor`.
    The patch is drawn once per process; per-frame cost of the gaze
    cursor is then just the alpha composite of a 64x64 patch.

    BGRA byte order (cv2 native), alpha = 0 in the corners so the
    halo's circular shape doesn't paint over the wallpaper outside
    the marker.
    """
    diameter = _GAZE_PATCH_RADIUS * 2
    patch = np.zeros((diameter, diameter, 4), dtype=np.uint8)
    centre = (_GAZE_PATCH_RADIUS, _GAZE_PATCH_RADIUS)

    # 1. Outer halo: filled circle, low alpha.  Painted first so the
    #    later passes layer cleanly on top.
    cv2.circle(
        patch, centre, _GAZE_HALO_RADIUS,
        (*_GAZE_COLOR_BGR, _GAZE_HALO_ALPHA),
        thickness=-1, lineType=cv2.LINE_AA,
    )
    # 2. Mid ring: stroked circle (thickness=_GAZE_RING_STROKE), mid alpha.
    cv2.circle(
        patch, centre, _GAZE_RING_RADIUS,
        (*_GAZE_COLOR_BGR, _GAZE_RING_ALPHA),
        thickness=_GAZE_RING_STROKE, lineType=cv2.LINE_AA,
    )
    # 3. Inner dot: small filled circle, high alpha.
    cv2.circle(
        patch, centre, _GAZE_DOT_RADIUS,
        (*_GAZE_COLOR_BGR, _GAZE_DOT_ALPHA),
        thickness=-1, lineType=cv2.LINE_AA,
    )
    return patch


# Module-level cache for the BGRA gaze patch (built once on first call).
_GAZE_PATCH_CACHE: dict[str, np.ndarray] = {}


def draw_gaze_cursor(frame: np.ndarray, x: int, y: int) -> None:
    """Composite the gaze-marker glyph onto `frame` centred at (x, y).

    Mutates `frame` in place.  Off-canvas placements are silently
    clipped -- if the gaze drifts to the very edge of the screen the
    visible portion of the marker still composites correctly.

    Args:
        frame: BGR uint8 image, mutated in place.  Should be the
               renderer's final output -- the gaze cursor belongs at
               the top of the paint order so it never gets occluded
               by the status bar or a sliding notification.
        x, y:  centre of the marker in canvas pixels.

    The patch is cached on first call so steady-state cost is just
    the alpha composite of a 64x64 region -- under a millisecond on
    the M2.  Colour-space convention: BGR in, BGRA patch -> BGR
    composite using the patch's alpha channel.
    """
    patch = _GAZE_PATCH_CACHE.get("default")
    if patch is None:
        patch = _build_gaze_patch()
        _GAZE_PATCH_CACHE["default"] = patch

    ph, pw = patch.shape[:2]
    paste_x = x - pw // 2
    paste_y = y - ph // 2

    frame_h, frame_w = frame.shape[:2]
    x0 = max(0, paste_x)
    y0 = max(0, paste_y)
    x1 = min(frame_w, paste_x + pw)
    y1 = min(frame_h, paste_y + ph)
    if x1 <= x0 or y1 <= y0:
        return

    sx0 = x0 - paste_x
    sy0 = y0 - paste_y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    src_bgr   = patch[sy0:sy1, sx0:sx1, :3].astype(np.float32)
    src_alpha = patch[sy0:sy1, sx0:sx1,  3].astype(np.float32) * (1.0 / 255.0)
    src_alpha = src_alpha[..., np.newaxis]
    dst = frame[y0:y1, x0:x1].astype(np.float32)
    out = src_bgr * src_alpha + dst * (1.0 - src_alpha)
    np.clip(out, 0.0, 255.0, out=out)
    frame[y0:y1, x0:x1] = out.astype(np.uint8)


# ============================================================================
# HiDPI display wrapper
# ============================================================================
#
# cv2.imshow on macOS Retina passes our BGR buffer to an NSImage and
# AppKit upscales it to the window's physical backing layer.  The
# upscale uses AppKit's default interpolation, which on text-heavy
# content reads as soft / fuzzy.  Pre-scaling the buffer to 2x with
# cv2.INTER_LANCZOS4 gives AppKit a higher-resolution source to start
# from; Lanczos has a steeper frequency response than bicubic so glyph
# edges land sharper.
#
# This is NOT the same as rendering the whole pipeline at 2x (which
# would actually add information).  It's a "best effort with the
# pixels we already have" pass that costs ~15-25ms per frame to upscale
# 1440x900 -> 2880x1800, in exchange for crisper-than-default text.

_HIDPI_SCALE: Final[int] = 2


def hidpi_imshow(window_name: str, frame: np.ndarray) -> None:
    """imshow with a Lanczos pre-upscale for crisper Retina display.

    The cv2 window backing on a Retina Mac renders at twice the logical
    pixel grid.  When we pass a 1x buffer to imshow, AppKit upsamples
    it with its default bicubic, which makes 12-14px text read as soft.
    Pre-upscaling with INTER_LANCZOS4 (sharper than bicubic) and then
    letting AppKit display the already-scaled buffer 1:1 lands with
    visibly crisper glyph edges.

    Args:
        window_name: cv2 window identifier (same one used by namedWindow).
        frame:       BGR uint8 buffer at logical resolution.  Not mutated.

    Cost: ~15-25ms per call on a 1440x900 source.  Counted against the
    per-frame budget.  Phases that don't need it (Phase 1's canvas-only
    demo) can keep calling cv2.imshow directly.
    """
    h, w = frame.shape[:2]
    scaled = cv2.resize(
        frame,
        (w * _HIDPI_SCALE, h * _HIDPI_SCALE),
        interpolation=cv2.INTER_LANCZOS4,
    )
    cv2.imshow(window_name, scaled)
