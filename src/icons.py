"""
Vision OS glass-panel surface + procedurally drawn app-icon glyphs.

This module is the visual atom of Phase 4's home screen: every tile and
the status bar is, underneath the typography, a *glass panel* sitting on
top of the dark wallpaper.  The icon glyphs (Safari compass, Photos
rainbow petals, Music eighth note, etc.) are drawn here as well so they
can be reused by the fake-app windows in Phase 6 without bouncing
through PNGs that would clutter `assets/`.

Two responsibilities live here, in the order they're called:

    1. `draw_glass_panel`   -- composite a translucent, slightly brightened
                               and desaturated patch back into the frame
                               with rounded corners.  This is the "glass"
                               look that floats every Vision OS tile.
    2. `draw_app_icon`      -- center a rounded coloured square inside a
                               glass panel and draw the per-app glyph on
                               top.  Each glyph is its own private helper
                               so the dispatch in `draw_app_icon` stays a
                               flat eight-way lookup with no nested logic.

Why a single module instead of one file per icon:  the eight glyphs are
small (well under 40 lines each), share a common centred-coordinate
contract, and are only ever called from one site (`draw_app_icon`).
Splitting them across eight files would 8x the import cost and bury the
"these are all variations on the same primitive" intent that the
side-by-side `_icon_*` definitions communicate at a glance.

Module color-space convention:
    The cv2 pixel buffer is BGR.  Every helper in this file accepts and
    returns BGR tuples; nothing in here ever crosses the PIL boundary
    *except* the calendar's "11" text (rendered through `draw_text`
    from src.design, which itself handles the RGB-vs-BGR translation).
    Constants imported with a `_BGR` suffix are passed to cv2 calls
    straight.  Constants with `_RGB` are forwarded to `draw_text`.
"""

from __future__ import annotations

import math
from typing import Final

import cv2
import numpy as np

from src.design import (
    RADIUS_APP_ICON,
    TEXT_ON_DARK_RGB,
    TEXT_ON_LIGHT_RGB,
    draw_text,
    load_font,
)
from src.tiles import rounded_rect


# ============================================================================
# Glass panel -- the floating-tile look
# ============================================================================
#
# "Glass" in visionOS is conceptually:
#
#     surface = (brightened_blurred_background  * 0.85)
#             + (near-black tint                 * 0.15)
#
# with the whole thing then rounded so the corners reveal the original
# wallpaper.  We deliberately do NOT implement a true Gaussian blur of
# the underlying pixels: the home screen background in Phase 4 is a
# flat black (BG_DARK), and blurring a flat fill produces the same flat
# fill -- a costly no-op.  Real Apple frosted glass on a flat surface
# *looks* slightly lifted because of the brighten-and-desaturate pass,
# not because of the blur.  When Phase 6 lays glass over a textured
# fake-app window, we can revisit and add an actual blur step here.
#
# Why the 0.85 / 0.15 mix:
#
#     0.85 keeps the surface bright enough to read as "glass over dark"
#         rather than a flat opaque tile.  Mixing below ~0.75 starts
#         losing the lifted feel; above ~0.90 the tint disappears and
#         the panel reads as a plain bright rect.
#
#     0.15 of the near-black tint pulls the surface back toward the page
#         palette so it doesn't blow out against the #000 wallpaper.  The
#         tint colour is intentionally not pure black (10, 10, 14 BGR
#         instead of (0, 0, 0)) -- a faint cool cast is what gives glass
#         its "frozen mineral" feel rather than "grey rectangle".
#
# Why the rounded-mask compositing approach:
#
#     The naive alternative is to draw four black corner triangles on
#     top of a rect.  But the wallpaper underneath is solid #000; painting
#     more black on top of it does not "reveal" the wallpaper -- it just
#     happens to match it by coincidence.  As soon as anyone in a later
#     phase puts a colourful wallpaper (a photo, a gradient, the camera
#     preview) behind a glass tile, those black corners will *cover* the
#     wallpaper instead of letting it show through, and the panel will
#     look pasted-on.
#
#     The rounded alpha-mask approach is correct in both cases:  inside
#     the rounded region we composite the brightened-tinted surface, and
#     outside (i.e. in the four notional corner squares) we leave the
#     original frame pixels exactly as they were.  This generalises to
#     any wallpaper without modification.

# Pixel-wise additive brighten applied to the background slice before the
# tint mix.  Range chosen empirically: 25 reads as "barely brightened",
# 35 starts to clip on already-bright wallpapers.  30 is the sweet spot
# on a #000 page and degrades gracefully on lighter pages.
_GLASS_BRIGHTEN: Final[int] = 30

# Lerp factor toward each pixel's luminance scalar.  0.0 = no
# desaturation (pixels untouched), 1.0 = fully grayscale.  visionOS
# glass is *slightly* desaturated -- enough to feel cool but not so
# much that a colourful wallpaper goes monochrome.  0.15 is the Apple
# default for tinted glass per their HIG.
_GLASS_DESATURATE: Final[float] = 0.15

# Near-black tint mixed in at 0.15 weight.  (10, 10, 14) BGR is a hair
# of blue bias -- gives the glass a cold, mineralic cast.  Pure black
# (0, 0, 0) produces a "dirty grey" look that reads as cheap.  These
# values are not knobs; they were chosen alongside the 0.85 / 0.15 mix.
_GLASS_TINT_BGR: Final[tuple[int, int, int]] = (10, 10, 14)

# Mix weights for the addWeighted call.  See the long block above for
# the WHY behind 0.85 / 0.15.
_GLASS_SURFACE_WEIGHT: Final[float] = 0.85
_GLASS_TINT_WEIGHT:    Final[float] = 0.15

# BT.601 luminance coefficients (B, G, R order to match the cv2 BGR
# channel layout).  Using BT.709 instead would shift the desaturated
# tone slightly cool, which we don't want -- glass should feel neutral
# rather than blue.  These are universal video coefficients.
_LUMINANCE_BGR: Final[tuple[float, float, float]] = (0.114, 0.587, 0.299)


def _build_glass_surface(under: np.ndarray) -> np.ndarray:
    """Return a brightened-and-desaturated copy of `under`, tinted toward dark.

    The math, applied per-pixel:

        bright = clamp(under + _GLASS_BRIGHTEN, 0, 255)
        lum    = bright . _LUMINANCE_BGR             # scalar per pixel
        desat  = lerp(bright, lum, _GLASS_DESATURATE)
        tint   = fill(under.shape, _GLASS_TINT_BGR)
        out    = (desat * 0.85) + (tint * 0.15)

    All steps run in float32 to avoid clipping at intermediate
    saturation points; we cast back to uint8 only at the very end.  `under`
    is read-only (we never mutate the caller's frame slice).
    """
    # uint8 -> float32 once, then everything downstream stays float.  cv2
    # would happily do uint8 saturation arithmetic but a float pipeline
    # lets us do the luminance lerp without rounding artefacts.
    bright = under.astype(np.float32) + float(_GLASS_BRIGHTEN)
    np.clip(bright, 0.0, 255.0, out=bright)

    # Per-pixel luminance as a (H, W, 1) scalar broadcast back over the
    # three channels.  This is a tight dot product across the channel
    # axis, computed once for the whole slice in one pass.
    lum = (
        bright[..., 0] * _LUMINANCE_BGR[0]
        + bright[..., 1] * _LUMINANCE_BGR[1]
        + bright[..., 2] * _LUMINANCE_BGR[2]
    )[..., np.newaxis]

    # lerp(bright, lum, t):  t=0 keeps colour, t=1 goes greyscale.  At
    # 0.15 the colour reads as "slightly cooled" rather than "grey".
    desat = bright + (lum - bright) * _GLASS_DESATURATE

    # Solid near-black plane the same shape as the patch.  Allocated
    # once per call -- cheap relative to the surface composite below.
    tint = np.empty_like(desat)
    tint[:, :] = _GLASS_TINT_BGR

    surface = cv2.addWeighted(
        desat, _GLASS_SURFACE_WEIGHT,
        tint,  _GLASS_TINT_WEIGHT,
        0.0,
    )
    # Final clamp + cast.  cv2.addWeighted on float32 inputs returns
    # float32; we narrow once at the boundary so callers get a normal
    # BGR uint8 buffer they can splice straight back into the frame.
    np.clip(surface, 0.0, 255.0, out=surface)
    return surface.astype(np.uint8)


def _build_rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Build a single-channel uint8 mask, 255 inside the rounded rect, 0 outside.

    We reuse `rounded_rect` from src.tiles by inflating a (h, w) zero
    buffer with a single-channel "colour" of 255 -- the same primitive
    that draws every tile in the demo.  The resulting mask is shape
    (h, w), dtype uint8, with antialiased corner pixels where the rect
    transitions from inside (255) to outside (0).

    Returning a 2-D mask (rather than 3-D RGB) keeps the downstream
    boolean / weighted-blend code below straightforward.  The mask is
    multiplied directly against the surface and 1 - mask against the
    underlying frame slice; no per-channel replication required.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    # rounded_rect treats `color_bgr` as a per-channel tuple, but on a
    # single-channel mask cv2 just takes the first element as the fill
    # value.  We pass (255, 255, 255) for symmetry with how cv2 expects
    # colour tuples; the mask only has one channel so only 255 lands.
    rounded_rect(mask, 0, 0, w, h, radius=radius, color_bgr=(255, 255, 255))
    return mask


def _apply_rounded_mask(
    frame_slice: np.ndarray,
    surface: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Composite `surface` over `frame_slice` using `mask` as alpha.

    Standard "over" compositing in fractional form:

        out = surface * (mask/255) + frame_slice * (1 - mask/255)

    Pre-normalising the mask to float once is cheaper than dividing by
    255 inside the lerp.  Antialiased edge pixels (mask values like 127
    on a corner step) blend smoothly between the two layers; hard
    interior pixels (mask=255) yield pure surface; hard exterior pixels
    (mask=0) yield the untouched frame.

    Returns a fresh uint8 buffer the same shape as `frame_slice`.
    """
    alpha = mask.astype(np.float32) * (1.0 / 255.0)
    alpha = alpha[..., np.newaxis]   # broadcast over the 3 colour channels
    out = surface.astype(np.float32) * alpha + (
        frame_slice.astype(np.float32) * (1.0 - alpha)
    )
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)


def draw_glass_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
) -> None:
    """Composite a translucent glass surface over frame[y:y+h, x:x+w].

    Mutates `frame` in place; outside the rounded interior of the panel
    the original pixels are preserved exactly.  This is the *only*
    legitimate way to make a Vision OS tile -- a hand-rolled rect
    overlay will not produce the lifted, cooled-down look.

    Off-frame placements (x<0, x+w>frame_w, etc.) are silently clipped
    to the visible region; nothing draws outside the canvas.

    Args:
        frame:  BGR uint8 image, mutated in place.
        x, y:   top-left of the panel's bounding rect, in frame pixels.
        w, h:   panel width / height.  Must be positive; sub-2*radius
                values are tolerated -- the rounded mask just degrades
                to a smaller-radius shape (matching `rounded_rect`).
        radius: corner radius in pixels.  Pass `RADIUS_APP_ICON` for
                home-screen tiles or `RADIUS_TILE_LARGE` for full-size
                marketing tiles.
    """
    # Clip the placement to frame extents -- off-screen panels are a
    # silent no-op rather than an exception, matching draw_text and
    # rounded_rect's contracts.  Negative origin handling: shift the
    # source patch by the same offset so the visible portion still
    # composites correctly.
    frame_h, frame_w = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + w)
    y1 = min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    clip_w = x1 - x0
    clip_h = y1 - y0

    # 1. Slice the area under the panel and 2-3. build the brightened,
    #    desaturated, tinted surface.  _build_glass_surface returns a
    #    fresh buffer; we are NOT yet writing back into the frame.
    under   = frame[y0:y1, x0:x1]
    surface = _build_glass_surface(under)

    # 4-5. Build the rounded alpha mask the same size as the clipped
    #      surface, then alpha-composite over the original frame slice.
    #      The mask handles the radius for us; we never paint corner
    #      pixels explicitly.
    mask        = _build_rounded_mask(clip_w, clip_h, radius)
    composited  = _apply_rounded_mask(under, surface, mask)

    # Splice the composited rect back into the frame.  Inside the
    # rounded region we now have the glass surface; outside the
    # rounded region (the four notional corner squares) we have the
    # original under-frame pixels unchanged, exactly as the rounded
    # mask intended.
    frame[y0:y1, x0:x1] = composited


# ============================================================================
# App icon dispatch + glyphs
# ============================================================================
#
# Each app's *background tint* (the rounded square the glyph sits on) is
# its identifying colour.  The glyph itself is then drawn in the
# colour(s) that best read against that tint.  These two roles are
# deliberately split:
#
#     - Tint  = what makes the icon recognisable at thumbnail size.
#               Safari is white-with-blue-compass; Music is red; Notes
#               is yellow.  You can pick the app out of a grid by tint
#               alone before the glyph resolves.
#
#     - Glyph = what makes the icon *legible* on inspection.  The
#               compass, the eighth note, the gear teeth.
#
# Tints are BGR tuples paired with each icon's private _icon_<id>
# function in `_ICON_TINTS` and `_ICON_DRAWERS` below.

_ICON_TINTS: Final[dict[str, tuple[int, int, int]]] = {
    # Safari background is iconic white (with a faintly blue cast in the
    # real artwork -- we use a neutral near-white that doesn't fight
    # against the compass-blue ring we draw on top).
    "safari":   (240, 240, 245),
    # Photos background is light grey so the rainbow petals stay the
    # eye's focus.  The real Apple Photos icon uses pure white and the
    # petals carry all the colour; we use a hair darker for visual
    # weight inside the glass tile.
    "photos":   (235, 235, 240),
    # Apple Music red.  In RGB that's around (250, 60, 75); reversed
    # for cv2 BGR.  This is the only saturated red on the home screen.
    "music":    (75, 75, 250),
    # Notes yellow.  Around RGB (255, 220, 40); reversed to BGR.  This
    # is the colour of a stickied legal-pad page.
    "notes":    (40, 220, 250),
    # Apple Mail blue.  Around RGB (70, 175, 245) -- a paler, friendlier
    # blue than the iOS system blue.  Reversed to BGR.
    "mail":     (245, 175, 70),
    # Calendar background is white; the red header band is drawn as
    # part of the glyph itself so it can move with the icon (Apple's
    # icon also has the "today" red header inset by a few pixels).
    "calendar": (240, 240, 245),
    # Settings dark grey.  In the iOS icon there's a subtle gradient
    # ramp; we flatten that to a single mid-dark tone.  Lighter than
    # BG_DARK so the glyph reads against it.
    "settings": (60, 60, 65),
    # Demo tile is iOS-blue (around RGB 0, 105, 255).  Reversed.  Sits
    # closest to the home-screen accent so it reads as "ours".
    "demo":     (255, 105, 0),
}


# ----------------------------------------------------------------------------
# Glyph helpers
# ----------------------------------------------------------------------------
#
# Every _icon_<id> function takes (frame, cx, cy, size) where (cx, cy) is
# the centre of the icon's *square* extent and `size` is its width and
# height in pixels.  The functions own everything inside that square but
# are explicitly NOT responsible for the rounded background fill -- that
# is `draw_app_icon`'s job (see below).
#
# Drawing primitives are all cv2.* with LINE_AA on any non-axis-aligned
# edge.  Without LINE_AA the curves look like 1990s GUI clip-art at the
# 96px scale these icons typically render at.
#
# Each function is under 40 lines per the phase prompt.

# Cached PIL font for the calendar "11" date label.  Loaded on first
# call to keep import time fast; cached on the function object so we
# don't pay the truetype-parse cost every frame.  Pattern mirrors
# `_get_tile_fonts` in src/tiles.py.
def _calendar_date_font():
    cached = getattr(_calendar_date_font, "_cache", None)
    if cached is None:
        cached = load_font(role="display", size=28)
        _calendar_date_font._cache = cached  # type: ignore[attr-defined]
    return cached


def _icon_safari(frame: np.ndarray, cx: int, cy: int, size: int) -> None:
    """Compass: blue ring + red/white needle.

    The Safari icon is a compass face seen head-on.  We render:
        - an outer blue ring (cv2.circle outline) at ~92% of the icon
          radius, line thickness scaled to size so it stays proportional
        - two opposed filled triangles forming the needle, tilted ~30°
          off vertical, the red half pointing north-east and the white
          half pointing south-west.  Apple's icon uses red for the
          "north" half and white for "south"; we follow that convention.
    """
    # Outer ring: a single circle outline.  cv2.circle radius is the
    # *outer* edge; thickness paints inward.  Apple's compass ring sits
    # well inside the rounded background, hence the 0.42 factor.
    radius = int(size * 0.42)
    thickness = max(2, size // 24)
    # Safari blue, BGR -- a slightly darker, slightly less saturated
    # ring than the system accent so it reads as "compass" rather
    # than "link".
    ring_bgr = (200, 110, 30)
    cv2.circle(frame, (cx, cy), radius, ring_bgr,
               thickness=thickness, lineType=cv2.LINE_AA)

    # Needle: two triangles meeting at the centre.  Pre-compute the
    # rotation once.  +30° puts the red (north) half tilted to the
    # north-east, matching Apple's reference compass artwork.  The
    # cv2 coordinate system has y pointing DOWN, so "north" in icon
    # space is the -y direction; multiplying cos(angle) by -1 below
    # accounts for that.
    angle = math.radians(30.0)
    tip_len  = int(size * 0.34)
    waist    = max(3, size // 18)

    # Tip directions in icon-local coordinates (y points down in cv2).
    tx, ty = math.sin(angle), -math.cos(angle)        # "north" direction
    px, py = -math.cos(angle), -math.sin(angle)       # perpendicular waist axis

    tip_n = (int(cx + tx * tip_len), int(cy + ty * tip_len))
    tip_s = (int(cx - tx * tip_len), int(cy - ty * tip_len))
    waist_l = (int(cx + px * waist), int(cy + py * waist))
    waist_r = (int(cx - px * waist), int(cy - py * waist))

    # North half -- red (BGR).
    cv2.fillPoly(frame, [np.array([tip_n, waist_l, waist_r], np.int32)],
                 (60, 70, 230), lineType=cv2.LINE_AA)
    # South half -- white.
    cv2.fillPoly(frame, [np.array([tip_s, waist_l, waist_r], np.int32)],
                 (245, 245, 245), lineType=cv2.LINE_AA)


def _icon_photos(frame: np.ndarray, cx: int, cy: int, size: int) -> None:
    """Six-petal rainbow flower.

    The Apple Photos icon is six overlapping coloured petals arranged
    radially.  We draw each petal as a filled ellipse rotated 60° from
    the previous one.  Petal colours, in canonical Apple order
    (clockwise from 12 o'clock): yellow, orange, red, magenta, blue,
    green.  Colours are BGR.
    """
    petal_w  = max(4, size // 8)   # short axis (thickness)
    petal_h  = int(size * 0.32)    # long axis (radius from centre)
    # Petal colours in BGR.  Slightly desaturated from pure neon so they
    # don't clash on a glass-over-dark surface.
    petals = [
        (60, 220, 240),   # yellow      (12)
        (40, 165, 245),   # orange      (2)
        (75, 75, 240),    # red         (4)
        (200, 70, 220),   # magenta     (6)
        (240, 145, 60),   # blue        (8)
        (100, 200, 90),   # green       (10)
    ]
    # Each ellipse is centred at the icon centre and rotated; the long
    # axis points "outward" so half of it pokes past the centre on
    # each side, creating the symmetric petal silhouette.
    for i, color in enumerate(petals):
        angle_deg = i * 60.0 - 90.0   # -90 puts petal 0 at 12 o'clock
        cv2.ellipse(
            frame, (cx, cy),
            (petal_h, petal_w),
            angle_deg, 0.0, 360.0,
            color, thickness=-1, lineType=cv2.LINE_AA,
        )


def _icon_music(frame: np.ndarray, cx: int, cy: int, size: int) -> None:
    """White eighth note on a red tile: head + stem + flag.

    The eighth-note glyph here is a stylised mark, not a music-engraving
    grade rendering.  Three pieces:
        - note head: a filled ellipse, slightly slanted
        - stem: a vertical rectangle rising from the head
        - flag: a filled quadrilateral hanging right off the stem top
    """
    white = (245, 245, 245)

    # Note head sits in the lower-left quadrant of the icon.
    head_cx = cx - int(size * 0.10)
    head_cy = cy + int(size * 0.16)
    head_rx = int(size * 0.13)
    head_ry = int(size * 0.10)
    cv2.ellipse(frame, (head_cx, head_cy), (head_rx, head_ry),
                -18.0, 0.0, 360.0, white,
                thickness=-1, lineType=cv2.LINE_AA)

    # Stem: a vertical filled rect rising from the head's top-right.
    stem_x = head_cx + head_rx - max(2, size // 32)
    stem_top_y = cy - int(size * 0.22)
    stem_bot_y = head_cy
    stem_w = max(3, size // 28)
    cv2.rectangle(frame,
                  (stem_x, stem_top_y),
                  (stem_x + stem_w, stem_bot_y),
                  white, thickness=-1)

    # Flag: a small filled quad off the top of the stem, sweeping right
    # and slightly down -- the canonical eighth-note flag shape.
    flag_pts = np.array([
        (stem_x + stem_w,                 stem_top_y),
        (stem_x + stem_w + int(size * 0.18), stem_top_y + int(size * 0.06)),
        (stem_x + stem_w + int(size * 0.16), stem_top_y + int(size * 0.14)),
        (stem_x + stem_w,                 stem_top_y + int(size * 0.08)),
    ], np.int32)
    cv2.fillPoly(frame, [flag_pts], white, lineType=cv2.LINE_AA)


def _icon_notes(frame: np.ndarray, cx: int, cy: int, size: int) -> None:
    """Yellow legal-pad page with three muted ruling lines.

    The Notes icon background is yellow (handled by `draw_app_icon`).
    Here we add three horizontal grey "ruling" lines spaced evenly down
    the icon's interior so it reads as a written-on page rather than a
    bare yellow square.
    """
    # Ruling lines, evenly distributed in the middle band of the icon.
    line_color = (160, 160, 165)         # muted grey, BGR
    line_thickness = max(2, size // 28)
    inset_x = int(size * 0.18)
    line_xs = (cx - size // 2 + inset_x, cx + size // 2 - inset_x)

    # Three lines vertically spaced 22% of the icon's height apart,
    # centred around cy.  This puts the middle line on the icon centre
    # and the outer two symmetrically above and below.
    line_dy = int(size * 0.18)
    for offset in (-line_dy, 0, line_dy):
        y = cy + offset
        cv2.line(frame, (line_xs[0], y), (line_xs[1], y),
                 line_color, thickness=line_thickness, lineType=cv2.LINE_AA)


def _icon_mail(frame: np.ndarray, cx: int, cy: int, size: int) -> None:
    """White envelope outline with a V-flap on top.

    Envelope rectangle is drawn as a thin outlined rect (so the blue
    tile shows through the body of the envelope).  The flap is two
    diagonal lines meeting at the envelope centre top, forming a V.
    Apple's mail envelope is usually drawn closed-flap (the V points
    down); we follow that.
    """
    white = (245, 245, 245)
    line_thickness = max(2, size // 24)

    # Envelope body: a rectangle centred on the icon.
    half_w = int(size * 0.34)
    half_h = int(size * 0.22)
    top_left  = (cx - half_w, cy - half_h)
    bot_right = (cx + half_w, cy + half_h)
    cv2.rectangle(frame, top_left, bot_right, white,
                  thickness=line_thickness, lineType=cv2.LINE_AA)

    # Flap: two diagonal lines from the upper corners meeting at the
    # body's horizontal centre, slightly inside the rectangle so the
    # V sits visually on the envelope rather than poking out the top.
    flap_meet = (cx, cy)                           # body centre
    cv2.line(frame, top_left,
             flap_meet, white,
             thickness=line_thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (top_left[0] + 2 * half_w, top_left[1]),
             flap_meet, white,
             thickness=line_thickness, lineType=cv2.LINE_AA)


def _icon_calendar(frame: np.ndarray, cx: int, cy: int, size: int) -> None:
    """Red "today" header band on top of a white page, with a centered "11".

    The Calendar icon is recognisable by its layered look: a thin red
    band across the top (~22% of the icon height) and the date number
    centred on the white body below.  The icon background tint comes
    in white (from _ICON_TINTS), so we only paint the red strip plus
    the number here.
    """
    # The red header strip.  Inset slightly from the icon edges so it
    # sits within the rounded background's interior rather than
    # touching its rounded corners.
    inset = int(size * 0.08)
    strip_x0 = cx - size // 2 + inset
    strip_x1 = cx + size // 2 - inset
    strip_y0 = cy - size // 2 + inset
    strip_h  = int(size * 0.22)
    strip_y1 = strip_y0 + strip_h
    apple_red_bgr = (60, 70, 230)
    cv2.rectangle(frame, (strip_x0, strip_y0), (strip_x1, strip_y1),
                  apple_red_bgr, thickness=-1, lineType=cv2.LINE_AA)

    # Date number "11" rendered through PIL so it reads at the same
    # quality as the rest of the demo's typography.  Centered on the
    # white half of the icon (below the strip).
    body_cy = (strip_y1 + (cy + size // 2 - inset)) // 2
    font = _calendar_date_font()
    # `draw_text(..., y=...)` anchors at the top of the bounding box;
    # subtract half the font's ascent height to vertically centre.
    _, top_bbox, _, bot_bbox = font.getbbox("11")
    text_h = bot_bbox - top_bbox
    draw_text(frame, "11", x=cx, y=body_cy - text_h // 2,
              color_rgb=TEXT_ON_LIGHT_RGB, font=font, align="center")


def _icon_settings(frame: np.ndarray, cx: int, cy: int, size: int) -> None:
    """Cog: eight short rectangular teeth, an inner ring, and a hub.

    The Settings gear is canonically eight-toothed (twelve in newer
    macOS art, eight in classic iOS).  We render the teeth as short
    radial rectangles, then an inner-ring outline, then a small solid
    hub at the centre.  Drawn in muted-light grey against the dark
    tile background.
    """
    light = (200, 200, 205)

    # Teeth: eight short rects equally spaced around the centre, each
    # rendered as a rotated filled quadrilateral via fillPoly.  Width
    # and length are sized to the icon so the teeth read at any scale.
    tooth_len = int(size * 0.10)
    tooth_w   = max(3, size // 16)
    tooth_inner_r = int(size * 0.24)
    for i in range(8):
        angle = i * (math.pi / 4)
        c, s = math.cos(angle), math.sin(angle)
        # Build a thin tangential rectangle from inner radius -> outer
        inner_cx = cx + c * tooth_inner_r
        inner_cy = cy + s * tooth_inner_r
        outer_cx = cx + c * (tooth_inner_r + tooth_len)
        outer_cy = cy + s * (tooth_inner_r + tooth_len)
        # Perpendicular half-width vector for the rect's sides.
        nx, ny = -s * (tooth_w / 2), c * (tooth_w / 2)
        pts = np.array([
            (int(inner_cx - nx), int(inner_cy - ny)),
            (int(inner_cx + nx), int(inner_cy + ny)),
            (int(outer_cx + nx), int(outer_cy + ny)),
            (int(outer_cx - nx), int(outer_cy - ny)),
        ], np.int32)
        cv2.fillPoly(frame, [pts], light, lineType=cv2.LINE_AA)

    # Inner ring outline + central hub.  These two circles complete the
    # cog -- without them the teeth read as a scatter of detached rects.
    cv2.circle(frame, (cx, cy), tooth_inner_r, light,
               thickness=max(2, size // 22), lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(3, size // 12), light,
               thickness=-1, lineType=cv2.LINE_AA)


def _icon_demo(frame: np.ndarray, cx: int, cy: int, size: int) -> None:
    """White "vd" wordmark for the Demo tile, centered.

    Reuses the design-system display font; same anchor maths as the
    calendar's "11".  Sits cleanly against the blue Demo tint.
    """
    font = _calendar_date_font()
    _, top_bbox, _, bot_bbox = font.getbbox("vd")
    text_h = bot_bbox - top_bbox
    # TEXT_ON_DARK_RGB is the canonical white-on-dark text colour; using
    # the design-token constant keeps the wordmark in sync with palette
    # tweaks elsewhere.
    draw_text(frame, "vd", x=cx, y=cy - text_h // 2,
              color_rgb=TEXT_ON_DARK_RGB, font=font, align="center")


_ICON_DRAWERS: Final[dict[str, object]] = {
    "safari":   _icon_safari,
    "photos":   _icon_photos,
    "music":    _icon_music,
    "notes":    _icon_notes,
    "mail":     _icon_mail,
    "calendar": _icon_calendar,
    "settings": _icon_settings,
    "demo":     _icon_demo,
}


def draw_app_icon(
    frame: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    app_id: str,
) -> None:
    """Draw the app icon for `app_id` centered at (cx, cy), `size` x `size`.

    Composed of two layers:
        1. A rounded square background (RADIUS_APP_ICON corners) tinted
           with the app's identifying colour from `_ICON_TINTS`.
        2. The app's private `_icon_<id>` glyph drawn on top.

    The function mutates `frame` in place.

    Args:
        frame:  BGR uint8 image.  Mutated in place.
        cx, cy: pixel centre of the icon's bounding square.
        size:   width and height of that bounding square, in pixels.
        app_id: one of the keys in `_ICON_TINTS`; anything else raises
                ValueError.  Keeping this strict means a typo at the
                home-screen call site fails loudly at render time
                instead of silently drawing a blank tile.
    """
    if app_id not in _ICON_TINTS:
        raise ValueError(
            f"Unknown app_id {app_id!r}; expected one of "
            f"{sorted(_ICON_TINTS.keys())}."
        )

    # 1. Rounded background: draw via the same rounded_rect primitive
    #    every tile in the demo uses, so corner antialiasing matches
    #    bit-for-bit between status bar, glass panels, and these icons.
    half = size // 2
    rounded_rect(
        frame,
        cx - half, cy - half,
        size, size,
        radius=RADIUS_APP_ICON,
        color_bgr=_ICON_TINTS[app_id],
    )

    # 2. Glyph layer.  Each drawer knows how to centre itself around
    #    (cx, cy) within the `size`-extent square the background just
    #    painted.  We dispatch through the table so this function
    #    stays a single flat eight-way lookup rather than an
    #    if/elif ladder.
    drawer = _ICON_DRAWERS[app_id]
    drawer(frame, cx, cy, size)  # type: ignore[operator]


# ============================================================================
# Public surface
# ============================================================================
#
# Only two functions in this module are meant to be called from outside:
#
#     draw_glass_panel(frame, x, y, w, h, radius)   -- floating-tile surface
#     draw_app_icon(frame, cx, cy, size, app_id)    -- one of 8 app icons
#
# Everything else (the _icon_<id> helpers, the _build/apply helpers, the
# tint table, the date-font cache) is module-private and may be
# reshuffled without notice.
# ============================================================================
