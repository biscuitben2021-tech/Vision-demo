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

# ============================================================================
# Liquid Glass constants (WWDC 2025 design language)
# ============================================================================
#
# Liquid Glass replaces the earlier flat tinted-rect treatment with six
# layered visual components.  None of them alone sells the effect; all
# six together produce a panel that reads as a physical translucent
# surface rather than a darker rectangle painted on the wallpaper.  See
# the long WHY block at each helper below.

# Frost tint -- near-white BGR plane that gets alpha-blended onto the
# glass surface at 10-12% opacity.  This is the single most important
# component: without it, the panel looks like "tinted glass over content"
# rather than "frosted glass with content underneath".  (240, 240, 245)
# BGR has a hair of blue lift in BGR -- a faint cool cast that
# distinguishes it from pure paper white.
_FROST_TINT_BGR: Final[tuple[int, int, int]] = (240, 240, 245)

# Rim highlight colour.  Slightly cooler than pure #fff so the rim
# doesn't read as a "fluorescent strip" against the warm content
# behind it.  Drawn at 1px with cv2.LINE_AA along the top edge + top
# corner arcs, then composited at 50% opacity.
_RIM_COLOR_BGR: Final[tuple[int, int, int]] = (245, 245, 250)

# Blur kernel range.  Even values must be incremented to odd because
# cv2.GaussianBlur requires an odd kernel size.  Intensity 1.0 picks
# 31, intensity 0.0 picks 21; lower intensity for app icons / cards
# where the source region is small and a smaller kernel is plenty.
_BLUR_KSIZE_MAX: Final[int] = 31
_BLUR_KSIZE_MIN: Final[int] = 21

# Skip-blur threshold.  When the region under the panel has
# region.std() < this, the underlying content is essentially uniform
# (typically BG_DARK on the home screen) and blurring it would do
# nothing visible -- the saved ~5-10ms of GaussianBlur time goes
# straight into the per-frame headroom.  The rest of the components
# still sell the glass effect on their own.
_BLUR_SKIP_STD: Final[float] = 5.0

# Brightness lift (HSV V channel).  Real glass passes most light
# through but picks up energy.  We add to V (lightness) in HSV so the
# hue / saturation don't shift -- adding to all three BGR channels
# would chase highlights toward white.  Range 12..18 by intensity.
_V_LIFT_BASE:  Final[int] = 12
_V_LIFT_RANGE: Final[int] = 6

# Frost tint alpha range.  10% on a small icon tile reads as a
# polish; 12% on the status bar reads as glass.  Scaled by intensity.
_FROST_ALPHA_BASE:  Final[float] = 0.10
_FROST_ALPHA_RANGE: Final[float] = 0.02

# Top inner gradient.  Vertical alpha ramp from 0.06 at the top edge
# fading to 0.0 at TOP_GRADIENT_DEPTH rows.  Blends toward pure white
# to add light, not toward grey (which would dirty the panel).
_TOP_GRADIENT_DEPTH: Final[int] = 12
_TOP_GRADIENT_ALPHA: Final[float] = 0.06

# Bottom inner shadow.  Vertical alpha ramp from 0.0 to 0.15 over
# BOTTOM_SHADOW_DEPTH rows at the bottom edge.  Blends toward black
# to deepen the bottom and sell the "floating physical surface" look.
_BOTTOM_SHADOW_DEPTH: Final[int] = 3
_BOTTOM_SHADOW_ALPHA: Final[float] = 0.15

# Rim opacity (alpha used when blending the rim copy back into the
# surface).  50% reads as a soft white edge; 100% would look like a
# hard CAD stroke.
_RIM_ALPHA: Final[float] = 0.5


# Module-level rounded-rect mask cache.  Keys are (w, h, radius).
# Rounded-rect masks are pure geometry -- same inputs always produce
# the same array -- so they're a textbook memoise target.  Generating
# one via rounded_rect costs ~5ms per call on a 240x64 panel; with 9
# panels per frame (status bar + 8 app icons) that's 45ms saved per
# steady-state frame after the first.  Module-level dict (rather than
# a function attribute) so the cache survives across the few callers
# in the codebase without per-call setup.
_MASK_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


# ============================================================================
# Warm aurora wallpaper -- visionOS atmospheric backdrop
# ============================================================================
#
# Pure-black home wallpaper makes the Liquid Glass effect nearly invisible
# (blurring black produces black; the frost tint adds nothing visible on
# a flat fill).  visionOS itself draws passthrough -- whatever's behind
# the user's head -- with a soft tint applied.  We approximate that on a
# laptop demo with three heavily-blurred radial color blobs over a
# near-black baseline: warm purple upper-left, soft pink upper-right,
# cool blue lower.  The result reads as "subtle ambient color" rather
# than wallpaper-of-the-month: glass panels get varied content to
# refract through, and `_glass_base`'s blur path activates (region.std()
# > 5) so the refraction effect is actually visible.
#
# Composed at low resolution (64x40) then upsampled with Lanczos to the
# full frame, plus a heavy Gaussian blur on the final result.  Cached
# per (width, height) -- the wallpaper is static for the life of a
# resolution; recomputing it every frame would burn ~30ms on the
# 101x101 blur of a 1440x900 buffer.
_AURORA_CACHE: dict[tuple[int, int], np.ndarray] = {}

# Aurora blob palette.  All BGR.  Picked to land on the warm-purple-to-
# cool-blue diagonal Apple uses for passthrough tints in marketing
# renders.  Floats so we can lerp; the floor and scale below land them
# in 0..255 uint8 space.
_AURORA_BASELINE_BGR: Final[tuple[int, int, int]] = (12, 10, 14)
_AURORA_BLOBS: Final[tuple[tuple[float, float, tuple[int, int, int], float], ...]] = (
    # (cx_norm, cy_norm, color_bgr, sigma_norm)  in [0..1] normalized coords
    (0.22, 0.18, (170, 55, 95),   0.32),   # warm purple,  upper-left
    (0.78, 0.20, (195, 110, 180), 0.30),   # soft pink,    upper-right
    (0.55, 0.78, (210, 145, 70),  0.36),   # cool blue,    lower-center
)


def _build_warm_aurora(w: int, h: int) -> np.ndarray:
    """Build the warm-aurora wallpaper for a (w, h) frame.

    Computed at low res for speed -- the heavy blur kills any
    high-frequency detail anyway, so 64x40 is plenty of source even at
    a 1440x900 final size.  Returned as a BGR uint8 array ready to
    `frame[:] = result`.

    Each blob is a Gaussian falloff: `exp(-((x-cx)^2 + (y-cy)^2) /
    (2 * sigma^2))`.  Blob centres + sigmas are in normalised [0..1]
    coords so the same colour layout holds at any final resolution.
    """
    sw, sh = 64, 40
    aurora = np.full(
        (sh, sw, 3), _AURORA_BASELINE_BGR, dtype=np.float32,
    )
    # numpy meshgrid in row-major (y, x) order, normalised to [0..1]
    yy = np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, sw, dtype=np.float32)[None, :]

    for cx, cy, color_bgr, sigma in _AURORA_BLOBS:
        # Gaussian intensity per pixel -- broadcast across the (sh, sw)
        # grid in one numpy pass.
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        falloff = np.exp(-d2 / (2.0 * sigma * sigma))[..., None]
        aurora = aurora + falloff * np.array(color_bgr, dtype=np.float32)

    np.clip(aurora, 0.0, 255.0, out=aurora)
    aurora_u8 = aurora.astype(np.uint8)

    # Lanczos upscale to full size, then a heavy Gaussian blur to wash
    # away any banding that the upscale introduced.  Kernel size scales
    # with frame width so the visual softness is consistent across
    # resolutions; we round to the nearest odd int because cv2
    # requires odd kernels.
    upsampled = cv2.resize(
        aurora_u8, (w, h), interpolation=cv2.INTER_LANCZOS4,
    )
    ksize = max(51, (w // 28) | 1)   # |1 forces odd
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(upsampled, (ksize, ksize), 0)

    # Damp brightness so the aurora reads as atmospheric ambient
    # colour rather than as a saturated photograph.  0.55 keeps the
    # warm tones legible without competing with the foreground glass
    # tiles.  Cast back to uint8 at the boundary.
    damped = (blurred.astype(np.float32) * 0.55).astype(np.uint8)
    return damped


def paint_warm_aurora(frame: np.ndarray, w: int, h: int) -> None:
    """Fill `frame[:h, :w]` with the cached warm-aurora wallpaper.

    Replaces the pure-black home wallpaper.  The aurora is static --
    same colours and gradient every frame -- so the bulk of the work
    is done once per resolution and cached.  Per-frame cost is just a
    single `np.copyto`, which is essentially free relative to the rest
    of the compose pipeline.

    `frame` is BGR uint8.  Mutated in place.
    """
    if w <= 0 or h <= 0:
        return
    key = (w, h)
    cached = _AURORA_CACHE.get(key)
    if cached is None:
        cached = _build_warm_aurora(w, h)
        _AURORA_CACHE[key] = cached
    # np.copyto for in-place fill; faster than frame[:] = cached because
    # it avoids a fresh allocation if shapes match exactly.
    np.copyto(frame[:h, :w], cached)


def _blur_kernel_size(intensity: float) -> int:
    """Pick an odd Gaussian kernel size for the given intensity.

    intensity = 1.0  ->  31  (strongest refraction; status bar)
    intensity = 0.85 ->  29  (app icon tiles)
    intensity = 0.9  ->  30 -> 31 (notifications; even => bumped to 31)
    intensity = 0.0  ->  21  (minimum; preserves any blur but cheap)

    Returns an odd int in the range [_BLUR_KSIZE_MIN, _BLUR_KSIZE_MAX].
    cv2.GaussianBlur requires odd kernel sizes; we bump even values
    upward rather than truncating downward so the higher-intensity
    setting always picks the stronger blur.
    """
    raw = _BLUR_KSIZE_MIN + intensity * (_BLUR_KSIZE_MAX - _BLUR_KSIZE_MIN)
    ksize = int(round(raw))
    if ksize % 2 == 0:
        ksize += 1
    return max(_BLUR_KSIZE_MIN, min(_BLUR_KSIZE_MAX, ksize))


def _glass_base(region: np.ndarray, intensity: float) -> np.ndarray:
    """Build the Liquid Glass base layer from `region`: blur, brighten, frost.

    Three of the six visual components of Liquid Glass land in this
    function.  The other three (top gradient, bottom shadow, rim
    highlight) are layered onto the base by their respective helpers
    AFTER the orchestrator has composited the base into the frame.

    Colour-space convention: input and output are BGR uint8.  The HSV
    conversion in the middle of the pipeline is transient -- the
    function never returns an HSV buffer.  This matters because the
    rest of the codebase assumes every numpy frame slice is BGR.

    Args:
        region:    BGR uint8 region under the panel.  Read-only; we
                   return a fresh buffer rather than mutating it.
        intensity: 0.0..1.0 scalar controlling the strength of each
                   component.  1.0 = status bar.  0.85 = app icon
                   backgrounds.  0.9 = notification cards.

    Returns:
        Fresh BGR uint8 array, same shape as `region`.
    """
    # 1. Refraction blur.  Gaussian blur simulates light scattering
    #    through real glass -- "blur simulates refraction through real
    #    glass".  Skip the blur entirely when the region is near-
    #    uniform (region.std() < 5): on a BG_DARK home screen the
    #    panel sits over pure black, blurring pure black is a pure
    #    no-op, and the saved 5-10ms goes straight back to the FPS
    #    budget.  The tint + rim + gradient + shadow still sell the
    #    effect alone -- the blur matters only when there's varied
    #    content underneath.
    if float(region.std()) >= _BLUR_SKIP_STD:
        ksize = _blur_kernel_size(intensity)
        blurred = cv2.GaussianBlur(region, (ksize, ksize), 0)
    else:
        blurred = region.copy()

    # 2. Brightness lift.  Real glass passes most light through but
    #    picks up energy doing so -- the surface reads a touch brighter
    #    than what's behind it.  We add a constant to the V channel of
    #    HSV (lightness axis) rather than to each BGR channel because
    #    V lifts perceived lightness without shifting hue or
    #    saturation; B+G+R addition pushes highlights toward neutral
    #    white and washes colour out of the underlying content.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    v_lift = _V_LIFT_BASE + int(round(_V_LIFT_RANGE * intensity))
    v = hsv[..., 2].astype(np.int16) + v_lift
    hsv[..., 2] = np.clip(v, 0, 255).astype(np.uint8)
    lifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Frost tint.  The single most important glass component:
    #    alpha-blending toward a near-white plane is what makes the
    #    panel read as "glass surface" rather than "darker rectangle".
    #    10-12% opacity is tuned so the underlying content stays
    #    legible while the panel acquires a soft frosted look.  At
    #    >15% the panel starts to look like opaque frosted plastic.
    tint_alpha = _FROST_ALPHA_BASE + _FROST_ALPHA_RANGE * intensity
    tint = np.empty_like(lifted)
    tint[:] = _FROST_TINT_BGR
    return cv2.addWeighted(lifted, 1.0 - tint_alpha, tint, tint_alpha, 0.0)


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


def _get_rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Return a cached rounded-rect alpha mask of shape (h, w), uint8.

    Mask values are 255 inside the rounded rectangle, 0 outside, with
    antialiased corner pixels.  The result depends only on (w, h,
    radius) -- pure geometry -- so the array is safe to memoise and
    share across frames.

    Why caching matters: building a mask via `_build_rounded_mask`
    costs ~5ms per call on a 240x64 panel.  With 9 glass panels per
    frame (status bar + 8 app icons) that's ~45ms per steady-state
    frame thrown away on rebuilding identical arrays.  This dict
    lookup amortises that to a one-time cost after warmup.

    The returned array is treated as read-only by every caller in
    this module.  Slicing it is fine (numpy returns a view); none of
    the downstream code writes back into the mask.
    """
    key = (w, h, radius)
    cached = _MASK_CACHE.get(key)
    if cached is None:
        cached = _build_rounded_mask(w, h, radius)
        _MASK_CACHE[key] = cached
    return cached


def _glass_top_gradient(surface: np.ndarray, mask: np.ndarray) -> None:
    """Apply a soft top-edge inner gradient to `surface`, in place.

    Visual purpose: glass picks up a brighter wash at its top edge
    from the implied light source above.  Without this the panel
    reads as a flat sticker; with it the panel reads as a thin
    physical surface catching daylight.

    Implementation: a vertical alpha ramp from 0.06 at the topmost
    row fading to 0.0 at `_TOP_GRADIENT_DEPTH` rows, modulated by the
    rounded mask's top strip so the gradient respects the panel's
    actual silhouette (corner pixels with mask=0 contribute nothing).
    Blends toward pure white -- the highlight should add LIGHT, not
    shift hue.

    Colour-space convention: `surface` is panel-local BGR uint8;
    `mask` is panel-local single-channel uint8.  Both mutated only
    via the slice indexing semantics of numpy (no in-place HSV or
    channel-order surprises).
    """
    h, w = surface.shape[:2]
    depth = min(_TOP_GRADIENT_DEPTH, h)
    if depth <= 0:
        return
    # Alpha ramp shape (depth, 1, 1) broadcasts cleanly against the
    # (depth, w, 3) surface slice.  np.linspace endpoints inclusive.
    ramp = np.linspace(
        _TOP_GRADIENT_ALPHA, 0.0, depth, dtype=np.float32,
    )[:, None, None]
    m = mask[:depth, :, None].astype(np.float32) * (1.0 / 255.0)
    alpha = ramp * m
    region = surface[:depth].astype(np.float32)
    region = region * (1.0 - alpha) + 255.0 * alpha
    np.clip(region, 0.0, 255.0, out=region)
    surface[:depth] = region.astype(np.uint8)


def _glass_bottom_shadow(surface: np.ndarray, mask: np.ndarray) -> None:
    """Apply a faint bottom-edge inner shadow to `surface`, in place.

    Visual purpose: gives the glass panel physical depth.  The
    bottom edge sits in slight shadow because the implied light
    source above doesn't reach it as strongly.  Without this the
    panel reads as a flat sticker; with it the panel reads as a
    thin floating physical surface.

    Implementation: a vertical alpha ramp from 0.0 to 0.15 across
    the last `_BOTTOM_SHADOW_DEPTH` rows, modulated by the rounded
    mask's bottom strip.  Blends toward pure black to deepen the
    edge rather than toward grey (which would dirty the panel).

    Same panel-local BGR uint8 / single-channel uint8 conventions as
    `_glass_top_gradient`.
    """
    h, w = surface.shape[:2]
    depth = min(_BOTTOM_SHADOW_DEPTH, h)
    if depth <= 0:
        return
    ramp = np.linspace(
        0.0, _BOTTOM_SHADOW_ALPHA, depth, dtype=np.float32,
    )[:, None, None]
    m = mask[-depth:, :, None].astype(np.float32) * (1.0 / 255.0)
    alpha = ramp * m
    region = surface[-depth:].astype(np.float32)
    # Blend toward black: out = region * (1 - alpha) + 0 * alpha.
    region = region * (1.0 - alpha)
    surface[-depth:] = region.astype(np.uint8)


def _glass_rim(surface: np.ndarray, w: int, h: int, radius: int) -> None:
    """Draw a 1px near-white rim along the top ~40% of the perimeter, in place.

    Visual purpose: glass catches its edge in light from above.  The
    rim follows: top straight edge between the two corner centres +
    the top-left rounded corner arc + the top-right rounded corner
    arc.  Drawn at 50% opacity so it reads as a soft white edge, not
    a hard CAD stroke -- the trick is to paint the rim onto a copy at
    full white and then `cv2.addWeighted` the copy back at 0.5.
    Untouched pixels stay identical (0.5*p + 0.5*p == p); only the
    rim-coloured pixels shift halfway toward white.

    OpenCV ellipse angle convention: 0° points along +x, angles
    increase clockwise in image (y-down) coordinates.  Top-left
    corner arc therefore spans 180° (leftmost) through 270° (topmost);
    top-right corner arc spans 270° (topmost) through 360°
    (rightmost).

    Coordinates are panel-local: (0, 0) is the top-left of `surface`.
    `surface` is BGR uint8 and mutated in place.
    """
    if w < 2 * radius or h < 2:
        return  # panel too small for a meaningful rim
    rim = surface.copy()
    color = _RIM_COLOR_BGR
    # Top straight edge between the two corner centres.  cv2.line is
    # subpixel-exact with LINE_AA; we don't need to inset by 0.5px.
    cv2.line(
        rim, (radius, 0), (w - radius, 0), color, 1, cv2.LINE_AA,
    )
    # Top-left corner arc: 180° -> 270°.
    cv2.ellipse(
        rim, (radius, radius), (radius, radius),
        0.0, 180.0, 270.0, color, 1, cv2.LINE_AA,
    )
    # Top-right corner arc: 270° -> 360°.
    cv2.ellipse(
        rim, (w - radius, radius), (radius, radius),
        0.0, 270.0, 360.0, color, 1, cv2.LINE_AA,
    )
    cv2.addWeighted(rim, _RIM_ALPHA, surface, 1.0 - _RIM_ALPHA, 0.0, dst=surface)


def draw_glass_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
    *,
    intensity: float = 1.0,
) -> None:
    """Composite a Liquid Glass surface over frame[y:y+h, x:x+w].

    The successor to the earlier flat tinted-rect treatment, this
    builds all six Liquid Glass components in sequence:

        1. Refraction blur     -- skipped when the underlying region
                                  is near-uniform (region.std() < 5).
        2. Brightness lift     -- +12..+18 on the HSV V channel,
                                  scaled by intensity.
        3. Frost tint          -- 10-12% alpha blend toward near-white
                                  (240, 240, 245) BGR.
        4. Top inner gradient  -- 0.06 -> 0 over 12 rows, blended
                                  toward white.
        5. Bottom inner shadow -- 0.0 -> 0.15 over 3 rows, blended
                                  toward black.
        6. Top-edge rim        -- 1px near-white along the top edge +
                                  top corner arcs at 50% opacity.

    Mutates `frame` in place; outside the rounded interior of the
    panel the original pixels are preserved exactly.

    Off-frame placements (x < 0, x + w > frame_w, etc.) are silently
    clipped to the visible region; nothing draws outside the canvas.
    When clipping occurs the rim is skipped because its corner arcs
    are only correct when the full panel rectangle is on-screen.

    Args:
        frame:     BGR uint8 image, mutated in place.
        x, y:      top-left of the panel's bounding rect, in frame
                   pixels.
        w, h:      panel width / height.
        radius:    corner radius in pixels.
        intensity: keyword-only.  Scales every visual component.
                   1.0  = status bar (strongest blur, biggest V lift,
                          biggest tint, full rim).
                   0.85 = app icon backgrounds.
                   0.9  = notification cards.
                   Default 1.0 preserves backward compatibility with
                   every existing caller written against the old
                   signature.
    """
    # Clip the placement to frame extents.  Off-screen panels are a
    # silent no-op, matching `draw_text` and `rounded_rect`.  When
    # the origin is negative we still produce the visible slice
    # (consumes mask rows/cols from the appropriate offset).
    frame_h, frame_w = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + w)
    y1 = min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    clip_w = x1 - x0
    clip_h = y1 - y0

    # Mask is cached over the FULL (w, h, radius) shape; we slice
    # whichever portion of it survived the clipping.  Reusing the
    # full-size cache key means a status bar that occasionally drifts
    # off-canvas still hits the cache for its on-screen frames.
    full_mask = _get_rounded_mask(w, h, radius)
    mask = full_mask[
        (y0 - y):(y0 - y + clip_h),
        (x0 - x):(x0 - x + clip_w),
    ]

    # 1-3. Build the brightened + blurred + frosted base surface.
    under = frame[y0:y1, x0:x1]
    surface = _glass_base(under, intensity)

    # 4-5-6. Layer the remaining three Liquid Glass components onto
    #        the surface in panel-local coords.  Each mutates the
    #        surface buffer in place; ordering is gradient -> shadow
    #        -> rim so the rim (topmost visual layer) sits over the
    #        gradient, not under it.
    _glass_top_gradient(surface, mask)
    _glass_bottom_shadow(surface, mask)
    if clip_w == w and clip_h == h:
        # Rim's corner arcs are only correct when the panel is fully
        # on-screen.  On a clipped panel the rim is skipped rather
        # than drawn at the wrong coordinates.
        _glass_rim(surface, w, h, radius)

    # Final composite: paint the surface over the original under-
    # region through the rounded alpha mask.  Outside the rounded
    # interior the original frame pixels survive untouched -- this is
    # what makes the panel's corners read as carved rather than
    # square-stamped onto the wallpaper.
    frame[y0:y1, x0:x1] = _apply_rounded_mask(under, surface, mask)


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
