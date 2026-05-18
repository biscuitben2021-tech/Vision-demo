"""
Fake app contents for the Phase 6 "tap a tile, see a fullscreen app" state.

Each public function in this module renders ONE app's faked content into a
caller-supplied BGR frame.  They are pure "fill this rect with this app's
pixels" renderers: no state, no animation, no input handling.  Phase 6's
main loop owns the state machine that decides WHICH app to render; this
module just paints whichever one main() points it at.

Signature contract (every render function):

    def render_<app_id>(frame: np.ndarray, w: int, h: int) -> None

    frame:  BGR uint8 image of shape (h, w, 3).  Mutated in place.
    w, h:   width and height of the frame in pixels.  Passed explicitly so
            each renderer can lay itself out responsively without
            re-querying frame.shape (and accidentally swapping the two,
            which is the single most common visual bug in this whole
            codebase -- frame.shape is (h, w, ...), not (w, h, ...)).

The eight render functions paint mock-ups for the eight home-screen apps:
Safari, Photos, Music, Notes, Mail, Calendar, Settings, Demo.  Each is
designed to read as "an Apple app the audience already knows" without
asking the user to interact with it.  They are NOT functional apps -- the
date is hard-coded, the song list is hard-coded, the inbox is hard-coded.
The demo is a stage set, not a product.

Module color-space convention:
    The cv2 pixel buffer is BGR.  Every helper in this file accepts and
    returns BGR tuples; PIL crosses into the module only through
    `draw_text` (which itself flips RGB->BGR internally).  Constants
    imported with a `_BGR` suffix go straight to cv2 calls; constants
    with `_RGB` go to draw_text.  Same convention as every other file
    in src/.

Each render function caps at 60 lines per the Phase 6 prompt.  Where
content -- like Mail's inbox rows -- needs more lines, the per-row work
is split into a private helper sibling at module scope (also < 30 lines).
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from src.design import (
    ACCENT_LIGHT_BGR,
    ACCENT_LIGHT_RGB,
    BG_DARK_BGR,
    BG_LIGHT_BGR,
    BG_NEUTRAL_BGR,
    CTA_GAP,
    GAP_TILE,
    GAP_VIEWPORT,
    HAIRLINE_BGR,
    RADIUS_TILE_SMALL,
    TEXT_MUTED_RGB,
    TEXT_ON_DARK_RGB,
    TEXT_ON_LIGHT_RGB,
    TEXT_TERTIARY_RGB,
    draw_text,
    load_font,
)
from src.icons import draw_glass_panel
from src.tiles import rounded_rect


# ============================================================================
# Font cache -- per process, lazy-loaded
# ============================================================================
#
# Same memoisation pattern src/tiles.py and src/icons.py use.  PIL truetype
# loads are not free; doing them every frame on every render call adds up
# fast when the user is hopping between apps.  All eight apps share a tiny
# pool of typographic sizes (titles at 32-64px, body at 14-21px), so a
# single shared cache covers every renderer here.
#
# The keys are (role, size) pairs and the values are the FreeTypeFont
# objects load_font hands back.  Cache lives on the function object so
# the module surface stays free of mutable globals -- same exception
# CLAUDE.md tolerates elsewhere (a pure cache that returns the same
# object on the same key has no behavioural side effects).

def _get_font(role: str, size: int):
    """Return a cached PIL truetype font for (role, size).

    Internally, the cache is a dict on this function's __dict__.  We
    look up by tuple key; on miss, we load and store.  No locking
    needed -- cv2's main loop is single-threaded and the mouse
    callback only ever reads HoverState, not fonts.
    """
    cache = getattr(_get_font, "_cache", None)
    if cache is None:
        cache = {}
        _get_font._cache = cache  # type: ignore[attr-defined]
    key = (role, size)
    if key not in cache:
        cache[key] = load_font(role=role, size=size)
    return cache[key]


# ============================================================================
# Small geometry helpers -- shared by multiple render functions
# ============================================================================
#
# Each helper is < 30 lines and exists once here rather than copy-pasted
# into the renderers.  They are module-private (underscore prefix); the
# public surface of this file is the eight render_<id> functions plus
# RENDERERS at the bottom.


def _fill(frame: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    """Fill `frame` edge-to-edge with `color_bgr`.

    A one-liner, but extracted so the call sites read as "paint the
    wallpaper" rather than "slice-assign the whole array".  The
    alternative `frame[:, :] = color` is faster than `cv2.rectangle`
    for whole-frame fills because numpy broadcasts a tuple over a
    contiguous block.
    """
    frame[:, :] = color_bgr


def _measure_height(text: str, font) -> int:
    """Return the visible glyph height for `text` rendered with `font`.

    Wraps `font.getbbox` so callers don't have to remember which two of
    the four return values are the vertical pair.  We deliberately use
    `bottom - top` (the visible glyph extent) rather than `font.size`
    (the nominal point size) -- the latter is consistently off by a
    few pixels and is what causes "almost vertically centred" bugs.
    """
    _, top, _, bottom = font.getbbox(text)
    return bottom - top


# ============================================================================
# Safari -- frozen apple.com-style hero
# ============================================================================
#
# Pattern, top to bottom in the viewport's middle band:
#
#     [Eyebrow]      "iPhone 16 Pro"          (21px, on-light, centered)
#     [H1]           A magical new way ...    (Display Semibold 64px, on-light)
#                    to interact with iPhone. (wraps to a second line, centered)
#     [Subhead]      "Hello, Apple Intelligence."  (21px, muted)
#     [CTA row]      Learn more >    Buy >    (17px, accent, centered)
#
# The hero sits ~30% from the top of the viewport -- enough room above
# the eyebrow that the page reads as airy "marketing page", not "app
# window with content jammed against the chrome".

def _wrap_h1(text: str, font, max_w: int) -> list[str]:
    """Greedy word-wrap `text` to at most two lines fitting `max_w`.

    Same algorithm as src/tiles._wrap_two_lines (intentionally a
    duplicate -- the apps module is meant to be self-contained for
    Phase 6 to demo without an `import _wrap_two_lines` dance, and
    inlining the 12-line greedy split keeps the dependency graph
    shallow).  Marketing copy that overflows the second line is left
    as overflow -- loud failure beats silent truncation.
    """
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if int(font.getlength(candidate)) <= max_w or not current:
            current = candidate
            continue
        if len(lines) >= 1:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines or [""]


def render_safari(frame: np.ndarray, w: int, h: int) -> None:
    """Render the frozen apple.com-style Safari hero.

    Five vertically stacked centered elements: eyebrow, two-line H1,
    subhead, CTA row.  Vertical anchor is ~30% down the viewport for
    the eyebrow; each subsequent block sits below the previous with a
    fixed 12 / 12 / 24px rhythm (eyebrow -> H1, H1 -> subhead,
    subhead -> CTAs) -- the same rhythm Phase 3's marketing tile uses
    internally, just laid out on the full viewport rather than a tile.
    """
    _fill(frame, BG_LIGHT_BGR)

    eyebrow_font  = _get_font("text",    21)
    headline_font = _get_font("display", 64)
    subhead_font  = _get_font("text",    21)
    cta_font      = _get_font("text",    17)

    # Eyebrow sits ~30% down the viewport.  Centering uses align="center"
    # so we just pass the canvas midpoint as the anchor x.
    cx = w // 2
    cursor_y = int(h * 0.30)
    draw_text(frame, "iPhone 16 Pro",
              x=cx, y=cursor_y,
              color_rgb=TEXT_ON_LIGHT_RGB, font=eyebrow_font, align="center")
    cursor_y += _measure_height("iPhone 16 Pro", eyebrow_font) + 12

    # H1 wraps to two lines if it exceeds 80% of the viewport width.
    # 80% leaves comfortable side gutters at common screen sizes
    # (1920, 2560) without the wrap collapsing on a small laptop.
    max_h1_w = int(w * 0.80)
    headline = "A magical new way to interact with iPhone."
    lines = _wrap_h1(headline, headline_font, max_h1_w)
    line_h = int(round(headline_font.size * 1.05))   # H2/H1 line-height 1.05
    for i, line in enumerate(lines):
        draw_text(frame, line,
                  x=cx, y=cursor_y + i * line_h,
                  color_rgb=TEXT_ON_LIGHT_RGB, font=headline_font,
                  align="center")
    cursor_y += (len(lines) - 1) * line_h
    cursor_y += _measure_height(lines[-1], headline_font) + 12

    # Subhead: muted, immediately below the H1 stack.
    subhead = "Hello, Apple Intelligence."
    draw_text(frame, subhead,
              x=cx, y=cursor_y,
              color_rgb=TEXT_MUTED_RGB, font=subhead_font, align="center")
    cursor_y += _measure_height(subhead, subhead_font) + 24

    # CTA row: "Learn more >    Buy >".  Chevron is the literal U+203A;
    # the two CTAs are spaced CTA_GAP apart and the pair is centered
    # within the viewport.  Same idiom Phase 3's _draw_cta_row uses.
    left  = "Learn more ›"
    right = "Buy ›"
    left_w  = int(round(cta_font.getlength(left)))
    right_w = int(round(cta_font.getlength(right)))
    group_w = left_w + CTA_GAP + right_w
    start_x = cx - group_w // 2
    draw_text(frame, left,  x=start_x, y=cursor_y,
              color_rgb=ACCENT_LIGHT_RGB, font=cta_font, align="left")
    draw_text(frame, right, x=start_x + left_w + CTA_GAP, y=cursor_y,
              color_rgb=ACCENT_LIGHT_RGB, font=cta_font, align="left")


# ============================================================================
# Photos -- 4x3 grid of solid muted-color tiles
# ============================================================================
#
# A faked photo library: 12 solid-colour rounded rects standing in for
# thumbnails.  The colours are warm Apple-ish greys / blues / beiges that
# read as photo tones rather than as UI swatches.

# 12 muted "photo" tones in BGR order.  Chosen to feel like film stills:
# warm desert beige, sky blue, dusty rose, mid grey, etc.  Each one sits
# in the same lightness band so the grid reads as a coherent library
# rather than a rainbow palette.
#
# BUG FIX: 10 of these tuples were previously stored as (R, G, B) in a
# slot named `_PHOTO_COLORS_BGR` and passed to rounded_rect's color_bgr
# argument -- a BGR/RGB byte-order mixup.  Result: every "warm beige"
# rendered as a cool blue-grey and every "dusk blue" rendered as a
# warm tan, because the channels were transposed before cv2 read them.
# Each warm/cool labelled tuple below has been reversed to true BGR so
# the labels match the rendered colour.  The "bleached cloud" and
# "paper" entries were ambiguous / already correct and are left untouched.
_PHOTO_COLORS_BGR: Final[list[tuple[int, int, int]]] = [
    (180, 200, 210),   # warm grey-beige
    (200, 215, 220),   # bone
    (150, 165, 180),   # deep beige
    (165, 180, 200),   # sand
    (195, 175, 155),   # dusk blue
    (215, 200, 185),   # pale sky
    (215, 220, 225),   # bleached cloud  (neutral; unchanged)
    (185, 175, 170),   # cool grey
    (215, 205, 195),   # winter morning
    (195, 180, 165),   # dawn
    (170, 155, 140),   # storm
    (225, 230, 235),   # paper           (already BGR-correct; unchanged)
]


def render_photos(frame: np.ndarray, w: int, h: int) -> None:
    """Render the Library: title plus a 4-column x 3-row grid of colour tiles.

    The title anchors at (GAP_VIEWPORT, GAP_VIEWPORT + 8); the grid
    starts >=24px below the title's last glyph bottom and spans the
    full width minus a GAP_VIEWPORT gutter on each side, with GAP_TILE
    between cells.  The grid's row height is computed to fill the
    remaining viewport height with the same 3 rows, so the tiles look
    proportional regardless of canvas aspect ratio.
    """
    _fill(frame, BG_LIGHT_BGR)

    title_font = _get_font("display", 48)
    title = "Library"
    title_x = GAP_VIEWPORT
    title_y = GAP_VIEWPORT + 8
    draw_text(frame, title, x=title_x, y=title_y,
              color_rgb=TEXT_ON_LIGHT_RGB, font=title_font, align="left")

    # Grid origin sits 24px below the title's visible bottom; the spec
    # says ">= 24px gap" so we pad by getbbox-bottom plus 24.
    title_h = _measure_height(title, title_font)
    grid_y0 = title_y + title_h + 24
    grid_x0 = GAP_VIEWPORT

    cols, rows = 4, 3
    grid_w = w - 2 * GAP_VIEWPORT
    grid_h = h - grid_y0 - GAP_VIEWPORT
    cell_w = (grid_w - (cols - 1) * GAP_TILE) // cols
    cell_h = (grid_h - (rows - 1) * GAP_TILE) // rows
    if cell_w <= 0 or cell_h <= 0:
        return   # canvas too small for a meaningful grid; bail out cleanly

    # Twelve solid colour tiles.  rounded_rect handles the corner AA.
    # _PHOTO_COLORS_BGR is length 12, exactly cols*rows; we iterate in
    # roster order so a future-me reading the list sees the on-screen
    # arrangement at a glance.
    for i in range(cols * rows):
        col = i % cols
        row = i // cols
        cx = grid_x0 + col * (cell_w + GAP_TILE)
        cy = grid_y0 + row * (cell_h + GAP_TILE)
        rounded_rect(frame, cx, cy, cell_w, cell_h,
                     radius=RADIUS_TILE_SMALL,
                     color_bgr=_PHOTO_COLORS_BGR[i])


# ============================================================================
# Music -- dark playlist with a now-playing bar
# ============================================================================
#
# Six song rows over a black wallpaper, capped by a translucent
# "now playing" glass bar pinned at the bottom.  Each row is a 60px-tall
# strip showing a 40px swatch + song title + artist + duration.

# Six tracks for the fake playlist.  Format: (title, artist, duration).
# Order matters: the first track is the "now playing" candidate the
# bottom bar shows.  Curated to feel like a personal late-night mix --
# nothing that screams "stock library" or "demo content".
_MUSIC_TRACKS: Final[list[tuple[str, str, str]]] = [
    ("Bloom",       "Radiohead",     "5:15"),
    ("Take Five",   "Dave Brubeck",  "5:24"),
    ("Lovers Rock", "TV Girl",       "3:33"),
    ("Heroes",      "David Bowie",   "6:07"),
    ("Time",        "Pink Floyd",    "7:01"),
    ("Strobe",      "deadmau5",     "10:32"),
]

# Per-track swatch colours in BGR.  Match the muted-photo palette in
# tone but lean cooler since the wallpaper is pure black.  Index aligns
# with _MUSIC_TRACKS so swatch i sits to the left of track i.
_MUSIC_SWATCHES_BGR: Final[list[tuple[int, int, int]]] = [
    (140, 100, 80),   # ocean
    (90, 110, 160),   # amber
    (110, 90, 150),   # rose
    (150, 120, 90),   # dusk
    (95, 95, 130),    # heather
    (140, 80, 200),   # neon plum
]


def _draw_music_row(frame: np.ndarray, w: int, y: int,
                    title: str, artist: str, duration: str,
                    swatch_bgr: tuple[int, int, int],
                    title_font, meta_font, duration_font) -> None:
    """Paint one 60px-tall song row at vertical offset `y`.

    Layout left-to-right: 24px gutter -> 40x40 swatch -> 16px gap ->
    title -> 12px gap -> artist (slightly muted) -> ... -> duration
    flush against the right edge with a 24px gutter.

    Pulled out as a sibling helper so `render_music` stays comfortably
    under the 60-line cap; the dispatching loop is then a four-liner.
    """
    swatch_size = 40
    swatch_x = 24
    swatch_y = y + (60 - swatch_size) // 2
    rounded_rect(frame, swatch_x, swatch_y, swatch_size, swatch_size,
                 radius=RADIUS_TILE_SMALL, color_bgr=swatch_bgr)

    # Title and artist sit on the same baseline; vertical centre is
    # the row midline at `y + 30`.  Subtract half the title's bbox
    # height so the visible glyphs sit centred rather than the font's
    # nominal baseline.
    title_h = _measure_height(title, title_font)
    text_y = y + (60 - title_h) // 2
    title_x = swatch_x + swatch_size + 16
    draw_text(frame, title, x=title_x, y=text_y,
              color_rgb=TEXT_ON_DARK_RGB, font=title_font, align="left")

    artist_x = title_x + int(title_font.getlength(title)) + 12
    draw_text(frame, artist, x=artist_x, y=text_y,
              color_rgb=TEXT_MUTED_RGB, font=meta_font, align="left")

    # Duration aligns to the right edge minus a 24px gutter.
    draw_text(frame, duration, x=w - 24, y=text_y,
              color_rgb=TEXT_MUTED_RGB, font=duration_font, align="right")


def _draw_music_now_playing(frame: np.ndarray, w: int, h: int) -> None:
    """Paint the bottom glass bar showing the currently playing track.

    80px tall, full width minus 24px gutter on each side, sat
    GAP_VIEWPORT from the bottom.  Inside: a 56px swatch, the track
    title + artist, and a thin progress bar showing ~30% played.
    """
    bar_h = 80
    bar_w = w - 2 * 24
    bar_x = 24
    bar_y = h - bar_h - GAP_VIEWPORT
    draw_glass_panel(frame, x=bar_x, y=bar_y,
                     w=bar_w, h=bar_h, radius=RADIUS_TILE_SMALL)

    # Now-playing track is _MUSIC_TRACKS[0].  Swatch + title + artist
    # mirror a row, just at slightly larger scale.
    title_font = _get_font("text", 17)
    artist_font = _get_font("text", 14)
    swatch_size = 56
    swatch_x = bar_x + 12
    swatch_y = bar_y + (bar_h - swatch_size) // 2
    rounded_rect(frame, swatch_x, swatch_y, swatch_size, swatch_size,
                 radius=RADIUS_TILE_SMALL,
                 color_bgr=_MUSIC_SWATCHES_BGR[0])

    title, artist, _ = _MUSIC_TRACKS[0]
    text_x = swatch_x + swatch_size + 16
    title_h = _measure_height(title, title_font)
    artist_h = _measure_height(artist, artist_font)
    block_h = title_h + 4 + artist_h
    text_y = bar_y + (bar_h - block_h) // 2
    draw_text(frame, title, x=text_x, y=text_y,
              color_rgb=TEXT_ON_DARK_RGB, font=title_font, align="left")
    draw_text(frame, artist, x=text_x, y=text_y + title_h + 4,
              color_rgb=TEXT_MUTED_RGB, font=artist_font, align="left")

    # Thin progress bar across the bottom of the glass panel.  ~30%
    # played feels like "they just started this one" without being so
    # short it disappears.
    track_y = bar_y + bar_h - 14
    track_x0 = text_x
    track_x1 = bar_x + bar_w - 24
    cv2.line(frame, (track_x0, track_y), (track_x1, track_y),
             HAIRLINE_BGR, thickness=2, lineType=cv2.LINE_AA)
    fill_x = track_x0 + int((track_x1 - track_x0) * 0.30)
    cv2.line(frame, (track_x0, track_y), (fill_x, track_y),
             (247, 245, 245), thickness=2, lineType=cv2.LINE_AA)


def render_music(frame: np.ndarray, w: int, h: int) -> None:
    """Render the Music app: six song rows plus a bottom now-playing bar.

    Wallpaper is pure black.  Song rows stack from the top with 60px
    pitch; the now-playing bar floats over them at the bottom.  The
    rows could overlap the bar at narrow canvas heights -- by Phase 6
    the realistic viewport is at least 800px tall so we accept that
    edge case rather than complicate the layout math.
    """
    _fill(frame, BG_DARK_BGR)

    title_font = _get_font("text", 17)
    meta_font  = _get_font("text", 14)
    duration_font = _get_font("text", 14)

    # Row 0 sits 80px down the canvas -- below where a future status
    # bar would land and well clear of the close button at top-left.
    for i, (title, artist, duration) in enumerate(_MUSIC_TRACKS):
        y = 80 + i * 60
        _draw_music_row(frame, w, y, title, artist, duration,
                        _MUSIC_SWATCHES_BGR[i],
                        title_font, meta_font, duration_font)

    _draw_music_now_playing(frame, w, h)


# ============================================================================
# Notes -- single note open
# ============================================================================
#
# A paper-feeling page with a note title and four lorem paragraphs.  The
# canvas is BG_LIGHT (paper); lines are kept single-line on the wide
# canvas (no word-wrap, per the prompt) so a narrow window just clips.

_NOTES_PARAGRAPHS: Final[list[str]] = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
]


def render_notes(frame: np.ndarray, w: int, h: int) -> None:
    """Render the Notes app: a title + four lorem paragraphs.

    Title sits at (GAP_VIEWPORT + 24, GAP_VIEWPORT + 24) -- a slightly
    larger inset than Photos / Mail so the page reads as a note WITHIN
    a writing app, not a header bar.  Paragraphs follow at 12px
    vertical pitch using the visible bbox height of each line; if
    a line happens to clip on the right (narrow window) we leave the
    overflow alone per the prompt's "keep it simple" guidance.
    """
    _fill(frame, BG_LIGHT_BGR)

    title_font = _get_font("display", 24)
    body_font  = _get_font("text",    17)

    title = "Vision Demo"
    title_x = GAP_VIEWPORT + 24
    title_y = GAP_VIEWPORT + 24
    draw_text(frame, title, x=title_x, y=title_y,
              color_rgb=TEXT_ON_LIGHT_RGB, font=title_font, align="left")

    # 16px gap from title bottom to first paragraph top.
    cursor_y = title_y + _measure_height(title, title_font) + 16
    for paragraph in _NOTES_PARAGRAPHS:
        draw_text(frame, paragraph, x=title_x, y=cursor_y,
                  color_rgb=TEXT_ON_LIGHT_RGB, font=body_font, align="left")
        cursor_y += _measure_height(paragraph, body_font) + 12


# ============================================================================
# Mail -- inbox list
# ============================================================================
#
# Eight sender/preview rows alternating row backgrounds; each row has an
# avatar circle, sender name, preview, and date.
#
# KNOWN COMPROMISE: the prompt asks for "Sender (Text Semibold 17px ->
# Text Regular 17px since we don't have semibold for 'text')".  load_font
# in src/design.py exposes only `text -> Regular` for body sizes; no
# Text Semibold weight is wired up.  We use Text Regular for the sender
# name and accept that the hierarchy cue between sender and preview
# comes from colour (TEXT_ON_LIGHT vs TEXT_MUTED) rather than weight.
# This is the same compromise documented in src/tiles.EYEBROW_SIZE.

_MAIL_ROWS: Final[list[tuple[str, str, str]]] = [
    ("Tim Cook",          "Welcome to the team.",    "Today"),
    ("Jony Ive",          "Re: enclosure radius",    "Today"),
    ("Susan",             "Apple Park reservation",  "Yesterday"),
    ("WWDC",              "Save the date",           "Mon"),
    ("Mom",               "see you tonight",         "Mon"),
    ("Stripe",            "Receipt: $1,299.00",      "Sun"),
    ("The New York Times","Morning briefing",        "Sat"),
    ("GitHub",            "Security alert",          "Fri"),
]

# Avatar tints in BGR; one per row index.  Chosen warm/cool alternation
# so adjacent rows are visually distinct without any single colour
# dominating.  Could be derived from a hash of the sender name but
# hard-coding lets each row's colour be tuned per the demo's palette.
_MAIL_AVATARS_BGR: Final[list[tuple[int, int, int]]] = [
    (245, 175, 70),   # blue (Mail tint -- "you")
    (130, 130, 200),  # rose
    (180, 200, 130),  # mint
    (200, 130, 200),  # lilac
    (140, 200, 230),  # peach
    (210, 180, 110),  # cyan
    (90, 90, 90),     # NYT slate
    (110, 110, 110),  # github grey
]


def _draw_mail_row(frame: np.ndarray, w: int, y: int,
                   sender: str, preview: str, date: str,
                   avatar_bgr: tuple[int, int, int],
                   row_bg_bgr: tuple[int, int, int]) -> None:
    """Paint one 64px-tall mail row at vertical offset `y`.

    Row background fills first (so the alternating-stripe look reads
    as a list), then the avatar circle, then the three text labels.
    Pulled out as a sibling helper so `render_mail` stays under
    its 60-line cap.
    """
    row_h = 64
    cv2.rectangle(frame, (0, y), (w, y + row_h), row_bg_bgr,
                  thickness=-1)

    # Avatar: 40px filled circle.  Centre x sits 24 + 20 = 44 from
    # the left edge so the circle sits flush in a comfortable gutter.
    av_r = 20
    av_cx = 24 + av_r
    av_cy = y + row_h // 2
    cv2.circle(frame, (av_cx, av_cy), av_r, avatar_bgr,
               thickness=-1, lineType=cv2.LINE_AA)

    sender_font  = _get_font("text", 17)   # Regular -- see KNOWN COMPROMISE
    preview_font = _get_font("text", 14)
    date_font    = _get_font("text", 13)

    # Sender + preview stack: sender on top, preview just below.
    text_x = av_cx + av_r + 16
    sender_h  = _measure_height(sender,  sender_font)
    preview_h = _measure_height(preview, preview_font)
    block_h = sender_h + 4 + preview_h
    sender_y = y + (row_h - block_h) // 2
    draw_text(frame, sender, x=text_x, y=sender_y,
              color_rgb=TEXT_ON_LIGHT_RGB, font=sender_font, align="left")
    draw_text(frame, preview, x=text_x, y=sender_y + sender_h + 4,
              color_rgb=TEXT_MUTED_RGB, font=preview_font, align="left")

    # Date right-aligned with a 24px gutter.
    date_y = y + (row_h - _measure_height(date, date_font)) // 2
    draw_text(frame, date, x=w - 24, y=date_y,
              color_rgb=TEXT_TERTIARY_RGB, font=date_font, align="right")


def render_mail(frame: np.ndarray, w: int, h: int) -> None:
    """Render the Mail inbox: title bar + 8 alternating rows."""
    _fill(frame, BG_LIGHT_BGR)

    title_font = _get_font("display", 32)
    title = "Inbox"
    title_x = GAP_VIEWPORT + 24
    title_y = GAP_VIEWPORT + 16
    draw_text(frame, title, x=title_x, y=title_y,
              color_rgb=TEXT_ON_LIGHT_RGB, font=title_font, align="left")

    list_y0 = title_y + _measure_height(title, title_font) + 24

    # Eight rows, alternating BG_LIGHT / BG_NEUTRAL.  Even rows use
    # the lighter background so row 0 (Tim Cook) is the brighter of
    # the two; this matches Apple Mail's default zebra striping.
    for i, (sender, preview, date) in enumerate(_MAIL_ROWS):
        row_y = list_y0 + i * 64
        row_bg = BG_LIGHT_BGR if i % 2 == 0 else BG_NEUTRAL_BGR
        _draw_mail_row(frame, w, row_y, sender, preview, date,
                       _MAIL_AVATARS_BGR[i], row_bg)


# ============================================================================
# Calendar -- November 2025 month view
# ============================================================================
#
# 7-column x 5-row grid.  Week starts Sunday.  Nov 2025 calendar:
#
#     Sun Mon Tue Wed Thu Fri Sat
#                              1     <- row 0, col 6
#      2   3   4   5   6   7   8     <- row 1
#      9  10  11  12  13  14  15
#     16  17  18  19  20  21  22
#     23  24  25  26  27  28  29
#     30                              <- row 5, col 0 (overflow row)
#
# We render rows 0..4 (Nov 1..29) for the canonical 5-row grid;
# Nov 30 lives on a sixth row that we draw too so the month doesn't
# silently truncate.  The "today" callout circles Nov 12 -- the centre
# of the month, requested by the prompt as a deterministic anchor.

_CALENDAR_DAY_HEADERS: Final[list[str]] = [
    "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
]


def _calendar_grid_origins(w: int, h: int, list_y0: int) -> tuple[int, int, int, int]:
    """Return (origin_x, origin_y, cell_w, cell_h) for the month grid.

    Splits out the geometry so render_calendar stays focussed on
    drawing.  Returns floats rounded to int so the placeholder
    positioning matches between the header row and the data rows.
    """
    cols, rows = 7, 6
    grid_w = w - 2 * GAP_VIEWPORT
    grid_h = h - list_y0 - GAP_VIEWPORT
    cell_w = grid_w // cols
    cell_h = max(48, grid_h // (rows + 1))   # +1 leaves space for the header
    origin_x = GAP_VIEWPORT + (grid_w - cell_w * cols) // 2
    origin_y = list_y0
    return origin_x, origin_y, cell_w, cell_h


def render_calendar(frame: np.ndarray, w: int, h: int) -> None:
    """Render the November 2025 month view, with today (Nov 12) circled."""
    _fill(frame, BG_LIGHT_BGR)

    title_font  = _get_font("display", 32)
    day_font    = _get_font("text", 17)
    header_font = _get_font("text", 13)

    title = "November 2025"
    title_x = GAP_VIEWPORT + 24
    title_y = GAP_VIEWPORT + 16
    draw_text(frame, title, x=title_x, y=title_y,
              color_rgb=TEXT_ON_LIGHT_RGB, font=title_font, align="left")
    list_y0 = title_y + _measure_height(title, title_font) + 24

    origin_x, origin_y, cell_w, cell_h = _calendar_grid_origins(w, h, list_y0)

    # Day headers across the top row.  Each header sits centred within
    # its column at the top of the header band, in muted tertiary grey.
    header_h = _measure_height("Sun", header_font)
    for col, name in enumerate(_CALENDAR_DAY_HEADERS):
        hx = origin_x + col * cell_w + cell_w // 2
        draw_text(frame, name, x=hx, y=origin_y,
                  color_rgb=TEXT_TERTIARY_RGB, font=header_font, align="center")

    # Date cells.  Nov 1 lands at row 0 col 6 (Saturday); subsequent
    # days wrap weekly.  We pre-compute (row, col) for each day so the
    # drawing loop is a flat 30-iteration count.
    grid_y0 = origin_y + header_h + 12

    today = 12   # the date to circle, per prompt
    for day in range(1, 31):
        # (day + 5) // 7 because day 1 is at col 6 (= (1 - 1 + 6) % 7
        # gives col 6 on row 0); generalising: position_in_grid = day + 5,
        # row = position // 7, col = position % 7 with `position = day - 1 + 6`.
        position = (day - 1) + 6
        row = position // 7
        col = position % 7
        cx = origin_x + col * cell_w + cell_w // 2
        cy = grid_y0 + row * cell_h + cell_h // 2

        if day == today:
            # Today: filled accent-blue circle behind a white digit.
            # ACCENT_LIGHT_BGR is the BGR-suffix design token for the
            # canonical "Learn more >" link blue (#0066cc).  Using the
            # token rather than the raw tuple keeps every blue in the
            # demo synced if a future palette tweak shifts the hue.
            cv2.circle(frame, (cx, cy), 16, ACCENT_LIGHT_BGR,
                       thickness=-1, lineType=cv2.LINE_AA)
            text_color = TEXT_ON_DARK_RGB
        else:
            text_color = TEXT_ON_LIGHT_RGB
        text = str(day)
        text_y = cy - _measure_height(text, day_font) // 2
        draw_text(frame, text, x=cx, y=text_y,
                  color_rgb=text_color, font=day_font, align="center")


# ============================================================================
# Settings -- pill rows for the most common toggles
# ============================================================================

_SETTINGS_ROWS: Final[list[str]] = [
    "Wi-Fi", "Bluetooth", "Display", "Sound", "General", "About",
]


def render_settings(frame: np.ndarray, w: int, h: int) -> None:
    """Render Settings: title + six pill rows with a right-side chevron.

    Each row is 56px tall, full width minus 48px gutter, with rounded
    corners filled in BG_NEUTRAL.  The prompt suggests glass-ish but
    flat is fine; flat fill matches Apple's iOS Settings cells which
    are themselves opaque rather than translucent.
    """
    _fill(frame, BG_LIGHT_BGR)

    title_font  = _get_font("display", 32)
    row_font    = _get_font("text", 17)
    chevron_font = _get_font("text", 17)

    title = "Settings"
    title_x = GAP_VIEWPORT + 24
    title_y = GAP_VIEWPORT + 16
    draw_text(frame, title, x=title_x, y=title_y,
              color_rgb=TEXT_ON_LIGHT_RGB, font=title_font, align="left")
    list_y0 = title_y + _measure_height(title, title_font) + 24

    row_h = 56
    row_x = 24
    row_w = w - 2 * 24
    for i, label in enumerate(_SETTINGS_ROWS):
        row_y = list_y0 + i * (row_h + GAP_TILE)
        rounded_rect(frame, row_x, row_y, row_w, row_h,
                     radius=RADIUS_TILE_SMALL, color_bgr=BG_NEUTRAL_BGR)

        # Label: 16px from the left edge, vertically centred.
        label_h = _measure_height(label, row_font)
        label_y = row_y + (row_h - label_h) // 2
        draw_text(frame, label, x=row_x + 16, y=label_y,
                  color_rgb=TEXT_ON_LIGHT_RGB, font=row_font, align="left")

        # Chevron right-aligned with a 16px gutter from the row edge.
        chevron = "›"   # U+203A, same chevron used elsewhere
        ch_y = row_y + (row_h - _measure_height(chevron, chevron_font)) // 2
        draw_text(frame, chevron, x=row_x + row_w - 16, y=ch_y,
                  color_rgb=TEXT_TERTIARY_RGB, font=chevron_font,
                  align="right")


# ============================================================================
# Demo -- the splash screen for the demo itself
# ============================================================================
#
# Centered H2, subhead, and a single CTA.  This is what someone sees if
# they tap the "Demo" tile -- the meta-tile that explains the whole
# project.  Voice and tone follow the rest: terse, confident, no emoji.

def render_demo(frame: np.ndarray, w: int, h: int) -> None:
    """Render the Demo splash: centered H2 + subhead + CTA.

    All three blocks are centered both horizontally and vertically as a
    group.  The vertical anchor is computed by stacking the three
    blocks with the spec'd 12 / 24px gaps and then offsetting the top
    so the stack's centre lands at h / 2.
    """
    _fill(frame, BG_LIGHT_BGR)

    h2_font      = _get_font("display", 48)
    subhead_font = _get_font("text", 21)
    cta_font     = _get_font("text", 17)

    h2 = "vision demo loaded"
    subhead = "Tap to begin."
    cta = "Begin ›"

    h2_h      = _measure_height(h2, h2_font)
    subhead_h = _measure_height(subhead, subhead_font)
    cta_h     = _measure_height(cta, cta_font)
    stack_h = h2_h + 12 + subhead_h + 24 + cta_h
    top = (h - stack_h) // 2

    cx = w // 2
    draw_text(frame, h2, x=cx, y=top,
              color_rgb=TEXT_ON_LIGHT_RGB, font=h2_font, align="center")
    sub_y = top + h2_h + 12
    draw_text(frame, subhead, x=cx, y=sub_y,
              color_rgb=TEXT_MUTED_RGB, font=subhead_font, align="center")
    cta_y = sub_y + subhead_h + 24
    draw_text(frame, cta, x=cx, y=cta_y,
              color_rgb=ACCENT_LIGHT_RGB, font=cta_font, align="center")


# ============================================================================
# Public surface
# ============================================================================
#
# Phase 6's main loop dispatches by app_id; the dict below is the
# canonical source of truth for "what apps exist".  Keep it in sync with
# `_ICON_TINTS` in src/icons.py and `APPS` in phase4_home_screen.py --
# a missing key here means tapping that home tile would crash, which
# is the loud-failure mode this codebase prefers.

RENDERERS: Final[dict[str, object]] = {
    "safari":   render_safari,
    "photos":   render_photos,
    "music":    render_music,
    "notes":    render_notes,
    "mail":     render_mail,
    "calendar": render_calendar,
    "settings": render_settings,
    "demo":     render_demo,
}
