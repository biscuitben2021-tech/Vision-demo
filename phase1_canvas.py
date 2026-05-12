"""
Phase 1 -- Fullscreen canvas with FPS counter.

Goals:
    * Open an OpenCV window in true fullscreen on the primary display.
    * Fill it with BG_LIGHT (the same warm near-white that backs Apple's
      marketing pages -- never pure #fff).
    * Render an FPS counter in the top-right corner using SF Pro Text
      via PIL, NOT cv2.putText, which produces blocky 1980s blitter
      glyphs no matter what hinting you ask for.
    * Quit on ESC or Q.

This is the foundation phase.  If fullscreen, background colour, and PIL
text rendering all look right here, every subsequent phase inherits good
defaults.  If any of them look wrong -- gray instead of warm near-white,
jaggy text, the menu bar peeking through -- STOP and fix it here.  They
do not get easier to debug once the tile grid is sitting on top.

Module color-space convention:
    The cv2 pixel buffer is BGR.
    The FPS color is named with `_RGB` because it crosses into PIL.
    Every spot where we cross the cv2/PIL boundary uses the suffix to make
    the conversion explicit, exactly as src.design documents.
"""

from __future__ import annotations

import time
from typing import Final

import cv2
import numpy as np

from src.design import (
    BG_LIGHT_BGR,
    TEXT_TERTIARY_RGB,
    draw_text,
    load_font,
)


# ----------------------------------------------------------------------------
# Window constants
# ----------------------------------------------------------------------------
#
# Making an OpenCV window genuinely fullscreen on macOS is a two-step
# dance:
#
#     cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
#                                  cv2.WINDOW_FULLSCREEN)
#
# WINDOW_NORMAL has to come first.  It tells OpenCV we want a resizable
# window backing -- which on macOS is the only kind that can subsequently
# be promoted to fullscreen via setWindowProperty.  If you skip
# WINDOW_NORMAL (or use WINDOW_AUTOSIZE, which is the cv2 default), the
# fullscreen flag is silently ignored and you get a small floating
# window with the menu bar peeking over the top.  This is a macOS-
# specific OpenCV quirk; the docs do not cover it.

WINDOW_NAME:   Final[str] = "Vision OS"
QUIT_KEY_ESC:  Final[int] = 27
QUIT_KEY_Q_L:  Final[int] = ord("q")
QUIT_KEY_Q_U:  Final[int] = ord("Q")

# Fallback canvas size used for the brief moment after the window opens
# but before macOS reports back its real fullscreen rect.  Picked to be a
# reasonable mid-range Retina logical size; it is replaced with the true
# screen rect within one frame.
DEFAULT_W: Final[int] = 1920
DEFAULT_H: Final[int] = 1080

# FPS counter placement and styling.
FPS_FONT_SIZE: Final[int] = 12   # matches the "Small / footnote" type token
FPS_MARGIN:    Final[int] = 16   # px gap from the top and right edges

# Exponential-moving-average smoothing for the FPS counter.  Pure
# instantaneous FPS bounces between ~58 and ~62 on a healthy 60Hz loop;
# the EMA pins the reading to its steady state with minimal lag.
FPS_EMA_ALPHA: Final[float] = 0.1


def make_fullscreen_window(name: str) -> None:
    """Open `name` as a true fullscreen, resizable-backing OpenCV window."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
    )


def screen_size(window_name: str) -> tuple[int, int]:
    """Return the (width, height) of the realized fullscreen window.

    cv2.getWindowImageRect returns (x, y, w, h) but can briefly report
    a (0, 0, 0, 0) rect immediately after promotion to fullscreen --
    macOS is still animating the transition.  We fall back to DEFAULT_W /
    DEFAULT_H for that one frame rather than blocking on it; the next
    loop iteration will pick up the real values.
    """
    rect = cv2.getWindowImageRect(window_name)
    if not rect:
        return DEFAULT_W, DEFAULT_H
    _, _, w, h = rect
    if w <= 0 or h <= 0:
        return DEFAULT_W, DEFAULT_H
    return w, h


def make_canvas(width: int, height: int) -> np.ndarray:
    """Return a fresh BGR canvas painted with BG_LIGHT.

    BG_LIGHT is #fbfbfd, never pure #ffffff.  Apple's restraint: pure
    white reads as a fluorescent bulb and crushes the highlight on every
    rounded tile we draw later; the warm near-white gives the design
    room to breathe.  This single line is the foundation the entire
    aesthetic stands on.

    Shape is (h, w, 3) uint8, the canonical cv2 BGR buffer layout.
    """
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = BG_LIGHT_BGR
    return canvas


def render_fps(
    canvas: np.ndarray,
    fps_text: str,
    font,  # PIL ImageFont; type left implicit to avoid pulling PIL into the signature
    canvas_w: int,
) -> None:
    """Draw the FPS string right-aligned to (canvas_w - FPS_MARGIN, FPS_MARGIN)."""
    # `font.getlength` gives the rendered advance width -- the proper
    # number to subtract from the right edge for tight right-alignment.
    # getbbox would over-include side bearings and leave the text drifted
    # a couple pixels off the edge.
    text_w = int(font.getlength(fps_text))
    draw_text(
        canvas,
        fps_text,
        x=canvas_w - text_w - FPS_MARGIN,
        y=FPS_MARGIN,
        color_rgb=TEXT_TERTIARY_RGB,
        font=font,
    )


def main() -> None:
    """Run the Phase 1 fullscreen loop until ESC or Q is pressed."""
    make_fullscreen_window(WINDOW_NAME)
    fps_font = load_font(role="text", size=FPS_FONT_SIZE)

    width, height = screen_size(WINDOW_NAME)
    canvas = make_canvas(width, height)

    # Seed `last_t` one frame in the past so the first dt is non-zero and
    # we don't get a 1e6-fps spike that takes the EMA dozens of frames
    # to bleed off.  60 Hz is a reasonable assumed cadence.
    last_t: float = time.perf_counter() - (1.0 / 60.0)
    fps_ema: float = 0.0

    while True:
        # Adapt the canvas if the display rect changed (e.g. user plugged
        # in a projector during the demo).  In the steady state this is a
        # no-op; we just wipe and reuse the same buffer.
        cur_w, cur_h = screen_size(WINDOW_NAME)
        if (cur_w, cur_h) != (width, height):
            width, height = cur_w, cur_h
            canvas = make_canvas(width, height)
        else:
            # Repaint background to wipe the previous frame's FPS string.
            # Cheap: a 1920x1080x3 fill is ~6 MB of memcpy, trivial next
            # to imshow.  No need for a scissored rect here in Phase 1.
            canvas[:, :] = BG_LIGHT_BGR

        # FPS measurement.  perf_counter is monotonic and sub-microsecond
        # on Apple Silicon; ideal for a per-frame timer.
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

        # Right-aligned tertiary text, top-right corner.
        render_fps(canvas, f"{fps_ema:5.1f} fps", fps_font, width)

        cv2.imshow(WINDOW_NAME, canvas)

        # waitKey(1) gives the event loop ~1ms to pump.  Without this
        # call cv2 windows on macOS do not refresh -- imshow alone is
        # not enough.  The mask is the standard idiom for normalising
        # the returned int to a plain ASCII code.
        key = cv2.waitKey(1) & 0xFF
        if key in (QUIT_KEY_ESC, QUIT_KEY_Q_L, QUIT_KEY_Q_U):
            break

        # If the user closed the window some other way (e.g. cmd-W in a
        # rare non-fullscreen state), bail out gracefully instead of
        # spinning forever on a window that no longer exists.
        if cv2.getWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_VISIBLE,
        ) < 1.0:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
