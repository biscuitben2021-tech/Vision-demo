"""Canonical demo entry point.  Boots the eye + hand tracking OS.

`run.py` -> `phase8_hand.main()` is the full experience: eye-tracked
cursor, hand-pinch click, the polished OS chrome from Phase 7
underneath.  On the way in, Phase 8 will:

    1. Probe the connected cameras and let you pick one
       (skipped if only one webcam is available).
    2. Open a windowed calibration screen.  Look at the centre dot
       and press SPACE to capture your "looking-at-centre" baseline.
       Press ESC instead to skip calibration and run mouse-only.
    3. Open the fullscreen OS.  Your gaze drives the cursor; pinch
       thumb-to-index with one hand to click whatever the cursor is
       hovering.  Hold the pinch and swipe horizontally to flip
       home pages.

If the camera, MediaPipe, or calibration fails at any step the OS
still boots -- it just falls back to mouse-driven cursor and click,
matching Phase 7.  ESC and Q quit cleanly at any time.

For a mouse-only demo (no camera, no eye tracking), run
`phase7_polish.py` directly instead of `run.py`.
"""

from phase8_hand import main

if __name__ == "__main__":
    main()
