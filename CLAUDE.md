# Vision Demo — Project Context for Claude Code

## What we're building

A fake "Vision OS"-style desktop that runs fullscreen on a MacBook Air M2 (8GB).
The audience sees what looks like an Apple operating system; they never see
the real macOS desktop. Eventually the user controls it by looking (gaze
cursor from the built-in webcam) and pinching (MediaPipe Hands on the same
webcam). For now we're focused on getting the OS visuals right.

The build is in three parts, in this order:

1. **Part 1 — the OS** (pure UI, mouse-driven for testing). Phases 1–7.
2. **Part 2 — eye tracking** (clone the folder, drop in the existing gaze
   model from the `eye-mouse/` project, improve calibration, add an optional
   pretrained mode for guests at the show). Phases 8–12.
3. **Part 3 — hand tracking** (pinch click, pinch drag, both running on
   the same webcam feed as the face tracker). Phases 13–15.

This ordering matters: by the end of Part 1 there's already a shippable
demo. If Parts 2 or 3 break, the user still has a polished fake OS to
show on stage.

## Background and dependencies on other work

This is a continuation of the user's `eye-mouse/` project. They've already
built phases 1–8 of an eye-controlled cursor: Ridge regression
calibration, gaze cursor, blink click/drag. They have a working
`calibration.pkl` file and a `src/` library with `gaze.py`, `smooth.py`,
`calibrate.py`, `blink.py`.

They've also built (on a separate Windows machine) a hand-tracking demo
with pinch/drag/swipe/zoom/rotate.

**Important:** Phases 1–7 of *this* project do NOT depend on those folders.
Part 1 is pure UI work and stands alone. Only at Phase 8 do we copy
`calibration.pkl` and `src/gaze.py` into this folder. The instruction
"clone the folder" in Phase 8 means: snapshot the entire `vision-demo/`
working state to `vision-demo-eye/` so the OS-only version stays safe
while we add tracking.

## Important context about this project

This is for a show. Time-sensitive but also a learning project. Priorities:

1. **Ship-able demo first.** Phase 7 should already feel like a working
   fake OS without any tracking. Part 1 ships even if Parts 2 and 3
   collapse.
2. **Apple-grade visual restraint.** No clutter. No drop shadows on
   tiles. No "fun" gradients. Read the design tokens below and stick to
   them. The look comes from what you *don't* put on screen.
3. **Heavy comments explaining WHY.** Same rule as the eye-mouse
   project. Don't write `# fill background`. Write `# Apple's restraint:
   never pure white. #fbfbfd reads as paper, #ffffff reads as a fluorescent
   bulb.`
4. **Type hints on every function.**
5. **One concept per file.** Don't merge the tile renderer and the
   home-screen layout into one module just because both run in the main
   loop.
6. **Each phase runnable standalone.** `python phase3_tile_grid.py` must
   show the tile grid by itself with no other dependencies.

When you write code, briefly explain in chat what each new piece does and
why. Don't dump a 300-line file with no narration.

## Tech stack

- Python 3.12
- `opencv-python` — fullscreen window, drawing primitives, camera capture
  (later)
- `Pillow` (PIL) — text rendering. OpenCV's built-in fonts cannot render
  SF Pro / Helvetica Neue at the quality this design needs. PIL is an
  image-processing library, not a GUI framework, so it's allowed.
- `numpy` — array math, background blending
- `mediapipe` — face mesh + hands (Part 2 and Part 3 only)
- `scikit-learn` — Ridge regression (Part 2, reused from `eye-mouse/`)
- `pickle` — load `calibration.pkl` (Part 2)

**Still NOT allowed:** PyTorch / TensorFlow (only in Phase 12 behind a
feature flag, optional stretch goal), GUI frameworks (Tkinter, PyQt,
Kivy), Electron, web UI, async frameworks, animation libraries. Everything
renders into a numpy array and gets shown via `cv2.imshow`.

## Apple design tokens — use these literally

Translated from the `apple_SKILL.md` file (web → OpenCV/PIL). Apple's
homepage is the reference. The trick is *restraint*: every tile is composed
like a print ad — single subject, oceans of negative space, no decorative
chrome. Borders, shadows, dividers, and gradients are essentially absent;
visual separation comes from the 16px gap between tiles and the radius
that crops each tile.

These tokens belong in `src/design.py` as constants, imported by every
phase. Don't define colors inline.

### Palette

OpenCV uses BGR. PIL uses RGB. Store each color in both forms in
`src/design.py` so there's never an off-by-one byte order bug. Comment
each one with WHY it's that exact value.

```python
# Page surfaces
BG_LIGHT_RGB    = (251, 251, 253)   # #fbfbfd — never pure white. Pure white is harsh; near-white reads as paper.
BG_LIGHT_BGR    = (253, 251, 251)
BG_DARK_RGB     = (0, 0, 0)         # #000000 — alternates with BG_LIGHT for ~half of tiles
BG_DARK_BGR     = (0, 0, 0)
BG_NEUTRAL_RGB  = (245, 245, 247)   # #f5f5f7 — footers, secondary tiles
BG_NEUTRAL_BGR  = (247, 245, 245)

# Text
TEXT_ON_LIGHT_RGB  = (29, 29, 31)    # #1d1d1f — near-black, never pure black
TEXT_ON_LIGHT_BGR  = (31, 29, 29)
TEXT_ON_DARK_RGB   = (245, 245, 247) # #f5f5f7 — same as BG_NEUTRAL, intentional
TEXT_ON_DARK_BGR   = (247, 245, 245)
TEXT_MUTED_RGB     = (134, 134, 139) # #86868b — subheads, secondary
TEXT_MUTED_BGR     = (139, 134, 134)
TEXT_TERTIARY_RGB  = (110, 110, 115) # #6e6e73 — disclaimers, footnotes
TEXT_TERTIARY_BGR  = (115, 110, 110)

# Accents (the "Learn more ›" link blue)
ACCENT_LIGHT_RGB   = (0, 102, 204)   # #0066cc — links on light surfaces
ACCENT_LIGHT_BGR   = (204, 102, 0)
ACCENT_DARK_RGB    = (41, 151, 255)  # #2997ff — links on dark tiles
ACCENT_DARK_BGR    = (255, 151, 41)
ACCENT_HOVER_RGB   = (0, 119, 237)   # #0077ed — hover state for light-tile links
ACCENT_HOVER_BGR   = (237, 119, 0)

# Hairlines / dividers (rarely used)
HAIRLINE_RGB       = (210, 210, 215) # #d2d2d7 — only in footer and form inputs
HAIRLINE_BGR       = (215, 210, 210)
```

### Typography

On macOS, SF Pro Display is at
`/System/Library/Fonts/SF-Pro-Display-Semibold.otf`. If unavailable, fall
back to `Helvetica Neue` (always installed on Macs). Load with
`PIL.ImageFont.truetype(path, size)`.

| Style | Family | Weight | Size | Color | Notes |
|---|---|---|---|---|---|
| H1 hero | SF Pro Display | Semibold (600) | 80px desktop | TEXT_ON_LIGHT or TEXT_ON_DARK | letter-spacing -0.5px (manual kerning if needed); line-height 1.05 |
| H2 tile headline | SF Pro Display | Semibold (600) | 48px | same | letter-spacing -0.3px |
| H3 eyebrow | SF Pro Text | Semibold (600) | 21px | TEXT_ON_LIGHT or product-family color | line-height 1.19 |
| Body / subhead | SF Pro Text | Regular (400) | 21px | TEXT_MUTED | line-height 1.38 |
| Small / footnote | SF Pro Text | Regular (400) | 12px | TEXT_TERTIARY | line-height 1.33 |

PIL doesn't do native letter-spacing. For headlines where it matters,
either accept the default or render character-by-character with manual
offsets (only if you have time).

### Spacing tokens

```python
GAP_TILE             = 16   # between tiles
GAP_VIEWPORT         = 16   # gutter from tile group to viewport edge
PAD_TILE_TOP         = 80   # top padding inside a tile (desktop)
PAD_TILE_X           = 40   # horizontal padding inside a tile (desktop)
RADIUS_TILE_LARGE    = 24   # corner radius for big tiles
RADIUS_TILE_SMALL    = 18   # corner radius for small/mobile tiles
RADIUS_APP_ICON      = 28   # corner radius for Vision OS app icons
MAX_CONTENT_WIDTH    = 980  # inner content column max width
CTA_GAP              = 24   # gap between the two link CTAs ("Learn more ›  Buy ›")
NAV_HEIGHT           = 44   # the blurred top nav bar
```

### Visual rules — non-negotiable

- Background of the whole OS canvas: BG_LIGHT or BG_DARK (alternate
  across tiles).
- Tiles have **no border, no drop shadow, no outline**. Separation comes
  from the 16px gap and tile background color alone.
- Rounded corners on every tile. OpenCV has no native rounded rect —
  draw a filled rect plus four filled circles at the corners. Wrap this
  in `src/tiles.py` as `rounded_rect(img, x, y, w, h, radius, color)`.
- Vision OS home-screen tiles are **translucent glass** over the desktop
  background. Achieve this with
  `cv2.addWeighted(tile, 0.85, background, 0.15, 0)` blended into a
  rounded mask.
- The "Learn more ›" CTA pattern: blue text + literal `›` chevron
  character (Unicode U+203A), CTA_GAP between CTAs in a row. No
  backgrounds, no underline by default, underline on hover.
- No emoji as decoration anywhere. Plain text glyphs or nothing.
- Headlines max 12 characters per line. Wrap to a second line rather
  than shrinking the font.

### Motion tokens (Phase 5+)

```python
FADE_UP_DURATION_MS    = 800   # tile entry: translateY(24px → 0), opacity 0 → 1
FADE_UP_EASING         = (0.28, 0.11, 0.32, 1)  # Apple's "ease-emphasized" cubic-bezier
HOVER_SCALE_DURATION_MS = 200  # cursor-over-tile: scale 1.0 → 1.02
CTA_COLOR_DURATION_MS  = 200   # link hover color shift
NAV_BLUR_DURATION_MS   = 250   # nav backdrop blur ramp on first scroll
```

Implement easing with a small `ease(t, control_points)` helper in
`src/motion.py`. No animation libraries.

### Reduced motion

Hold `R` at startup to skip all animations (instant fades, no parallax,
no scale on hover). Useful for low-end machines and accessibility.
Document this in the Phase 5 file header.

## Voice and tone for fake-app copy

Same rules as Apple marketing. Confident, terse, aspirational. Never
explain a feature — name it. Sentences are short. No exclamation marks.
No emoji.

Pattern:

> Eyebrow: **Photos** — Headline: **Every memory, instantly.** — Subhead: Your library, at a glance.

> Eyebrow: **Music** — Headline: **Hello, soundtrack.** — Subhead: Pick up where you left off.

Bad: "Now featuring AI-powered photo organization with on-device machine
learning!"

Good: "Hello, memories."

## Phased build plan

### Part 1 — The OS (no tracking)

| Phase | File | What it does |
|-------|------|--------------|
| 1 | `phase1_canvas.py` | Fullscreen window, BG_LIGHT background, FPS counter in the top-right, quits on ESC or Q. Establishes that fullscreen-on-Mac works and the design.py palette is wired. |
| 2 | `phase2_typography.py` | Render H1 "Hello, vision." centered with a muted subhead, using SF Pro Display via PIL. Confirms type rendering looks Apple-grade, not OpenCV-grade. |
| 3 | `phase3_tile_grid.py` | Static 2-up grid of 4 rounded tiles, alternating light/dark. Each tile shows eyebrow + headline + subhead + two chevron CTAs. Builds `src/tiles.py`. |
| 4 | `phase4_home_screen.py` | Vision OS home screen: 6–8 floating glass app tiles in a grid (Safari, Photos, Music, Notes, Mail, Calendar, Settings, plus a "Demo" tile). Top status bar with time. Builds `src/icons.py`. |
| 5 | `phase5_motion.py` | Fade-up on entry, subtle parallax on a hero tile, cursor-driven hover state. Builds `src/motion.py`. **Uses the actual mouse for hover** — gaze comes in Part 2. |
| 6 | `phase6_app_window.py` | Click an app tile → tile expands into a fake app window. Each app shows different fake content (Photos: grid of color swatches; Music: playlist; Notes: lorem text; Mail: inbox list; Safari: a frozen apple.com-style page). X closes. |
| 7 | `phase7_polish.py` | Live clock that updates every second; fake notification slides in from top-right after 10 seconds; smooth transitions between home and app. **End of Part 1 — this is the demo-day fallback.** |

### Part 2 — Eye tracking

Before starting Phase 8: snapshot the working `vision-demo/` folder to
`vision-demo-eye/` (`cp -r vision-demo vision-demo-eye`). Work in the
new folder so the mouse-controlled Part 1 stays intact.

| Phase | File | What it does |
|-------|------|--------------|
| 8 | `phase8_gaze.py` | Copy `calibration.pkl`, `src/gaze.py`, `src/smooth.py` from `eye-mouse/`. Build phase8 based on phase7. Replace the mouse cursor with the gaze cursor. Highlight a tile on hover via gaze. Click is still mouse for now. |
| 9 | `phase9_one_euro.py` | Replace exponential smoothing with a 1€ filter. ~30 lines of public-domain Python, copy in directly — no new library. Cursor should feel noticeably smoother without lagging. |
| 10 | `phase10_recal_5x5.py` | "Recalibrate" mode triggered by holding C for 2 seconds: 5x5 grid (25 points) for better accuracy than the 3x3 in `eye-mouse/`. Save to `calibration.pkl`. |
| 11 | `phase11_guest_cal.py` | "Guest mode" — 3-point ~10-second quick calibration for someone trying the demo at the show (top-left, top-right, bottom-center). Saves to `guest_calibration.pkl` (separate file so it doesn't overwrite). UI prompt walks the guest through it. |
| 12 | `phase12_pretrained.py` | OPTIONAL stretch goal. Load L2CS-Net (PyTorch) for zero-calibration gaze. Gated behind a `--pretrained` flag. **Skip this phase if the show is less than 3 days away** — it's a research project disguised as a feature. Phase 11 covers the same need with no new dependencies. |

### Part 3 — Hand tracking

| Phase | File | What it does |
|-------|------|--------------|
| 13 | `phase13_hands.py` | Add MediaPipe Hands on the same webcam frame as the face mesh. Camera preview shrinks to 320x240 in the bottom-right corner with face + hand mesh drawn on it. Performance check: target 25 fps minimum. |
| 14 | `phase14_pinch_click.py` | Detect pinch via thumb-tip to index-tip distance normalized by hand size. Pinch while gazing at a tile = click that tile (replaces the mouse click). Click animation reuses Phase 6. |
| 15 | `phase15_pinch_drag.py` | Pinch and hold for 300ms+ = grab. Move hand to drag the tile. Release pinch = drop. Used for rearranging home-screen tiles. |

## Final file structure

```
vision-demo/                      (Part 1 working state, mouse-driven)
├── CLAUDE.md
├── README.md
├── requirements.txt
├── phase1_canvas.py
├── phase2_typography.py
├── phase3_tile_grid.py
├── phase4_home_screen.py
├── phase5_motion.py
├── phase6_app_window.py
├── phase7_polish.py
├── assets/
│   ├── SF-Pro-Display-Semibold.otf  # if shipped separately from system fonts
│   ├── SF-Pro-Display-Regular.otf
│   └── (no PNG icons — draw SF-Symbols-style glyphs procedurally)
└── src/
    ├── __init__.py
    ├── design.py       # palette + typography + spacing tokens
    ├── tiles.py        # rounded_rect + tile renderer
    ├── icons.py        # app icon glyphs (Safari, Photos, etc.) drawn in code
    ├── motion.py       # ease curves, fade_up, hover scale
    ├── apps.py         # fake app content (Photos grid, Music playlist, ...)
    └── compositor.py   # the master frame builder

vision-demo-eye/                  (Part 2, cloned from vision-demo)
├── ... (same as above plus:)
├── calibration.pkl       # copied from eye-mouse/
├── guest_calibration.pkl # generated by Phase 11
├── phase8_gaze.py
├── phase9_one_euro.py
├── phase10_recal_5x5.py
├── phase11_guest_cal.py
├── phase12_pretrained.py   # optional
└── src/
    └── ... (plus gaze.py, smooth.py, one_euro.py)

vision-demo-full/                 (Part 3, cloned from vision-demo-eye)
├── ... (same as above plus:)
├── phase13_hands.py
├── phase14_pinch_click.py
└── phase15_pinch_drag.py
└── src/
    └── ... (plus hands.py)
```

## Show-day defensiveness

- ESC kills the script instantly. Test this regularly. Make it actually
  exit cleanly even when something else is hung.
- Keep a screen recording of the working demo on a USB stick as a
  backup. If the live tracker fails on stage, play the video.
- Bring a power adapter. Fullscreen rendering + camera + MediaPipe will
  drain the M2 battery fast.
- Bring a portable USB light pointed at the user's face. Stage lighting
  is unpredictable and the eye tracker is sensitive to it.
- Recalibrate (Phase 10) in the actual stage lighting if at all
  possible, 5 minutes before going on.

## Coding style

- Type hints everywhere
- Top-of-file constants block with WHY comments (especially for design
  tokens — comment WHY they're those exact values, not just what they
  are)
- Functions under 30 lines where possible — split if longer
- No global mutable state
- Use `pathlib.Path`, not string paths
- `if __name__ == "__main__":` guard at the bottom of every phase script
- BGR vs RGB: pick one convention per module and document it in the
  module's docstring. Mixing them up causes "the design looks slightly
  off and I can't tell why" bugs.

## What you do at the start of every session

1. Confirm which part and phase the user is on by checking which
   `phase*.py` files already exist in the current folder.
2. Don't proactively jump ahead. If they're on Phase 3, work on Phase 3.
3. If a rendered result doesn't look Apple-grade, the most common
   culprits are: pure white instead of `#fbfbfd`, OpenCV font instead
   of PIL, drop shadow added "to make it pop," or too-tight margins.
   Check those first.
4. Read the attached `apple_SKILL.md` if it's still in the folder. The
   tokens here come from it, but the skill has more rationale.

## Anti-goals

- No web UI, no Electron, no browser stuff (rendering is pure OpenCV +
  PIL into a numpy array)
- No multi-face support
- No accessibility-api stuff (no AXUIElement, no Quartz event taps) —
  the gaze cursor controls the fake OS, never the real Mac cursor
- No premature optimization. Readability first, performance later if
  needed
- No emoji decoration
- No drop shadows on tiles — Apple specifically avoids these. Flatness
  is the point.
- No PNG icon assets if avoidable — draw glyphs procedurally with cv2
  primitives. Keeps the build dependency-free.
- No PyTorch / TensorFlow until Phase 12, and only behind a flag
- No animation libraries — implement ease curves yourself in
  `src/motion.py` (one cubic-bezier function plus a few lerps)
