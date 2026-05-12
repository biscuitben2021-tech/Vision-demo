# vision-demo (Part 1)

A fake Vision OS demo running fullscreen on macOS.  Mouse-driven for
now; eye tracking and hand tracking come in Parts 2 and 3.

## Setup

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python run.py
```

ESC or Q to quit.  Click any app tile to open it.  Click the X in the
top-left of any app to return to the home screen.

Hold `R` at startup or pass `--reduced-motion` to disable animations.

## What's in Part 1

* `phase1_canvas.py` — fullscreen window + design tokens + FPS counter
* `phase2_typography.py` — H1 hero + subhead + CTA stack via PIL
* `phase3_tile_grid.py` — 2x2 Apple-marketing-style tile grid
* `phase4_home_screen.py` — Vision OS home: 4x2 glass app tile grid
* `phase5_motion.py` — fade-up + hover scale with Apple's ease-emphasized
* `phase6_app_window.py` — click-to-open app windows + close button
* `phase7_polish.py` — live clock + sliding notifications + cross-fade

Each `phaseN_*.py` runs standalone for development; `run.py` boots the
polished Phase 7 build.

## Repository layout

```
phase1_canvas.py  ...  phase7_polish.py     # one runnable per phase
run.py                                       # demo-day entry point
src/
  design.py        # palette / spacing / typography tokens
  tiles.py         # rounded_rect + draw_tile
  icons.py         # glass panel + 8 procedural app icon glyphs
  motion.py        # cubic bezier + FadeUpState + HoverState
  apps.py          # 8 fake-app render functions
  compositor.py    # Compositor: owns OS state, drives all rendering
```
