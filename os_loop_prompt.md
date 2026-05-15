# vision-demo Part 1 — Autonomous OS build loop

Paste this whole document into Claude Code from inside the `vision-demo/`
folder, after `CLAUDE.md`, `apple_SKILL.md`, `src/design.py`, and
`phase1_canvas.py` already exist.

This prompt is meant for the parent Claude Code agent. The parent
dispatches each phase to the `diligent-elite-coder` subagent via the
Task tool.

---

## Mission

Autonomously build Phases 2 through 7 of Part 1 (the OS) for the
vision-demo project. Use the `diligent-elite-coder` subagent for the
actual coding. Loop sequentially through each phase, smoke-testing
between phases, retrying up to 3 times on failure. At the end the user
should be able to run `python run.py` and use the fake OS with their
normal mouse and keyboard — no eye or hand tracking yet.

## Override notice

CLAUDE.md says "After each phase, stop and let the user run it before
moving on" and "Don't proactively jump ahead." Those rules are
**explicitly suspended** for this prompt only. The user has authorized
full autonomy through Phase 7. Do NOT stop to ask for approval between
phases. Do NOT ask the user to test each one. Build straight through to
Phase 7 unless a phase fails 3 retries in a row.

All other CLAUDE.md rules stay in force — heavy WHY comments, type hints,
30-line function cap, design tokens from `src/design.py`, no banned
dependencies, no drop shadows on tiles, near-white backgrounds not pure
white, PIL for typography not cv2.putText.

## Setup verification (do this first, before any phase)

1. Verify these files exist in the current directory:
   - `CLAUDE.md`
   - `apple_SKILL.md`
   - `src/design.py`
   - `phase1_canvas.py`
   If any are missing, halt and tell the user which.
2. Verify the venv is active: `which python` should point inside the
   project folder. If not, halt and tell the user.
3. Smoke-test Phase 1: `timeout 3 python phase1_canvas.py 2>&1`. Expected
   outcome: the process either times out (good — UI was running) or
   exits cleanly. A traceback in stderr is a fail — halt and tell the
   user.

If all three checks pass, proceed to the phase loop.

## Per-phase loop pattern

For each phase N in [2, 3, 4, 5, 6, 7]:

1. Dispatch the phase spec to `diligent-elite-coder` as a Task with the
   prompt text from the "Phase N spec" section below.
2. When the subagent returns, smoke-test:
   ```bash
   timeout 3 python phaseN_<name>.py 2>&1 | tail -30
   ```
3. If stderr contains a Python traceback, capture the last 30 lines and
   send back to `diligent-elite-coder` as a follow-up Task:
   > Phase N broke at runtime. Here is the traceback:
   > ```
   > <traceback>
   > ```
   > Read CLAUDE.md again. Fix the file. Do not rewrite from scratch
   > unless necessary — make a minimal change. Return when fixed.
4. Retry up to 3 times per phase. If still failing after 3 retries,
   halt the loop and report which phase, the final traceback, and what
   was attempted.
5. On pass, commit with `git add -A && git commit -m "phase N: <short>"`.
   If the repo isn't a git repo yet, run `git init && git add -A &&
   git commit -m "initial"` first.
6. Move to the next phase.

After Phase 7 passes, do the post-loop steps (run.py, README, final
report) at the bottom of this document.

## Phase 2 spec

Dispatch to `diligent-elite-coder`:

> Read `CLAUDE.md` and `apple_SKILL.md`. You are building Phase 2 of the
> vision-demo OS. Phase 1 already exists.
>
> Goal: render Apple-grade typography on the off-white canvas from Phase
> 1. The hero headline must look the way apple.com looks, not the way
> cv2.putText looks.
>
> Files to write:
>
> 1. Extend `src/design.py`: add `draw_text(frame, text, x, y,
>    color_rgb, font, align="left")`. Renders text via PIL into a PIL
>    Image, converts to numpy RGB → BGR, composites into the numpy BGR
>    frame at (x, y). Handles alpha if PIL produces RGBA. Supports
>    align="left" | "center" | "right" (align is relative to the (x, y)
>    anchor). Type hints. Heavy WHY comments — especially on the
>    RGB-vs-BGR conversion, since that's the silent-bug zone.
>
> 2. `phase2_typography.py`: based on `phase1_canvas.py`. On the
>    BG_LIGHT canvas, render centered horizontally:
>    - H1 "Hello, vision." in SF Pro Display Semibold 80px, color
>      TEXT_ON_LIGHT_RGB, starting ~30% down the screen
>    - Subhead "Spatial computing, reimagined." in SF Pro Text Regular
>      21px, color TEXT_MUTED_RGB, 12px below the H1
>    - Two chevron CTAs side by side, 24px apart, 24px below the subhead:
>      "Learn more ›" and "Try the demo ›" in SF Pro Text Regular 17px,
>      color ACCENT_LIGHT_RGB
>    - Keep the FPS counter from Phase 1 in the top-right
>
> Rules: type hints, functions under 30 lines, heavy WHY comments,
> `if __name__ == "__main__":` guard. Do not draw drop shadows. Do not
> use cv2.putText anywhere. Do not change colors from the design tokens.
>
> Don't run the script yourself — the parent will smoke-test. Return
> when the files are written.

After this passes, commit and continue to Phase 3.

## Phase 3 spec

Dispatch to `diligent-elite-coder`:

> Read `CLAUDE.md` and `apple_SKILL.md`. Build Phase 3.
>
> Goal: a static 2-up grid of 4 rounded Apple-style tiles, alternating
> light and dark surfaces. Each tile shows eyebrow + headline + subhead
> + two chevron CTAs, exactly matching the apple.com pattern.
>
> Files to write:
>
> 1. `src/tiles.py`:
>    - `rounded_rect(frame, x, y, w, h, radius, color_bgr)`: draws a
>      filled rounded rectangle by drawing a main rect plus four corner
>      circles. Use `cv2.LINE_AA` for the corner circles.
>    - `draw_tile(frame, x, y, w, h, *, theme, eyebrow, headline,
>      subhead, cta_pair)`: renders a complete Apple tile. `theme` is
>      `"light"` or `"dark"`, which selects the background color and
>      the text colors. `cta_pair` is `tuple[str, str]` like
>      `("Learn more", "Buy")` — append the chevron `›` automatically,
>      do not require the caller to include it.
>    - Inside content is vertically anchored 80px from the top of the
>      tile, eyebrow first, then headline (max 12 chars per line —
>      wrap to a second line rather than shrinking), then subhead,
>      then a horizontal row of CTAs centered, 24px apart.
>    - All function signatures use type hints. No function over 30
>      lines — split helpers as needed.
>
> 2. `phase3_tile_grid.py`: based on phase2. Fills the screen with a
>    2-up grid of 4 tiles. Layout:
>    - Outer gutter: 16px from each viewport edge
>    - 2 columns × 2 rows
>    - 16px gap between tiles
>    - Tiles size themselves to fill the available area equally
>
>    Tile content:
>    - Top-left (light theme): eyebrow "Photos", headline "Every
>      memory, instantly.", subhead "Your library, at a glance.", CTAs
>      ("Learn more", "Open")
>    - Top-right (dark theme): eyebrow "Music", headline "Hello,
>      soundtrack.", subhead "Pick up where you left off.", CTAs
>      ("Learn more", "Open")
>    - Bottom-left (dark theme): eyebrow "Notes", headline "Think it.
>      Keep it.", subhead "Everywhere you go.", CTAs ("Learn more",
>      "Open")
>    - Bottom-right (light theme): eyebrow "Safari", headline "The
>      web, in a window.", subhead "Browse beautifully.", CTAs
>      ("Learn more", "Open")
>
> Visual rules from CLAUDE.md: 24px radius, no border, no drop shadow,
> no outline. Tile separation comes from the 16px gap alone.

After this passes, commit and continue to Phase 4.

## Phase 4 spec

Dispatch to `diligent-elite-coder`:

> Read `CLAUDE.md` and `apple_SKILL.md`. Build Phase 4.
>
> Goal: a Vision OS home screen with 8 floating glass app tiles in a
> grid, a top status bar with a clock placeholder, on a dark background.
>
> Files to write:
>
> 1. `src/icons.py`:
>    - `draw_glass_panel(frame, x, y, w, h, radius)`: translucent glass
>      effect over whatever is underneath. Sample the area under the
>      panel, brighten + slightly desaturate, blend back with
>      `cv2.addWeighted` at alpha 0.85 over a near-black overlay.
>      Rounded corners using `rounded_rect` from `src/tiles.py`.
>    - `draw_app_icon(frame, cx, cy, size, app_id)`: draws a procedural
>      icon glyph centered at (cx, cy). `app_id` is one of
>      `"safari" | "photos" | "music" | "notes" | "mail" | "calendar"
>      | "settings" | "demo"`. Use simple cv2 primitives — Safari:
>      compass with a red needle on white; Photos: rainbow petal
>      flower; Music: white music note on red; Notes: yellow page with
>      lines; Mail: white envelope on blue; Calendar: white card with
>      red header and date "11"; Settings: gear shape; Demo: a circle
>      with "vd" wordmark. Each icon sits on a rounded background
>      tinted with its identifying color.
>    - Each icon function under 40 lines.
>
> 2. `phase4_home_screen.py`: dark background (BG_DARK_BGR). At the
>    top, a 44px-tall glass status bar with "vision" wordmark on the
>    left (SF Pro Text Semibold 14px, TEXT_ON_DARK_RGB) and a clock
>    placeholder "10:30 AM" on the right. Below the status bar with
>    ~80px breathing room, a 4×2 grid of 8 app tiles. Each tile is
>    140×140px with RADIUS_APP_ICON corners, glass surface, icon
>    centered, and the app's display name in SF Pro Text Regular 13px
>    below the icon (color TEXT_ON_DARK_RGB).
>
>    Apps in this order: Safari, Photos, Music, Notes, Mail, Calendar,
>    Settings, Demo.
>
>    Grid is horizontally centered. 24px gap between tiles.

After this passes, commit and continue to Phase 5.

## Phase 5 spec

Dispatch to `diligent-elite-coder`:

> Read `CLAUDE.md` and `apple_SKILL.md`. Build Phase 5.
>
> Goal: add motion. The home screen tiles fade-up on entry. The mouse
> cursor's hover state scales the hovered tile to 1.02. All eased with
> Apple's ease-emphasized curve.
>
> Files to write:
>
> 1. `src/motion.py`:
>    - `cubic_bezier(t: float, p1x: float, p1y: float, p2x: float,
>      p2y: float) -> float`: Apple's ease-emphasized curve evaluated
>      at parameter t in [0, 1]. Newton-Raphson is fine; ~15 lines.
>    - `ease_emphasized(t: float) -> float`: calls cubic_bezier with
>      the fixed control points (0.28, 0.11, 0.32, 1).
>    - `FadeUpState` dataclass with fields `start_ms: int`,
>      `duration_ms: int` and a method `value(now_ms) -> tuple[float,
>      float]` returning (opacity 0..1, y_offset_px 24..0).
>    - `HoverState` class tracking which tile_id is currently hovered.
>      Method `set_hover(tile_id)` and `scale_for(tile_id, now_ms)`
>      returning the smoothly-eased scale value (target 1.0 or 1.02,
>      eased over HOVER_SCALE_DURATION_MS).
>
> 2. `phase5_motion.py`: based on phase4. On script start, the 8 app
>    tiles fade-up over 800ms with a 50ms stagger (tile 0 at t=0, tile
>    1 at t=50ms, etc.). The mouse cursor hovering an app tile scales
>    that tile to 1.02 over 200ms. Leaving the tile scales back to 1.0
>    over 200ms.
>
>    Use `cv2.setMouseCallback` for mouse tracking. Use
>    `cv2.getTickCount() / cv2.getTickFrequency()` for time in seconds,
>    multiply by 1000 for ms.
>
>    Hold `R` at startup (check `sys.argv` for `--reduced-motion` or
>    handle a keypress before the main loop starts) to disable all
>    animation — tiles appear instantly at full opacity, no hover
>    scaling.
>
> Implementing scale: render the tile to a sub-image, resize by the
> scale factor, paste back centered on its slot. Or use an affine
> transform. Pick whichever is simpler.

After this passes, commit and continue to Phase 6.

## Phase 6 spec

Dispatch to `diligent-elite-coder`:

> Read `CLAUDE.md` and `apple_SKILL.md`. Build Phase 6.
>
> Goal: clicking an app tile opens a fullscreen fake app window. Each
> app has different fake content. A close button returns to the home
> screen.
>
> Files to write:
>
> 1. `src/apps.py`:
>    Per-app render functions, one per app_id. Each takes
>    `(frame: np.ndarray, w: int, h: int) -> None` and fills the frame
>    with that app's fake content. Use the design tokens for everything.
>
>    - `render_safari(frame, w, h)`: frozen apple.com-style hero —
>      eyebrow "iPhone 16 Pro" centered top, H1 "A magical new way to
>      interact with iPhone." centered, subhead "Hello, Apple
>      Intelligence." in TEXT_MUTED, two chevron CTAs ("Learn more",
>      "Buy"). Light background.
>    - `render_photos(frame, w, h)`: 4×3 grid of solid muted-color
>      tiles (use a palette of warm Apple-ish greys, blues, beiges).
>      Title "Library" in H2 at the top-left.
>    - `render_music(frame, w, h)`: vertical list of 6 song rows. Each
>      row: cover swatch + track name + artist + duration on the right.
>      "Now Playing" bar pinned at the bottom showing the current
>      track. Dark background.
>    - `render_notes(frame, w, h)`: single note open. 24px title "Vision
>      Demo" near the top. 4 paragraphs of lorem ipsum below. Light
>      background, paper-feeling.
>    - `render_mail(frame, w, h)`: inbox list with 8 sender/preview
>      rows, alternating subtle row backgrounds. Top bar says "Inbox".
>    - `render_calendar(frame, w, h)`: month view (5 rows × 7 columns)
>      with today highlighted as a circled date in the center. Month
>      name at the top: "November 2025".
>    - `render_settings(frame, w, h)`: vertical stack of 6 pill-shaped
>      settings rows: "Wi-Fi", "Bluetooth", "Display", "Sound",
>      "General", "About". Each pill has a faint glass surface.
>    - `render_demo(frame, w, h)`: centered "vision demo loaded" in H2,
>      subhead "Tap to begin.", one chevron CTA "Begin ›".
>
>    Each render function under 60 lines.
>
> 2. `phase6_app_window.py`: based on phase5. Add a state machine with
>    two states:
>    - `HOME`: shows the home screen from phase5
>    - `APP_OPEN`: shows the open app, with a 32×32 close button (an
>      X glyph on a glass circle) in the top-left corner
>
>    Transitions: clicking an app tile in HOME → APP_OPEN with that
>    app_id. Clicking the close button in APP_OPEN → HOME. No animated
>    transition yet — that's Phase 7. Just a hard cut.
>
>    Click detection: use cv2 mouse callback, check on
>    `cv2.EVENT_LBUTTONDOWN` whether the click is inside any app tile's
>    rect (HOME state) or inside the close button rect (APP_OPEN state).

After this passes, commit and continue to Phase 7.

## Phase 7 spec

Dispatch to `diligent-elite-coder`:

> Read `CLAUDE.md` and `apple_SKILL.md`. Build Phase 7. **This is the
> end of Part 1 — the demo-day fallback. After this passes the user
> should be able to use the fake OS with normal mouse and keyboard.**
>
> Goal: polish. Live updating clock. Fake notification slides in from
> the top-right after 10 seconds. Smooth 300ms cross-fade between HOME
> and APP_OPEN.
>
> Files to write:
>
> 1. `src/compositor.py`:
>    - `Compositor` class that owns OS state:
>      - current screen (`"home"`, `"app"`, `"transitioning"`)
>      - current app_id when applicable
>      - notification queue: list of `Notification` dataclass instances,
>        each with `title`, `body`, `spawn_ms`, `lifetime_ms` (4000)
>      - transition state: from_screen, to_screen, start_ms,
>        duration_ms (300)
>    - `compose_frame(now_ms: int, mouse_xy: tuple[int, int],
>      mouse_pressed: bool) -> np.ndarray`: returns the next frame to
>      display. Drives all rendering and state transitions.
>    - `enqueue_notification(title: str, body: str, now_ms: int)`: adds
>      a notification to the queue.
>    - Renders notifications as glass cards (240×64px, RADIUS_TILE_SMALL
>      corners) sliding in from the right edge of the screen over
>      300ms, holding for `lifetime_ms - 600ms`, sliding out over
>      300ms.
>
> 2. `phase7_polish.py`: pulls everything together. The clock in the
>    status bar updates every second using `datetime.datetime.now()`
>    formatted as "10:30 AM" (system locale). After 10 seconds of
>    runtime, one fake notification is enqueued — title "Apple Music",
>    body "Now playing: Bloom — Radiohead". 20 seconds later, another:
>    title "Messages", body "Mom: see you tonight". Multiple
>    notifications stack vertically with a 12px gap.
>
>    Transition between home and app states cross-fades over 300ms.
>
>    ESC quits cleanly. Q also quits.
>
> **Acceptance criteria: running `python phase7_polish.py` should look
> like a polished fake operating system. Mouse fully controls
> everything. No eye or hand tracking is required.** If you ship Phase
> 7 and nothing else from Parts 2 or 3, the demo is still impressive.

After this passes, run the post-loop steps below.

## Post-loop steps

After Phase 7 passes its smoke test:

1. Final integration smoke test:
   ```bash
   timeout 5 python phase7_polish.py 2>&1 | tail -30
   ```
   Must exit cleanly (no traceback in stderr). If it fails, dispatch
   one more retry to `diligent-elite-coder` describing the failure.

2. Verify the file tree matches CLAUDE.md's Part 1 structure:
   ```
   phase1_canvas.py through phase7_polish.py
   src/design.py, src/tiles.py, src/icons.py, src/motion.py,
   src/apps.py, src/compositor.py
   ```
   If anything is missing, halt and report.

3. Create `run.py` as the canonical Part 1 entry point. Two lines:
   ```python
   from phase7_polish import main
   main()
   ```

4. Create `README.md`:
   ```markdown
   # vision-demo (Part 1)

   A fake Vision OS demo running fullscreen on macOS. Mouse-driven for
   now. Eye tracking and hand tracking come later (Parts 2 and 3).

   ## Setup

       python3.12 -m venv venv
       source venv/bin/activate
       pip install -r requirements.txt

   ## Run

       python run.py

   ESC or Q to quit. Click any app tile to open it. Click the X in the
   top-left of any app to return to the home screen.
   ```

5. Create or update `requirements.txt`:
   ```
   opencv-python
   numpy
   Pillow
   ```

6. Final commit: `git add -A && git commit -m "Part 1 complete: OS shippable"`.

## Final report format

Output exactly this format when done:

```
Part 1 complete.

Phases built:
  ✓ Phase 1: src/design.py + phase1_canvas.py
  ✓ Phase 2: phase2_typography.py (+ draw_text in design.py)
  ✓ Phase 3: src/tiles.py + phase3_tile_grid.py
  ✓ Phase 4: src/icons.py + phase4_home_screen.py
  ✓ Phase 5: src/motion.py + phase5_motion.py
  ✓ Phase 6: src/apps.py + phase6_app_window.py
  ✓ Phase 7: src/compositor.py + phase7_polish.py
  ✓ run.py + README.md + requirements.txt

Smoke tests: <N passed, M failed>
Retries needed: <list of phases that retried, and how many times>
Compromises or known issues: <bullet list, or "none">
Anti-goals violated: <list, or "none">

To run the OS: `python run.py`
ESC or Q quits. Click apps to open. Click X to close back to home.
```

If any phase needed all 3 retries and you halted, report which phase,
the final traceback, and the last instruction sent to
`diligent-elite-coder`. Do NOT silently move on.

## Halt conditions

Stop the loop immediately and report to the user if any of these happen:

- A phase fails 3 retries in a row
- `diligent-elite-coder` reports it cannot find `CLAUDE.md` or
  `apple_SKILL.md`
- A phase requires a library not in the allowed list (CLAUDE.md)
- The git working tree gets into a state where the commit fails
- More than 30 minutes of wall-clock time elapses (the user expects this
  to be a coffee break, not an overnight job)

End of prompt.
