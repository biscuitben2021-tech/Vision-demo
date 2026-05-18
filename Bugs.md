# Bug Report -- 2026-05-18

## Review Summary
- **Files Reviewed**:
  - `phase7_demo.py` (Agent 1)
  - `phase8_hand.py`, `run.py` (Agent 2)
  - `src/hands.py`, `src/pretrained_gaze.py` (Agent 3)
  - `src/compositor.py`, `src/tiles.py` (Agent 4)
  - `src/design.py`, `src/icons.py`, `src/motion.py`, `src/apps.py` (Agent 5)
- **Verification Pass**: FINAL (5th sweep)
- **Claimed Fixes Audited**: 20
- **Claimed Fixes Verified Real**: 20
- **Hallucinated / Missing Fixes**: 0
- **New Bugs Found**: 2 (1 LOW, 1 LOW)
- **Critical**: 0 | **High**: 0 | **Medium**: 0 | **Low**: 2
- **Static Checks**: All files `py_compile` clean. Every `import` exercised and clean.
- **Smoke Tests**: Compositor instantiated, `compose_frame` runs across `home -> transitioning -> app` states; `paint_warm_aurora` paints a 64x40 canvas without error; every app renderer (`render_safari`, `render_photos`, `render_music`, `render_notes`, `render_mail`, `render_calendar`, `render_settings`, `render_demo`) paints a 1280x720 frame; `cubic_bezier(0)==0.0` and `cubic_bezier(1)==1.0`; `rounded_rect` with `radius=0`, `radius>w/2`, `w=0` all run without exception; `_nearest_tile_index([])` returns `None`; `set_gaze_lock(99)` and `set_gaze_lock(-1)` are silently coerced to `None`.

---

## Claim-by-Claim Verification

### Agent 1 -- `phase7_demo.py` (3/3 verified)

| # | Claim | Status | Notes |
|---|---|---|---|
| 1 | `verify_camera_at_startup` at lines 301-342 with diagnostics + sys.exit(1) on failure, cleanup before exit | VERIFIED | Function present at lines 301-342, opens cap, reads a frame, prints the documented diagnostic block, releases cap + destroyAllWindows before sys.exit(1). |
| 2 | SPACE-driven calibration with `CALIBRATION_MIN_GAP`, `FLASH_FRAMES`, `KEY_SPACE`, `run_calibration_step`, optional `reject_predicate`, `run_ready_screen` | VERIFIED | `CALIBRATION_SECONDS`/`READY_SECONDS` are gone; `CALIBRATION_MIN_GAP=0.05` (line 122), `FLASH_FRAMES=60` (line 128), `KEY_SPACE=32` (line 169) present; `run_calibration_step` lives at line 469 with the documented `reject_predicate` parameter; `run_ready_screen` at line 520. |
| 3 | `_process_camera_frame` + `_render_calibration_frame` factor out per-frame thumbnail rendering with red/white border + landmarks; called from every calibration step | VERIFIED | `_process_camera_frame` at line 401; `_render_calibration_frame` at line 424; called from both `run_calibration_step` (line 501) and `run_ready_screen` (line 536). Thumbnail border is `COLOR_WHITE` when hand present, `COLOR_RED` when absent (line 672). |

### Agent 2 -- `phase8_hand.py` + `run.py` (3/3 verified)

| # | Claim | Status | Notes |
|---|---|---|---|
| 4 | `try/finally` widened to wrap everything after `_try_construct_hand_input` | VERIFIED | `try:` opens at line 961 immediately after the hand-input construction (line 948); `finally:` at line 1200 releases the camera. The fullscreen window creation, `_wait_for_fullscreen_geometry`, `_run_eye_calibration`, `Compositor(...)`, `setMouseCallback`, and the entire render loop all sit inside the try block. |
| 5 | New `_wait_for_fullscreen_geometry(window_name)` helper waits for two consecutive equal nonzero `screen_size` reads | VERIFIED | Function at line 296. Polls up to `max_iterations=12` times at `poll_ms=20` each. Exits early when `cur_w == last_w and cur_h == last_h and cur_w > 0 and cur_h > 0` (line 335). Falls back to best-effort return after the budget. |
| 6 | Added `Any` import + type hints on `_no_hand_frame`, `_try_construct_hand_input`, `_run_eye_calibration` | VERIFIED | Line 46: `from typing import Any, Final, Optional`. Line 1218: `def _no_hand_frame() -> Any:`. Line 453: `_try_construct_hand_input(camera_index: Optional[int], pretrained_gaze: Optional[Any] = None) -> Optional[Any]`. Line 728: `_run_eye_calibration(hand_input: Any, screen_w: int, screen_h: int) -> bool`. |

### Agent 3 -- `src/hands.py` + `src/pretrained_gaze.py` (5/5 verified)

| # | Claim | Status | Notes |
|---|---|---|---|
| 7 | `fit_calibration`: `np.linalg.matrix_rank(a_rows) < 3` guard + `LinAlgError` try/except; preserves prior matrix on failure | VERIFIED | `src/hands.py` line 1028: `if np.linalg.matrix_rank(a_rows) < 3: return False`. Line 1032-1039: lstsq calls wrapped in `try/except np.linalg.LinAlgError: return False`. `_calibration_matrix` is only assigned inside `with self._lock:` *after* both solves succeed (line 1041-1042). Prior matrix preserved on every early `return False`. |
| 8 | `HandInput.__init__`: model constructors wrapped in try/except; on failure releases Hands + cap then re-raises as `HandInputUnavailable`. Pre-init `_hands_model = None` / `_face_mesh = None` | VERIFIED | Lines 728-729: pre-init to `None`. Lines 730-760: model construction inside `try:`. Lines 761-776: except branch releases `_hands_model` (with None guard) and `_cap`, then raises `HandInputUnavailable`. |
| 9 | `gaze_cursor`: three branches compute `(sx, sy)` as floats; `math.isfinite` check before `int(round(...))`; returns `None` on non-finite | VERIFIED | Lines 1101-1126: three branches (pretrained / matrix / baseline) each assign `sx, sy` as floats. Line 1136: `if not (math.isfinite(sx) and math.isfinite(sy)): return None`. Clamping to canvas bounds follows after the finite check (lines 1140-1141). |
| 10 | `_publish_results`: gaze-state writes deferred into the publish-lock block; `update_gaze=False` on no-face preserves stale-on-loss | VERIFIED | Lines 1440-1464: `smoothed_to_publish`, `latest_to_publish`, `update_gaze` computed in locals BEFORE the lock. Lines 1550-1552 inside `with self._lock:` block: `if update_gaze: self._latest_gaze_norm = ...; self._smoothed_gaze_norm = ...`. No-face branch sets `update_gaze=False` (line 1464); the publish block then skips the write so the previous smoothed reading persists. |
| 11 | `PretrainedGaze._ensure_weights`: `weights.parent.mkdir` wrapped in `try/except OSError`; returns `None` on failure | VERIFIED | `src/pretrained_gaze.py` lines 99-107: `try: weights.parent.mkdir(parents=True, exist_ok=True)` wrapped, `except OSError as exc: print(...); return None`. |

### Agent 4 -- `src/compositor.py` + `src/tiles.py` (4/4 verified)

| # | Claim | Status | Notes |
|---|---|---|---|
| 12 | `Final` added to the `from typing` import (line 63) | VERIFIED | Line 63: `from typing import Final, Literal, Optional`. All `_GAZE_LOCK_*` constants on lines 93-97 reference `Final[...]` and resolve. |
| 13 | `_paint_gaze_lock_chip` docstring rewritten to reference the named constants | VERIFIED | Docstring on lines 649-670 mentions `_GAZE_LOCK_PAD (12px)`, `_GAZE_LOCK_INTENSITY (0.9)`, `_GAZE_LOCK_BORDER_PX (2px)`, `_GAZE_LOCK_BORDER_ALPHA (80%)` by name. |
| 14 | Comment blocks + `chip_r` inline comment match current values (PAD=12, BORDER_PX=2, 0.9 intensity, 0.80 alpha) | VERIFIED | Header comment lines 87-92 names PAD=12, BORDER_PX=2, BORDER_ALPHA=80%. Inline comment at line 688-691 explains `RADIUS_APP_ICON + pad // 2 = 34` for PAD=12 -- math holds. |
| 15 | `src/tiles.py` audited but unchanged | VERIFIED | File mtime is May 11 22:33; every other modified `src/` file is May 18. Confirmed read-only audit. |

### Agent 5 -- `src/design.py` + `src/icons.py` + `src/motion.py` + `src/apps.py` (5/5 verified)

| # | Claim | Status | Notes |
|---|---|---|---|
| 16 | `_get_supersampled_font` forwards `index=getattr(font, "index", 0)` so Helvetica Neue Bold (face index 2) is preserved | VERIFIED | `src/design.py` line 260: `index=getattr(font, "index", 0)`. Smoke test exercised the SFNS path (index=0) and confirmed both the supersampled `.size` (42 vs 21) and the Semibold variation (heavier glyph widths: SS-Semibold renders "Hello" at 94px vs plain 42px Regular at 89px) are preserved. |
| 17 | `_FROST_TINT_BGR` reversed to `(245, 240, 240)` | VERIFIED | `src/icons.py` line 131. Header comment lines 122-130 documents the original `(240, 240, 245)` value as the broken state. |
| 18 | `_RIM_COLOR_BGR` reversed to `(250, 245, 245)` | VERIFIED | `src/icons.py` line 139. Header comment lines 133-138 documents the prior `(245, 245, 250)` value. |
| 19 | `_AURORA_BLOBS` blob #1 -> `(95, 55, 170)`, blob #2 -> `(180, 110, 195)`, blob #3 untouched | VERIFIED | `src/icons.py` lines 232-234. Blob #3 is `(210, 145, 70)` (already cool-blue dominant in BGR -- B=210). Header comment lines 226-231 calls out the reversal. |
| 20 | `_PHOTO_COLORS_BGR`: 10 of 12 entries reversed | VERIFIED | `src/apps.py` lines 267-280. Header comment lines 254-266 confirms `_PHOTO_COLORS_BGR` was previously stored as RGB and is now true BGR; entries 7 ("bleached cloud") and 12 ("paper") are left as ambiguous / already-correct neutrals. |

---

## Cross-cutting risk audit

### Concurrent-edit consistency

* **`gaze_cursor` returning `None` on non-finite**: caller in `phase8_hand.py` lines 1068-1101 stores the result in `gaze_xy: Optional[tuple[int, int]] = None`, then every downstream branch is gated by `if gaze_xy is not None`. The fall-through `else` (line 1121) safely clears `set_gaze_lock(None)` + `set_close_button_focused(False)`. No None-deref risk.
* **`fit_calibration` returning `False`**: every caller in `phase8_hand.py` (lines 852, 874, 877) consumes the return value via `fitted = hand_input.fit_calibration()` and either logs success/failure or returns it. False translates to "no affine fitted"; `gaze_cursor()` then returns None for the iris-affine branch and the OS falls back to the mouse. Correct.
* **`HandInput.__init__` raising `HandInputUnavailable`**: caller `_try_construct_hand_input` catches it on line 484 and prints a single-line warning. The catch-all on line 491 mops up unexpected exception types. Falls back to mouse-only mode. Correct.
* **BGR color reversals in `icons.py` and `apps.py`**: every changed constant is private (underscore prefix) and used only within its own module (`grep -rn` confirms no external references). No tests or other modules depend on the old values.
* **`set_gaze_lock(...)` validity**: clamps non-None values outside `0..7` to `None` (line 606); `_paint_gaze_lock_chip` also re-checks `0 <= self._gaze_lock_tile_id < len(self.geometry.tile_rects)` (line 677). Double-guarded.
* **`compositor.geometry` may be None on the very first frame**: phase8_hand line 1093 checks `compositor.geometry is not None` before reading `compositor.geometry.tile_rects`. Compose_frame is called AFTER the gaze logic on that frame -- on the first iteration, the gaze branch falls into the `else` clause and clears the lock without crashing.
* **Background thread vs lock ordering**: lock created at `src/hands.py` line 780; thread spawned at line 850 (after the lock and all `self._latest_*` fields exist). No "thread reads before init" race.
* **Edge-flag double-fire**: `step()` clears `click_now` / `drag_just_started` / `drag_just_ended` after read (lines 1193-1210). The 60Hz OS loop polling the 30Hz camera thread cannot accidentally re-fire a pinch.
* **Stale-on-loss preservation**: on face loss `update_gaze=False` skips the smoothed-gaze write, so `_smoothed_gaze_norm` retains the prior value. `gaze_cursor` reads the same locked snapshot. On the iris path the cursor freezes where it was instead of snapping to `(0,0)`. On the pretrained path `_latest_pretrained_yp` IS cleared every no-face frame (line 1385), which is the documented L2CS behaviour.

### Hallucinated/missing fixes

None. Every claim located in source at the documented line range and matches the described behaviour.

---

## New bugs found

### LOW -- Import structure interleaved with module constant
- **File**: `phase8_hand.py`
- **Line(s)**: 51-77
- **Code**:
```python
from src.compositor import Compositor
from src.design import draw_fps_hud, draw_text, load_font
from phase6_app_window import _close_button_rect


# How big the gaze-snap zone around the close button is.  ...
_CLOSE_SNAP_PX: Final[int] = 280
from phase1_canvas import (
    FPS_EMA_ALPHA,
    QUIT_KEY_ESC,
    ...
)
```
- **What it does**: Declares a module-level `Final` constant in the middle of the import block. The `from phase1_canvas import ...` block sits BELOW `_CLOSE_SNAP_PX` instead of above.
- **What's wrong**: PEP-8 ("Imports are always put at the top of the file") and the codebase's own convention (every other phase script keeps imports contiguous). It's structurally lazy: the constant was clearly inserted where it was first needed, not where the rest of the file's "Constants" block lives (line 80+).
- **Impact**: None at runtime. Static checkers (`ruff E402`, `pylint wrong-import-position`) would flag it. Future readers do a double-take. No risk for the show.
- **The Roast**: Line 63 is a `Final[int]` sandwiched between two `from ... import` blocks like a piece of bacon someone forgot to put on the burger. The CLAUDE.md insists on "one concept per file" but apparently "imports together at the top" was the optional clause.
- **Fix**: Move `_CLOSE_SNAP_PX` down to join the other `Final` constants near line 115 (where `_CLI_FLAG_REDUCED_MOTION` lives) and pull the `phase1_canvas` import up.

---

### LOW -- `_ICON_TINTS` BGR-ordering ambiguity carries the same shape as the bugs Agent 5 just fixed
- **File**: `src/icons.py`
- **Line(s)**: 813, 831
- **Code**:
```python
_ICON_TINTS: Final[dict[str, tuple[int, int, int]]] = {
    "safari":   (240, 240, 245),
    ...
    "calendar": (240, 240, 245),
    ...
}
```
- **What it does**: Sets the Safari and Calendar icon background tints. The dict is named `_ICON_TINTS` and the inline comments describe BGR convention.
- **What's wrong**: `(240, 240, 245)` interpreted as BGR is `B=240, G=240, R=245` -- a faint warm/red bias. The comment for Safari says "we use a neutral near-white that doesn't fight against the compass-blue ring we draw on top", which scans as "intentional neutral, not blue". Calendar's comment is "Calendar background is white". So neither tint claims to be blue-cast; the tuples are NOT incorrect by their own documented intent. BUT -- this is the *exact* tuple shape Agent 5 just identified as a BGR/RGB byte-order bug in `_FROST_TINT_BGR` (`(240, 240, 245)` -> `(245, 240, 240)`). If the original author meant "near-white with a hint of blue" (which an RGB reading would give), this is the same byte-order mistake Agent 5 caught elsewhere; if they meant "near-white with a hint of warmth" the current value is correct. The comments lean toward "neutral", which makes the current value defensible but not above suspicion.
- **Impact**: Visual only, and so close to neutral the difference is sub-perceptible. Demo-day audience will not notice.
- **The Roast**: Two icons in a dict literally next to four constants Agent 5 just reversed for being-BGR-in-name-only. Either the original author had a consistent BGR/RGB mistake (in which case Agent 5 missed two) or they deliberately picked warm-cast for two icons while everyone else picked cool-cast -- which is the kind of inconsistency the CLAUDE.md "Apple-grade visual restraint" section was written to prevent.
- **Fix** (optional, not blocking): If the design intent really is "neutral near-white", swap to a pure neutral like `(244, 244, 244)`. If "faintly cool" was the intent (matching the real Safari and Calendar icon art), swap to `(245, 240, 240)` to match `_FROST_TINT_BGR`. Leave as-is if "faintly warm" was deliberate; in that case add a one-line comment saying so to defuse future suspicion.

---

## Run-readiness statement

**GREEN: `phase7_demo.py` and `phase8_hand.py` are both runnable end-to-end. All 20 claimed fixes verified present and correct. No new bugs that block running.**

The two findings above are LOW-severity cosmetic / consistency notes; neither prevents the demo from shipping. Static compile clean across every file. Every import path exercised. Compositor smoke test runs `home -> transitioning -> app` cleanly with `set_gaze_lock` and `set_close_button_focused` toggled. Every app renderer paints a 1280x720 frame. `gaze_cursor` returns `None` on the calibration-not-fitted path; `fit_calibration` returns `False` on rank-deficient samples; `HandInputUnavailable` is caught by the phase8 fallback path. `try/finally` around the main loop releases the camera handle on every exit path including calibration / compositor-init exceptions. The OS boots in mouse-only mode if any tracking dep fails. The defensive show-day rule ("ESC kills the script instantly; OS always boots") holds.

You can ship this on stage with confidence.
