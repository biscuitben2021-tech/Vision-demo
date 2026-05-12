# Apple visual skill — boiled down for this project

This file was reconstructed inside the `vision-demo/` folder when the
original `apple_SKILL.md` was not present.  All authoritative design
tokens live in `CLAUDE.md` and `src/design.py`; this document records
the *rationale* behind those tokens so anyone (human or subagent) reading
just this file understands the WHY.

## The trick

Apple's marketing aesthetic is dominated by what isn't on the page.
A tile is composed like a print ad: a single subject, oceans of negative
space, no decorative chrome.  Borders, drop shadows, dividers, and
gradients are essentially absent.  Visual separation comes from the
**16-pixel gap between tiles** and the **24-pixel radius** that crops
each tile -- nothing else.

If a render looks "almost Apple but not quite", check first:

1. Background is `#fbfbfd`, not pure white.  Pure `#ffffff` reads as a
   fluorescent bulb and crushes every rounded-tile highlight.
2. Text is `#1d1d1f` on light, `#f5f5f7` on dark.  Pure black and pure
   white feel cheap; the warm near-extremes feel like Apple.
3. Tiles have **no** border, **no** drop shadow, **no** outline.
4. Headlines use SF Pro Display Semibold rendered through PIL.  Anything
   coming out of `cv2.putText` will betray the entire design.
5. Tile interior padding is 80px from the top, 40px on the sides.
   Cramped is the opposite of Apple.

## Palette (mirrors CLAUDE.md)

| Token             | Hex      | Use                                             |
|-------------------|----------|-------------------------------------------------|
| BG_LIGHT          | #fbfbfd  | The default page surface.  Never pure white.    |
| BG_DARK           | #000000  | Alternating tile rows; OS home wallpaper.       |
| BG_NEUTRAL        | #f5f5f7  | Footers, secondary tiles, search inputs.        |
| TEXT_ON_LIGHT     | #1d1d1f  | Primary headlines on a light surface.           |
| TEXT_ON_DARK      | #f5f5f7  | Primary headlines on a black surface.           |
| TEXT_MUTED        | #86868b  | Subheads and body copy below headlines.         |
| TEXT_TERTIARY     | #6e6e73  | Disclaimers, footnotes, FPS counter.            |
| ACCENT_LIGHT      | #0066cc  | "Learn more >" link on light surface.           |
| ACCENT_DARK       | #2997ff  | "Learn more >" link on black surface.           |
| ACCENT_HOVER      | #0077ed  | Hover state for ACCENT_LIGHT.                   |
| HAIRLINE          | #d2d2d7  | Footer dividers and form input borders only.    |

## Typography rules

| Style          | Family             | Weight   | Size  | Color           |
|----------------|--------------------|----------|-------|-----------------|
| H1 hero        | SF Pro Display     | Semibold | 80px  | TEXT_ON_*       |
| H2 tile        | SF Pro Display     | Semibold | 48px  | TEXT_ON_*       |
| H3 eyebrow     | SF Pro Text        | Semibold | 21px  | TEXT_ON_* / brand |
| Body / subhead | SF Pro Text        | Regular  | 21px  | TEXT_MUTED      |
| Footnote       | SF Pro Text        | Regular  | 12px  | TEXT_TERTIARY   |

* Headlines cap at 12 characters per line; wrap to a second line before
  shrinking the size.
* Line height: H1 = 1.05, H2 / H3 = 1.19, body = 1.38, footnote = 1.33.
* Letter spacing: H1 -0.5px, H2 -0.3px.  PIL has no native tracking; if
  needed, render glyph-by-glyph -- only if you have time.

## Voice and tone

Confident, terse, aspirational.  Never explain a feature -- name it.
Short sentences.  No exclamation marks.  No emoji.

```
Eyebrow:  Photos
Headline: Every memory, instantly.
Subhead:  Your library, at a glance.
```

Bad: "Now featuring AI-powered photo organization with on-device ML!"
Good: "Hello, memories."

## CTA pattern

"Learn more `›`" — blue text + literal `›` chevron character (U+203A).
24px gap between CTAs in a row.  No backgrounds, no underline by default,
underline only on hover.

## Motion

Use Apple's "ease-emphasized" cubic bezier (0.28, 0.11, 0.32, 1.0) for
all motion.  Tile fade-up: translateY(24 → 0) + opacity (0 → 1) over
800ms.  Hover scale: 1.0 → 1.02 over 200ms.  Nav blur ramp: 250ms.

Hold `R` at startup to skip animations.

## Anti-goals

* No drop shadows on tiles.  Apple deliberately avoids them.
* No emoji as decoration.
* No PNG icon assets if avoidable -- draw glyphs procedurally.
* No animation libraries.  Implement easing yourself in `src/motion.py`.
* No web UI, no Electron, no GUI frameworks.  Pure OpenCV + PIL into a
  numpy array.

End of skill notes.  When in doubt, the design tokens in
`src/design.py` are the source of truth; this file explains the WHY.
