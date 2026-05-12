"""Vision OS demo internals.

This package holds the design system (palette, typography, spacing tokens)
and -- as later phases land -- the tile renderer, app-icon glyphs, motion
curves, fake-app content, and the master frame compositor.

Nothing in this package performs I/O at import time.  Importing
`src.design` is cheap and side-effect-free; fonts are only resolved when
`load_font(...)` is actually called.
"""
