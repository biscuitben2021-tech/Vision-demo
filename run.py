"""Canonical Part 1 entry point.  Boots the polished Phase 7 OS.

End users on demo day run `python run.py`; everything below the surface
is implemented in `phase7_polish.py`, which composes the full home /
app / notification stack on top of Phases 1 through 6.
"""

from phase7_polish import main

if __name__ == "__main__":
    main()
