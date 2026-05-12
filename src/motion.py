"""
Easing curves + per-element animation state for the Vision OS demo.

This module owns *time-based* visual interpolation.  Every animation in
the demo -- tile fade-up on entry, hover scale on cursor-over, the future
nav blur ramp -- evaluates through one of the helpers in this file.  No
phase script is allowed to roll its own easing math; doing so is how
"the hover scales linearly but the fade-up eases" inconsistencies creep
into the build.

Three responsibilities live here, in the order they're called:

    1. `cubic_bezier`     -- generic CSS-style cubic bezier evaluator.
                             Takes a normalised time `t` in [0, 1] and the
                             four control-point coordinates and returns
                             the eased *progress*.  Single source of truth
                             for "what does easing look like" in the demo.
    2. `ease_emphasized`  -- the only curve this codebase uses today, the
                             FADE_UP_EASING from src/design.  Future
                             curves are siblings to this function, not
                             alternate paths through cubic_bezier
                             scattered through phase files.
    3. `FadeUpState`,     -- per-element animation state.  These types
       `HoverState`         answer "given the wall-clock time, what does
                             this element look like right now?" without
                             callers having to track their own timers.

Module color-space convention:
    This module never touches pixels -- it only emits scalars (opacity,
    y-offset, scale).  No RGB / BGR concerns.  Pixel-space conversions
    are the renderer's job, not the easing curve's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.design import (
    FADE_UP_DURATION_MS,
    FADE_UP_EASING,
    HOVER_SCALE_DURATION_MS,
)


# ============================================================================
# Cubic-bezier easing
# ============================================================================
#
# A CSS-style cubic bezier is parameterised by two control points
# (p1x, p1y) and (p2x, p2y) with the endpoints fixed at P0 = (0, 0) and
# P3 = (1, 1).  The full bezier as a function of a parameter `u` in [0,1]:
#
#     x(u) = 3(1-u)^2 u p1x + 3(1-u) u^2 p2x + u^3
#     y(u) = 3(1-u)^2 u p1y + 3(1-u) u^2 p2y + u^3
#
# The catch:  the public API takes the *time* coordinate `t` (x-axis) and
# returns the *progress* (y-axis), but the bezier is parameterised by
# `u`, not by x.  So we must SOLVE for the u where x(u) = t, then plug
# that u into y(...).  The "ease(t)" curve a designer thinks of is a
# vertical slice of the bezier at x = t.
#
# Newton-Raphson is the standard root-finder for this: x(u) - t is
# monotonic in u for valid CSS bezier control points (those with p1x and
# p2x both in [0, 1]), so a few iterations from the obvious seed u = t
# converge to within a pixel of accuracy.  Eight iterations is enough --
# CSS engines (Blink, WebKit, Gecko) all cap at four to eight depending
# on the precision requested.  We fall back to a binary search if Newton
# is misbehaving (which shouldn't happen for the curves we use, but
# guarding against it costs nothing).


# Newton-Raphson convergence tolerance.  At 1e-6 the resulting eased y
# is accurate to well under a pixel for any practical curve length;
# any tighter and we'd just be spending more CPU per frame for no
# visible benefit.
_NEWTON_TOLERANCE: Final[float] = 1e-6

# Maximum Newton iterations before we fall back to bisection.  Eight is
# the upper bound the major browser easing engines use; for the
# well-behaved curves in this codebase, two or three usually suffice.
_NEWTON_MAX_ITERATIONS: Final[int] = 8

# Maximum bisection iterations.  At 32 the search interval has shrunk by
# 2^32 ~= 4e9, which is overkill for 1e-6 accuracy -- but bisection is
# the fallback path, not the hot path, so we can afford the safety.
_BISECTION_MAX_ITERATIONS: Final[int] = 32


def _bezier_x(u: float, p1x: float, p2x: float) -> float:
    """Evaluate the cubic bezier's x-component at parameter `u`.

    Bernstein form with P0 = 0 and P3 = 1 collapsed in:

        x(u) = 3(1-u)^2 u p1x  +  3(1-u) u^2 p2x  +  u^3

    Kept as a private helper because both the Newton step and the
    bisection fallback need it, and inlining would duplicate the math.
    """
    one_minus_u = 1.0 - u
    return (
        3.0 * one_minus_u * one_minus_u * u * p1x
        + 3.0 * one_minus_u * u * u * p2x
        + u * u * u
    )


def _bezier_y(u: float, p1y: float, p2y: float) -> float:
    """Evaluate the cubic bezier's y-component at parameter `u`.

    Same Bernstein form as `_bezier_x` but threaded with the y control-
    point coordinates.  Returns the eased progress in (approximately)
    [0, 1] for well-formed CSS bezier control points.
    """
    one_minus_u = 1.0 - u
    return (
        3.0 * one_minus_u * one_minus_u * u * p1y
        + 3.0 * one_minus_u * u * u * p2y
        + u * u * u
    )


def cubic_bezier(
    t: float,
    p1x: float, p1y: float,
    p2x: float, p2y: float,
) -> float:
    """Return the eased progress at time `t` for a CSS-style cubic bezier.

    Arguments:
        t: input time, normalised to [0, 1].  Out-of-range inputs are
           clamped before solving (an animation at t < 0 hasn't started;
           at t > 1 it's over).
        p1x, p1y: first control point in time-vs-progress space.
        p2x, p2y: second control point in time-vs-progress space.

    Returns the bezier y at the parameter `u` where x(u) == t.  Endpoints
    short-circuit to exactly 0.0 / 1.0 so we never miss them by a Newton
    rounding artefact.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0

    # Newton-Raphson on f(u) = x(u) - t.
    #
    # The derivative dx/du of the cubic bezier is, after collecting terms:
    #
    #     dx/du = 3(1-u)^2 p1x  +  6(1-u) u (p2x - p1x)  +  3 u^2 (1 - p2x)
    #
    # Heavy-WHY comment as the prompt asks: this is the partial derivative
    # of the Bernstein form of x(u) with respect to u.  Each term comes
    # from differentiating the corresponding cubic Bernstein basis
    # function and multiplying by the control-point x.  P0_x = 0 so its
    # term drops; the remaining three terms above are P1, P2, P3
    # respectively.  This derivative is the slope of the time axis at
    # the current u guess -- exactly what Newton-Raphson needs to take
    # a step toward the root.  Seeding u = t is reasonable because the
    # bezier x is monotonic and roughly tracks identity for the
    # standard near-linear easings we use.
    u = t
    for _ in range(_NEWTON_MAX_ITERATIONS):
        x = _bezier_x(u, p1x, p2x) - t
        if abs(x) < _NEWTON_TOLERANCE:
            return _bezier_y(u, p1y, p2y)
        one_minus_u = 1.0 - u
        dx = (
            3.0 * one_minus_u * one_minus_u * p1x
            + 6.0 * one_minus_u * u * (p2x - p1x)
            + 3.0 * u * u * (1.0 - p2x)
        )
        if dx == 0.0:
            break  # Slope vanished; bail to bisection rather than divide by zero.
        u -= x / dx

    # Bisection fallback.  Used either when Newton hit a zero derivative
    # or when it failed to converge in _NEWTON_MAX_ITERATIONS steps.
    # The bezier x is monotonic in u for valid CSS control points, so a
    # simple interval halving always terminates.
    low, high = 0.0, 1.0
    for _ in range(_BISECTION_MAX_ITERATIONS):
        u = 0.5 * (low + high)
        x = _bezier_x(u, p1x, p2x)
        if abs(x - t) < _NEWTON_TOLERANCE:
            break
        if x < t:
            low = u
        else:
            high = u
    return _bezier_y(u, p1y, p2y)


def ease_emphasized(t: float) -> float:
    """Apple's "ease-emphasized" curve, evaluated at normalised time `t`.

    Wraps `cubic_bezier` with the four control points pulled from
    src/design.FADE_UP_EASING -- (0.28, 0.11, 0.32, 1.0).  This is the
    only easing curve the codebase uses today; every fade, scale, and
    blur ramp threads through here so future-Apple-curve experiments
    swap a single function rather than scattering control points
    throughout phase files.

    If a future phase needs a *different* curve (a snappier "ease-out"
    for cursor-pressed feedback, say), add a sibling helper here --
    don't call cubic_bezier directly from a phase.  The motion module
    is the source-of-truth boundary between "what we know about
    animation" and "which animation a given UI element gets."
    """
    p1x, p1y, p2x, p2y = FADE_UP_EASING
    return cubic_bezier(t, p1x, p1y, p2x, p2y)


# ============================================================================
# FadeUpState -- per-tile entry animation
# ============================================================================
#
# A tile that fades up has two simultaneous animated properties: opacity
# and vertical offset.  Both share the same start time, duration, and
# easing curve, so it makes sense to bundle them behind one state object
# rather than spawning two per tile.
#
# We deliberately keep this as a plain dataclass with a single `value`
# method rather than splitting "opacity_state" and "y_offset_state" --
# they always animate together, always have identical timing, and the
# tuple return is cheaper than two separate calls per tile per frame.

# Pixel offset the tile starts BELOW its final resting position.  The
# tile is rendered with y = base_y + y_offset, so a positive value means
# the tile sits LOWER on screen at t=0 and rises up to base_y at t=1.
# 24px is the apple_SKILL.md spec; smaller values look static, larger
# values overshoot what the eye reads as "lifting up gently".
_FADE_UP_DELTA_Y: Final[float] = 24.0


@dataclass
class FadeUpState:
    """Tracks one tile's fade-up entry: opacity 0->1 and y-offset 24->0.

    Both properties animate from their starting values to their final
    values over `duration_ms` milliseconds, eased through
    `ease_emphasized`.  Before `start_ms` the tile is invisible and
    24px below its base position; after `start_ms + duration_ms` the
    tile is fully visible at its base position.

    Fields:
        start_ms:    wall-clock millisecond at which the animation
                     begins.  Typically (process_start_ms + i*50) for
                     the i-th tile in a staggered home-screen entry.
        duration_ms: how long the fade takes.  Defaults to
                     FADE_UP_DURATION_MS (800ms) from src/design.
    """

    start_ms: int
    duration_ms: int = FADE_UP_DURATION_MS

    def value(self, now_ms: int) -> tuple[float, float]:
        """Return (opacity, y_offset_px) at the wall-clock time `now_ms`.

        opacity in [0.0, 1.0]; the tile alpha-blends against the
        background at this factor.  y_offset_px in [0.0, 24.0]; the
        tile is drawn at base_y + y_offset, so a positive value pushes
        it DOWN visually -- exactly what we want for the "rises into
        place" feel as the offset eases from 24 to 0.

        Before the animation has started, returns (0.0, 24.0) -- tile
        is invisible and at its starting position.  After the
        animation has finished, returns (1.0, 0.0) -- tile is fully
        visible at its resting position.  Both clamps make the
        endpoints exact rather than relying on the bezier short-circuit.
        """
        elapsed = now_ms - self.start_ms
        if elapsed <= 0:
            return 0.0, _FADE_UP_DELTA_Y
        if elapsed >= self.duration_ms:
            return 1.0, 0.0

        # Normalise to [0, 1] and look up the eased progress.  The
        # progress drives BOTH properties: opacity grows from 0 to 1,
        # y-offset shrinks from 24 to 0, both on the same eased curve.
        t = elapsed / self.duration_ms
        eased = ease_emphasized(t)
        opacity = eased
        y_offset = _FADE_UP_DELTA_Y * (1.0 - eased)
        return opacity, y_offset


# ============================================================================
# HoverState -- per-tile cursor-over scaling
# ============================================================================
#
# When the mouse cursor enters a tile, that tile scales from 1.0 to 1.02
# over HOVER_SCALE_DURATION_MS (200ms), eased.  When it leaves, the tile
# scales back to 1.0 over the same duration.  Only one tile can be
# "current" at a time, but multiple tiles can be mid-transition (e.g.
# the user sweeps the cursor across the grid in a single motion -- tile
# A is decaying back to 1.0 while tile B is rising to 1.02).
#
# The subtle bit is the "interrupted transition" case:  if tile B's
# scale is mid-rise toward 1.02 and the cursor leaves it, the new
# "from" value for the decay must be B's CURRENT eased scale -- NOT a
# fresh 1.02.  Otherwise the scale snaps to 1.02 the instant the
# cursor leaves, and decays from there, producing a visible "kick".
# The discontinuity-prevention logic below records the current eased
# value as the new transition's starting point.

# Target scales for the two hover states.  1.02 is the apple_SKILL.md
# spec -- a 2% scale change reads as "this is reactive to your
# cursor" without ever looking like a button press.  Larger scales
# read as janky / mobile-app-like and break the visionOS feel.
_HOVER_SCALE_OFF: Final[float] = 1.0
_HOVER_SCALE_ON:  Final[float] = 1.02


@dataclass
class _HoverTransition:
    """One tile's most recent scale transition: from -> to over duration.

    Tracking both `from_scale` and `target_scale` (rather than just the
    target plus a "started at scale 1.0" assumption) is the entire fix
    for the interrupted-transition janky-kick bug described in the
    HoverState class doc.  When a transition is interrupted, we record
    the CURRENT eased scale as the new transition's `from_scale`.

    `start_ms` is the wall-clock millisecond at which this transition
    began.  duration is always HOVER_SCALE_DURATION_MS (constant), so
    we don't store it.
    """

    from_scale: float
    target_scale: float
    start_ms: int


class HoverState:
    """Tracks which tile is hovered + per-tile scale transitions.

    Public surface is two methods:
        - set_hover(tile_id, now_ms): the mouse callback calls this on
          every cursor move.  Internally it diffs the new tile_id
          against the previous hover; if they differ, it records a
          decay transition (target 1.0) on the OLD tile and a rise
          transition (target 1.02) on the NEW tile.  Identical
          tile_ids are a no-op so we don't reset mid-transition on
          every pixel of mouse motion.
        - scale_for(tile_id, now_ms): the renderer calls this for
          every visible tile each frame, threading the eased scale
          into its sub-image resize step.

    The internal state is two pieces:
        - _hovered: the currently-hovered tile_id, or None for "cursor
                    not over any tile".  Set/cleared by set_hover.
        - _transitions: dict mapping tile_id -> _HoverTransition.  A
                        tile that has never been hovered does not
                        appear here and scale_for returns the resting
                        scale (1.0) for it.

    Implementation note:  the prompt sketches the per-tile entry as
    `(target_scale, change_ms)` -- but that pair alone is insufficient
    to implement the discontinuity-prevention behaviour the same
    prompt requires ("when set_hover interrupts a mid-transition, the
    new 'from' is the current eased value").  Reproducing the eased
    current value from only (target, time) would require remembering
    the previous transition's target as well, which is exactly
    `from_scale` under a different name.  We make `from_scale`
    explicit instead, captured at the moment of transition.

    Why a class instead of free functions:  hover state is genuinely
    stateful (the previous hover target is what makes set_hover work),
    and threading that state through call sites as a dict + a
    "previous" int would be ugly.  CLAUDE.md's "no global mutable
    state" rule targets module-level mutable singletons; an explicit
    state object instantiated by the main loop and passed into the
    paint helpers is the structured alternative the rule allows.
    """

    def __init__(self) -> None:
        self._hovered: int | None = None
        self._transitions: dict[int, _HoverTransition] = {}

    def set_hover(self, tile_id: int | None, now_ms: int) -> None:
        """Update the hover target.  Idempotent for identical tile_ids.

        Three cases:
            1. tile_id == self._hovered:  no-op.  Don't reset a tile's
               transition just because the cursor wiggled by a pixel
               within it.
            2. tile_id != self._hovered AND old hover was a tile:
               schedule a decay transition (target 1.0) on the OLD
               tile using its CURRENT eased scale as the starting
               point (interrupted-transition handling, see below).
            3. tile_id != self._hovered AND new hover is a tile:
               schedule a rise transition (target 1.02) on the NEW
               tile, again using its CURRENT eased scale as the
               starting point.

        The "starting point = current eased value" trick is the
        load-bearing piece of jank-prevention.  Without it, sweeping
        the cursor across the grid fast enough produces visible
        snap-to-1.02-then-decay flickers as each tile's transition
        gets re-anchored at a fresh 1.02 instead of where it really
        was when the cursor left.  This is the bit that turns "the
        hover looks fine in isolated tests but feels wrong when you
        sweep" into "this feels native."
        """
        if tile_id == self._hovered:
            return

        # Schedule decay on the OUTGOING tile, anchored at its CURRENT
        # eased scale rather than a fresh 1.02.  This is the
        # interrupted-transition fix.
        if self._hovered is not None:
            current = self.scale_for(self._hovered, now_ms)
            self._transitions[self._hovered] = _HoverTransition(
                from_scale=current,
                target_scale=_HOVER_SCALE_OFF,
                start_ms=now_ms,
            )

        # Schedule rise on the INCOMING tile, again anchored at its
        # current eased scale (which may be 1.0 if it has never been
        # hovered, or somewhere between 1.0 and 1.02 if a previous
        # decay was still in flight).
        if tile_id is not None:
            current = self.scale_for(tile_id, now_ms)
            self._transitions[tile_id] = _HoverTransition(
                from_scale=current,
                target_scale=_HOVER_SCALE_ON,
                start_ms=now_ms,
            )

        self._hovered = tile_id

    def scale_for(self, tile_id: int, now_ms: int) -> float:
        """Return the eased scale for `tile_id` at wall-clock `now_ms`.

        A tile with no recorded transition is at its resting scale
        (1.0).  Otherwise, the transition's elapsed time is normalised
        against HOVER_SCALE_DURATION_MS and run through
        `ease_emphasized` to produce the eased blend factor; the
        scale then lerps from `from_scale` to `target_scale` by that
        factor.

        Elapsed times beyond the duration clamp to the target scale
        (the transition has completed); elapsed times before the
        start_ms (shouldn't happen in practice, since transitions are
        only recorded at the current `now_ms`) clamp to from_scale.
        """
        transition = self._transitions.get(tile_id)
        if transition is None:
            return _HOVER_SCALE_OFF

        elapsed = now_ms - transition.start_ms
        if elapsed <= 0:
            return transition.from_scale
        if elapsed >= HOVER_SCALE_DURATION_MS:
            return transition.target_scale

        t = elapsed / HOVER_SCALE_DURATION_MS
        eased = ease_emphasized(t)
        # Lerp: from + (target - from) * eased.
        return (
            transition.from_scale
            + (transition.target_scale - transition.from_scale) * eased
        )


# ============================================================================
# Public surface
# ============================================================================
#
# Only four symbols in this module are meant to be called from outside:
#
#     cubic_bezier(t, p1x, p1y, p2x, p2y) -> float
#         The generic CSS cubic-bezier evaluator.  Phases should prefer
#         ease_emphasized; reach for this only when adding a NEW curve.
#     ease_emphasized(t) -> float
#         The single curve every animation in the demo uses.
#     FadeUpState(start_ms, duration_ms=800)
#         Per-tile entry animation state.
#     HoverState()
#         Per-tile cursor-over scaling state.
#
# Everything else (_HoverTransition, _bezier_x, _bezier_y, the iteration
# / tolerance constants) is module-private and may be reshuffled without
# notice.
# ============================================================================
