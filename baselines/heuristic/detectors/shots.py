"""Shots.

For every shot event in the feed, find the release frame from tracking. Three
scored strategies (ball high with the shooter as last handler, shooter holding a
rising ball, shooter close to a high ball) plus a tip/putback strategy tried first
for tips. The winning frame is traced back to the release, then walked forward
to the apex for layups and dunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from baselines.heuristic.config import HeuristicConfig
from baselines.heuristic.inputs import Action, Event, Frame, dist, dist_to_hoop, frames_between

LAYUP_FAMILY = {"layup", "dunk", "tip"}


@dataclass
class Candidate:
    frame: Frame
    score: float
    method: str


def detect_shots(
    frames: list[Frame], shot_events: list[Event], team: str, hoops, config: HeuristicConfig
) -> list[Action]:
    """``frames`` are one possession's frames plus a margin; ``team`` is the team in possession."""
    shots = []
    for event in sorted(shot_events, key=lambda e: e.t):
        if event.player is None:
            continue
        window = frames_between(frames, event.t - config.shot_window, event.t + config.shot_window) or frames
        best = _tip_candidate(window, event, hoops, config) if event.subtype == "tip" else None
        best = best or _standard_candidate(window, event, config)
        if best is None:
            continue
        release = (
            best.frame
            if best.method == "tip_putback"
            else _trace_back_release(window, best.frame, event.player, config)
        )
        if event.subtype in LAYUP_FAMILY and best.method != "tip_putback":
            release = _walk_forward_to_apex(window, release, event.player, config)
        x, y = release.location_of(event.player)
        shots.append(
            Action(
                "shot",
                release,
                event.player,
                team,
                x=x,
                y=y,
                outcome=event.outcome,
                method=best.method,
                event_id=event.event_id,
            )
        )
    return shots


def detect_putbacks_from_rebounds(
    frames: list[Frame],
    shot_events: list[Event],
    detected: list[Action],
    rebounds: list[Action],
    team: str,
    hoops,
    config: HeuristicConfig,
) -> list[Action]:
    """A tip event with no detected shot, right after an offensive rebound by the same player, becomes a putback."""
    found = {s.event_id for s in detected if s.event_id is not None}
    offensive = [r for r in rebounds if r.subtype == "offensive" and r.player is not None]
    tips = {e.event_id: e for e in shot_events if e.subtype == "tip"}
    out = []
    for event in sorted(tips.values(), key=lambda e: e.t):
        if event.event_id in found or event.player is None:
            continue
        candidates = [
            r
            for r in offensive
            if r.player == event.player
            and r.team == event.team
            and -config.putback_before_rebound <= event.t - r.t <= config.putback_after_rebound
        ]
        if not candidates:
            continue
        rebound = min(candidates, key=lambda r: abs(event.t - r.t))
        frame = _putback_frame(frames, event.player, rebound.t, hoops, config)
        if frame is None:
            continue
        x, y = frame.location_of(event.player)
        out.append(
            Action(
                "shot",
                frame,
                event.player,
                team,
                x=x,
                y=y,
                outcome=event.outcome,
                method="putback_after_rebound",
                event_id=event.event_id,
                confidence=0.85,
            )
        )
        found.add(event.event_id)
    return out


def _putback_frame(frames: list[Frame], shooter_id: str, rebound_t: float, hoops, config: HeuristicConfig):
    """The first frame after the rebound with the shooter on the ball near a hoop, else the first frame after it."""
    near = config.putback_search_fallback
    after = [f for f in frames if rebound_t < f.t <= rebound_t + config.putback_search_after] or frames_between(
        frames, rebound_t - near, rebound_t + near
    )
    for f in after:
        shooter = f.player(shooter_id)
        if (
            shooter
            and dist_to_hoop(f.ball_x, f.ball_y, hoops) <= config.tip_basket_radius
            and dist(shooter.x, shooter.y, f.ball_x, f.ball_y) <= config.tip_proximity
        ):
            return f
    return after[0] if after else None


# ---------------------------------------------------------------- candidates


def _standard_candidate(window: list[Frame], event: Event, config: HeuristicConfig) -> Candidate | None:
    best: Candidate | None = None
    holder_history = [(f.holder, f.t) for f in window if f.holder is not None]
    for i, f in enumerate(window):
        previous = next((h for h, t in reversed(holder_history) if t <= f.t), None)
        if f.ball_z <= config.shot_height and f.holder is not None:
            previous = f.holder
        closeness = 1.0 / (1.0 + abs(f.t - event.t))
        if f.ball_z >= config.shot_height:
            if previous is not None and previous.id == event.player:
                best = _better(best, Candidate(f, 10.0 + f.ball_z + 2.0 * closeness, "high_ball_prev_handler"))
            d = f.distance_to_ball(event.player)
            if d is not None and d <= config.shot_proximity:
                proximity = (config.shot_proximity - d) / config.shot_proximity
                best = _better(
                    best, Candidate(f, 8.0 + f.ball_z + 3.0 * proximity + 2.0 * closeness, "proximity_based")
                )
            continue
        if f.holder is not None and f.holder.id == event.player:
            future = [g.ball_z for g in window[i + 1 : i + 1 + config.shot_rise_lookahead_frames]]
            peak = max(future, default=f.ball_z)
            if peak > f.ball_z or peak > config.shot_height:
                best = _better(
                    best, Candidate(f, 10.0 + f.ball_z + 0.1 * peak + 2.0 * closeness, "player_with_ball_trajectory")
                )
    return best


def _tip_candidate(window: list[Frame], event: Event, hoops, config: HeuristicConfig) -> Candidate | None:
    best: Candidate | None = None
    for f in window:
        offset = f.t - event.t
        if offset < -config.tip_window_before or offset > config.tip_window_after or f.ball_z < config.tip_height:
            continue
        to_hoop = dist_to_hoop(f.ball_x, f.ball_y, hoops)
        d = f.distance_to_ball(event.player)
        if to_hoop > config.tip_basket_radius or d is None or d > config.tip_proximity:
            continue
        score = (
            10.0 / (1.0 + abs(offset))
            + 3.0 * (config.tip_proximity - d) / config.tip_proximity
            + 2.0 * (config.tip_basket_radius - to_hoop) / config.tip_basket_radius
            + min(f.ball_z, 12.0)
        )
        best = _better(best, Candidate(f, score, "tip_putback"))
    return best


def _better(current: Candidate | None, new: Candidate) -> Candidate:
    return new if current is None or new.score > current.score else current


def _trace_back_release(window: list[Frame], detection: Frame, shooter: str, config: HeuristicConfig) -> Frame:
    start = window.index(detection)
    for f in reversed(window[max(0, start - config.release_walk_back_frames) : start]):
        if f.ball_z < config.shot_height:
            if f.holder is not None and f.holder.id == shooter:
                return f
            d = f.distance_to_ball(shooter)
            if d is not None and d <= config.shot_proximity:
                return f
    return detection


def _walk_forward_to_apex(window: list[Frame], release: Frame, shooter: str, config: HeuristicConfig) -> Frame:
    start = window.index(release)
    current, z = release, release.ball_z
    for f in window[start + 1 : start + 1 + config.release_walk_forward_frames]:
        d = f.distance_to_ball(shooter)
        if d is None or d > config.apex_max_distance or f.ball_z < z - config.apex_drop_tolerance:
            break
        current, z = f, f.ball_z
    return current
