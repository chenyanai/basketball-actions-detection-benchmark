"""Rebounds: the feed says who rebounded; tracking says when.

The relevant shot is found first, then the first frame where the ball has peaked,
come back down, and the rebounder is next to it. A chain of increasingly relaxed
strategies ends at a fixed delay after the shot.
"""

from __future__ import annotations

from dataclasses import dataclass

from baselines.heuristic.config import HeuristicConfig
from baselines.heuristic.inputs import Action, Event, Frame, closest_frame, dist_to_hoop


@dataclass
class KnownShot:
    t: float
    team: str | None
    from_feed: bool = False  # known from the feed only, so its time is the feed's


def known_shots(detected: list[Action], shot_events: list[Event], config: HeuristicConfig) -> list[KnownShot]:
    """Detected shots, plus the feed's shots that no known shot is close to."""
    shots = [KnownShot(s.t, s.team) for s in detected]
    for e in shot_events:
        if not any(abs(e.t - s.t) < config.rebound_shot_match_window for s in shots):
            shots.append(KnownShot(e.t, e.team, from_feed=True))
    return sorted(shots, key=lambda s: s.t)


def detect_rebounds(
    frames: list[Frame], rebound_events: list[Event], shots: list[KnownShot], hoops, config: HeuristicConfig
) -> list[Action]:
    """``frames`` are one possession's frames plus a margin."""
    rebounds = []
    for event in rebound_events:
        shot = _relevant_shot(shots, event.t, config)
        offensive = event.subtype == "offensive"
        if shot is not None and shot.team is not None and event.team is not None:
            offensive = event.team == shot.team
        eligible = []
        if shot is not None:
            start = shot.t - (config.rebound_feed_shot_lookback if shot.from_feed else 0.0)
            eligible = [f for f in frames if start < f.t <= shot.t + config.rebound_window]
        if not eligible:
            eligible = [
                f
                for f in frames
                if event.t - config.rebound_fallback_before < f.t <= event.t + config.rebound_fallback_after
            ]
        found = _search(eligible, event, offensive, shot, hoops, config) if eligible else None
        if found is None:
            delayed = shot is not None and bool(eligible)
            found = _last_resort(frames, shot.t + config.rebound_fallback_delay if delayed else event.t)
        if found is None:
            continue
        frame, method = found
        rebounds.append(
            Action(
                "rebound",
                frame,
                event.player,
                event.team,
                x=frame.ball_x,
                y=frame.ball_y,
                subtype="offensive" if offensive else "defensive",
                method=method,
                event_id=event.event_id,
                confidence=0.5 if method == "play_by_play_time" else 0.9,
            )
        )
    return rebounds


def _relevant_shot(shots: list[KnownShot], rebound_t: float, config: HeuristicConfig) -> KnownShot | None:
    margin = config.rebound_window + config.rebound_shot_margin
    for shot in reversed(shots):
        if shot.t < rebound_t:
            if rebound_t - shot.t <= margin:
                return shot
            break
    if shots and rebound_t - shots[0].t <= margin:
        return shots[0]
    return None


def _search(frames: list[Frame], event: Event, offensive: bool, shot: KnownShot | None, hoops, config: HeuristicConfig):
    player = event.player
    if player is None:
        return None
    description = event.description.lower()
    if shot is not None and ("tip" in description or "putback" in description):
        found = _via_tip(frames, player, shot.t, hoops, config)
        if found:
            return found
    if offensive:
        found = _via_ball_holder(frames, player, config)
        if found:
            return found
    return (
        _via_proximity(frames, player, config.rebound_proximity, config.rebound_min_frames, config)
        or _via_proximity(frames, player, config.rebound_proximity_relaxed, 1, config)
        or _via_simple_proximity(frames, player, config.rebound_proximity_relaxed)
    )


def _via_tip(frames: list[Frame], player: str, shot_t: float, hoops, config: HeuristicConfig):
    """A tip barely comes down: a tighter window, the ball under rim level, the rebounder on it near a hoop."""
    for f in frames:
        if f.t > shot_t + config.rebound_tip_window:
            break
        d = f.distance_to_ball(player)
        if (
            f.ball_z < config.rebound_tip_height
            and d is not None
            and d <= config.rebound_proximity
            and dist_to_hoop(f.ball_x, f.ball_y, hoops) <= config.rebound_tip_basket_radius
        ):
            return f, "tip_proximity"
    return None


def _via_proximity(frames: list[Frame], player: str, radius: float, min_frames: int, config: HeuristicConfig):
    peaked, close = False, []
    for f in frames:
        if f.ball_z >= config.rebound_peak_height:
            peaked = True
            continue
        if not peaked or f.ball_z >= config.rebound_possession_height:
            continue
        d = f.distance_to_ball(player)
        if d is not None and d <= radius:
            close.append(f)
            if len(close) >= min_frames:
                return close[0], "trajectory_proximity"
    return None


def _via_ball_holder(frames: list[Frame], player: str, config: HeuristicConfig):
    peaked, held, first = False, [], None
    for i, f in enumerate(frames):
        if f.ball_z >= config.rebound_peak_height:
            peaked = True
            continue
        if not peaked or f.ball_z >= config.rebound_possession_height:
            continue
        if f.holder is not None and f.holder.id == player:
            held.append(f)
            first = i if first is None else first
            if len(held) >= config.rebound_hold_min_frames and i - first >= config.rebound_hold_min_span_frames:
                return held[min(config.rebound_hold_min_frames, len(held) - 1)], "trajectory_ball_holder"
    return None


def _via_simple_proximity(frames: list[Frame], player: str, radius: float):
    for f in frames:
        d = f.distance_to_ball(player)
        if d is not None and d <= radius:
            return f, "simple_proximity"
    return None


def _last_resort(frames: list[Frame], target: float):
    """The first frame at or after the target time, else the closest one."""
    after = [f for f in frames if f.t >= target]
    frame = min(after, key=lambda f: f.t) if after else closest_frame(frames, target)
    return (frame, "play_by_play_time") if frame is not None else None
