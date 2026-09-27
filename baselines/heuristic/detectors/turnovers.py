"""Turnovers: walk back from the play-by-play time to the frame where the ball left the player's hands.

A bad pass also yields an incomplete pass action, to the stealer when the feed names one.
"""

from __future__ import annotations

from baselines.heuristic.config import HeuristicConfig
from baselines.heuristic.inputs import CLOCK_STOPPED, Action, Event, Frame, closest_frame, frames_between


def detect_turnovers(
    frames: list[Frame], turnover_events: list[Event], config: HeuristicConfig
) -> tuple[list[Action], list[Action]]:
    """``frames`` are one possession's frames plus a margin. Returns (turnovers, incomplete passes)."""
    turnovers, incomplete = [], []
    for event in turnover_events:
        lookback = {"bad_pass": config.bad_pass_lookback, "lost_ball": config.lost_ball_lookback}.get(
            event.subtype, config.violation_lookback
        )
        found, method = (
            _ball_left_hands(frames, event.player, event.t, lookback, config) if event.player else (None, "")
        )
        frame = found or closest_frame(frames, event.t)
        if frame is None:
            continue
        if event.subtype in ("violation", "offensive_foul") and _clock_stopped(
            frames, event.t, lookback + config.clock_stopped_extra_lookback
        ):
            method = f"{event.subtype}_clock_verified"
        elif found is None:
            method = "play_by_play_time"
        x, y = frame.location_of(event.player)
        turnovers.append(
            Action(
                "turnover",
                frame,
                event.player,
                event.team,
                player2=event.player2,
                x=x,
                y=y,
                subtype=event.subtype,
                method=method,
                event_id=event.event_id,
            )
        )
        if event.subtype == "bad_pass":
            incomplete.append(
                Action(
                    "pass",
                    frame,
                    event.player,
                    event.team,
                    player2=event.player2,
                    x=x,
                    y=y,
                    outcome="incomplete",
                    method="bad_pass_turnover",
                )
            )
    return turnovers, incomplete


def _ball_left_hands(
    frames: list[Frame], player: str, t: float, lookback: float, config: HeuristicConfig
) -> tuple[Frame | None, str]:
    window = frames_between(frames, t - lookback, t + config.turnover_window_after)
    last_with_ball = None
    for f, nxt in zip(window, window[1:]):
        if f.holder is not None and f.holder.id == player and (nxt.holder is None or nxt.holder.id != player):
            last_with_ball = f
    if last_with_ball is not None:
        return last_with_ball, "handler_transition"
    for f in reversed(window):
        if f.holder is not None and f.holder.id == player:
            return f, "handler_last_frame"
    best, best_d = None, float("inf")
    for f in window:
        d = f.distance_to_ball(player)
        if (
            d is not None
            and d < best_d
            and d <= config.turnover_proximity
            and f.ball_z < config.turnover_max_ball_height
        ):
            best, best_d = f, d
    return (best, "handler_proximity") if best is not None else (None, "")


def _clock_stopped(frames: list[Frame], t: float, lookback: float) -> bool:
    window = frames_between(frames, t - lookback, t)
    return any(abs(a.clock - b.clock) < CLOCK_STOPPED for a, b in zip(window, window[1:]))
