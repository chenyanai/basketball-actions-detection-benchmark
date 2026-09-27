"""Fouls come straight from the feed, time included; tracking only adds where the fouler stood."""

from __future__ import annotations

from baselines.heuristic.config import HeuristicConfig
from baselines.heuristic.inputs import Action, Event, Frame, closest_frame, frames_between


def detect_fouls(frames: list[Frame], foul_events: list[Event], config: HeuristicConfig) -> list[Action]:
    """``frames`` are one possession's frames plus a margin."""
    fouls = []
    for event in foul_events:
        near = frames_between(frames, event.t - config.foul_frame_window, event.t + config.foul_frame_window)
        frame = closest_frame(near or frames, event.t)
        if frame is None:
            continue
        x, y = frame.location_of(event.player)
        fouls.append(
            Action(
                "foul",
                frame,
                event.player,
                event.team,
                player2=event.player2,
                x=x,
                y=y,
                subtype=event.subtype,
                method="play_by_play_time",
                event_id=event.event_id,
                time=event.t,
            )
        )
    return fouls
