"""The actions stage: every action is detected inside its possession.

The game is first split into possessions (possessions.py), each holding its own frames and its own feed
events. The feed says which shots, rebounds, turnovers and fouls a possession holds and who made them;
tracking says when and where. Each possession is handled on its own frames plus a margin, its detections
are kept only near its own time span, and one event is never reported twice. Passes are pure tracking,
from the possession's own frames.
Every action carries the number of the possession it was found in.
"""

from __future__ import annotations

import bisect

import pandas as pd

import schema
from baselines import pbp_clock
from baselines.heuristic.ball_handler import annotate, teams_from_possessions
from baselines.heuristic.config import HeuristicConfig, default_config
from baselines.heuristic.detectors.fouls import detect_fouls
from baselines.heuristic.detectors.passes import detect_passes
from baselines.heuristic.detectors.rebounds import detect_rebounds, known_shots
from baselines.heuristic.detectors.shots import detect_putbacks_from_rebounds, detect_shots
from baselines.heuristic.detectors.turnovers import detect_turnovers
from baselines.heuristic.inputs import (
    Action,
    Event,
    Frame,
    actions_to_table,
    apply_ball_handler,
    closest_frame,
    events_from_feed,
    frames_from_game,
)
from baselines.heuristic.possessions import Possession, find_possessions
from data_loader.game import Game

TYPES = ("pass", "shot", "rebound", "turnover", "foul")
FEED_TYPES = ("shot", "rebound", "turnover", "foul")


def detect_actions(
    game: Game,
    pbp: pd.DataFrame,
    possessions: pd.DataFrame | None = None,
    ball_handler: pd.DataFrame | None = None,
    config: HeuristicConfig | None = None,
    types: tuple[str, ...] = TYPES,
) -> pd.DataFrame:
    """The actions of the benchmark's third stage; ``types`` keeps only some of them.

    ``possessions`` is not read: the heuristic always finds its own, because each of its possessions holds the
    feed events it searches for, and a possessions table has no events.
    """
    config = config or default_config()
    frames = frames_from_game(game, config)
    own = find_possessions(game, frames, pbp, config)
    if ball_handler is not None:
        apply_ball_handler(frames, ball_handler)
    else:
        annotate(frames, teams_from_possessions(frames, own), config)
    shot_events = [e for e in events_from_feed(pbp, game) if e.kind == "shot"]
    actions = _detect_per_possession(game, frames, own, shot_events, config)
    return schema.normalize(schema.ACTIONS, actions_to_table([a for a in actions if a.kind in types]))


def match_pbp(game: Game, pbp: pd.DataFrame, config: HeuristicConfig | None = None) -> pd.DataFrame:
    """Play-by-play matching: each feed row goes where its detector found it; rows the detectors skip fall back
    to the clock-only placement."""
    actions = detect_actions(game, pbp=pbp, config=config, types=FEED_TYPES)
    found = actions.dropna(subset=["event_id"]).drop_duplicates("event_id")[["event_id", "frame_idx", "x", "y"]]
    fallback = pbp_clock.match_pbp(game, pbp)
    return pd.concat([found, fallback[~fallback["event_id"].isin(found["event_id"])]], ignore_index=True)


def _detect_per_possession(
    game: Game, frames: list[Frame], possessions: list[Possession], shot_events: list[Event], config: HeuristicConfig
) -> list[Action]:
    """``shot_events`` are all the feed's shots: a rebound may follow a shot filed with an earlier possession."""
    times = [f.t for f in frames]
    actions: list[Action] = []
    earlier_shots: list[Action] = []
    attempted, reported = set(), set()  # shot events already tried; events already reported
    kept: list[Action] = []  # shots and rebounds reported so far, for the duplicate check

    for p in possessions:
        done = {"shot": attempted, "rebound": reported, "turnover": reported, "foul": reported}
        mine = {kind: [e for e in p.events_of(kind) if e.event_id not in done[kind]] for kind in FEED_TYPES}
        attempted |= {e.event_id for e in mine["shot"]}
        if not p.frames and not any(mine.values()):
            continue
        start, end = p.bounds
        lo = bisect.bisect_left(times, start - config.possession_margin)
        hi = bisect.bisect_right(times, end + config.possession_margin)
        found = _detect_in_possession(game, frames[lo:hi], p, mine, shot_events, earlier_shots, frames, config)

        found = [a for a in found if _belongs(a, p, config)]
        for a in found:
            a.possession = p.number
        earlier_shots += [a for a in found if a.kind == "shot"]
        for a in found:
            if a.kind == "pass":
                actions.append(a)
                continue
            if a.event_id is not None and a.event_id in reported:
                continue
            if a.kind in ("shot", "rebound"):
                if any(_same_action(a, b, config.duplicate_window) for b in kept):
                    continue
                kept.append(a)
            if a.event_id is not None:
                reported.add(a.event_id)
            actions.append(a)
    return sorted(actions, key=lambda a: a.t)


def _detect_in_possession(
    game: Game,
    window: list[Frame],
    p: Possession,
    mine: dict[str, list[Event]],
    all_shot_events: list[Event],
    earlier_shots: list[Action],
    all_frames: list[Frame],
    config: HeuristicConfig,
) -> list[Action]:
    hoops = game.court.hoops
    if not window:
        return [_at_feed_time(e, all_frames) for e in mine["turnover"] + mine["foul"]]

    shots = detect_shots(window, mine["shot"], p.team, hoops, config)
    rebounds = []
    if mine["rebound"]:
        known = known_shots(earlier_shots + shots, all_shot_events, config)
        rebounds = detect_rebounds(window, mine["rebound"], known, hoops, config)
    shots += detect_putbacks_from_rebounds(
        window, mine["shot"] + all_shot_events, shots, rebounds, p.team, hoops, config
    )
    turnovers, incomplete = detect_turnovers(window, mine["turnover"], config)
    fouls = detect_fouls(window, mine["foul"], config)
    passes = detect_passes(p.frames, shots, config)
    return sorted(shots + rebounds + turnovers + incomplete + fouls + passes, key=lambda a: a.t)


def _belongs(a: Action, p: Possession, config: HeuristicConfig) -> bool:
    """Detectors search a margin around the possession; keep what falls near enough to its own time span."""
    start, end = p.bounds
    if a.kind == "pass":
        return start <= a.t < end and a.team == p.team
    if a.kind in ("shot", "rebound"):
        return start - config.early_shot_margin <= a.t <= end + config.possession_margin
    return start - config.boundary_margin <= a.t <= end + config.boundary_margin


def _same_action(a: Action, b: Action, window: float) -> bool:
    return a.kind == b.kind and a.player == b.player and a.frame.period == b.frame.period and abs(a.t - b.t) <= window


def _at_feed_time(event: Event, frames: list[Frame]) -> Action:
    """No tracking around the possession: the event keeps the feed's time."""
    frame = closest_frame(frames, event.t)
    return Action(
        event.kind,
        frame,
        event.player,
        event.team,
        player2=event.player2,
        x=frame.ball_x,
        y=frame.ball_y,
        subtype=event.subtype,
        method="play_by_play_time",
        event_id=event.event_id,
        time=event.t,
    )
