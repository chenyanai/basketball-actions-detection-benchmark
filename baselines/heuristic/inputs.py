"""Internal representation the heuristic detectors work on: frames, events, actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import schema
from baselines.heuristic.config import HeuristicConfig, default_config
from data_loader.game import Game

CLOCK_STOPPED = 0.001  # two frames this close in game clock show a stopped clock


@dataclass(slots=True, eq=False)
class Player:
    id: str
    team: str
    x: float
    y: float


@dataclass(slots=True, eq=False)
class Frame:
    idx: int
    period: int
    clock: float
    shot_clock: float | None
    t: float
    ball_x: float
    ball_y: float
    ball_z: float
    players: list[Player]
    holder: Player | None = None

    def player(self, player_id: str) -> Player | None:
        for p in self.players:
            if p.id == player_id:
                return p
        return None

    def location_of(self, player_id: str | None) -> tuple[float, float]:
        """The player's position on this frame, or the ball's when the player is not on court."""
        p = self.player(player_id)
        return (p.x, p.y) if p else (self.ball_x, self.ball_y)

    def distance_to_ball(self, player_id: str) -> float | None:
        p = self.player(player_id)
        return None if p is None else dist(p.x, p.y, self.ball_x, self.ball_y)


@dataclass(slots=True, eq=False)
class Event:
    kind: str
    subtype: str | None
    period: int
    clock: float
    t: float
    team: str
    player: str | None
    player2: str | None
    outcome: str | None
    event_id: int | None = None
    team2: str | None = None  # team of player2
    description: str = ""

    @property
    def made(self) -> bool:
        return self.kind == "shot" and self.outcome == "made"

    @property
    def missed(self) -> bool:
        return self.kind == "shot" and self.outcome == "missed"

    @property
    def is_shooting_foul(self) -> bool:
        return self.kind == "foul" and self.subtype == "shooting"


@dataclass
class Action:
    kind: str
    frame: Frame
    player: str | None
    team: str
    player2: str | None = None
    x: float | None = None
    y: float | None = None
    subtype: str | None = None
    outcome: str | None = None
    confidence: float = 1.0
    method: str = ""
    event_id: int | None = None
    time: float | None = None  # when it differs from the frame's time: a foul keeps the feed's time
    possession: int | None = None  # the number of the possession it was found in

    @property
    def t(self) -> float:
        return self.frame.t if self.time is None else self.time


def frames_from_game(game: Game, config: HeuristicConfig | None = None) -> list[Frame]:
    """The frames the detectors work on, in time order: one per game-clock reading, plus the moments before a restart.

    A stopped clock repeats its reading, so keeping one frame per reading leaves live play only.
    ``pre_inbound_seconds`` adds back the last moments of each stoppage, where the inbound pass is thrown.
    """
    config = config or default_config()
    f = game.frames
    period, clock, idx = f["period"].to_numpy(), f["game_clock"].to_numpy(dtype=float), f["frame_idx"].to_numpy()
    shot_clock = f["shot_clock"].to_numpy(dtype=float)
    if config.backfill_shot_clock:
        shot_clock = _backfill(shot_clock, period)

    order = np.lexsort((idx, -clock, period))
    first_reading = np.r_[True, (period[order][1:] != period[order][:-1]) | (clock[order][1:] != clock[order][:-1])]
    keep = np.zeros(len(f), dtype=bool)
    keep[order[first_reading]] = True
    keep |= _before_restart(period, clock, idx, config.pre_inbound_seconds * game.fps)

    players = game.players.sort_values("frame_idx", kind="stable")
    p_frame = players["frame_idx"].to_numpy()
    p_id, p_team = players["player_id"].to_numpy(), players["team_id"].to_numpy()
    p_x, p_y = players["x"].to_numpy(), players["y"].to_numpy()
    lo, hi = np.searchsorted(p_frame, idx, side="left"), np.searchsorted(p_frame, idx, side="right")
    ball = f[["ball_x", "ball_y", "ball_z"]].to_numpy(dtype=float)

    frames = []
    for i in order[keep[order]]:
        on_court = [Player(str(p_id[j]), str(p_team[j]), float(p_x[j]), float(p_y[j])) for j in range(lo[i], hi[i])]
        sc = shot_clock[i]
        frames.append(
            Frame(
                int(idx[i]),
                int(period[i]),
                float(clock[i]),
                None if np.isnan(sc) else float(sc),
                game.absolute_time(int(period[i]), float(clock[i])),
                *ball[i],
                on_court,
            )
        )
    return frames


def _before_restart(period: np.ndarray, clock: np.ndarray, idx: np.ndarray, max_frames: float) -> np.ndarray:
    """Stopped-clock frames at most ``max_frames`` before the clock runs again in the same period."""
    n = len(clock)
    if n < 2 or max_frames <= 0:
        return np.zeros(n, dtype=bool)
    same = (period[1:] == period[:-1]) & (clock[1:] == clock[:-1])
    stopped = np.r_[False, same]
    run_end = np.where(np.r_[~same, True], np.arange(n), n)
    run_end = np.minimum.accumulate(run_end[::-1])[::-1]  # last frame of the run each frame is in
    restart = np.minimum(run_end + 1, n - 1)
    restarts = (run_end + 1 < n) & (period[restart] == period)
    return stopped & restarts & (idx[restart] - idx <= max_frames)


def apply_ball_handler(frames: list[Frame], ball_handler: pd.DataFrame) -> None:
    holder = dict(zip(ball_handler["frame_idx"], ball_handler["player_id"]))
    for f in frames:
        pid = holder.get(f.idx)
        f.holder = None if pd.isna(pid) else f.player(str(pid))


TRANSPARENT = ("substitution", "timeout")  # feed rows every rule looks through


def events_from_feed(feed: pd.DataFrame, game: Game) -> list[Event]:
    """Feed rows as events, in period and clock order. Rebounds are relabelled from the team of the last miss."""
    team_of = game.team_by_player
    events = []
    for r in feed.itertuples(index=False):
        player2 = _none(r.player2_id)
        events.append(
            Event(
                r.event_type,
                _none(r.subtype),
                int(r.period),
                float(r.clock),
                game.absolute_time(int(r.period), float(r.clock)),
                _none(r.team_id),
                _none(r.player_id),
                player2,
                _none(r.outcome),
                getattr(r, "event_id", None),
                team_of.get(player2),
                _none(getattr(r, "description", None)) or "",
            )
        )
    events.sort(key=lambda e: (e.period, -e.clock))
    _relabel_rebounds(events)
    return events


def _relabel_rebounds(events: list[Event]) -> None:
    """A rebound is offensive when the rebounder's team took the last missed shot or free throw."""
    last_miss, period = None, None
    for e in events:
        if e.period != period:
            last_miss, period = None, e.period
        if e.outcome == "missed" and e.kind in ("shot", "free_throw"):
            last_miss = e.team
        elif e.made:
            last_miss = None
        elif e.kind == "rebound":
            if e.team is not None and last_miss is not None:
                e.subtype = "offensive" if e.team == last_miss else "defensive"
            if e.subtype != "offensive":
                last_miss = None


def actions_to_table(actions: list[Action]) -> pd.DataFrame:
    rows = [
        {
            "period": a.frame.period,
            "frame_idx": a.frame.idx,
            "end_frame": None,
            "action_type": a.kind,
            "player_id": a.player,
            "player2_id": a.player2,
            "team_id": a.team,
            "x": a.x,
            "y": a.y,
            "subtype": a.subtype,
            "outcome": a.outcome,
            "confidence": a.confidence,
            "method": a.method,
            "event_id": a.event_id,
            "possession": a.possession,
        }
        for a in actions
    ]
    return pd.DataFrame(rows, columns=list(schema.COLUMNS[schema.ACTIONS]) + ["method", "event_id", "possession"])


def frames_between(frames: list[Frame], start: float, end: float) -> list[Frame]:
    return [f for f in frames if start <= f.t <= end]


def closest_frame(frames: list[Frame], t: float) -> Frame | None:
    return min(frames, key=lambda f: abs(f.t - t)) if frames else None


def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def dist_to_hoop(x: float, y: float, hoops) -> float:
    return min(dist(x, y, hx, hy) for hx, hy in hoops)


def _backfill(values: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Fill each blank with the next reading in the same period."""
    return pd.Series(values).groupby(periods).bfill().to_numpy()


def _none(value) -> str | None:
    """A feed cell as text, or None when it is empty."""
    return None if pd.isna(value) else str(value)
