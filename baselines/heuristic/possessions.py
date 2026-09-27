"""Possession detection, the first step of everything else: the game is split into possessions, and each
possession holds its own frames and its own feed events, so later steps know who shot, rebounded, lost the
ball or fouled in it.

A "team with the ball" walk over the feed's events decides who has possession after each
one and which events belong to it, consecutive same-team spans are merged, and every
boundary is snapped to the shot-clock reset seen in tracking.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import pandas as pd

from baselines.heuristic.config import HeuristicConfig, default_config
from baselines.heuristic.inputs import CLOCK_STOPPED, TRANSPARENT, Event, Frame, events_from_feed, frames_from_game
from data_loader.game import Game


@dataclass
class Possession:
    period: int
    team: str
    start: float  # feed times, whole seconds
    end: float
    events: list[Event] = field(default_factory=list)  # the feed events filed with this possession, in feed order
    tracking_start: float | None = None  # the same boundaries moved onto tracking
    tracking_end: float | None = None
    frames: list[Frame] = field(default_factory=list)
    number: int = 0  # position in the game; detected actions carry it

    @property
    def bounds(self) -> tuple[float, float]:
        start = self.start if self.tracking_start is None else self.tracking_start
        end = self.end if self.tracking_end is None else self.tracking_end
        return min(start, end), max(start, end)

    def events_of(self, kind: str) -> list[Event]:
        """This possession's shots, rebounds, turnovers or fouls.

        A defensive rebound is filed with both possessions it sits between; it counts for the rebounder's.
        """
        return [e for e in self.events if e.kind == kind and (kind != "rebound" or e.team == self.team)]


def detect_possessions(game: Game, pbp: pd.DataFrame, config: HeuristicConfig | None = None) -> pd.DataFrame:
    config = config or default_config()
    return to_table(find_possessions(game, frames_from_game(game, config), pbp, config))


def find_possessions(game: Game, frames: list[Frame], pbp: pd.DataFrame, config: HeuristicConfig) -> list[Possession]:
    walked = walk_events(events_from_feed(pbp, game), game)
    possessions = snap_to_tracking(merge_same_team(walked), frames, config, 1.0 / game.fps)
    assign_frames(possessions, frames)
    return possessions


def assign_frames(possessions: list[Possession], frames: list[Frame]) -> None:
    """Each frame goes to the possession whose [start, end) holds its time; frames between possessions go nowhere."""
    possessions.sort(key=_start)
    for number, p in enumerate(possessions):
        p.frames, p.number = [], number
    i = 0
    for f in frames:
        while i < len(possessions) and f.t >= _end(possessions[i]):
            i += 1
        if i == len(possessions):
            break
        if _start(possessions[i]) <= f.t:
            possessions[i].frames.append(f)


def to_table(possessions: list[Possession]) -> pd.DataFrame:
    rows = [
        {
            "period": p.period,
            "start_frame": min(f.idx for f in p.frames),
            "end_frame": max(f.idx for f in p.frames),
            "team_id": p.team,
            "possession": p.number,
        }
        for p in possessions
        if p.frames
    ]
    return pd.DataFrame(rows, columns=["period", "start_frame", "end_frame", "team_id", "possession"])


def events_table(possessions: list[Possession]) -> pd.DataFrame:
    """What the feed says happened in each possession and who did it, one row per event."""
    rows = [
        {
            "possession": p.number,
            "period": p.period,
            "team_id": p.team,
            "event_id": e.event_id,
            "event_type": e.kind,
            "subtype": e.subtype,
            "player_id": e.player,
            "outcome": e.outcome,
        }
        for p in possessions
        for kind in ("shot", "rebound", "turnover", "foul")
        for e in p.events_of(kind)
    ]
    columns = ["possession", "period", "team_id", "event_id", "event_type", "subtype", "player_id", "outcome"]
    return pd.DataFrame(rows, columns=columns).sort_values(["possession", "event_id"], ignore_index=True)


def _start(p: Possession) -> float:
    return p.start if p.tracking_start is None else p.tracking_start


def _end(p: Possession) -> float:
    return p.end if p.tracking_end is None else p.tracking_end


# ---------------------------------------------------------------- play-by-play walk


def walk_events(events: list[Event], game: Game) -> list[Possession]:
    """Read the feed in time order, tracking which team has the ball; each change of team ends a possession.

    Who has the ball after an event:
      - made shot or turnover: the other team
      - rebound: the rebounding team
      - missed shot: still the shooting team (the rebound decides)
      - foul with free throws: the free throws and everything between them are skipped. The team in
        possession keeps the ball when the fouled team rebounds its own missed last free throw;
        otherwise the fouling team gets it
      - foul without free throws: the fouled team (it inbounds)
      - anything else: no change
    The first such event of a period opens its first possession. The event that ends a possession belongs
    to it, a defensive rebound also belongs to the possession it starts, and every event in between belongs
    to the possession it happens in. A turnover that restates an offensive foul (same player, same moment)
    is filed with the possession the foul just ended.
    Times are the feed's clock (whole seconds); snap_to_tracking later moves each boundary onto tracking.
    """

    home, away = game.home_team_id, game.away_team_id

    def other(team: str | None) -> str | None:
        return away if team == home else home if team == away else None

    out: list[Possession] = []
    for period in sorted({e.period for e in events}):
        period_events = [e for e in events if e.period == period]
        team, start, own, skip_until = None, None, [], -1
        for i, event in enumerate(period_events):
            if i <= skip_until:
                continue
            if _duplicate_offensive_foul(period_events[i - 1] if i else None, event):
                closed = out[-1] if out else None
                if closed and event.team and closed.team == event.team and abs(closed.end - event.t) <= 0.01:
                    if event not in closed.events:
                        closed.events.append(event)
                continue
            ends = True
            if event.made or event.kind == "turnover":
                after = other(event.team)
            elif event.kind == "rebound":
                after = event.team
            elif event.kind == "foul":
                if event.is_shooting_foul or _free_throws_follow(period_events, i):
                    skip_until = max(skip_until, _free_throw_sequence_end(period_events, i))
                    keeps = _offense_keeps_after_free_throws(period_events, i, event.team2)
                    after, ends = (team, False) if keeps else (event.team, True)
                else:
                    after = other(event.team)
            elif event.missed:
                after, ends = event.team, False
            else:
                continue
            if not after:
                continue
            if after == team:
                own.append(event)
                continue
            if team is not None:
                if ends:
                    own.append(event)
                out.append(Possession(period, team, start, event.t, own))
            team, start, own = after, event.t, []
            if event.kind == "rebound" and event.subtype != "offensive":
                own.append(event)
        if team is not None:
            out.append(Possession(period, team, start, game.absolute_time(period, 0.0), own))
    return out


def _duplicate_offensive_foul(prev: Event | None, event: Event) -> bool:
    return (
        prev is not None
        and event.kind == "turnover"
        and event.subtype == "offensive_foul"
        and prev.kind == "foul"
        and abs(prev.t - event.t) <= 0.01
        and prev.player == event.player
        and prev.team == event.team
    )


def _free_throws_follow(events: list[Event], i: int) -> bool:
    for e in events[i + 1 :]:
        if e.kind == "free_throw":
            return True
        if e.kind != "rebound" and e.kind not in TRANSPARENT:
            return False
    return False


def _free_throw_sequence_end(events: list[Event], i: int) -> int:
    end = i
    for j in range(i + 1, len(events)):
        if events[j].kind in ("free_throw", "rebound") or events[j].kind in TRANSPARENT:
            end = j
        else:
            break
    return end


def _offense_keeps_after_free_throws(events: list[Event], foul_index: int, fouled_team: str | None) -> bool:
    """Missed last free throw followed by an offensive rebound keeps the possession alive."""
    last_ft = None
    for j in range(foul_index + 1, len(events)):
        if events[j].kind == "free_throw":
            last_ft = j
        elif events[j].kind != "rebound" and events[j].kind not in TRANSPARENT:
            break
    if last_ft is None or events[last_ft].outcome != "missed":
        return False
    nxt = next((e for e in events[last_ft + 1 :] if e.kind not in TRANSPARENT), None)
    return nxt is not None and nxt.kind == "rebound" and nxt.subtype == "offensive" and nxt.team == fouled_team


# ---------------------------------------------------------------- merging and refinement


def merge_same_team(possessions: list[Possession]) -> list[Possession]:
    """The other team never had the ball between two possessions of the same team, so they are one."""
    merged: list[Possession] = []
    for p in possessions:
        if merged and merged[-1].team == p.team and merged[-1].period == p.period:
            merged[-1].end = p.end
            merged[-1].events += p.events
        else:
            merged.append(Possession(p.period, p.team, p.start, p.end, list(p.events)))
    return merged


def snap_to_tracking(
    possessions: list[Possession], frames: list[Frame], config: HeuristicConfig, frame_seconds: float
) -> list[Possession]:
    """Move each boundary from the feed's whole second to the shot-clock reset, or the dead ball, seen near it.

    The game's first possession starts at the first frame of its period and its last ends after the last
    frame. A boundary between periods keeps the feed's times.
    """
    times = [f.t for f in frames]
    for i, p in enumerate(possessions):
        in_period = [f for f in frames if f.period == p.period]
        if i == 0:
            p.tracking_start = in_period[0].t if in_period else p.start
        if i == len(possessions) - 1:
            p.tracking_end = in_period[-1].t + frame_seconds if in_period else p.end
            break
        nxt = possessions[i + 1]
        p.tracking_end, nxt.tracking_start = p.end, nxt.start
        if nxt.period != p.period:
            continue
        lo = bisect.bisect_left(times, p.end - config.boundary_search_before)
        hi = bisect.bisect_right(times, p.end + config.boundary_search_after)
        window = frames[lo:hi]
        if len(window) < 2:
            continue
        reset = _first_reset(window, config)
        if reset is None:
            reset = _dead_ball_start(window)
        elif reset <= p.tracking_start + config.min_snap_possession:
            continue
        if reset is not None:
            p.tracking_end = nxt.tracking_start = reset
    return possessions


def _first_reset(window: list[Frame], config: HeuristicConfig) -> float | None:
    for prev, cur in zip(window, window[1:]):
        if (
            prev.shot_clock is not None
            and cur.shot_clock is not None
            and cur.shot_clock - prev.shot_clock > config.shot_clock_reset_jump
        ):
            return cur.t
    return None


def _dead_ball_start(window: list[Frame]) -> float | None:
    """The time the clock stopped, if it stopped inside the window."""
    start = None
    for prev, cur in zip(window, window[1:]):
        if abs(cur.clock - prev.clock) < CLOCK_STOPPED:
            start = prev.t if start is None else start
        elif start is not None:
            break
    return start
