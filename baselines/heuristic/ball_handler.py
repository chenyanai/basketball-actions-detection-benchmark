"""Ball handler: the possession-team player within reach of the ball, with hysteresis.

Inside a possession only its team is eligible; between possessions anyone is. The previous
handler keeps the ball while still within reach, so dribbles do not flicker the annotation.
"""

from __future__ import annotations

import bisect

import pandas as pd

from baselines.heuristic.config import HeuristicConfig, default_config
from baselines.heuristic.inputs import Frame, Player, dist, frames_from_game
from baselines.heuristic.possessions import Possession
from data_loader.game import Game


def detect_ball_handler(
    game: Game,
    pbp: pd.DataFrame | None = None,
    possessions: pd.DataFrame | None = None,
    config: HeuristicConfig | None = None,
) -> pd.DataFrame:
    """The handler follows the possessions it is given; the feed itself is not read here."""
    config = config or default_config()
    frames = frames_from_game(game, config)
    annotate(frames, teams_from_table(frames, possessions), config)
    holder = {f.idx: (f.holder.id if f.holder else None) for f in frames}
    frame_ids = game.frames["frame_idx"]
    return pd.DataFrame({"frame_idx": frame_ids, "player_id": frame_ids.map(holder)})


def teams_from_table(frames: list[Frame], possessions: pd.DataFrame | None) -> list[str | None]:
    """The team in possession on every frame, None between possessions."""
    teams: list[str | None] = [None] * len(frames)
    if possessions is None:
        return teams
    position = {f.idx: i for i, f in enumerate(frames)}
    for p in possessions.itertuples(index=False):
        for idx in range(int(p.start_frame), int(p.end_frame) + 1):
            i = position.get(idx)
            if i is not None and teams[i] is None:
                teams[i] = str(p.team_id)
    return teams


def teams_from_possessions(frames: list[Frame], possessions: list[Possession]) -> list[str | None]:
    """The same from time bounds; where possessions overlap the earlier one wins."""
    times = [f.t for f in frames]
    teams: list[str | None] = [None] * len(frames)
    for p in reversed(possessions):
        start, end = p.bounds
        lo, hi = bisect.bisect_left(times, start), bisect.bisect_left(times, end)
        teams[lo:hi] = [p.team] * (hi - lo)
    return teams


def annotate(frames: list[Frame], teams: list[str | None], config: HeuristicConfig) -> None:
    """Set ``frame.holder`` in place. The previous handler is forgotten at a period, a time gap or a new possession."""
    previous: Player | None = None
    last: Frame | None = None
    last_team: str | None = None
    for f, team in zip(frames, teams):
        if last is not None and (f.period != last.period or f.t - last.t > config.time_gap_reset or team != last_team):
            previous = None
        f.holder = _handler(f, team, previous, config.handler_radius)
        previous = f.holder or previous  # remembered while the ball is loose, so a bounce does not lose the handler
        last, last_team = f, team


def _handler(f: Frame, team: str | None, previous: Player | None, radius: float) -> Player | None:
    close = {p.id: (p, dist(p.x, p.y, f.ball_x, f.ball_y)) for p in f.players if (team is None or p.team == team)}
    close = {pid: (p, d) for pid, (p, d) in close.items() if d <= radius}
    if previous is not None and previous.id in close:
        return close[previous.id][0]
    return min(close.values(), key=lambda pd_: pd_[1])[0] if close else None
