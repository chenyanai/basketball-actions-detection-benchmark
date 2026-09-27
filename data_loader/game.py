"""In-memory representation of one game: tracking frames, rosters, court geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FPS = 25.0
PERIOD_SECONDS = 600.0
OVERTIME_SECONDS = 300.0

LIVE_CLOCK_STEP = 0.02  # seconds the game clock must drop between frames to count as running


def mark_live(frames: pd.DataFrame) -> pd.Series:
    """True where the game clock is moving: the frame before or after shows a different reading."""
    clock, period = frames["game_clock"].to_numpy(), frames["period"].to_numpy()
    moved = np.r_[False, (clock[:-1] - clock[1:] > LIVE_CLOCK_STEP) & (period[1:] == period[:-1])]
    return pd.Series(moved | np.r_[moved[1:], False], index=frames.index)


FRAME_COLUMNS = ["frame_idx", "period", "game_clock", "shot_clock", "ball_x", "ball_y", "ball_z"]
PLAYER_COLUMNS = ["frame_idx", "player_id", "team_id", "x", "y"]
ROSTER_COLUMNS = ["player_id", "team_id", "jersey", "name"]


@dataclass(frozen=True)
class CourtGeometry:
    """Feet, origin at center court, x along the length."""

    length: float
    width: float
    hoop_x: float
    three_point_radius: float
    three_point_corner_y: float
    key_width: float = 16.1
    key_length: float = 19.0

    @property
    def hoops(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (-self.hoop_x, 0.0), (self.hoop_x, 0.0)


FIBA_COURT = CourtGeometry(length=91.86, width=49.21, hoop_x=40.8, three_point_radius=22.15, three_point_corner_y=21.65)


@dataclass
class Game:
    game_id: str
    home_team_id: str
    away_team_id: str
    team_names: dict[str, str]
    roster: pd.DataFrame
    frames: pd.DataFrame
    players: pd.DataFrame
    court: CourtGeometry = FIBA_COURT
    fps: float = FPS
    period_seconds: float = PERIOD_SECONDS
    overtime_seconds: float = OVERTIME_SECONDS

    @property
    def team_by_player(self) -> dict[str, str]:
        return dict(zip(self.roster["player_id"], self.roster["team_id"]))

    def team_of(self, player_id: str) -> str | None:
        return self.team_by_player.get(str(player_id))

    def jersey_of(self, player_id: str) -> str:
        row = self.roster.loc[self.roster["player_id"] == str(player_id), "jersey"]
        return str(row.iloc[0]) if len(row) else ""

    def absolute_time(self, period: int, game_clock: float) -> float:
        """Seconds since tip-off, counting periods as fully elapsed."""
        if period <= 4:
            return (period - 1) * self.period_seconds + (self.period_seconds - game_clock)
        return 4 * self.period_seconds + (period - 5) * self.overtime_seconds + (self.overtime_seconds - game_clock)

    def positions(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Dense view: (frame_idx[n], xy[n, players, 2] with NaN when absent, player ids)."""
        frame_ids = self.frames["frame_idx"].to_numpy()
        player_ids = list(self.roster["player_id"])
        frame_pos = {f: i for i, f in enumerate(frame_ids)}
        player_pos = {p: i for i, p in enumerate(player_ids)}
        xy = np.full((len(frame_ids), len(player_ids), 2), np.nan)
        rows = self.players
        fi = rows["frame_idx"].map(frame_pos).to_numpy()
        pi = rows["player_id"].map(player_pos).to_numpy()
        xy[fi, pi, 0] = rows["x"].to_numpy()
        xy[fi, pi, 1] = rows["y"].to_numpy()
        return frame_ids, xy, player_ids


@dataclass
class GroundTruth:
    """SkillCorner's labels in the output schema, plus the raw event tables for extra attributes."""

    game_id: str
    possessions: pd.DataFrame
    ball_handler: pd.DataFrame
    actions: pd.DataFrame
    events: dict[str, list[dict]] = field(default_factory=dict, repr=False)
