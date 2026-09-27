"""The feed alone: place every feed row at the middle of its game-clock second, without reading the tracking.

The feed's clock is floored, so a row at 431 happened while the clock read 431.0 to 432.0.
The location is the named player's position at that frame, or the ball's.
It is the floor of play-by-play matching, and as a detector it shows how much of a detection score the feed
already gives: a detector earns what it scores above this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import schema
from data_loader.game import Game


def match_pbp(game: Game, pbp: pd.DataFrame) -> pd.DataFrame:
    placed = []
    for period, feed in pbp[pbp["event_type"] != "free_throw"].groupby("period"):
        frames = game.frames[game.frames["period"] == period].reset_index(drop=True)
        elapsed = -frames["game_clock"].to_numpy()  # ascending, so searchsorted applies
        at = np.searchsorted(elapsed, -(feed["clock"].to_numpy() + 0.5)).clip(0, len(frames) - 1)
        at_frames = frames.iloc[at][["frame_idx", "ball_x", "ball_y"]]
        placed.append(at_frames.assign(event_id=feed["event_id"].to_numpy(), player_id=feed["player_id"].to_numpy()))
    rows = pd.concat(placed, ignore_index=True)
    rows = rows.merge(game.players[["frame_idx", "player_id", "x", "y"]], on=["frame_idx", "player_id"], how="left")
    # No player named, or not on court: use the ball's position.
    rows["x"], rows["y"] = rows["x"].fillna(rows["ball_x"]), rows["y"].fillna(rows["ball_y"])
    return rows[["event_id", "frame_idx", "x", "y"]]


def detect_actions(game: Game, pbp: pd.DataFrame, possessions=None, ball_handler=None) -> pd.DataFrame:
    """The same placement as an actions table: every feed row is an action at its clock second."""
    rows = match_pbp(game, pbp).merge(pbp, on="event_id")
    columns = ["period", "frame_idx", "event_type", "player_id", "team_id", "x", "y"]
    return schema.concat(schema.ACTIONS, [rows[columns].rename(columns={"event_type": "action_type"})])
