"""The simplest detector: tracking only, the play-by-play feed is ignored. Deliberately small; it is the floor
of the leaderboard.

Ball handler: nearest player within a few feet of the ball.
Possessions: runs of frames where the handler's team does not change.
Actions: a pass when the handler changes between teammates, a shot when the ball rises above the rim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import schema
from data_loader.game import Game

HANDLER_RADIUS_FT = 3.0
SHOT_HEIGHT_FT = 10.0
MIN_POSSESSION_SECONDS = 2.0


def detect_ball_handler(
    game: Game, pbp: pd.DataFrame | None = None, possessions: pd.DataFrame | None = None
) -> pd.DataFrame:
    frame_ids, xy, player_ids = game.positions()
    ball = game.frames[["ball_x", "ball_y"]].to_numpy()[:, None, :]
    dist = np.linalg.norm(xy - ball, axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = dist.argmin(axis=1)
    holder = np.array(player_ids, dtype=object)[nearest]
    holder[dist[np.arange(len(frame_ids)), nearest] > HANDLER_RADIUS_FT] = None
    return pd.DataFrame({"frame_idx": frame_ids, "player_id": holder})


def detect_possessions(game: Game, pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    handler = detect_ball_handler(game)
    team = handler["player_id"].map(game.team_by_player)
    frames = game.frames[["frame_idx", "period"]].assign(team=team.to_numpy())
    frames["team"] = frames.groupby("period")["team"].ffill()
    frames = frames.dropna(subset=["team"])
    run = (frames["team"] != frames["team"].shift()) | (frames["period"] != frames["period"].shift())
    runs = frames.groupby(run.cumsum()).agg(
        period=("period", "first"),
        start_frame=("frame_idx", "min"),
        end_frame=("frame_idx", "max"),
        team_id=("team", "first"),
    )
    long_enough = (runs["end_frame"] - runs["start_frame"]) >= MIN_POSSESSION_SECONDS * game.fps
    return runs[long_enough].reset_index(drop=True)


def detect_actions(
    game: Game,
    pbp: pd.DataFrame | None = None,
    possessions: pd.DataFrame | None = None,
    ball_handler: pd.DataFrame | None = None,
) -> pd.DataFrame:
    handler = ball_handler if ball_handler is not None else detect_ball_handler(game)
    frames = game.frames.merge(handler, on="frame_idx")

    # A pass is the handler changing between teammates within a period, placed at the passer's last frame.
    held = frames.dropna(subset=["player_id"]).reset_index(drop=True)
    prev = held.shift(1)
    team, prev_team = held["player_id"].map(game.team_by_player), prev["player_id"].map(game.team_by_player)
    is_pass = (held["player_id"] != prev["player_id"]) & (held["period"] == prev["period"]) & (team == prev_team)
    passes = pd.DataFrame(
        {
            "period": prev["period"],
            "frame_idx": prev["frame_idx"],
            "action_type": "pass",
            "player_id": prev["player_id"],
            "player2_id": held["player_id"],
            "team_id": prev_team,
            "x": prev["ball_x"],
            "y": prev["ball_y"],
            "outcome": "complete",
        }
    )[is_pass]

    # A shot is the ball rising through SHOT_HEIGHT_FT, credited to the last player who held it.
    shooter = frames["player_id"].ffill()
    rising = (frames["ball_z"] >= SHOT_HEIGHT_FT) & (frames["ball_z"].shift(1) < SHOT_HEIGHT_FT) & shooter.notna()
    shots = pd.DataFrame(
        {
            "period": frames["period"],
            "frame_idx": frames["frame_idx"],
            "action_type": "shot",
            "player_id": shooter,
            "team_id": shooter.map(game.team_by_player),
            "x": frames["ball_x"],
            "y": frames["ball_y"],
        }
    )[rising]
    return schema.concat(schema.ACTIONS, [passes, shots])
