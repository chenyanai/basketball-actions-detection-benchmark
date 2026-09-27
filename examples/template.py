"""Starting point for a submission. Keep the functions you implement, delete the rest.

Run it on one game and see the score:
    python runner.py detect examples/template.py --games 114099 --check
"""

import pandas as pd

import schema
from data_loader import Game


def detect_possessions(game: Game, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per possession: period, start_frame, end_frame, team_id."""
    return schema.empty(schema.POSSESSIONS)


def detect_ball_handler(game: Game, pbp: pd.DataFrame, possessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per frame: frame_idx, player_id (None when nobody controls the ball)."""
    return schema.empty(schema.BALL_HANDLER)


def detect_actions(
    game: Game, pbp: pd.DataFrame, possessions: pd.DataFrame | None = None, ball_handler: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per action; see schema.ACTION_TYPES for the vocabulary and schema.COLUMNS for the columns."""
    return schema.empty(schema.ACTIONS)
