"""Build on the heuristic: keep its stages, replace one action type with your own, and add new ones.

Run it on one game and see the score:
    python runner.py detect examples/extend_heuristic.py --games 114099 --check

The heuristic's possessions and ball handler are this submission's own, so all three stages are scored.
Delete those two lines and the stages are skipped: the runner passes None, and the heuristic builds its
own possessions and ball handler from the feed and tracking.

The two new types show both shapes in schema.ACTION_TYPES. A screen is instant (one frame), matched
within 1 s; a drive is an interval (frame_idx to end_frame), matched by temporal IoU. SkillCorner's
annotations score both, though no baseline detects them yet.
"""

import pandas as pd

import schema
from baselines import heuristic
from data_loader import Game

detect_possessions = heuristic.detect_possessions
detect_ball_handler = heuristic.detect_ball_handler


def detect_actions(game: Game, pbp: pd.DataFrame, possessions=None, ball_handler=None) -> pd.DataFrame:
    rest = heuristic.detect_actions(game, pbp, ball_handler=ball_handler, types=("pass", "rebound", "turnover", "foul"))
    mine = [my_shots(game, pbp, ball_handler), screens(game, ball_handler), drives(game, ball_handler)]
    return schema.concat(schema.ACTIONS, [rest, *mine])


def my_shots(game: Game, pbp: pd.DataFrame, ball_handler: pd.DataFrame | None) -> pd.DataFrame:
    """Stand-in for your own shot model: here, the heuristic's shot rule."""
    return heuristic.detect_actions(game, pbp, ball_handler=ball_handler, types=("shot",))


def screens(game: Game, ball_handler: pd.DataFrame | None) -> pd.DataFrame:
    # One dict per screen: period, frame_idx, action_type="screen", subtype="on_ball" (a pick) or "off_ball",
    # player_id (the screener), player2_id (the teammate it frees), team_id, x, y (where it was set).
    rows = []
    return schema.concat(schema.ACTIONS, [pd.DataFrame(rows)])


def drives(game: Game, ball_handler: pd.DataFrame | None) -> pd.DataFrame:
    # One dict per drive: period, frame_idx (its start), end_frame, action_type="drive", player_id (the driver),
    # team_id, x, y.
    rows = []
    return schema.concat(schema.ACTIONS, [pd.DataFrame(rows)])
