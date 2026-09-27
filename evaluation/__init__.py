"""Scoring of detections against SkillCorner ground truth."""

from evaluation.actions import SUBSETS, ActionSettings, evaluate_actions
from evaluation.ball_handler import evaluate_ball_handler
from evaluation.matching import Matches, match_instant, match_interval, temporal_iou
from evaluation.possessions import evaluate_complete_possessions, evaluate_possessions
from evaluation.report import (
    PBP_MATCHING,
    Settings,
    evaluate_game,
    evaluate_pbp_matching,
    evaluate_run,
    leaderboard,
    load,
    save,
    to_markdown,
)

__all__ = [
    "SUBSETS",
    "ActionSettings",
    "evaluate_actions",
    "evaluate_ball_handler",
    "evaluate_complete_possessions",
    "leaderboard",
    "Matches",
    "match_instant",
    "match_interval",
    "temporal_iou",
    "evaluate_pbp_matching",
    "evaluate_possessions",
    "PBP_MATCHING",
    "Settings",
    "evaluate_game",
    "evaluate_run",
    "load",
    "save",
    "to_markdown",
]
