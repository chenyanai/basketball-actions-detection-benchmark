"""The heuristic detector: play-by-play for identity, tracking for time and place.

Three stage functions with the benchmark's signatures, one detector per action
type underneath, every threshold in ``HeuristicConfig``.
"""

from baselines.heuristic.actions import TYPES, detect_actions, match_pbp
from baselines.heuristic.ball_handler import detect_ball_handler
from baselines.heuristic.config import HeuristicConfig, default_config
from baselines.heuristic.possessions import detect_possessions

__all__ = [
    "TYPES",
    "HeuristicConfig",
    "default_config",
    "detect_actions",
    "detect_ball_handler",
    "detect_possessions",
    "match_pbp",
]
