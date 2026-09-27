"""Settings of the heuristic detector.

The values live only in config.yaml, grouped by stage; this class names and types them.

Your own values: write a yaml with just the settings you change (same sections) and pass
``HeuristicConfig.from_yaml("mine.yaml")`` as ``config=``; everything else comes from config.yaml.
One setting in code: ``dataclasses.replace(default_config(), shot_height=9.0)``. A config never changes in place.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class HeuristicConfig:
    # ball handler
    handler_radius: float
    time_gap_reset: float

    # passes
    min_pass_interval: float
    repeat_pass_interval: float
    max_shot_flight: float
    shot_flight_low_ratio: float
    shot_flight_low_window: float

    # shots
    shot_height: float
    shot_proximity: float
    shot_window: float
    tip_height: float
    tip_proximity: float
    tip_basket_radius: float
    tip_window_before: float
    tip_window_after: float
    release_walk_back_frames: int
    release_walk_forward_frames: int
    shot_rise_lookahead_frames: int
    apex_max_distance: float
    apex_drop_tolerance: float
    putback_before_rebound: float
    putback_after_rebound: float
    putback_search_after: float
    putback_search_fallback: float

    # rebounds
    rebound_window: float
    rebound_peak_height: float
    rebound_possession_height: float
    rebound_proximity: float
    rebound_proximity_relaxed: float
    rebound_min_frames: int
    rebound_hold_min_frames: int
    rebound_hold_min_span_frames: int
    rebound_shot_match_window: float
    rebound_feed_shot_lookback: float
    rebound_shot_margin: float
    rebound_fallback_before: float
    rebound_fallback_after: float
    rebound_fallback_delay: float
    rebound_tip_window: float
    rebound_tip_height: float
    rebound_tip_basket_radius: float

    # turnovers
    bad_pass_lookback: float
    lost_ball_lookback: float
    violation_lookback: float
    turnover_proximity: float
    turnover_window_after: float
    turnover_max_ball_height: float
    clock_stopped_extra_lookback: float

    # fouls
    foul_frame_window: float

    # actions
    possession_margin: float
    early_shot_margin: float
    boundary_margin: float
    duplicate_window: float

    # possessions
    boundary_search_before: float
    boundary_search_after: float
    shot_clock_reset_jump: float
    min_snap_possession: float

    # frame preparation
    pre_inbound_seconds: float
    backfill_shot_clock: bool

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_PATH) -> HeuristicConfig:
        """config.yaml, overridden by the settings in ``path``. Sections only group the file."""
        values = _read(DEFAULT_PATH) | _read(path)
        names = {f.name for f in fields(cls)}
        if unknown := set(values) - names:
            raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
        if missing := names - set(values):
            raise ValueError(f"{DEFAULT_PATH}: missing settings {sorted(missing)}")
        return cls(**values)


@lru_cache(maxsize=1)
def default_config() -> HeuristicConfig:
    return HeuristicConfig.from_yaml(DEFAULT_PATH)


def _read(path: str | Path) -> dict:
    sections = yaml.safe_load(Path(path).read_text()) or {}
    return {key: value for section in sections.values() for key, value in (section or {}).items()}
