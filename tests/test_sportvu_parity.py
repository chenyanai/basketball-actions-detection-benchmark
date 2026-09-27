"""The heuristic reproduces the reference SportVU pipeline (sportvu-actions) on its bundled sample game.

Needs py7zr and a checkout of sportvu-actions, found through SPORTVU_ACTIONS_ROOT or next to this repository.
"""

import dataclasses
import os
from pathlib import Path

import pandas as pd
import pytest

from baselines import heuristic
from baselines.heuristic.actions import detect_actions
from baselines.heuristic.inputs import frames_from_game
from baselines.heuristic.possessions import find_possessions
from data_loader import sportvu

ROOT = Path(os.environ.get("SPORTVU_ACTIONS_ROOT", Path(__file__).parents[2] / "sportvu-actions"))
SAMPLE, GOLDEN = ROOT / "tests" / "data" / "sample_game", ROOT / "tests" / "data" / "golden"
CONFIG = dataclasses.replace(heuristic.default_config(), pre_inbound_seconds=0.0)


@pytest.fixture(scope="module")
def sample():
    pytest.importorskip("py7zr")
    if not GOLDEN.is_dir():
        pytest.skip(f"sportvu-actions not found at {ROOT}")
    return sportvu.load_game(SAMPLE / "0021500001_q1.7z"), sportvu.build_feed(SAMPLE / "0021500001_q1_pbp.csv")


def test_possessions_match_the_reference(sample):
    game, feed = sample
    found = find_possessions(game, frames_from_game(game, CONFIG), feed, CONFIG)
    golden = pd.read_csv(GOLDEN / "possessions.csv", dtype={"team_id": str})
    ours = [(p.team, p.start, p.end, round(p.tracking_start, 6), round(p.tracking_end, 6)) for p in found]
    theirs = list(
        zip(
            golden["team_id"],
            golden["start_time"],
            golden["end_time"],
            golden["tracking_start_time"].round(6),
            golden["tracking_end_time"].round(6),
        )
    )
    assert ours == theirs


def test_actions_match_the_reference(sample):
    game, feed = sample
    actions = detect_actions(game, pbp=feed, config=CONFIG)
    clock = game.frames.set_index("frame_idx")["game_clock"]
    actions["t"] = (720.0 - actions["frame_idx"].map(clock)).round(2)
    fouls = actions["action_type"] == "foul"  # a foul keeps the feed's time, not its frame's
    actions.loc[fouls, "t"] = actions.loc[fouls, "event_id"].map(dict(zip(feed["event_id"], 720.0 - feed["clock"])))
    golden = pd.read_csv(GOLDEN / "actions.csv", dtype={"player_id": str})
    for kind in ("pass", "shot", "rebound", "turnover", "foul"):
        ours = sorted(zip(actions.loc[actions["action_type"] == kind, "t"], _players(actions, kind)))
        theirs = sorted(zip(golden.loc[golden["action_type"] == kind, "timestamp"].round(2), _players(golden, kind)))
        if kind == "pass":  # the reference stamps a pass released at time 0.0 with its catch instead
            ours, theirs = ours[1:], theirs[1:]
        assert ours == theirs, kind


def _players(table: pd.DataFrame, kind: str) -> list[str]:
    return list(table.loc[table["action_type"] == kind, "player_id"].fillna("unknown").astype(str))
