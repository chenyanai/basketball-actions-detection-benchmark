from pathlib import Path

import pandas as pd
import pytest

from data_loader import Game, build_feed, ground_truth, load_events, load_game

EXCERPT = Path(__file__).parent / "data" / "excerpt"
GAME_ID = "114099"


MISSING = "test data not found at"


def pytest_terminal_summary(terminalreporter):
    """One clear line when the local test data is missing, instead of dozens of silent skips."""
    skipped = [r for r in terminalreporter.stats.get("skipped", []) if MISSING in str(r.longrepr)]
    if skipped:
        terminalreporter.write_line(
            f"WARNING: {len(skipped)} tests skipped: no test excerpt at tests/data/excerpt/{GAME_ID}. "
            "The data cannot be shipped, so these tests run only where it exists locally (see the README).",
            yellow=True,
        )


@pytest.fixture(scope="session")
def match_dir() -> Path:
    """Every data fixture goes through this one, so tests that need the excerpt skip without it."""
    folder = EXCERPT / GAME_ID
    if not folder.is_dir():
        pytest.skip(f"{MISSING} {folder}")
    return folder


@pytest.fixture(scope="session")
def game(match_dir):
    return load_game(match_dir)


@pytest.fixture(scope="session")
def truth(match_dir, game):
    return ground_truth(match_dir, game)


@pytest.fixture(scope="session")
def feed(match_dir, game):
    return build_feed(load_events(match_dir), game.home_team_id, game.away_team_id)


@pytest.fixture
def fake_game():
    """A Game with a roster and a clock, for tests that need no tracking."""

    def build(home_team_id: str, away_team_id: str, roster: list[tuple[str, str, str]]) -> Game:
        frames = pd.DataFrame({"frame_idx": range(0, 15000), "period": 1})
        frames["game_clock"] = 600.0 - frames["frame_idx"] / 25.0
        return Game(
            game_id="test",
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            team_names={home_team_id: "Home", away_team_id: "Away"},
            roster=pd.DataFrame(roster, columns=["player_id", "team_id", "name"]).assign(jersey=""),
            frames=frames,
            players=pd.DataFrame(columns=["frame_idx", "player_id", "team_id", "x", "y"]),
        )

    return build
