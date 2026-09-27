import dataclasses
from types import SimpleNamespace

import pytest

import schema
from baselines import heuristic, naive
from baselines.heuristic.inputs import Event, frames_from_game
from baselines.heuristic.possessions import Possession, events_table, find_possessions, merge_same_team, walk_events
from evaluation import evaluate_game


@pytest.fixture(scope="module")
def heuristic_tables(game, feed):
    possessions = heuristic.detect_possessions(game, feed)
    handler = heuristic.detect_ball_handler(game, feed, possessions)
    actions = heuristic.detect_actions(game, feed, possessions, handler)
    return {"possessions": possessions, "ball_handler": handler, "actions": actions}


def test_naive_outputs_validate(game):
    p = schema.validate(schema.POSSESSIONS, naive.detect_possessions(game))
    h = schema.validate(schema.BALL_HANDLER, naive.detect_ball_handler(game))
    a = schema.validate(schema.ACTIONS, naive.detect_actions(game, None, p, h))
    assert len(p) > 10 and len(h) == len(game.frames)
    assert set(a["action_type"]) == {"pass", "shot"}


def test_heuristic_outputs_validate(heuristic_tables):
    for kind, table in heuristic_tables.items():
        schema.validate(kind, table)
    assert set(heuristic_tables["actions"]["action_type"]) == {"pass", "shot", "rebound", "turnover", "foul"}


def test_heuristic_pinned_scores(game, truth, heuristic_tables):
    """Change these on purpose when the detector changes."""
    r = evaluate_game(heuristic_tables, truth, game)
    assert r["possessions"]["recall"] == pytest.approx(0.591, abs=1e-3)
    assert r["possessions"]["frame_team_accuracy"] == pytest.approx(0.920, abs=0.005)
    assert r["ball_handler"]["frame_accuracy"] == pytest.approx(0.753, abs=0.005)
    shots, passes = r["actions"]["shot"], r["actions"]["pass"]
    assert shots["recall"] == pytest.approx(17 / 18, abs=1e-3) and shots["precision"] == 1.0  # the fouled miss is found
    assert shots["timing"]["mean_abs_s"] < 0.25
    assert passes["f1"] == pytest.approx(0.756, abs=0.01)
    assert r["actions"]["rebound"]["recall"] == pytest.approx(5 / 7, abs=1e-3)


def test_action_type_filter(game, feed, heuristic_tables):
    only_shots = heuristic.detect_actions(
        game, feed, heuristic_tables["possessions"], heuristic_tables["ball_handler"], types=("shot",)
    )
    assert set(only_shots["action_type"]) == {"shot"}


def test_frame_selection(game):
    live_only = frames_from_game(game, dataclasses.replace(heuristic.default_config(), pre_inbound_seconds=0.0))
    readings = [(f.period, f.clock) for f in live_only]
    assert len(set(readings)) == len(readings) < len(game.frames)
    assert [f.t for f in live_only] == sorted(f.t for f in live_only)
    with_inbounds = frames_from_game(game)
    assert len(live_only) < len(with_inbounds) < len(game.frames)
    assert all(f.shot_clock is not None for f in with_inbounds[:200])


def _event(event_id, kind, t, team, player, subtype=None, outcome=None):
    return Event(kind, subtype, 1, 600.0 - t, float(t), team, player, None, outcome, event_id)


def test_merge_same_team():
    first, second, third = (_event(i, "shot", i, "a", "a1", outcome="missed") for i in (1, 2, 3))
    merged = merge_same_team(
        [
            Possession(1, "a", 0, 5, [first]),
            Possession(1, "a", 5, 9, [second, third]),
            Possession(1, "b", 9, 12),
            Possession(2, "b", 0, 3),
        ]
    )
    assert [(p.team, p.start, p.end) for p in merged] == [("a", 0, 9), ("b", 9, 12), ("b", 0, 3)]
    assert merged[0].events == [first, second, third]


def test_each_possession_holds_its_own_events():
    """The feed says who shot, rebounded and lost the ball in each possession."""
    events = [
        _event(0, "shot", 2, "b", "b1", outcome="made"),  # opens the period: team a inbounds
        _event(1, "shot", 10, "a", "a1", outcome="missed"),
        _event(2, "rebound", 11, "b", "b1", subtype="defensive"),
        _event(3, "shot", 20, "b", "b2", outcome="made"),
        _event(4, "turnover", 30, "a", "a2", subtype="bad_pass"),
        _event(5, "shot", 40, "b", "b1", outcome="missed"),
        _event(6, "rebound", 41, "b", "b3", subtype="offensive"),
        _event(7, "shot", 42, "b", "b3", outcome="made"),
    ]
    game = SimpleNamespace(home_team_id="a", away_team_id="b", absolute_time=lambda period, clock: 600.0 - clock)
    possessions = merge_same_team(walk_events(events, game))

    def players(kind):
        return [[e.player for e in p.events_of(kind)] for p in possessions]

    assert [(p.team, p.start, p.end) for p in possessions] == [
        ("a", 2, 11),
        ("b", 11, 20),
        ("a", 20, 30),
        ("b", 30, 42),
        ("a", 42, 600),
    ]
    assert players("shot") == [["a1"], ["b2"], [], ["b1", "b3"], []]
    assert players("rebound") == [[], ["b1"], [], ["b3"], []]  # the defensive rebound counts for the rebounder's
    assert players("turnover") == [[], [], ["a2"], [], []]


def test_actions_come_from_their_possession(game, feed, heuristic_tables):
    config = heuristic.default_config()
    possessions = find_possessions(game, frames_from_game(game, config), feed, config)
    filed = events_table(possessions)
    assert set(filed["event_type"]) == {"shot", "rebound", "turnover", "foul"}
    assert not filed.duplicated("event_id").any()

    found = heuristic_tables["actions"].dropna(subset=["event_id"])
    found = found.astype({"event_id": int, "possession": int})
    expected = filed.set_index("event_id")
    assert len(found) > 30
    assert (expected.loc[found["event_id"], "possession"].to_numpy() == found["possession"].to_numpy()).all()
    players = found.dropna(subset=["player_id"])
    assert (expected.loc[players["event_id"], "player_id"].to_numpy() == players["player_id"].to_numpy()).all()
    assert set(heuristic_tables["actions"]["possession"].dropna()) <= set(heuristic_tables["possessions"]["possession"])


def test_own_yaml_overrides_only_what_it_sets(tmp_path):
    default = heuristic.default_config()
    custom = tmp_path / "mine.yaml"
    custom.write_text("shots:\n  shot_height: 9.0\n")
    assert heuristic.HeuristicConfig.from_yaml(custom) == dataclasses.replace(default, shot_height=9.0)
    custom.write_text("shots:\n  no_such_setting: 1\n")
    with pytest.raises(ValueError, match="unknown settings"):
        heuristic.HeuristicConfig.from_yaml(custom)


def test_config_never_changes_in_place():
    with pytest.raises(dataclasses.FrozenInstanceError):
        heuristic.default_config().shot_height = 9.0


def test_config_is_honoured(game, feed):
    strict = dataclasses.replace(heuristic.default_config(), handler_radius=0.5)
    handler = heuristic.detect_ball_handler(game, config=strict)
    assert handler["player_id"].notna().mean() < 0.3
