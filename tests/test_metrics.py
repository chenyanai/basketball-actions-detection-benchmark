import dataclasses
import re

import pandas as pd
import pytest

from evaluation import (
    Settings,
    evaluate_actions,
    evaluate_ball_handler,
    evaluate_game,
    evaluate_possessions,
    evaluate_run,
    to_markdown,
)
from evaluation.matching import match_instant, match_interval, temporal_iou


def tag(table: pd.DataFrame) -> pd.DataFrame:
    """The scoring functions below the report expect every table to carry game_id."""
    return table.assign(game_id="114099")


def test_oracle_scores_perfectly(game, truth):
    tables = {"possessions": truth.possessions, "ball_handler": truth.ball_handler, "actions": truth.actions}
    r = evaluate_game(tables, truth, game)
    assert r["possessions"]["frame_team_accuracy"] == 1.0 and r["possessions"]["f1"] == 1.0
    assert r["ball_handler"]["frame_accuracy"] == 1.0 and r["ball_handler"]["no_ball_accuracy"] == 1.0
    for kind, m in r["actions"].items():
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["strict_recall"] == 1.0, kind
        timing = m["timing"]
        assert timing["rmse_s"] == 0.0 and timing["mse_s2"] == 0.0


def test_complete_possessions(game, truth):
    tables = {"possessions": truth.possessions, "ball_handler": truth.ball_handler, "actions": truth.actions}
    assert evaluate_game(tables, truth, game)["complete_possessions"]["complete"] == 1.0
    no_shots = {**tables, "actions": truth.actions[truth.actions["action_type"] != "shot"]}
    c = evaluate_game(no_shots, truth, game)["complete_possessions"]
    assert "shot" not in c["action_types"] and c["complete"] == 1.0
    shots_dropped = truth.actions.drop(truth.actions[truth.actions["action_type"] == "shot"].index[:5])
    c = evaluate_game({**tables, "actions": shots_dropped}, truth, game)["complete_possessions"]
    assert c["complete"] < 1.0 and c["split_correct"] == 1.0 and c["all_found_by_type"]["shot"] < 1.0
    actions_only = evaluate_game({"actions": truth.actions}, truth, game)["complete_possessions"]
    assert actions_only["requires_split"] is False and actions_only["complete"] == 1.0


def test_missing_tables_are_not_attempted(game, truth):
    r = evaluate_game({}, truth, game)
    assert r["possessions"]["status"] == "not_attempted"
    assert r["ball_handler"]["status"] == "not_attempted"
    assert all(m["status"] == "not_attempted" for m in r["actions"].values())
    assert r["complete_possessions"]["status"] == "not_attempted"


def test_attempted_is_decided_over_the_whole_run(game, truth):
    """A game where the submission found no shot scores zero shot recall; it did not skip shots."""
    other = dataclasses.replace(game, game_id="other")
    no_shots = truth.actions[truth.actions["action_type"] != "shot"]
    report = evaluate_run([(game, truth, {"actions": truth.actions}), (other, truth, {"actions": no_shots})])
    shots = report["games"]["other"]["actions"]["shot"]
    assert shots["status"] == "scored" and shots["recall"] == 0.0
    assert report["games"]["other"]["complete_possessions"]["all_found_by_type"]["shot"] == 0.0
    assert report["aggregate"]["actions"]["shot"]["recall"] == 0.5


def test_timing_error_and_attribution(truth):
    shots = truth.actions[truth.actions["action_type"] == "shot"].copy()
    shifted = shots.copy()
    shifted["frame_idx"] += 10  # 0.4 s late
    shifted.loc[shifted.index[0], "player_id"] = "someone_else"
    m = evaluate_actions(tag(shifted), tag(truth.actions))["shot"]
    assert m["recall"] == 1.0 and m["timing"]["mean_signed_s"] == pytest.approx(0.4)
    assert m["timing"]["mse_s2"] == pytest.approx(0.16) and m["timing"]["rmse_s"] == pytest.approx(0.4)
    assert m["player_accuracy"] == pytest.approx(17 / 18, abs=1e-3)
    assert m["strict_recall"] == pytest.approx(17 / 18, abs=1e-3)
    assert m["subsets"]["made"]["truth"] + m["subsets"]["missed"]["truth"] == 18


def test_average_precision_uses_confidence(truth):
    shots = truth.actions[truth.actions["action_type"] == "shot"].copy()
    fake = shots.head(3).copy()
    fake["frame_idx"] += 5000
    fake["confidence"] = 0.1
    shots["confidence"] = 0.9
    m = evaluate_actions(tag(pd.concat([shots, fake])), tag(truth.actions))["shot"]
    assert m["precision"] == pytest.approx(18 / 21, abs=1e-3) and m["average_precision"] == 1.0


def test_possession_frame_accuracy(game, truth):
    wrong = truth.possessions.copy()
    wrong["team_id"] = wrong["team_id"].map({"1315": "1316", "1316": "1315"})
    m = evaluate_possessions(tag(wrong), tag(truth.possessions), tag(game.frames))
    assert m["frame_team_accuracy"] == 0.0 and m["recall"] == 0.0


def test_ball_handler_rates(game, truth):
    nobody = truth.ball_handler.assign(player_id=None)
    m = evaluate_ball_handler(tag(nobody), tag(truth.ball_handler), game.team_by_player)
    assert m["no_handler_rate"] == 1.0 and m["in_touch_accuracy"] == 0.0 and m["no_ball_accuracy"] == 1.0
    assert m["frame_accuracy"] == pytest.approx(1 - m["frames_in_touch"] / m["frames"], abs=1e-3)


def test_run_report_and_markdown(game, truth):
    tables = {"possessions": truth.possessions, "ball_handler": truth.ball_handler, "actions": truth.actions}
    report = evaluate_run([(game, truth, tables), (game, truth, tables)], Settings(), meta={"name": "oracle"})
    assert report["aggregate"]["actions"]["shot"]["truth"] == 36
    assert set(report["games"]) == {"114099"}
    md = to_markdown(report)
    assert "# oracle" in md and re.search(r"\| shot +\| +36 \|", md)
    assert report["aggregate"]["actions"]["shot"]["f1_by_window"] == {"0.5": 1.0, "1": 1.0, "2": 1.0}
    assert "F1 by matching window" in md


def test_leaderboard_lists_only_default_settings(game, truth):
    from evaluation.actions import ActionSettings
    from evaluation.report import leaderboard

    tables = {"possessions": truth.possessions, "ball_handler": truth.ball_handler, "actions": truth.actions}
    official = evaluate_run([(game, truth, tables)], Settings(), meta={"name": "official"})
    loose = evaluate_run(
        [(game, truth, tables)], Settings(actions=ActionSettings(window_seconds=3.0)), meta={"name": "loose"}
    )
    board = leaderboard([official, loose])
    assert "official" in board and "loose" not in board
    row = next(line for line in board.splitlines() if "official" in line)
    assert "**100.0%**" in row.split("|")[2], "every stage column is a bold percentage"


def test_live_play_filter():
    from data_loader.game import mark_live
    from evaluation.report import live_frames, near_live_play

    clock = [600.0, 599.96, 599.92, 599.92, 599.92, 599.92, 599.88]
    frames = pd.DataFrame({"frame_idx": range(7), "period": 1, "game_clock": clock})
    frames["live"] = mark_live(frames)
    assert list(frames["live"]) == [True, True, True, False, False, True, True]
    assert list(live_frames(frames)["frame_idx"]) == [0, 1, 2, 5, 6]

    frames = pd.DataFrame({"frame_idx": range(100), "live": [True] * 10 + [False] * 80 + [True] * 10})
    actions = pd.DataFrame({"frame_idx": [5, 30, 50, 70, 95]})
    assert list(near_live_play(tag(actions), tag(frames), 25)["frame_idx"]) == [5, 30, 70, 95]


# ---------------------------------------------------------------- blind matching


def rows(*frames, period=1, player="p", team="t", game="g"):
    return pd.DataFrame(
        [{"game_id": game, "period": period, "frame_idx": f, "player_id": player, "team_id": team} for f in frames]
    )


def test_greedy_prefers_closest_and_never_reuses():
    det = rows(100, 104)
    truth = rows(102)
    m = match_instant(det, truth, window_frames=25)
    assert m.pairs == [(0, 0)] and m.unmatched_detections == [1] and m.fn == 0


def test_window_and_period_are_respected():
    assert match_instant(rows(100), rows(140), window_frames=25).tp == 0
    assert match_instant(rows(100, period=1), rows(100, period=2), window_frames=25).tp == 0


def test_require_player():
    det, truth = rows(100, player="a"), rows(100, player="b")
    assert match_instant(det, truth, 25).tp == 1
    assert match_instant(det, truth, 25, require_player=True).tp == 0


def test_missing_players_compare_equal_when_required():
    det, truth = rows(100, player=None), rows(100, player=None)
    assert match_instant(det, truth, 25, require_player=True).tp == 1


def test_game_id_isolates_games():
    det, truth = rows(100, game="g1"), rows(100, game="g2")
    assert match_instant(det, truth, 25).tp == 0


def test_interval_matching():
    det = pd.DataFrame([{"game_id": "g", "period": 1, "frame_idx": 0, "end_frame": 100, "team_id": "t"}])
    truth = pd.DataFrame(
        [
            {"game_id": "g", "period": 1, "frame_idx": 20, "end_frame": 120, "team_id": "t"},
            {"game_id": "g", "period": 1, "frame_idx": 300, "end_frame": 400, "team_id": "t"},
        ]
    )
    m = match_interval(det, truth, iou_threshold=0.5)
    assert m.pairs == [(0, 0)] and m.unmatched_truth == [1]
    assert match_interval(det, truth.assign(team_id="other"), 0.5).tp == 0
    assert temporal_iou(0, 100, 20, 120) == 80 / 120
