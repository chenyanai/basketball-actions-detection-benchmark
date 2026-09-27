import json
from pathlib import Path

import pandas as pd

import schema
from baselines import heuristic, pbp_clock
from data_loader import build_feed, feed_answers, load_events, load_game, write_feed
from evaluation import evaluate_pbp_matching, leaderboard
from runner import main

GAME = "114099"


def answers_of(match_dir, truth):
    return feed_answers(load_events(match_dir), truth.actions).assign(game_id=GAME)


def test_feed_rows_have_ids_and_answers(match_dir, truth, feed):
    answers = answers_of(match_dir, truth)
    assert list(feed["event_id"]) == list(range(len(feed)))
    scored = feed[feed["event_type"] != "free_throw"]
    assert list(answers["event_id"]) == list(scored["event_id"])
    assert list(answers["event_type"]) == list(scored["event_type"])
    assert answers["x"].notna().mean() > 0.9
    true_frames = set(zip(truth.actions["action_type"], truth.actions["frame_idx"]))
    assert set(zip(answers["event_type"], answers["frame_idx"])) <= true_frames


def test_true_frames_never_reach_the_feed(feed):
    assert "frame" not in feed.columns and "true_frame" not in feed.columns and "frame_idx" not in feed.columns


def test_perfect_matcher_scores_perfectly(match_dir, truth):
    answers = answers_of(match_dir, truth)
    all_rows = evaluate_pbp_matching(answers[["game_id", "event_id", "frame_idx", "x", "y"]], answers)["all"]
    assert all_rows["matched"] == 1.0 and all_rows["within_0.5s"] == 1.0 and all_rows["mean_abs_s"] == 0.0
    assert all_rows["f1"] == 1.0


def test_unmatched_rows_are_misses(match_dir, truth):
    answers = answers_of(match_dir, truth)
    one = evaluate_pbp_matching(answers[["game_id", "event_id", "frame_idx", "x", "y"]].head(1), answers)["all"]
    assert one["rows"] == len(answers)
    assert one["within_1s"] == one["recall"] == round(1 / len(answers), 4)
    assert one["precision"] == 1.0  # the one row returned is right: leaving rows out costs recall, not precision


def test_late_frames_and_unknown_ids(match_dir, truth):
    answers = answers_of(match_dir, truth)
    late = answers[["game_id", "event_id", "frame_idx", "x", "y"]].assign(frame_idx=answers["frame_idx"] + 30)
    junk = pd.DataFrame({"game_id": [GAME], "event_id": [10**6], "frame_idx": [0], "x": [0.0], "y": [0.0]})
    m = evaluate_pbp_matching(pd.concat([late, junk]), answers)["all"]
    assert m["within_1s"] == 0.0 and m["within_2s"] == 1.0 and m["mean_abs_s"] == 1.2


def test_baselines_place_every_scored_row(game, feed, match_dir, truth):
    answers = answers_of(match_dir, truth)
    scores = {}
    for name, matcher in (("clock", pbp_clock), ("heuristic", heuristic)):
        matches = schema.validate(schema.PBP_MATCHES, matcher.match_pbp(game, feed)).assign(game_id=GAME)
        assert set(matches["event_id"]) == set(answers["event_id"])
        scores[name] = evaluate_pbp_matching(matches, answers)
    assert scores["clock"]["all"]["within_2s"] > 0.7
    assert scores["heuristic"]["shot"]["within_0.5s"] > scores["clock"]["shot"]["within_0.5s"]


def test_cli_match_then_evaluate(tmp_path, match_dir, truth):
    data, run, out = str(match_dir.parent), str(tmp_path / "run"), str(tmp_path / "clock-pbp.json")
    feeds = Path(tmp_path / "feeds")
    game, events = load_game(match_dir), load_events(match_dir)
    # the excerpt has no ACB data: its feed and answers are built from the events, for the test only
    write_feed(build_feed(events, game.home_team_id, game.away_team_id), feeds / f"{match_dir.name}.csv")
    answers = tmp_path / "answers"  # kept apart from the feeds, as fetch-acb does
    write_feed(feed_answers(events, truth.actions), answers / f"{match_dir.name}.csv")
    feeds = str(feeds)
    assert main(["match-pbp", "baselines.pbp_clock", "--data", data, "--out", run, "--feeds", feeds]) == 0
    assert main(["evaluate", run, "--data", data, "--out", out, "--answers", str(answers)]) == 0
    report = json.loads((tmp_path / "clock-pbp.json").read_text())
    assert report["meta"]["task"] == "pbp_matching" and report["aggregate"]["all"]["matched"] == 1.0
    assert "## Play-by-Play Matching" in (tmp_path / "leaderboard.md").read_text()


def test_leaderboard_separates_the_two_tasks(match_dir, truth):
    answers = answers_of(match_dir, truth)
    matching = {"meta": {"name": "m", "task": "pbp_matching"}, "aggregate": evaluate_pbp_matching(answers, answers)}
    text = leaderboard([matching])
    assert "## Play-by-Play Matching" in text and "## Detection" not in text
