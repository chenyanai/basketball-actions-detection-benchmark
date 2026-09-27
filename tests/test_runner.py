import json

import pytest

from runner import (
    STAGES,
    load_functions,
    load_run_tables,
    main,
    run_games,
    run_submission,
    save_run_tables,
    update_readme_leaderboard,
)


def test_load_submission_from_module_and_file(tmp_path):
    name, stages = load_functions("baselines.naive", STAGES)
    assert name == "naive" and len(stages) == 3
    path = tmp_path / "mine.py"
    path.write_text("from baselines.naive import detect_actions\n")
    name, stages = load_functions(str(path), STAGES)
    assert name == "mine" and list(stages) == ["detect_actions"]


def test_skipped_stages_pass_none_never_the_ground_truth(tmp_path, game, feed):
    """A stage the submission skips hands None on: no detector ever sees SkillCorner's possessions or handler."""
    path = tmp_path / "actions_only.py"
    path.write_text(
        "import schema\nseen = []\n"
        "def detect_actions(game, pbp, possessions=None, ball_handler=None):\n"
        "    seen.append((possessions, ball_handler))\n"
        "    return schema.empty(schema.ACTIONS)\n"
    )
    result = run_submission(load_functions(str(path), STAGES)[1], game, feed)
    from actions_only import seen

    assert seen == [(None, None)]
    assert list(result) == ["actions"]


def test_run_tables_round_trip(tmp_path, game, feed):
    result = run_submission(load_functions("baselines.naive", STAGES)[1], game, feed)
    save_run_tables(tmp_path, game.game_id, result)
    back = load_run_tables(tmp_path, game.game_id)
    assert run_games(tmp_path) == [game.game_id]
    for kind, table in result.items():
        assert len(back[kind]) == len(table)
    assert back["actions"]["player_id"].dtype == object or str(back["actions"]["player_id"].dtype) == "str"


def feeds_for(tmp_path, match_dir, feed) -> str:
    """The excerpt has no ACB feed, so the CLI tests give detect the fixture's."""
    folder = tmp_path / "feeds"
    folder.mkdir(exist_ok=True)
    feed.to_csv(folder / f"{match_dir.name}.csv", index=False)
    return str(folder)


def test_cli_detect_then_evaluate(tmp_path, match_dir, feed):
    data, run, out = str(match_dir.parent), str(tmp_path / "run"), str(tmp_path / "report.json")
    feeds = feeds_for(tmp_path, match_dir, feed)
    assert main(["detect", "baselines.naive", "--data", data, "--out", run, "--feeds", feeds]) == 0
    assert (tmp_path / "run" / "run.json").exists()
    assert main(["evaluate", run, "--data", data, "--out", out]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["meta"]["name"] == "naive" and report["aggregate"]["actions"]["shot"]["status"] == "scored"
    assert not (tmp_path / "report.md").exists()  # the report is JSON only; evaluate prints its Markdown


def test_evaluating_part_of_the_games_leaves_results_alone(tmp_path, match_dir, feed, monkeypatch):
    """One game is not the benchmark: its report goes to the run folder, not to results/ and the leaderboard."""
    monkeypatch.chdir(tmp_path)  # results/ is relative to the working directory
    data, run = str(match_dir.parent), str(tmp_path / "run")
    feeds = feeds_for(tmp_path, match_dir, feed)
    assert main(["detect", "baselines.naive", "--data", data, "--out", run, "--feeds", feeds]) == 0
    assert main(["evaluate", run, "--data", data]) == 0
    assert (tmp_path / "run" / "report.json").exists() and not (tmp_path / "results").exists()


def test_detection_always_gets_the_feed(tmp_path, game, feed, truth):
    path = tmp_path / "spy.py"
    path.write_text(
        "seen = []\ndef detect_possessions(*args, **kwargs):\n    seen.append((args, kwargs))\n    raise SystemExit\n"
    )
    with pytest.raises(SystemExit):
        run_submission(load_functions(str(path), STAGES)[1], game, feed)
    from spy import seen

    assert len(seen) == 1 and seen[0][0][0] is game and seen[0][0][1] is feed and seen[0][1] == {}


def test_readme_leaderboard_is_replaced_between_its_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- leaderboard:start -->\nold\n<!-- leaderboard:end -->\noutro\n")
    board = "## Detection\n\n| detector |\n|---|\n| mine |\n"
    update_readme_leaderboard(readme, board)
    first = readme.read_text()
    assert "old" not in first and "### Detection" in first and first.startswith("intro") and first.endswith("outro\n")
    update_readme_leaderboard(readme, board)
    assert readme.read_text() == first
    plain = tmp_path / "PLAIN.md"
    plain.write_text("no markers here\n")
    update_readme_leaderboard(plain, board)
    assert plain.read_text() == "no markers here\n"
