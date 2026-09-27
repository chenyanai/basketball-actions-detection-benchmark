#!/usr/bin/env python
"""Command line for the benchmark: fetch data and the feed, run a detector or a play-by-play matcher,
evaluate, visualize.

A submission is a Python module defining any subset of these functions, with these signatures:

    def detect_possessions(game, pbp) -> DataFrame                                   # schema.POSSESSIONS
    def detect_ball_handler(game, pbp, possessions=None) -> DataFrame               # schema.BALL_HANDLER
    def detect_actions(game, pbp, possessions=None, ball_handler=None) -> DataFrame  # schema.ACTIONS
    def match_pbp(game, pbp) -> DataFrame   # schema.PBP_MATCHES, the matching task, scored on its own

``game`` is a data_loader.Game and ``pbp`` the play-by-play feed. A stage the submission skips passes
``None`` on, never the ground truth; to build on a baseline's possessions, call its stage functions.
A run folder holds run.json and one sub-folder per game with the submission's own tables as CSV.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import schema
from data_loader import (
    GAME_IDS,
    available,
    benchmark_feed,
    check_acb,
    fetch,
    fetch_acb,
    ground_truth,
    load_acb,
    load_events,
    load_game,
    match_dir,
    read_feed,
    write_feed,
)
from evaluation import (
    PBP_MATCHING,
    ActionSettings,
    Settings,
    evaluate_pbp_matching,
    evaluate_run,
    leaderboard,
    load,
    save,
    to_markdown,
)

DATA = "data/matches"
FEEDS = "data/play_by_play"
ANSWERS = "data/answers"  # the true frame of each feed row: read only by evaluate, never next to the feed
ACB = "data/acb"
REPO = Path(__file__).parent
LEADERBOARD_MARKERS = ("<!-- leaderboard:start -->", "<!-- leaderboard:end -->")
GET_DATA = (
    "Download it with `python runner.py fetch`, or clone https://github.com/SkillCorner/opendata-basketball, "
    "run `git lfs pull` in it, and pass --data path/to/opendata-basketball/data/matches"
)
STAGES = {
    "detect_possessions": schema.POSSESSIONS,
    "detect_ball_handler": schema.BALL_HANDLER,
    "detect_actions": schema.ACTIONS,
}
RUN_FILE = "run.json"
ID_COLUMNS = ("player_id", "player2_id", "team_id", "action_type", "outcome")  # read back as text, never as numbers


# ---------------------------------------------------------------- submissions and run folders


def load_functions(target: str, names) -> tuple[str, dict]:
    """The submission's name and those of ``names`` it defines. ``target`` is a .py file or a dotted module."""
    if target.endswith(".py"):
        path = Path(target).resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = sys.modules[path.stem] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    found = {n: getattr(module, n) for n in names if callable(getattr(module, n, None))}
    if not found:
        raise ValueError(f"{target} defines none of {', '.join(names)}")
    return (Path(target).stem if target.endswith(".py") else target.rsplit(".", 1)[-1]), found


def run_submission(stages: dict, game, pbp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the stages a submission defines, in order; each gets the tables before it, None for a skipped one."""
    tables, earlier = {}, []
    for fn_name, kind in STAGES.items():
        if fn_name in stages:
            tables[kind] = schema.validate(kind, stages[fn_name](game, pbp, *earlier))
        earlier.append(tables.get(kind))
    return tables


def save_run_tables(run_dir: str | Path, game_id: str, tables: dict[str, pd.DataFrame]) -> None:
    folder = Path(run_dir) / str(game_id)
    folder.mkdir(parents=True, exist_ok=True)
    for kind, table in tables.items():
        table.to_csv(folder / f"{kind}.csv", index=False)


def load_run_tables(run_dir: str | Path, game_id: str) -> dict[str, pd.DataFrame]:
    folder = Path(run_dir) / str(game_id)
    return {
        kind: schema.validate(kind, pd.read_csv(folder / f"{kind}.csv", dtype={c: str for c in ID_COLUMNS}))
        for kind in schema.COLUMNS
        if (folder / f"{kind}.csv").exists()
    }


def run_games(run_dir: str | Path) -> list[str]:
    root = Path(run_dir)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.csv"))) if root.is_dir() else []


# ---------------------------------------------------------------- commands


def cmd_fetch(args) -> int:
    """Download the tracking. The feed is a separate record and comes from `fetch-acb`."""
    folders = fetch(args.data, args.games)
    print(f"{len(folders)} games in {args.data}")
    if not list(Path(args.feeds).glob("*.csv")):
        print(f"no play-by-play feed in {args.feeds} yet: run `python runner.py fetch-acb` (it needs an ACB API key)")
    return 0


def cmd_fetch_acb(args) -> int:
    """Download the official ACB play-by-play and build the feed from it. The matching answers go to their own
    folder, away from the feeds a detector reads."""
    games = _games(args)
    if args.api_key:
        fetch_acb(args.raw, args.api_key, games, args.contact)
    differs = check_acb(args.raw, games)
    if differs:
        print(f"warning: {len(differs)} file(s) differ from the hashes in data_loader/acb.py: {', '.join(differs)}")
    for game_id in games:
        folder = _game_dir(args.data, game_id)
        game = load_game(folder)
        truth = ground_truth(folder, game)
        feed, answers = benchmark_feed(load_acb(args.raw, game_id), game, load_events(folder), truth.actions)
        write_feed(feed, Path(args.feeds) / f"{game_id}.csv")
        write_feed(answers, Path(args.answers) / f"{game_id}.csv")
    print(f"{len(games)} play-by-play feeds in {args.feeds}")
    return 0


def cmd_detect(args) -> int:
    """The detector sees the tracking and the feed, never the ground truth."""
    name, stages = load_functions(args.target, STAGES)
    run_dir = Path(args.out or Path("runs") / name)
    entries = []
    for game_id in _games(args):
        folder = _game_dir(args.data, game_id)
        game = load_game(folder)
        truth = ground_truth(folder, game)
        own = run_submission(stages, game, _feed(args, game_id, game))
        save_run_tables(run_dir, game_id, own)
        if args.check:  # a game's tracking is ~100 MB: keep it only when it is scored below
            entries.append((game, truth, own))
        print(f"{game_id}: " + ", ".join(f"{k} {len(v)}" for k, v in own.items()))
    meta = {
        "name": name,
        "target": args.target,
        "stages": list(stages),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / RUN_FILE).write_text(json.dumps(meta, indent=2))
    print(f"run written to {run_dir}")
    if args.check:
        print(to_markdown(evaluate_run(entries, _settings(args), meta)))
    return 0


def cmd_match_pbp(args) -> int:
    """The matcher sees the tracking and the feed, never the ground truth."""
    name, found = load_functions(args.target, ["match_pbp"])
    match_pbp = found["match_pbp"]
    run_dir = Path(args.out or Path("runs") / f"{name}-pbp")
    for game_id in _games(args):
        game = load_game(_game_dir(args.data, game_id))
        matches = schema.validate(schema.PBP_MATCHES, match_pbp(game, _feed(args, game_id, game)))
        save_run_tables(run_dir, game_id, {schema.PBP_MATCHES: matches})
        print(f"{game_id}: {len(matches)} feed rows matched")
    meta = {"name": name, "target": args.target, "task": PBP_MATCHING}
    meta["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (run_dir / RUN_FILE).write_text(json.dumps(meta, indent=2))
    print(f"run written to {run_dir}")
    if args.check:
        print(to_markdown(_matching_report(run_dir, args, meta)))
    return 0


def _matching_report(run_dir: Path, args, meta: dict) -> dict:
    """Scored against the answers fetch-acb wrote for each feed row."""
    game_ids, matches, answers = args.games or run_games(run_dir), [], []
    for game_id in game_ids:
        path = Path(args.answers) / f"{game_id}.csv"
        if not path.exists():
            sys.exit(f"no matching answers at {path}; run `python runner.py fetch-acb`")
        matches.append(load_run_tables(run_dir, game_id)[schema.PBP_MATCHES].assign(game_id=game_id))
        answers.append(pd.read_csv(path).assign(game_id=game_id))
    matches, answers = pd.concat(matches, ignore_index=True), pd.concat(answers, ignore_index=True)
    games = {game_id: evaluate_pbp_matching(matches, answers, game_ids=[game_id]) for game_id in game_ids}
    return {"meta": meta, "aggregate": evaluate_pbp_matching(matches, answers), "games": games}


def cmd_evaluate(args) -> int:
    run_dir = Path(args.run)
    meta = json.loads((run_dir / RUN_FILE).read_text()) if (run_dir / RUN_FILE).exists() else {"name": run_dir.name}
    games = args.games or run_games(run_dir)
    if not games:
        sys.exit(f"no game tables under {run_dir}; run `python runner.py detect` or `match-pbp` first")
    if meta.get("task") == PBP_MATCHING:
        report = _matching_report(run_dir, args, meta)
        default_out = f"{meta['name']}-pbp.json"
    else:
        entries = []
        for game_id in games:
            folder = _game_dir(args.data, game_id)
            game = load_game(folder)
            entries.append((game, ground_truth(folder, game), load_run_tables(run_dir, game_id)))
        report = evaluate_run(entries, _settings(args), meta)
        default_out = f"{meta.get('name', run_dir.name)}.json"
    # results/ and the leaderboards take only reports that compare: every game, the default settings
    comparable = set(games) == set(GAME_IDS) and _settings(args) == Settings()
    if args.out:
        out = Path(args.out)
    else:
        out = Path("results") / default_out if comparable else run_dir / "report.json"
    save(report, out)
    if args.out or comparable:
        board = leaderboard([load(f) for f in sorted(out.parent.glob("*.json"))])
        (out.parent / "leaderboard.md").write_text(board)
        if out.parent.resolve() == (REPO / "results").resolve():
            update_readme_leaderboard(REPO / "README.md", board)
    print(to_markdown(report))
    print(f"report written to {out}")
    if not (args.out or comparable):
        print("not in results/ or the leaderboard: they take reports on every game with the default settings")
    return 0


def update_readme_leaderboard(readme: Path, board: str) -> None:
    """Copy the leaderboard into the README between its two markers, so the README never shows stale scores."""
    start, end = LEADERBOARD_MARKERS
    text = readme.read_text()
    if start not in text or end not in text:
        return
    board = re.sub(r"^## ", "### ", board, flags=re.MULTILINE)  # the README section is itself a level-2 heading
    before, rest = text.split(start, 1)
    readme.write_text(f"{before}{start}\n{board.strip()}\n{end}{rest.split(end, 1)[1]}")


def cmd_visualize(args) -> int:
    from visualization import render

    folder = _game_dir(args.data, args.game)
    game = load_game(folder)
    truth = ground_truth(folder, game) if args.truth else None
    tables = load_run_tables(args.run, args.game)
    out = render(
        game,
        tables,
        args.out,
        possession=args.possession,
        frames=args.frames,
        truth=truth,
        fps=args.fps,
        stride=args.stride,
    )
    print(f"wrote {out}")
    return 0


def _games(args) -> list[str]:
    games = args.games or available(args.data)
    if not games:
        sys.exit(f"no games under {args.data}. {GET_DATA}")
    return games


def _game_dir(data: str, game_id: str) -> Path:
    """The game's folder, or exit with how to get the data."""
    if str(game_id) not in available(data):
        sys.exit(f"game {game_id} is not under {data}, or its tracking file is a Git LFS pointer. {GET_DATA}")
    return match_dir(data, game_id)


def _feed(args, game_id: str, game):
    """The feed the run sees. Missing is an error: there is no other feed to fall back to."""
    path = Path(args.feeds) / f"{game_id}.csv"
    if not path.exists():
        sys.exit(
            f"no play-by-play feed at {path}. Run `python runner.py fetch-acb` to build it on the official "
            "ACB clock; it needs an ACB API key."
        )
    return read_feed(path)


def _settings(args) -> Settings:
    actions = ActionSettings(window_seconds=args.window, iou_threshold=args.iou)
    return Settings(actions=actions, possession_iou=args.iou, complete_window_seconds=args.complete_window)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="log progress")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the SkillCorner games", parents=[common])
    p.add_argument("--data", default=DATA, help="folder of SkillCorner match folders (default %(default)s)")
    p.add_argument("--games", nargs="*", default=None, help=f"default: all {len(GAME_IDS)}")
    p.add_argument("--feeds", default=FEEDS, help="where the play-by-play feed is written")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser(
        "fetch-acb", parents=[common], help="download the ACB play-by-play and build the feed on its clock"
    )
    p.add_argument("--data", default=DATA, help="folder of SkillCorner match folders (default %(default)s)")
    p.add_argument("--games", nargs="*", default=None)
    p.add_argument("--raw", default=ACB, help="where the raw ACB files are kept (default %(default)s)")
    p.add_argument("--feeds", default=FEEDS, help="where the feed is written (default %(default)s)")
    p.add_argument("--answers", default=ANSWERS, help="matching answers from fetch-acb (default %(default)s)")
    p.add_argument("--api-key", default=os.environ.get("ACB_API_KEY"), help="X-APIKEY, or set $ACB_API_KEY")
    p.add_argument("--contact", default=os.environ.get("ACB_CONTACT", ""), help="your email, sent in the User-Agent")
    p.set_defaults(func=cmd_fetch_acb)

    p = sub.add_parser("detect", parents=[common], help="run a submission and store its tables under runs/")
    p.add_argument("target", help="submission: a .py file or a dotted module, e.g. baselines.naive")
    p.add_argument("--data", default=DATA, help="folder of SkillCorner match folders (default %(default)s)")
    p.add_argument("--games", nargs="*", default=None)
    p.add_argument("--out", default=None, help="run directory (default runs/<submission name>)")
    p.add_argument("--feeds", default=FEEDS, help="play-by-play feeds from fetch-acb (default %(default)s)")
    p.add_argument("--check", action="store_true", help="also print the evaluation, without saving a report")
    _eval_args(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("match-pbp", parents=[common], help="run a play-by-play matcher and store its table under runs/")
    p.add_argument("target", help="a .py file or a dotted module defining match_pbp, e.g. baselines.pbp_clock")
    p.add_argument("--data", default=DATA, help="folder of SkillCorner match folders (default %(default)s)")
    p.add_argument("--games", nargs="*", default=None)
    p.add_argument("--out", default=None, help="run directory (default runs/<name>-pbp)")
    p.add_argument("--feeds", default=FEEDS, help="play-by-play feeds from fetch-acb (default %(default)s)")
    p.add_argument("--answers", default=ANSWERS, help="matching answers from fetch-acb (default %(default)s)")
    p.add_argument("--check", action="store_true", help="also print the evaluation, without saving a report")
    p.set_defaults(func=cmd_match_pbp)

    p = sub.add_parser("evaluate", parents=[common], help="score a run directory against the ground truth")
    p.add_argument("run")
    p.add_argument("--data", default=DATA, help="folder of SkillCorner match folders (default %(default)s)")
    p.add_argument("--games", nargs="*", default=None)
    p.add_argument(
        "--out", default=None, help="report path (default results/<name>.json; <run>/report.json if partial or custom)"
    )
    p.add_argument("--answers", default=ANSWERS, help="matching answers from fetch-acb (default %(default)s)")
    _eval_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("visualize", parents=[common], help="animate one possession or frame range of a run")
    p.add_argument("run")
    p.add_argument("game")
    p.add_argument("--data", default=DATA, help="folder of SkillCorner match folders (default %(default)s)")
    p.add_argument("--possession", type=int, default=None, help="index into the run's possessions table")
    p.add_argument("--frames", default=None, help="frame range start:end")
    p.add_argument("--truth", action="store_true", help="overlay the SkillCorner events")
    p.add_argument("--out", required=True, help=".gif or .mp4")
    p.add_argument("--fps", type=float, default=12.5)
    p.add_argument("--stride", type=int, default=2)
    p.set_defaults(func=cmd_visualize)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    return args.func(args)


def _eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--window", type=float, default=1.0, help="matching window for instant actions, seconds")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for intervals and possessions")
    p.add_argument(
        "--complete-window", type=float, default=2.0, help="window for the complete-possession metric, seconds"
    )


if __name__ == "__main__":
    raise SystemExit(main())
