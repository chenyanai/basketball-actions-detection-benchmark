"""SkillCorner's open data: download a game, and load it into a Game and its ground truth.

For each game three files are fetched over plain HTTPS into ``<root>/<game_id>/``: the
game metadata and the Dynamic Events from raw.githubusercontent.com, and the tracking
file, which the SkillCorner repo stores in Git LFS, from GitHub's LFS media endpoint.
No git, git-lfs or account is needed, files already present are skipped, and nothing
is modified. A clone of the SkillCorner repo made with ``git lfs pull`` has the same
layout under ``data/matches`` and can be used instead by passing it as ``--data``.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import schema
from data_loader.game import FIBA_COURT, FRAME_COLUMNS, PLAYER_COLUMNS, ROSTER_COLUMNS, Game, GroundTruth, mark_live

logger = logging.getLogger(__name__)

RAW = "https://raw.githubusercontent.com/SkillCorner/opendata-basketball/main/data/matches"
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata-basketball/main/data/matches"
GAME_IDS = ["114243", "114234", "114169", "114099", "114086", "178442", "179612", "184439", "188630", "191313"]

FILES = {
    "game_data": ("{id}_game_data.json", RAW),
    "events": ("{id}_dynamic_events.json", RAW),
    "tracking": ("{id}_tracking_data.jsonl.gz", MEDIA),
}


# ---------------------------------------------------------------- download


def match_dir(root: str | Path, game_id: str) -> Path:
    return Path(root) / str(game_id)


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def fetch(root: str | Path, game_ids: Iterable[str] | None = None) -> list[Path]:
    """Write ``<root>/<game_id>/`` with the three per-game files. Returns the match folders."""
    folders = []
    for game_id in [str(g) for g in (game_ids or GAME_IDS)]:
        folder = match_dir(root, game_id)
        folder.mkdir(parents=True, exist_ok=True)
        for name, base in FILES.values():
            target = folder / name.format(id=game_id)
            if target.exists() and not is_lfs_pointer(target):
                continue
            url = f"{base}/{game_id}/{name.format(id=game_id)}"
            logger.info("downloading %s", url)
            _download(url, target)
        folders.append(folder)
    return folders


def available(root: str | Path) -> list[str]:
    """Game ids under ``root`` that have all three files."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [folder / name.format(id=folder.name) for name, _ in FILES.values()]
        if all(f.exists() and not is_lfs_pointer(f) for f in files):
            found.append(folder.name)
    return found


def _download(url: str, target: Path) -> None:
    """Write to ``<target>.part`` and rename it when complete, so an interrupted download is never kept."""
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)


# ---------------------------------------------------------------- load


def load_game(folder: str | Path) -> Game:
    folder = Path(folder)
    game_id = folder.name
    meta = _read_json(folder / f"{game_id}_game_data.json")
    home, away = meta["homeTeam"], meta["awayTeam"]
    home_id, away_id = str(home["teamId"]), str(away["teamId"])
    roster = pd.DataFrame(
        [
            {
                "player_id": str(p["playerId"]),
                "team_id": str(team["teamId"]),
                "jersey": str(p.get("jersey", "")),
                "name": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
            }
            for team in (home, away)
            for p in team.get("players", [])
        ],
        columns=ROSTER_COLUMNS,
    )
    tracking = folder / f"{game_id}_tracking_data.jsonl.gz"
    if is_lfs_pointer(tracking):
        raise ValueError(
            f"{tracking} is a Git-LFS pointer, not the data. Run `git lfs pull` in your clone or use `runner.py fetch`."
        )
    frames, players = _read_tracking(tracking, home_id, away_id)
    return Game(
        game_id=game_id,
        home_team_id=home_id,
        away_team_id=away_id,
        team_names={home_id: home.get("teamName", ""), away_id: away.get("teamName", "")},
        roster=roster,
        frames=frames,
        players=players,
        court=FIBA_COURT,
    )


def load_events(folder: str | Path) -> dict[str, list[dict]]:
    folder = Path(folder)
    return _read_json(folder / f"{folder.name}_dynamic_events.json")


def ground_truth(folder: str | Path, game: Game | None = None) -> GroundTruth:
    """SkillCorner's Dynamic Events as possessions, per-frame ball handler, and actions."""
    folder = Path(folder)
    game = game or load_game(folder)
    events = load_events(folder)
    return GroundTruth(
        game_id=game.game_id,
        possessions=_possessions(events),
        ball_handler=_ball_handler(events, game),
        actions=_actions(events),
        events=events,
    )


# ---------------------------------------------------------------- tracking


def _read_tracking(path: Path, home_id: str, away_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_rows, player_rows = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ball = d.get("ball") or {}
            if not ball.get("xyz") or not d.get("homePlayers") or not d.get("awayPlayers"):
                continue
            idx = int(d["frameIdx"])
            frame_rows.append(
                (
                    idx,
                    int(d["period"]),
                    float(d["gameClock"]),
                    d.get("shotClock"),
                    ball["xyz"][0],
                    ball["xyz"][1],
                    ball["xyz"][2] if len(ball["xyz"]) > 2 else 0.0,
                )
            )
            for side, team in (("homePlayers", home_id), ("awayPlayers", away_id)):
                for p in d[side]:
                    player_rows.append((idx, str(p["playerId"]), team, p["xyz"][0], p["xyz"][1]))
    frames = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    frames["shot_clock"] = pd.to_numeric(frames["shot_clock"], errors="coerce")
    frames["live"] = mark_live(frames)
    players = pd.DataFrame(player_rows, columns=PLAYER_COLUMNS)
    return frames, players


# ---------------------------------------------------------------- ground truth


def _possessions(events: dict) -> pd.DataFrame:
    rows = [
        {"period": p["period"], "start_frame": p["startFrame"], "end_frame": p["endFrame"], "team_id": p["offTeamId"]}
        for p in events.get("possessions", [])
    ]
    return schema.validate(schema.POSSESSIONS, pd.DataFrame(rows, columns=list(schema.COLUMNS[schema.POSSESSIONS])))


def _ball_handler(events: dict, game: Game) -> pd.DataFrame:
    frame_ids = game.frames["frame_idx"].to_numpy()
    holder = pd.Series([None] * len(frame_ids), index=frame_ids, dtype=object)
    for t in events.get("touches", []):
        lo, hi = int(t["startFrame"]), int(t["endFrame"])
        holder.loc[(holder.index >= lo) & (holder.index <= hi)] = str(t["playerId"])
    table = pd.DataFrame({"frame_idx": frame_ids, "player_id": holder.to_numpy()})
    return schema.validate(schema.BALL_HANDLER, table)


def _actions(events: dict) -> pd.DataFrame:
    """Every SkillCorner event family as rows of one action table, one small converter per family.

    Columns a family does not set are filled by ``schema.validate``. Extra columns (inbounds,
    fouled, three, after_free_throw, handoff, attributable) are kept for the recall-by-subset metrics.
    """
    handoffs = [(h["setterId"], h.get("receiverId"), int(h["frame"])) for h in events.get("handoffs", [])]
    shot_team = shot_teams(events)
    converters = {
        "passes": lambda e: _pass(e, handoffs),
        "shots": _shot,
        "rebounds": lambda e: _rebound(e, shot_team),
        "turnovers": _turnover,
        "fouls": _foul,
        "picks": _on_ball_screen,
        "off_ball_screens": _off_ball_screen,
        "drives": lambda e: _ball_handler_interval(e, "drive"),
        "isolations": lambda e: _ball_handler_interval(e, "isolation"),
        "posts": lambda e: _ball_handler_interval(e, "post"),
        "closeouts": _closeout,
    }
    rows = [convert(e) for family, convert in converters.items() for e in events.get(family, [])]
    table = pd.DataFrame([row for row in rows if row is not None])

    # SkillCorner normalizes locations to the attacking direction; turn them back into the tracking frame.
    left_hoop = {p["id"]: bool(p.get("leftHoop")) for p in events.get("possessions", [])}
    flip = table.pop("possession_id").map(left_hoop).eq(False)
    table.loc[flip, ["x", "y"]] *= -1

    table["attributable"] = table["player_id"].notna()
    table = schema.validate(schema.ACTIONS, table)
    return table.sort_values(["period", "frame_idx", "action_type"], kind="stable").reset_index(drop=True)


def _action(e: dict, action_type: str, frame: int, player, team, location: str = "location", **columns) -> dict:
    """The columns every action has; ``columns`` adds the rest by their schema names."""
    x, y = _xy(e.get(location))
    row = {"period": e["period"], "frame_idx": frame, "action_type": action_type, "player_id": player, "team_id": team}
    return row | {"x": x, "y": y, "possession_id": e.get("possessionId")} | columns


def _pass(p: dict, handoffs: list[tuple]) -> dict:
    complete, receiver = p.get("complete"), p.get("receiverId")
    # A handoff event by the same two players within 25 frames (1 s) makes the pass a handoff.
    handoff = any(a == p["passerId"] and b == receiver and abs(f - p["startFrame"]) <= 25 for a, b, f in handoffs)
    return _action(
        p,
        "pass",
        p["startFrame"],
        p["passerId"],
        p["offTeamId"],
        location="passerLoc",
        player2_id=receiver if complete else None,
        outcome=None if complete is None else "complete" if complete else "incomplete",
        subtype="handoff" if handoff else None,
        inbounds=bool(p.get("inbounds")),
    )


def _shot(s: dict) -> dict:
    outcome = "made" if s.get("outcome") else "missed"
    extra = {"fouled": bool(s.get("fouled")), "three": bool(s.get("three"))}
    return _action(s, "shot", s["startFrame"], s["shooterId"], s["offTeamId"], outcome=outcome, **extra)


def _rebound(r: dict, shot_team: dict) -> dict:
    """A team rebound has no player."""
    player = r.get("rebounderId") if r.get("rebounded") else None
    subtype = rebound_subtype(r, shot_team)
    return _action(
        r, "rebound", r["frame"], player, r["teamId"], subtype=subtype, after_free_throw=not r.get("fgReb", True)
    )


def shot_teams(events: dict) -> dict:
    """Shot or free-throw id -> the team that took it, for rebound_subtype."""
    attempts = events.get("shots", []) + events.get("free_throws", [])
    return {a["id"]: a["offTeamId"] for a in attempts}


def rebound_subtype(r: dict, shot_team: dict) -> str:
    """Offensive when the rebounding team took the shot the rebound follows.

    SkillCorner's ``defensive`` flag is left unset on some unsecured team rebounds (14 of 768 over the
    ten games, all with ``rebounded`` false), so the linked shot decides and the flag is only the fallback.
    """
    team = shot_team.get(r.get("shotId"))
    if team is not None:
        return "offensive" if r["teamId"] == team else "defensive"
    return "defensive" if r.get("defensive") else "offensive"


def _turnover(t: dict) -> dict:
    """A team turnover (a violation) has no player."""
    player = None if t.get("teamTo") else t.get("turnedOverId")
    return _action(t, "turnover", t["frame"], player, t["teamId"], player2_id=t.get("stealerId"))


def _foul(f: dict) -> dict | None:
    """Technical fouls are not actions."""
    if f.get("isTechnical"):
        return None
    extra = {"player2_id": f.get("fouledId"), "subtype": foul_subtype(f)}
    return _action(f, "foul", f["frame"], f.get("foulerId"), f["foulerTeamId"], **extra)


def _on_ball_screen(k: dict) -> dict:
    extra = {"player2_id": k.get("ballhandlerId"), "subtype": "on_ball", "handoff": bool(k.get("handoffPick"))}
    return _action(k, "screen", k["frame"], k.get("screenerId"), k["offTeamId"], **extra)


def _off_ball_screen(o: dict) -> dict:
    return _action(
        o, "screen", o["frame"], o.get("screenerId"), o["offTeamId"], player2_id=o.get("cutterId"), subtype="off_ball"
    )


def _ball_handler_interval(e: dict, action_type: str) -> dict:
    """Drives, isolations and posts: the ball handler over a span of frames."""
    return _action(e, action_type, e["startFrame"], e["ballhandlerId"], e["offTeamId"], end_frame=e["endFrame"])


def _closeout(c: dict) -> dict:
    location = "startLoc" if c.get("startLoc") else "location"
    extra = {"player2_id": c.get("ballhandlerId"), "end_frame": c["endFrame"], "location": location}
    return _action(c, "closeout", c["startFrame"], c.get("ballhandlerDefId"), c["defTeamId"], **extra)


def _xy(value) -> tuple[float, float]:
    """SkillCorner's [x, y], or NaN when the event has no location."""
    return (float(value[0]), float(value[1])) if value and value[0] is not None else (np.nan, np.nan)


def foul_subtype(foul: dict) -> str:
    """shooting, offensive, flagrant or personal: one rule for the ground truth and the play-by-play feed."""
    kind = (foul.get("foulType") or "").lower()
    if foul.get("shooting") or kind == "shooting":
        return "shooting"
    return kind if kind in ("offensive", "flagrant") else "personal"


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
