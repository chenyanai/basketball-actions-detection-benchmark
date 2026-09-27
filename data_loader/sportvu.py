"""NBA SportVU tracking and play-by-play, loaded into a Game and a feed so the heuristic can run on NBA data.

``load_game`` reads a raw SportVU game (the public ``.json``, or its ``.7z`` when py7zr is installed) and
``build_feed`` a stats.nba.com play-by-play CSV, both into the shapes the heuristic takes:

    game, feed = sportvu.load_game("0021500001.json"), sportvu.build_feed("0021500001_pbp.csv")
    config = dataclasses.replace(heuristic.default_config(), pre_inbound_seconds=0.0)
    possessions = heuristic.detect_possessions(game, feed, config)
    actions = heuristic.detect_actions(game, pbp=feed, config=config)
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.game import CourtGeometry, Game

NBA_COURT = CourtGeometry(length=94.0, width=50.0, hoop_x=41.75, three_point_radius=23.75, three_point_corner_y=22.0)
CENTER_X, CENTER_Y = 47.0, 25.0  # SportVU's origin is a corner; the heuristic's is center court

MADE, MISSED, FREE_THROW, REBOUND, TURNOVER, FOUL, SUBSTITUTION, TIMEOUT = 1, 2, 3, 4, 5, 6, 8, 9
TIP_SHOTS = {97, 107, 72, 87}
LAYUP_SHOTS = {5, 7, 50, 52, 108, 40, 41, 42, 43, 44, 49, 71, 73, 74, 75, 76, 98, 99, 100}
TURNOVER_SUBTYPES = {
    1: "bad_pass",
    45: "bad_pass",
    2: "lost_ball",
    40: "lost_ball",
    41: "lost_ball",
    5: "offensive_foul",
}
SHOOTING_FOULS = {2, 29}


def load_game(path: str | Path) -> Game:
    raw = _read_raw(Path(path))
    first = raw["events"][0]
    home, away = str(first["home"]["teamid"]), str(first["visitor"]["teamid"])

    moments, seen = [], set()
    for event in raw["events"]:
        for m in event.get("moments") or []:
            if len(m) >= 6 and m[1] not in seen:
                seen.add(m[1])
                moments.append((int(m[0]), float(m[2]), 0.0 if m[3] is None else float(m[3]), m[5]))
    moments = pd.DataFrame(moments, columns=["period", "game_clock", "shot_clock", "locations"])
    moments = moments.sort_values(["period", "game_clock"], ascending=[True, False], kind="stable")
    moments = moments.drop_duplicates(["period", "game_clock"]).reset_index(drop=True)  # one capture per clock reading

    frames, players = [], []
    for idx, m in enumerate(moments.itertuples(index=False)):
        ball = next((o for o in m.locations if o[0] == -1 and o[1] == -1), (-1, -1, 0.0, 0.0, 0.0))
        frames.append((idx, m.period, m.game_clock, m.shot_clock, ball[2] - CENTER_X, ball[3] - CENTER_Y, ball[4]))
        for team, player, x, y, *_ in m.locations:
            if player not in (-1, 0):
                players.append((idx, str(int(player)), str(int(team)), x - CENTER_X, y - CENTER_Y))

    roster = pd.DataFrame(
        [
            (str(p["playerid"]), str(side["teamid"]), str(p.get("jersey", "")), f"{p['firstname']} {p['lastname']}")
            for event in raw["events"]
            for side in (event["home"], event["visitor"])
            for p in side.get("players", [])
        ],
        columns=["player_id", "team_id", "jersey", "name"],
    ).drop_duplicates("player_id")
    return Game(
        game_id=str(raw.get("gameid", "")),
        home_team_id=home,
        away_team_id=away,
        team_names={home: first["home"].get("name", home), away: first["visitor"].get("name", away)},
        roster=roster.reset_index(drop=True),
        frames=pd.DataFrame(
            frames,
            columns=["frame_idx", "period", "game_clock", "shot_clock", "ball_x", "ball_y", "ball_z"],
        ),
        players=pd.DataFrame(players, columns=["frame_idx", "player_id", "team_id", "x", "y"]),
        court=NBA_COURT,
        period_seconds=720.0,
        overtime_seconds=300.0,
    )


def build_feed(pbp_csv: str | Path) -> pd.DataFrame:
    """Every play-by-play row in file order, in the benchmark's feed columns plus ``description``."""
    rows = []
    for r in pd.read_csv(pbp_csv).itertuples(index=False):
        kind = int(r.EVENTMSGTYPE)
        action = int(r.EVENTMSGACTIONTYPE) if pd.notna(r.EVENTMSGACTIONTYPE) else 0
        minutes, seconds = str(r.PCTIMESTRING).split(":")
        text = " ".join(str(d) for d in (r.HOMEDESCRIPTION, r.VISITORDESCRIPTION, r.NEUTRALDESCRIPTION) if pd.notna(d))
        text = text.lower()
        player, team, player2 = _id(r.PLAYER1_ID), _id(r.PLAYER1_TEAM_ID), _id(r.PLAYER2_ID)
        if team is None and player is not None and 1610612700 <= int(player) <= 1610612799:
            player, team = None, player  # a team event carries the team id as its player
        row = {
            "event_id": int(r.EVENTNUM),
            "period": int(r.PERIOD),
            "clock": int(minutes) * 60 + int(float(seconds)),
            "event_type": "other",
            "subtype": None,
            "points": 0,
            "team_id": team,
            "player_id": player,
            "player2_id": None,
            "outcome": None,
            "description": text,
        }
        if kind in (MADE, MISSED):
            subtype = "tip" if action in TIP_SHOTS else "layup" if action in LAYUP_SHOTS else "jump_shot"
            row.update(event_type="shot", subtype=subtype, outcome="made" if kind == MADE else "missed")
            row.update(points=3 if "3pt" in text else 2, player2_id=player2 if kind == MADE else None)
        elif kind == FREE_THROW:
            count = re.search(r"(\d+) of (\d+)", text)
            row.update(event_type="free_throw", outcome="missed" if "miss" in text else "made")
            row.update(subtype=f"{count.group(1)} of {count.group(2)}" if count else None)
        elif kind == REBOUND:
            tally = re.search(r"off:(\d+)", text)
            offensive = (tally is not None and int(tally.group(1)) > 0) or "offensive" in text
            row.update(event_type="rebound", subtype="offensive" if offensive else "defensive")
        elif kind == TURNOVER:
            row.update(event_type="turnover", subtype=TURNOVER_SUBTYPES.get(action, "violation"), player2_id=player2)
        elif kind == FOUL:
            shooting = action in SHOOTING_FOULS or "s.foul" in text or "shooting" in text
            row.update(event_type="foul", subtype="shooting" if shooting else "personal", player2_id=player2)
        elif kind in (SUBSTITUTION, TIMEOUT):
            row.update(event_type="substitution" if kind == SUBSTITUTION else "timeout")
        rows.append(row)
    return pd.DataFrame(rows)


def _read_raw(path: Path) -> dict:
    if path.suffix.lower() != ".7z":
        return json.loads(path.read_text())
    import py7zr

    with py7zr.SevenZipFile(path, "r") as archive, tempfile.TemporaryDirectory() as folder:
        archive.extractall(folder)
        return json.loads(next(Path(folder).rglob("*.json")).read_text())


def _id(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) in ("", "0"):
        return None
    return str(int(float(value)))
