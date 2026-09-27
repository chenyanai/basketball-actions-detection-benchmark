"""SkillCorner's annotations as a list of play-by-play events: what happened and who did it.

One row per shot, free throw, rebound, turnover and foul, with team, player, outcome and SkillCorner's own
clock in whole seconds. This is not the benchmark's feed, and that clock leaks the answer. It serves two
purposes: data_loader/acb.py aligns it with the ACB scoresheet to find which fouls came on a missed shot,
and the tests use it on the bundled excerpt, which has no ACB data. ``feed_answers`` gives each row's true frame.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from data_loader.skillcorner import foul_subtype, rebound_subtype, shot_teams

COLUMNS = [
    "event_id",
    "period",
    "clock",
    "event_type",
    "subtype",
    "points",
    "team_id",
    "player_id",
    "player2_id",
    "outcome",
    "score_home",
    "score_away",
]

_ORDER = {"shot": 0, "foul": 1, "turnover": 2, "free_throw": 3, "rebound": 4}

SHOT_SUBTYPES = {"tip": "tip", "dunk": "dunk", "layup": "layup", "lob": "layup", "hook": "hook", "floater": "floater"}


def build_feed(events: dict, home_team_id: str, away_team_id: str) -> pd.DataFrame:
    """The events, on SkillCorner's own clock. That clock leaks the answer: only fetch-acb's feed is ever scored."""
    return _finalize(_rows(events), home_team_id, away_team_id)


def feed_answers(events: dict, actions: pd.DataFrame) -> pd.DataFrame:
    """The true frame and location of every feed row. Free throws are not actions, so they have no answer."""
    rows = pd.DataFrame(_rows(events)).rename_axis("event_id").reset_index()
    rows = rows[rows["event_type"] != "free_throw"]
    where = actions.drop_duplicates(["action_type", "frame_idx", "team_id"])[
        ["action_type", "frame_idx", "team_id", "x", "y"]
    ]
    where = where.rename(columns={"action_type": "event_type", "frame_idx": "true_frame"})
    out = rows.merge(where, on=["event_type", "true_frame", "team_id"], how="left")
    return out.rename(columns={"true_frame": "frame_idx"})[["event_id", "event_type", "frame_idx", "x", "y"]]


def _rows(events: dict) -> list[dict]:
    shots = {s["id"]: s for s in events.get("shots", [])}
    fouls = [f for f in events.get("fouls", []) if not f.get("isTechnical")]
    rows = []

    for s in shots.values():
        made = bool(s.get("outcome"))
        rows.append(
            _row(
                s,
                "shot",
                s["startFrame"],
                _shot_clock(s),
                _shot_subtype(s),
                3 if s.get("three") else 2,
                s["offTeamId"],
                s["shooterId"],
                s.get("passerId") if s.get("assisted") else None,
                "made" if made else "missed",
            )
        )

    foul_rows = {}
    for f in fouls:
        frame, clock = int(f["frame"]), float(f.get("gameClock") or 0.0)
        shot = shots.get(f.get("shotId"))
        if shot is not None:
            frame, clock = int(shot["startFrame"]), _shot_clock(shot)
        foul_rows[f["id"]] = _row(
            f, "foul", frame, clock, foul_subtype(f), 0, f["foulerTeamId"], f.get("foulerId"), f.get("fouledId"), None
        )
        foul_rows[f["id"]]["true_frame"] = int(f["frame"])
        rows.append(foul_rows[f["id"]])

    for t in events.get("turnovers", []):
        subtype, linked_foul = _turnover_subtype(t, events.get("passes", []), fouls)
        frame, clock = int(t["frame"]), float(t.get("gameClock") or 0.0)
        if linked_foul is not None:
            frame, clock = foul_rows[linked_foul["id"]]["frame"], foul_rows[linked_foul["id"]]["clock"]
        rows.append(
            _row(
                t,
                "turnover",
                frame,
                clock,
                subtype,
                0,
                t["teamId"],
                None if t.get("teamTo") else t.get("turnedOverId"),
                t.get("stealerId"),
                None,
            )
        )
        rows[-1]["true_frame"] = int(t["frame"])

    by_foul: dict = {}
    for ft in events.get("free_throws", []):
        by_foul.setdefault(ft.get("foulId") or ft["id"], []).append(ft)
    for group in by_foul.values():
        group.sort(key=lambda x: int(x["frame"]))
        for k, ft in enumerate(group, start=1):
            rows.append(
                _row(
                    ft,
                    "free_throw",
                    ft["frame"],
                    float(ft.get("gameClock") or 0.0),
                    f"{k} of {len(group)}",
                    1,
                    ft["offTeamId"],
                    ft["shooterId"],
                    None,
                    "made" if ft.get("outcome") else "missed",
                )
            )

    shot_team = shot_teams(events)
    for r in events.get("rebounds", []):
        rows.append(
            _row(
                r,
                "rebound",
                r["frame"],
                float(r.get("gameClock") or 0.0),
                rebound_subtype(r, shot_team),
                0,
                r["teamId"],
                r.get("rebounderId") if r.get("rebounded") else None,
                None,
                None,
            )
        )

    return sorted(rows, key=lambda r: (r["period"], r["frame"], _ORDER[r["event_type"]]))


def write_feed(feed: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    feed.to_csv(path, index=False)


def read_feed(path: str | Path) -> pd.DataFrame:
    feed = pd.read_csv(
        path, dtype={"team_id": str, "player_id": str, "player2_id": str, "subtype": str, "outcome": str}
    )
    for col in ("player_id", "player2_id", "subtype", "outcome"):
        feed[col] = feed[col].where(feed[col].notna(), None)
    return feed


def _row(e, event_type, frame, clock, subtype, points, team, player, player2, outcome) -> dict:
    return {
        "period": int(e["period"]),
        "frame": int(frame),
        "true_frame": int(frame),
        "clock": float(clock),
        "event_type": event_type,
        "subtype": subtype,
        "points": points,
        "team_id": str(team),
        "player_id": None if player is None else str(player),
        "player2_id": None if player2 is None else str(player2),
        "outcome": outcome,
    }


def _shot_clock(s: dict) -> float:
    return float(s["endGameClock"] if s.get("endGameClock") is not None else s.get("startGameClock") or 0.0)


def _shot_subtype(s: dict) -> str:
    for key in ("complexShotType", "shotType"):
        value = (s.get(key) or "").lower()
        if value in SHOT_SUBTYPES:
            return SHOT_SUBTYPES[value]
    return "jump"


def _turnover_subtype(t: dict, passes: list[dict], fouls: list[dict]) -> tuple[str, dict | None]:
    clock, period, player = float(t.get("gameClock") or 0.0), t.get("period"), t.get("turnedOverId")

    def near(value) -> bool:
        return value is not None and abs(float(value) - clock) <= 3.0

    if t.get("teamTo"):
        return "violation", None
    offensive = [
        f
        for f in fouls
        if (f.get("foulType") or "").lower() == "offensive"
        and f.get("period") == period
        and near(f.get("gameClock"))
        and f.get("foulerId") == player
    ]
    if offensive:
        return "offensive_foul", min(offensive, key=lambda f: abs(float(f["gameClock"]) - clock))
    if any(
        p.get("turnover")
        and p.get("period") == period
        and near(p.get("startGameClock"))
        and p.get("passerId") == player
        for p in passes
    ):
        return "bad_pass", None
    return "lost_ball", None


def _finalize(rows: list[dict], home: str, away: str) -> pd.DataFrame:
    score = {home: 0, away: 0}
    out, prev_period, prev_clock = [], None, None
    for event_id, r in enumerate(rows):
        clock = int(math.floor(max(0.0, r["clock"])))
        if prev_period == r["period"] and prev_clock is not None:
            clock = min(clock, prev_clock)
        prev_period, prev_clock = r["period"], clock
        if r["outcome"] == "made":
            score[r["team_id"]] = score.get(r["team_id"], 0) + r["points"]
        out.append({**r, "event_id": event_id, "clock": clock, "score_home": score[home], "score_away": score[away]})
    return pd.DataFrame(out, columns=COLUMNS)
