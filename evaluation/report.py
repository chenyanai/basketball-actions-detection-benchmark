"""Score a run of either task and pool its games; render its report and the leaderboards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tabulate import tabulate

import schema
from data_loader.game import FPS, Game, GroundTruth
from evaluation.actions import ActionSettings, evaluate_actions
from evaluation.ball_handler import evaluate_ball_handler
from evaluation.matching import f1, ratio, round4
from evaluation.possessions import evaluate_complete_possessions, evaluate_possessions
from schema import ACTION_TYPES, INTERVAL

TABLES = (schema.POSSESSIONS, schema.BALL_HANDLER, schema.ACTIONS)
PBP_MATCHING = "pbp_matching"
MATCHING_WINDOWS = (0.5, 1.0, 2.0)  # seconds: a feed row is also scored at each of these windows
MATCHING_TYPES = ("shot", "rebound", "turnover", "foul")  # the feed rows scored; free throws are not


@dataclass
class Settings:
    actions: ActionSettings = field(default_factory=ActionSettings)
    possession_iou: float = 0.5
    complete_window_seconds: float = 2.0
    live_tolerance_seconds: float = 1.0
    fps: float = FPS


Entry = tuple[Game, GroundTruth, dict[str, pd.DataFrame]]  # a game, its ground truth, the detector's tables


def evaluate_game(
    detections: dict[str, pd.DataFrame], truth: GroundTruth, game: Game, settings: Settings | None = None
) -> dict:
    return _evaluate(_stack([(game, truth, detections)]), settings or Settings())


def evaluate_run(entries: list[Entry], settings: Settings | None = None, meta: dict | None = None) -> dict:
    """One report over all games pooled, plus one per game scored on the same tables filtered to that game."""
    settings = settings or Settings()
    tables = _stack(entries)
    games = {game.game_id: _evaluate(tables, settings, game_ids=[game.game_id]) for game, _, _ in entries}
    return {
        "meta": meta or {},
        "settings": _settings_dict(settings),
        "aggregate": _evaluate(tables, settings),
        "games": games,
    }


@dataclass
class _Tables:
    """Every game's tables stacked into one per kind, each row tagged with its game_id."""

    detections: dict[str, pd.DataFrame]
    truth: dict[str, pd.DataFrame]
    frames: pd.DataFrame
    team: dict[str, str]  # player id -> team id

    def only(self, game_ids: list[str]) -> _Tables:
        def keep(table: pd.DataFrame) -> pd.DataFrame:
            return table[table["game_id"].isin(game_ids)]

        return _Tables(
            {k: keep(t) for k, t in self.detections.items()},
            {k: keep(t) for k, t in self.truth.items()},
            keep(self.frames),
            self.team,
        )


def _stack(entries: list[Entry]) -> _Tables:
    def stack(kind: str, tables: list[tuple[str, pd.DataFrame | None]]) -> pd.DataFrame:
        tagged = [t.assign(game_id=game_id) for game_id, t in tables if t is not None and len(t)]
        return pd.concat(tagged, ignore_index=True) if tagged else schema.empty(kind).assign(game_id=None)

    team = {}
    for game, _, _ in entries:
        team.update(game.team_by_player)
    return _Tables(
        detections={t: stack(t, [(game.game_id, det.get(t)) for game, _, det in entries]) for t in TABLES},
        truth={t: stack(t, [(game.game_id, getattr(truth, t)) for game, truth, _ in entries]) for t in TABLES},
        frames=pd.concat([game.frames.assign(game_id=game.game_id) for game, _, _ in entries], ignore_index=True),
        team=team,
    )


def _evaluate(tables: _Tables, settings: Settings, game_ids: list[str] | None = None) -> dict:
    """Score every game in ``tables``, or only ``game_ids`` when given."""
    attempted = set(tables.detections[schema.ACTIONS]["action_type"])  # the whole run's, before choosing games
    if game_ids:
        tables = tables.only(game_ids)
    det, truth, frames = tables.detections, tables.truth, tables.frames
    tolerance = int(round(settings.live_tolerance_seconds * settings.fps))
    live = live_frames(frames)
    possessions = det[schema.POSSESSIONS]
    handler = on_live_frames(det[schema.BALL_HANDLER], frames)
    actions = near_live_play(det[schema.ACTIONS], frames, tolerance)
    true_actions = near_live_play(truth[schema.ACTIONS], frames, tolerance)
    return {
        "possessions": evaluate_possessions(
            possessions, truth[schema.POSSESSIONS], live, settings.fps, settings.possession_iou
        ),
        "ball_handler": evaluate_ball_handler(handler, on_live_frames(truth[schema.BALL_HANDLER], frames), tables.team),
        "actions": evaluate_actions(actions, true_actions, settings.actions, attempted),
        "complete_possessions": evaluate_complete_possessions(
            possessions,
            truth[schema.POSSESSIONS],
            actions,
            true_actions,
            settings.complete_window_seconds,
            settings.possession_iou,
            settings.fps,
            attempted,
        ),
    }


# ---------------------------------------------------------------- live play
# Only live play is scored: frames where the game clock is moving. Free throws, the wait before an
# inbound, timeouts and the pause after a whistle are dead time. Frame-level metrics use live frames
# only; an action counts when it happens within a tolerance of live play, which keeps the whistle
# and the inbound pass itself.


def live_frames(frames: pd.DataFrame) -> pd.DataFrame:
    return frames[frames["live"]]


def on_live_frames(table: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Rows of a per-frame table that fall on live frames."""
    keys = ["game_id", "frame_idx"]
    return table.merge(live_frames(frames)[keys], on=keys) if len(table) else table


def near_live_play(actions: pd.DataFrame, frames: pd.DataFrame, tolerance_frames: int) -> pd.DataFrame:
    """Actions whose frame is within ``tolerance_frames`` of a live frame of the same game."""
    if actions.empty:
        return actions
    live = live_frames(frames)
    keep = np.zeros(len(actions), dtype=bool)
    for game_id, rows in actions.groupby("game_id").indices.items():
        ids = np.sort(live.loc[live["game_id"] == game_id, "frame_idx"].to_numpy())
        if not len(ids):
            continue
        at = actions["frame_idx"].to_numpy()[rows]
        right = np.clip(np.searchsorted(ids, at), 0, len(ids) - 1)
        left = np.clip(right - 1, 0, len(ids) - 1)
        keep[rows] = np.minimum(np.abs(ids[right] - at), np.abs(ids[left] - at)) <= tolerance_frames
    return actions[keep].reset_index(drop=True)


# ---------------------------------------------------------------- play-by-play matching


def evaluate_pbp_matching(
    matches: pd.DataFrame, answers: pd.DataFrame, fps: float = FPS, game_ids: list[str] | None = None
) -> dict:
    """How close each feed row's submitted frame is to its true one. A row the submission leaves out is a miss.

    ``matches``: game_id, event_id, frame_idx, x, y. ``answers``: the matching answers fetch-acb wrote, plus game_id.

    Scores every game in ``answers``, or only ``game_ids`` when given.
    """
    if game_ids:
        matches, answers = matches[matches["game_id"].isin(game_ids)], answers[answers["game_id"].isin(game_ids)]
    keys = ["game_id", "event_id"]
    found = matches[keys + ["frame_idx", "x", "y"]].drop_duplicates(keys)
    found = found.rename(columns={"frame_idx": "found_frame", "x": "found_x", "y": "found_y"})
    rows = answers.merge(found, on=keys, how="left")
    rows["dt"] = (rows["found_frame"] - rows["frame_idx"]).abs() / fps
    rows["loc"] = np.hypot(rows["found_x"] - rows["x"], rows["found_y"] - rows["y"])
    return {"all": _matching_metrics(rows)} | {
        t: _matching_metrics(rows[rows["event_type"] == t]) for t in MATCHING_TYPES
    }


def _matching_metrics(rows: pd.DataFrame) -> dict:
    dt = rows["dt"].dropna()
    out = {"rows": len(rows), "matched": round4(len(dt) / len(rows)) if len(rows) else None}
    for w in MATCHING_WINDOWS:
        out[f"within_{w:g}s"] = round4((dt <= w).sum() / len(rows)) if len(rows) else None
    # a row placed within 1 s is correct: precision over the rows returned, recall (= within 1 s) over all rows
    correct = int((dt <= 1.0).sum())
    out["precision"], out["recall"] = ratio(correct, len(dt)), ratio(correct, len(rows))
    out["f1"] = f1(out["precision"], out["recall"])
    out["mean_abs_s"] = round4(dt.mean())
    out["median_abs_s"] = round4(dt.median())
    out["p90_abs_s"] = round4(dt.quantile(0.9))
    out["median_location_error_ft"] = round4(rows["loc"].median())
    return out


def save(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- markdown


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def sec(x) -> str:
    return "–" if x is None else f"{x:.2f} s"


def ft(x) -> str:
    return "–" if x is None else f"{x:.1f} ft"


def table(rows: list[dict] | list[list], headers="keys") -> str:
    """A GitHub Markdown table; a dict row may leave out columns, which stay blank."""
    return tabulate(rows, headers=headers, tablefmt="github")


def markdown_page(title: str, bullets: list[str], sections: dict[str, str]) -> str:
    """``# title``, a bullet list, then one ``## heading`` per section."""
    head = "\n".join([f"# {title}", ""] + [f"- {b}" for b in bullets])
    return head + "\n" + "".join(f"\n## {heading}\n\n{body}\n" for heading, body in sections.items())


def to_markdown(report: dict) -> str:
    if report.get("meta", {}).get("task") == PBP_MATCHING:
        return _matching_markdown(report)
    meta, agg = report.get("meta", {}), report["aggregate"]
    sections = {
        "Actions": table([_action_row(kind, agg["actions"][kind]) for kind in ACTION_TYPES if kind in agg["actions"]]),
        "F1 by matching window": _window_table(agg["actions"])
        + f"\n\nThe official window is {report['settings']['window_seconds']:g} s. A rising row is late, not missed.",
        "Possessions": _stage_table(agg["possessions"], _possession_metrics),
        "Ball handler": _stage_table(agg["ball_handler"], _ball_handler_metrics),
        "Complete possessions": _stage_table(agg["complete_possessions"], _complete_metrics)
        + "\n\nComplete: split correctly and every attempted action found with the right player"
        " (defined in the README).",
    }
    if report.get("games"):
        per_game = [_game_row(game_id, g, _scored_types(agg)) for game_id, g in report["games"].items()]
        sections["Per game"] = table(per_game) + "\n\nAction cells are precision / recall."
    bullets = [f"{key}: {meta[key]}" for key in ("target", "created") if key in meta]
    if meta.get("stages"):
        bullets.append("stages: " + ", ".join(meta["stages"]))
    return markdown_page(meta.get("name", "run"), bullets, sections)


def _window_table(actions: dict) -> str:
    rows = [
        {"type": kind} | {f"{w} s": pct(v) for w, v in m["f1_by_window"].items()}
        for kind, m in actions.items()
        if "f1_by_window" in m
    ]
    return table(rows) if rows else "not attempted"


def _stage_table(metrics: dict, values: Callable[[dict], dict]) -> str:
    """A metric | value table, or "not attempted" when the submission produced nothing to score."""
    if metrics["status"] != "scored":
        return "not attempted"
    return table([{"metric": name, "value": value} for name, value in values(metrics).items()])


def _possession_metrics(p: dict) -> dict:
    boundary = p.get("boundary_error_s", {})
    return {
        "matched (IoU ≥ threshold)": f"{p['matched']} of {p['truth']}",
        "recall": pct(p["recall"]),
        "precision": pct(p["precision"]),
        "F1": pct(p.get("f1")),
        "frame-level team accuracy": pct(p["frame_team_accuracy"]),
        "mean IoU": pct(p.get("mean_iou")),
        "start error, mean abs": sec(boundary.get("start", {}).get("mean_abs_s")),
        "end error, mean abs": sec(boundary.get("end", {}).get("mean_abs_s")),
    }


def _ball_handler_metrics(h: dict) -> dict:
    return {
        "live frames": h["frames"],
        "frame accuracy (right player, or nobody when the ball is free)": pct(h["frame_accuracy"]),
        "inside touches: right player": pct(h["in_touch_accuracy"]),
        "inside touches: wrong teammate": pct(h["wrong_teammate_rate"]),
        "inside touches: wrong team": pct(h["wrong_team_rate"]),
        "inside touches: no handler": pct(h["no_handler_rate"]),
        "ball free: nobody named": pct(h["no_ball_accuracy"]),
    }


def _complete_metrics(c: dict) -> dict:
    return {
        "possessions": c["truth"],
        "action types": ", ".join(c["action_types"]),
        "window": f"{c['window_seconds']:g} s",
        "complete": pct(c["complete"]),
        "split correctly": pct(c["split_correct"]),
        "all actions found": pct(c["all_actions_found"]),
    } | {f"all {t} found": pct(v) for t, v in c["all_found_by_type"].items()}


def leaderboard(reports: list[dict]) -> str:
    """Two tables: detection, then play-by-play matching. Detection reports scored with other settings are left out."""
    matching = [r for r in reports if r.get("meta", {}).get("task") == PBP_MATCHING]
    official = _settings_dict(Settings())
    detection = [r for r in reports if r not in matching and r.get("settings") == official]
    parts = ["## Detection\n\n" + _detection_leaderboard(detection)] if detection else []
    if matching:
        parts.append("## Play-by-Play Matching\n\n" + _matching_leaderboard(matching))
    return "\n".join(parts)


def _detection_leaderboard(reports: list[dict]) -> str:
    """One row per report, best value per column in bold: the three stages, complete possessions, F1 per action type."""
    types = [
        t for t in ACTION_TYPES if any(r["aggregate"]["actions"].get(t, {}).get("status") == "scored" for r in reports)
    ]

    def complete(r: dict) -> float:
        return r["aggregate"].get("complete_possessions", {}).get("complete") or 0.0

    def own_possessions(r: dict) -> bool:
        return r["aggregate"]["possessions"].get("status") == "scored"

    reports = sorted(reports, key=lambda r: (not own_possessions(r), -complete(r)))  # full pipelines first
    rows = []
    for r in reports:
        agg, meta = r["aggregate"], r.get("meta", {})
        rows.append(
            {
                "detector": meta.get("name", "?"),
                "possession frames": agg["possessions"].get("frame_team_accuracy"),
                "possession F1": agg["possessions"].get("f1"),
                "ball handler": agg["ball_handler"].get("frame_accuracy"),
                # without its own possessions the split is not scored, so the value is shown but never the best
                "complete possessions": complete(r) if own_possessions(r) else f"{pct(complete(r))} (actions only)",
            }
            | {f"{t} F1": agg["actions"].get(t, {}).get("f1") for t in types}
        )
    for column in list(rows[0])[1:]:
        best = max((r[column] for r in rows if isinstance(r[column], float)), default=None)
        for r in rows:
            r[column] = _cell(r[column], best)
    return table(rows) + "\n"


def _cell(value, best) -> str:
    """A percentage, in bold when it is the column's best; text such as "(actions only)" passes through."""
    if isinstance(value, str):
        return value
    return f"**{pct(value)}**" if value is not None and value == best else pct(value)


def _matching_markdown(report: dict) -> str:
    meta = report.get("meta", {})
    by_type = [{"type": kind} | _matching_row(m) for kind, m in report["aggregate"].items()]
    per_game = [
        {"game": game_id, "rows": m["all"]["rows"]} | _matching_headline(m["all"])
        for game_id, m in report["games"].items()
    ]
    bullets = [f"{key}: {meta[key]}" for key in ("target", "task", "created") if key in meta]
    return markdown_page(
        meta.get("name", "run"), bullets, {"Play-by-play matching": table(by_type), "Per game": table(per_game)}
    )


def _matching_leaderboard(reports: list[dict]) -> str:
    """One row per matcher: F1 and timing over all feed rows, then F1 per event type."""
    rows = [
        {"matcher": r["meta"].get("name", "?")}
        | _matching_headline(r["aggregate"]["all"])
        | {f"{t} F1": pct(r["aggregate"][t].get("f1")) for t in MATCHING_TYPES}
        for r in sorted(reports, key=lambda r: -(r["aggregate"]["all"].get("f1") or 0))
    ]
    return table(rows) + "\n"


def _matching_headline(m: dict) -> dict:
    """The median, not the mean: a few rows placed a whole stoppage away would otherwise decide the column."""
    return {"F1": pct(m.get("f1")), "within 1 s": pct(m["within_1s"]), "median abs Δt": sec(m["median_abs_s"])}


def _matching_row(m: dict) -> dict:
    return (
        {"rows": m["rows"], "matched": pct(m["matched"])}
        | {"precision": pct(m.get("precision")), "F1": pct(m.get("f1"))}
        | {f"within {w:g} s": pct(m[f"within_{w:g}s"]) for w in MATCHING_WINDOWS}
        | {
            "mean abs Δt": sec(m["mean_abs_s"]),
            "median abs Δt": sec(m["median_abs_s"]),
            "p90 abs Δt": sec(m["p90_abs_s"]),
            "median loc. err": ft(m["median_location_error_ft"]),
        }
    )


def _action_row(kind: str, m: dict) -> dict:
    if m["status"] != "scored":
        return {"type": kind, "truth": m["truth"], "detected": "–", "precision": "not attempted"}
    timing = m.get("timing") or {}
    if ACTION_TYPES[kind].shape == INTERVAL:
        timing = timing.get("start", {})
    return {
        "type": kind,
        "truth": m["truth"],
        "detected": m["detected"],
        "precision": pct(m["precision"]),
        "recall": pct(m["recall"]),
        "F1": pct(m["f1"]),
        "strict recall": pct(m["strict_recall"]),
        "mean abs Δt": sec(timing.get("mean_abs_s")),
        "RMSE Δt": sec(timing.get("rmse_s")),
        "median loc. err": ft((m.get("location_error_ft") or {}).get("median")),
    }


def _game_row(game_id: str, g: dict, types: list[str]) -> dict:
    """Action cells are precision / recall."""
    row = {
        "game": game_id,
        "possessions": pct(g["possessions"].get("frame_team_accuracy")),
        "handler": pct(g["ball_handler"].get("frame_accuracy")),
    }
    for kind in types:
        m = g["actions"].get(kind, {})
        row[kind] = f"{pct(m.get('precision'))} / {pct(m.get('recall'))}" if m.get("status") == "scored" else "–"
    return row


def _scored_types(agg: dict) -> list[str]:
    return [k for k in ACTION_TYPES if agg["actions"].get(k, {}).get("status") == "scored"]


def _settings_dict(settings: Settings) -> dict:
    return {
        "window_seconds": settings.actions.window_seconds,
        "action_iou_threshold": settings.actions.iou_threshold,
        "complete_window_seconds": settings.complete_window_seconds,
        "live_tolerance_seconds": settings.live_tolerance_seconds,
        "possession_iou_threshold": settings.possession_iou,
    }
