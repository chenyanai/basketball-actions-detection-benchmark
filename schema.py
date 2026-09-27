"""The output contract: the three tables a detector produces and the evaluator scores.

Every table is a pandas DataFrame. Time is the game-wide frame index at 25 fps.
Coordinates are feet with the origin at center court, as in the SkillCorner data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

POSSESSIONS = "possessions"
BALL_HANDLER = "ball_handler"
ACTIONS = "actions"
PBP_MATCHES = "pbp_matches"

INSTANT = "instant"
INTERVAL = "interval"


@dataclass(frozen=True)
class ActionSpec:
    shape: str
    player: str
    player2: str | None = None
    subtypes: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()


ACTION_TYPES: dict[str, ActionSpec] = {
    "pass": ActionSpec(INSTANT, "passer", "receiver", subtypes=("handoff",), outcomes=("complete", "incomplete")),
    "shot": ActionSpec(INSTANT, "shooter", outcomes=("made", "missed")),
    "rebound": ActionSpec(INSTANT, "rebounder", subtypes=("offensive", "defensive")),
    "turnover": ActionSpec(INSTANT, "player who lost the ball", "stealer"),
    "foul": ActionSpec(INSTANT, "fouler", "fouled player", subtypes=("shooting", "personal", "offensive", "flagrant")),
    "screen": ActionSpec(INSTANT, "screener", "teammate the screen is set for", subtypes=("on_ball", "off_ball")),
    "drive": ActionSpec(INTERVAL, "ball handler"),
    "isolation": ActionSpec(INTERVAL, "ball handler"),
    "post": ActionSpec(INTERVAL, "ball handler"),
    "closeout": ActionSpec(INTERVAL, "defender", "ball handler"),
}

COLUMNS: dict[str, dict[str, str]] = {
    POSSESSIONS: {"period": "int", "start_frame": "int", "end_frame": "int", "team_id": "str"},
    BALL_HANDLER: {"frame_idx": "int", "player_id": "str?"},
    ACTIONS: {
        "period": "int",
        "frame_idx": "int",
        "end_frame": "int?",
        "action_type": "str",
        "player_id": "str?",
        "player2_id": "str?",
        "team_id": "str",
        "x": "float?",
        "y": "float?",
        "subtype": "str?",
        "outcome": "str?",
        "confidence": "float?",
    },
    PBP_MATCHES: {"event_id": "int", "frame_idx": "int", "x": "float?", "y": "float?"},
}


class SchemaError(ValueError):
    pass


def empty(kind: str) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=_pandas_dtype(spec)) for col, spec in COLUMNS[kind].items()})


def concat(kind: str, tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    tables = [t for t in tables if t is not None and len(t)]
    if not tables:
        return empty(kind)
    return normalize(kind, pd.concat(tables, ignore_index=True))


def normalize(kind: str, table: pd.DataFrame) -> pd.DataFrame:
    """Coerce column types, fill defaults, and order columns. Extra columns are kept."""
    out = table.copy()
    for col, spec in COLUMNS[kind].items():
        if col not in out.columns:
            if _optional(spec):
                out[col] = np.nan
            else:
                raise SchemaError(f"{kind}: missing required column '{col}'")
        out[col] = _coerce(out[col], spec)
    if kind == ACTIONS:
        out["confidence"] = out["confidence"].fillna(1.0)
    ordered = list(COLUMNS[kind]) + [c for c in out.columns if c not in COLUMNS[kind]]
    return out[ordered].reset_index(drop=True)


def validate(kind: str, table: pd.DataFrame) -> pd.DataFrame:
    """Return the normalized table or raise SchemaError naming the first bad row."""
    if kind not in COLUMNS:
        raise SchemaError(f"unknown table kind '{kind}'")
    out = normalize(kind, table)
    for col, spec in COLUMNS[kind].items():
        if not _optional(spec):
            _require_no_nulls(kind, out, col)
    if kind == POSSESSIONS:
        bad = out.index[out["end_frame"] < out["start_frame"]]
        _fail_if(bad, kind, "end_frame before start_frame")
    if kind == BALL_HANDLER:
        _fail_if(out.index[out["frame_idx"].duplicated()], kind, "duplicate frame_idx")
    if kind == PBP_MATCHES:
        _fail_if(out.index[out["event_id"].duplicated()], kind, "duplicate event_id")
    if kind == ACTIONS:
        unknown = out.index[~out["action_type"].isin(ACTION_TYPES)]
        _fail_if(unknown, kind, "unknown action_type")
        shapes = out["action_type"].map(lambda t: ACTION_TYPES[t].shape)
        _fail_if(out.index[(shapes == INTERVAL) & out["end_frame"].isna()], kind, "interval action without end_frame")
        _fail_if(out.index[(shapes == INSTANT) & out["end_frame"].notna()], kind, "instant action with end_frame")
        _fail_if(
            out.index[out["end_frame"].notna() & (out["end_frame"] < out["frame_idx"])],
            kind,
            "end_frame before frame_idx",
        )
        _fail_if(out.index[(out["confidence"] < 0) | (out["confidence"] > 1)], kind, "confidence outside [0, 1]")
    return out


def _require_no_nulls(kind: str, table: pd.DataFrame, col: str) -> None:
    _fail_if(table.index[table[col].isna()], kind, f"null in required column '{col}'")


def _fail_if(bad_index: pd.Index, kind: str, reason: str) -> None:
    if len(bad_index):
        raise SchemaError(f"{kind}: {reason} at row {int(bad_index[0])} ({len(bad_index)} rows)")


def _optional(spec: str) -> bool:
    """A trailing "?" marks a column that may be missing or null."""
    return spec.endswith("?")


def _pandas_dtype(spec: str) -> str:
    """int -> int64 (Int64 when optional, so it can hold nulls), float -> float64, str -> object."""
    dtypes = {"int": "Int64" if _optional(spec) else "int64", "float": "float64"}
    return dtypes.get(spec.rstrip("?"), "object")


def _coerce(series: pd.Series, spec: str) -> pd.Series:
    dtype = _pandas_dtype(spec)
    if dtype == "object":
        return series.map(_as_id)
    return pd.to_numeric(series, errors="coerce").astype(dtype)


def _as_id(value) -> str | None:
    """Ids as text: 59188, 59188.0 and "59188" all become "59188"; missing becomes None."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return str(value)
