"""Per-action-type metrics: precision, recall, timing error, location error, attribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader.game import FPS
from evaluation.matching import (
    Matches,
    error_stats,
    f1,
    ids,
    match_instant,
    match_interval,
    ratio,
    round4,
    span,
    temporal_iou,
)
from schema import ACTION_TYPES, INTERVAL

# Recall is also reported on these ground-truth subsets: name -> (column, value).
SUBSETS: dict[str, dict[str, tuple[str, object]]] = {
    "shot": {"made": ("outcome", "made"), "missed": ("outcome", "missed"), "fouled": ("fouled", True)},
    "rebound": {
        "offensive": ("subtype", "offensive"),
        "defensive": ("subtype", "defensive"),
        "after_free_throw": ("after_free_throw", True),
    },
    "pass": {"inbounds": ("inbounds", True), "not_inbounds": ("inbounds", False), "handoff": ("subtype", "handoff")},
    "screen": {"on_ball": ("subtype", "on_ball"), "off_ball": ("subtype", "off_ball"), "handoff": ("handoff", True)},
    "foul": {
        "shooting": ("subtype", "shooting"),
        "personal": ("subtype", "personal"),
        "offensive": ("subtype", "offensive"),
    },
}


F1_WINDOWS = (0.5, 1.0, 2.0)  # seconds; F1 is also reported at these windows, to tell late from missed


@dataclass
class ActionSettings:
    window_seconds: float = 1.0
    iou_threshold: float = 0.5
    fps: float = FPS

    def window_frames(self) -> int:
        return int(round(self.window_seconds * self.fps))


def evaluate_actions(
    detections: pd.DataFrame,
    truth: pd.DataFrame,
    settings: ActionSettings | None = None,
    attempted: set[str] | None = None,
) -> dict:
    """``attempted``: the types the submission returned anywhere in the run, by default those in ``detections``.

    A type it never returned is not attempted; a returned type with no detections here scores zero recall.
    """
    settings = settings or ActionSettings()
    attempted = set(detections["action_type"]) if attempted is None else attempted
    out = {}
    for kind, spec in ACTION_TYPES.items():
        gt = truth[truth["action_type"] == kind].reset_index(drop=True)
        det = detections[detections["action_type"] == kind].reset_index(drop=True)
        if gt.empty and det.empty:
            continue
        if kind not in attempted:
            out[kind] = {"status": "not_attempted", "truth": len(gt)}
            continue
        if spec.shape == INTERVAL:
            matches = match_interval(det, gt, settings.iou_threshold)
            strict = _strict_interval(det, gt, matches)
        else:
            matches = match_instant(det, gt, settings.window_frames())
            strict = match_instant(det, gt, settings.window_frames(), require_player=True).tp
        out[kind] = _metrics(kind, det, gt, matches, strict, settings)
        if spec.shape != INTERVAL:
            out[kind]["f1_by_window"] = {f"{w:g}": _f1_at(det, gt, int(round(w * settings.fps))) for w in F1_WINDOWS}
    return out


def _metrics(
    kind: str, det: pd.DataFrame, gt: pd.DataFrame, m: Matches, strict_tp: int, settings: ActionSettings
) -> dict:
    precision = ratio(m.tp, m.tp + m.fp)
    recall = ratio(m.tp, m.tp + m.fn)
    result = {
        "status": "scored",
        "truth": len(gt),
        "detected": len(det),
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "strict_recall": ratio(strict_tp, len(gt)),
    }
    if not m.pairs:
        return result
    d = det.iloc[[i for i, _ in m.pairs]].reset_index(drop=True)
    t = gt.iloc[[j for _, j in m.pairs]].reset_index(drop=True)
    result["timing"] = _timing(kind, d, t, settings.fps)
    result["location_error_ft"] = _location(d, t)
    result["player_accuracy"] = _agreement(d["player_id"], t["player_id"])
    if ACTION_TYPES[kind].player2:
        result["player2_accuracy"] = _agreement(d["player2_id"], t["player2_id"])
    if ACTION_TYPES[kind].subtypes:
        result["subtype_accuracy"] = _agreement(d["subtype"], t["subtype"])
    if "attributable" in gt.columns:
        attributable = gt["attributable"].fillna(True).astype(bool)
        result["attributable"] = _subset(m, attributable)
        result["team_events"] = _subset(m, ~attributable)
    for name, (column, value) in SUBSETS.get(kind, {}).items():
        if column in gt.columns:
            result.setdefault("subsets", {})[name] = _subset(m, (gt[column] == value).fillna(False).astype(bool))
    if det["confidence"].nunique() > 1:
        result["average_precision"] = _average_precision(det, m)
    return result


def _f1_at(det: pd.DataFrame, gt: pd.DataFrame, window_frames: int) -> float | None:
    m = match_instant(det, gt, window_frames)
    return f1(ratio(m.tp, m.tp + m.fp), ratio(m.tp, m.tp + m.fn))


def _timing(kind: str, d: pd.DataFrame, t: pd.DataFrame, fps: float) -> dict:
    start = (d["frame_idx"].to_numpy() - t["frame_idx"].to_numpy()) / fps
    out = error_stats(start)
    if ACTION_TYPES[kind].shape == INTERVAL:
        end = (d["end_frame"].to_numpy().astype(float) - t["end_frame"].to_numpy().astype(float)) / fps
        out = {
            "start": out,
            "end": error_stats(end),
            "mean_iou": float(temporal_iou(*span(d), *span(t)).mean()),
        }
    return out


def _location(d: pd.DataFrame, t: pd.DataFrame) -> dict | None:
    ok = d["x"].notna() & d["y"].notna() & t["x"].notna() & t["y"].notna()
    if not ok.any():
        return None
    dist = np.hypot(
        d.loc[ok, "x"].to_numpy() - t.loc[ok, "x"].to_numpy(), d.loc[ok, "y"].to_numpy() - t.loc[ok, "y"].to_numpy()
    )
    return {
        "n": int(ok.sum()),
        "mean": round4(dist.mean()),
        "median": round4(np.median(dist)),
        "p90": round4(np.percentile(dist, 90)),
    }


def _agreement(a: pd.Series, b: pd.Series) -> float | None:
    known = b.notna().to_numpy()
    if not known.any():
        return None
    return round4((ids(a)[known] == ids(b)[known]).mean())


def _subset(m: Matches, mask: pd.Series) -> dict:
    total = int(mask.sum())
    matched = sum(1 for _, j in m.pairs if mask.iloc[j])
    return {"truth": total, "tp": matched, "recall": ratio(matched, total)}


def _strict_interval(det: pd.DataFrame, gt: pd.DataFrame, m: Matches) -> int:
    d, t = ids(det["player_id"]), ids(gt["player_id"])
    return sum(1 for i, j in m.pairs if d[i] == t[j])


def _average_precision(det: pd.DataFrame, m: Matches) -> float:
    hit = np.zeros(len(det), dtype=bool)
    hit[[i for i, _ in m.pairs]] = True
    order = np.argsort(-det["confidence"].to_numpy(), kind="stable")
    hits = hit[order]
    precision_at = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    total_truth = m.tp + m.fn
    return round4(float(precision_at[hits].sum() / total_truth)) if total_truth else 0.0
