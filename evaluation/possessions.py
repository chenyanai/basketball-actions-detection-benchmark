"""Possession metrics: how well the game is split into possessions, and complete possessions.

The split is scored per live frame (does the detected possession name the right team) and per
possession (interval matching, with the error at the start and end).

A true possession is complete when
  1. a detected possession of the same team overlaps it with IoU >= threshold, and
  2. every true action inside it, of the types the submission attempts, is matched by a
     detection of the same type and player within ``window_seconds``.
When the submission detects no possessions only the second condition applies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader.game import FPS
from evaluation.matching import error_stats, f1, match_instant, match_interval, ratio, round4, span, temporal_iou
from schema import ACTION_TYPES, INTERVAL


def evaluate_possessions(
    detections: pd.DataFrame, truth: pd.DataFrame, frames: pd.DataFrame, fps: float = FPS, iou_threshold: float = 0.5
) -> dict:
    if detections.empty:
        return {"status": "not_attempted", "truth": len(truth)}
    det, gt = detections.reset_index(drop=True), truth.reset_index(drop=True)
    m = match_interval(det, gt, iou_threshold, start_col="start_frame")
    result = {
        "status": "scored",
        "truth": len(gt),
        "detected": len(det),
        "matched": m.tp,
        "precision": ratio(m.tp, len(det)),
        "recall": ratio(m.tp, len(gt)),
        "f1": f1(ratio(m.tp, len(det)), ratio(m.tp, len(gt))),
        "frame_team_accuracy": _frame_team_accuracy(det, gt, frames),
    }
    if m.pairs:
        d = det.iloc[[i for i, _ in m.pairs]].reset_index(drop=True)
        t = gt.iloc[[j for _, j in m.pairs]].reset_index(drop=True)
        start = (d["start_frame"].to_numpy() - t["start_frame"].to_numpy()) / fps
        end = (d["end_frame"].to_numpy() - t["end_frame"].to_numpy()) / fps
        result["mean_iou"] = round4(temporal_iou(*span(d, "start_frame"), *span(t, "start_frame")).mean())
        result["boundary_error_s"] = {"start": error_stats(start), "end": error_stats(end)}
    return result


def _frame_team_accuracy(det: pd.DataFrame, gt: pd.DataFrame, frames: pd.DataFrame) -> float | None:
    """Share of live frames inside a true possession whose detected possession names the same team."""
    covered = correct = 0
    for game_id, game_frames in frames.groupby("game_id"):
        frame_ids = game_frames["frame_idx"].to_numpy()
        truth_team = _team_per_frame(gt[gt["game_id"] == game_id], frame_ids)
        det_team = _team_per_frame(det[det["game_id"] == game_id], frame_ids)
        has_truth = np.array([t is not None for t in truth_team])
        covered += int(has_truth.sum())
        correct += int((truth_team[has_truth] == det_team[has_truth]).sum())
    return ratio(correct, covered)


def _team_per_frame(table: pd.DataFrame, frame_ids: np.ndarray) -> np.ndarray:
    team = np.full(len(frame_ids), None, dtype=object)
    for row in table.itertuples(index=False):
        team[(frame_ids >= row.start_frame) & (frame_ids <= row.end_frame)] = str(row.team_id)
    return team


# ---------------------------------------------------------------- complete possessions


def evaluate_complete_possessions(
    det_possessions: pd.DataFrame,
    true_possessions: pd.DataFrame,
    det_actions: pd.DataFrame,
    true_actions: pd.DataFrame,
    window_seconds: float = 2.0,
    iou_threshold: float = 0.5,
    fps: float = FPS,
    attempted: set[str] | None = None,
) -> dict:
    """``attempted``: the types the submission returned anywhere in the run, by default those in ``det_actions``."""
    attempted = set(det_actions["action_type"]) if attempted is None else attempted
    types = [t for t in ACTION_TYPES if t in attempted]
    if not types:
        return {"status": "not_attempted", "truth": len(true_possessions)}
    truth = true_possessions.reset_index(drop=True)
    found = _found_actions(det_actions, true_actions, types, int(round(window_seconds * fps)), iou_threshold)

    requires_split = not det_possessions.empty
    split_ok = pd.Series(True, index=truth.index)
    if requires_split:
        matched = {j for _, j in match_interval(det_possessions, truth, iou_threshold, start_col="start_frame").pairs}
        split_ok = pd.Series([j in matched for j in truth.index], index=truth.index)

    # Every true action inside each true possession; an action on a shared boundary frame counts for both.
    inside = truth[["game_id", "start_frame", "end_frame"]].reset_index(names="possession")
    inside = inside.merge(found[["game_id", "frame_idx", "action_type", "found"]], on="game_id")
    inside = inside[inside["frame_idx"].between(inside["start_frame"], inside["end_frame"])]
    all_found = inside.groupby(["possession", "action_type"])["found"].all()  # per possession and type
    actions_ok = all_found.groupby("possession").all().reindex(truth.index, fill_value=True)
    by_type = all_found.groupby("action_type").agg(["size", "sum"]).reindex(types, fill_value=0)
    complete = split_ok & actions_ok
    return {
        "status": "scored",
        "truth": len(truth),
        "window_seconds": window_seconds,
        "action_types": types,
        "requires_split": requires_split,
        "split_correct": ratio(split_ok.sum(), len(truth)) if requires_split else None,
        "all_actions_found": ratio(actions_ok.sum(), len(truth)),
        "complete": ratio(complete.sum(), len(truth)),
        "all_found_by_type": {t: ratio(by_type.at[t, "sum"], by_type.at[t, "size"]) for t in types},
    }


def _found_actions(
    det: pd.DataFrame, truth: pd.DataFrame, types: list[str], window_frames: int, iou: float
) -> pd.DataFrame:
    """True actions of the attempted types with a ``found`` flag."""
    parts = []
    for kind in types:
        gt = truth[truth["action_type"] == kind].reset_index(drop=True)
        d = det[det["action_type"] == kind].reset_index(drop=True)
        if ACTION_TYPES[kind].shape == INTERVAL:
            hit = {j for _, j in match_interval(d, gt, iou).pairs}
        else:
            hit = {j for _, j in match_instant(d, gt, window_frames, require_player=True).pairs}
        parts.append(gt.assign(found=[j in hit for j in gt.index]))
    return pd.concat(parts, ignore_index=True)
