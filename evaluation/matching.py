"""Blind matching of detections to ground truth. Nothing is matched by identity.

Every table here carries ``game_id``: a single game is just a run of one. The number helpers every
metric shares (rounding, ratios, error statistics) are at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Matches:
    pairs: list[tuple[int, int]]  # (detection row, ground-truth row), positional
    unmatched_detections: list[int]
    unmatched_truth: list[int]

    @property
    def tp(self) -> int:
        return len(self.pairs)

    @property
    def fp(self) -> int:
        return len(self.unmatched_detections)

    @property
    def fn(self) -> int:
        return len(self.unmatched_truth)


def match_instant(
    detections: pd.DataFrame, truth: pd.DataFrame, window_frames: int, require_player: bool = False
) -> Matches:
    """Greedy, closest first: same period, |frame difference| <= window, optionally same player."""
    det, tru = detections.reset_index(drop=True), truth.reset_index(drop=True)
    candidates = []
    for subset, pool in _same_game_and_period(det, tru):
        t_rows, t_frames = pool.index.to_numpy(), pool["frame_idx"].to_numpy()
        t_players = ids(pool["player_id"])
        for i, frame, player in zip(subset.index, subset["frame_idx"], ids(subset["player_id"])):
            delta = np.abs(t_frames - frame)
            ok = delta <= window_frames
            if require_player:
                ok &= t_players == player
            candidates.extend((int(delta[j]), int(i), int(t_rows[j])) for j in np.flatnonzero(ok))
    return _greedy(candidates, len(det), len(tru))


def match_interval(
    detections: pd.DataFrame,
    truth: pd.DataFrame,
    iou_threshold: float,
    start_col: str = "frame_idx",
    same_team: bool = True,
) -> Matches:
    """Greedy, highest temporal IoU first: same period, IoU >= threshold, optionally same team."""
    det, tru = detections.reset_index(drop=True), truth.reset_index(drop=True)
    candidates = []
    for subset, pool in _same_game_and_period(det, tru):
        (d_start, d_end), (t_start, t_end) = span(subset, start_col), span(pool, start_col)
        iou = temporal_iou(d_start[:, None], d_end[:, None], t_start, t_end)  # one row per detection
        ok = iou >= iou_threshold
        if same_team:
            ok &= ids(subset["team_id"])[:, None] == ids(pool["team_id"])
        i, j = np.nonzero(ok)
        candidates += zip((-iou[i, j]).tolist(), subset.index[i].tolist(), pool.index[j].tolist())
    return _greedy(candidates, len(det), len(tru))


def ids(series: pd.Series) -> np.ndarray:
    """Ids as strings with missing values as '' so they compare cleanly."""
    return series.fillna("").astype(str).to_numpy()


def _same_game_and_period(det: pd.DataFrame, truth: pd.DataFrame):
    """(detections, truth) pairs that share a game and period: rows only ever match within one."""
    det_groups = dict(list(det.groupby(["game_id", "period"])))
    for key, pool in truth.groupby(["game_id", "period"]):
        if key in det_groups:
            yield det_groups[key], pool


def span(table: pd.DataFrame, start_col: str = "frame_idx") -> tuple[np.ndarray, np.ndarray]:
    """Interval start and end frames as float arrays."""
    return table[start_col].to_numpy(float), table["end_frame"].to_numpy(float)


def temporal_iou(a_start, a_end, b_start, b_end):
    """IoU of frame intervals; works element-wise on numpy arrays, so broadcasting gives every pair."""
    inter = np.maximum(0, np.minimum(a_end, b_end) - np.maximum(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def _greedy(candidates: list[tuple[float, int, int]], n_det: int, n_truth: int) -> Matches:
    candidates.sort()
    used_det: set[int] = set()
    used_truth: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i in used_det or j in used_truth:
            continue
        used_det.add(i)
        used_truth.add(j)
        pairs.append((i, j))
    return Matches(
        pairs, [i for i in range(n_det) if i not in used_det], [j for j in range(n_truth) if j not in used_truth]
    )


# ---------------------------------------------------------------- numbers every metric reports


def round4(x) -> float | None:
    """Metrics are stored to 4 decimals; None and NaN become None."""
    return None if x is None or pd.isna(x) else round(float(x), 4)


def ratio(num, den) -> float | None:
    return round4(num / den) if den else None


def f1(p: float | None, r: float | None) -> float | None:
    return round4(2 * p * r / (p + r)) if p and r else (0.0 if p is not None and r is not None else None)


def error_stats(signed: np.ndarray) -> dict:
    """Summary of signed errors in seconds (detected minus true)."""
    abs_err = np.abs(signed)
    return {
        "n": int(len(signed)),
        "mean_abs_s": round4(abs_err.mean()),
        "median_abs_s": round4(np.median(abs_err)),
        "p90_abs_s": round4(np.percentile(abs_err, 90)),
        "mse_s2": round4(np.mean(signed**2)),
        "rmse_s": round4(np.sqrt(np.mean(signed**2))),
        "mean_signed_s": round4(signed.mean()),
        "within_0.5s": round4((abs_err <= 0.5).mean()),
        "within_1s": round4((abs_err <= 1.0).mean()),
    }
