"""Ball-handler metrics: per-frame agreement with SkillCorner touches.

The headline is accuracy over every live frame: the right player inside a touch, and nobody outside one.
"""

from __future__ import annotations

import pandas as pd

from evaluation.matching import ids, ratio


def evaluate_ball_handler(detections: pd.DataFrame, truth: pd.DataFrame, team: dict[str, str]) -> dict:
    """``team`` maps player id to team id."""
    if detections.empty:
        return {"status": "not_attempted", "truth_frames": int(truth["player_id"].notna().sum())}
    merged = truth.merge(detections, on=["game_id", "frame_idx"], how="left", suffixes=("_true", "_det"))
    true_p, det_p = merged["player_id_true"], merged["player_id_det"]
    in_touch = true_p.notna().to_numpy()
    has_det = det_p.notna().to_numpy()
    right_player = in_touch & (ids(det_p) == ids(true_p))
    right_nobody = ~in_touch & ~has_det
    same_team = (
        in_touch
        & has_det
        & ~right_player
        & (det_p.map(team).fillna("").to_numpy() == true_p.map(team).fillna("").to_numpy())
    )
    return {
        "status": "scored",
        "frames": len(merged),
        "frames_in_touch": int(in_touch.sum()),
        "frame_accuracy": ratio((right_player | right_nobody).sum(), len(merged)),
        "in_touch_accuracy": ratio(right_player.sum(), in_touch.sum()),
        "no_ball_accuracy": ratio(right_nobody.sum(), (~in_touch).sum()),
        "wrong_teammate_rate": ratio(same_team.sum(), in_touch.sum()),
        "wrong_team_rate": ratio((in_touch & has_det & ~right_player & ~same_team).sum(), in_touch.sum()),
        "no_handler_rate": ratio((in_touch & ~has_det).sum(), in_touch.sum()),
    }
