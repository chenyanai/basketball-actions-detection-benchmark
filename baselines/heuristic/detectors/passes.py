"""Passes: a handler change between teammates, stamped at the passer's last frame."""

from __future__ import annotations

from baselines.heuristic.config import HeuristicConfig
from baselines.heuristic.inputs import Action, Frame


def detect_passes(frames: list[Frame], shots: list[Action], config: HeuristicConfig) -> list[Action]:
    """``frames`` are one possession's own frames with ``holder`` set; ``shots`` keep ball flight from being a pass."""
    passes: list[Action] = []
    previous: Frame | None = None
    last_pass_t: float | None = None
    last_pair: tuple[str, str] | None = None
    last: Frame | None = None
    for f in frames:
        if last is not None and (f.period != last.period or f.t - last.t > config.time_gap_reset):
            previous, last_pass_t, last_pair = None, None, None
        last = f
        if f.holder is None:
            continue
        if previous is not None and _in_shot_flight(f, previous, shots, config):
            previous = f
            continue
        pair = (previous.holder.id, f.holder.id) if previous is not None else None
        since_last = f.t - last_pass_t if last_pass_t is not None else float("inf")
        if (
            previous is not None
            and f.holder.id != previous.holder.id
            and f.holder.team == previous.holder.team
            and not (pair == last_pair and since_last < config.repeat_pass_interval)
            and since_last > config.min_pass_interval
        ):
            passes.append(
                Action(
                    "pass",
                    previous,
                    previous.holder.id,
                    previous.holder.team,
                    player2=f.holder.id,
                    x=previous.ball_x,
                    y=previous.ball_y,
                    outcome="complete",
                    method="handler_change",
                )
            )
            last_pass_t, last_pair = f.t, pair
        previous = f
    return passes


def _in_shot_flight(f: Frame, previous: Frame, shots: list[Action], config: HeuristicConfig) -> bool:
    for shot in shots:
        gap = abs(f.t - shot.t)
        if gap <= config.max_shot_flight and previous.holder is not None and previous.holder.id == shot.player:
            if f.ball_z > config.shot_height or (
                f.ball_z > config.shot_flight_low_ratio * config.shot_height and gap < config.shot_flight_low_window
            ):
                return True
    return False
