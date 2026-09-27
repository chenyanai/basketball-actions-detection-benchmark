"""Animate a stretch of a game with a run's possessions, ball handler and actions, on a drawn court.

Actions show on the ball. It takes an action's colour while the action lasts,
as SkillCorner labelled it when its events are given, and each of the run's
detections is a soft glow of that colour spreading out from the ball. The
data is SkillCorner's, credited on every frame. Needs matplotlib and pillow (requirements.txt).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import schema
from data_loader.game import CourtGeometry, Game, GroundTruth

HOME, AWAY, BALL, RING = "#1F5FA8", "#C8362B", "#F28C28", "#FFD700"
COLORS = {"pass": "#16A34A", "shot": "#D62728", "rebound": "#0E9FB5", "turnover": "#E6AB02", "foul": "#B02CBF"}
SECONDS = {"pass": 0.9, "shot": 1.5}  # how long the ball keeps an action's colour, and trails
SECONDS_OTHER = 0.6
PULSE_SECONDS, PULSE_RADIUS = 0.4, 2.6  # a detection's glow spreads from the ball to this radius, in feet
PULSE_WAVES = (0.0, 0.3)  # each wave starts this far into the pulse
TRAIL_FRAMES = 12
BAR, BAR_TEXT, SHOT_CLOCK = "#23272E", "#F4F4F4", "#FF5A4F"
SPONSORS = ("Lenovo",)  # dropped from team names in the legend
CREDIT = "Data by SkillCorner"

LINE, FLOOR, PAINT = "#4A3A2A", "#E9CFA3", "#DDBB85"  # the court
CIRCLE, NO_CHARGE = 5.91, 4.10  # FIBA: 1.8 m centre and free-throw circles, 1.25 m no-charge arc


@dataclass
class Marker:
    kind: str
    frame: int
    detected: bool


def render(
    game: Game,
    tables: dict[str, pd.DataFrame],
    out: str | Path,
    possession: int | None = None,
    frames: str | None = None,
    truth: GroundTruth | None = None,
    fps: float = 12.5,
    stride: int = 2,
    dpi: int = 90,
    hold_seconds: float = 1.2,
) -> Path:
    start, end = _frame_range(tables, possession, frames)
    clip = game.frames[(game.frames["frame_idx"] >= start) & (game.frames["frame_idx"] <= end)].reset_index(drop=True)
    if clip.empty:
        raise ValueError("no frames in the requested range")
    players = game.players[game.players["frame_idx"].between(start, end)]
    held = _table(tables, schema.BALL_HANDLER)
    handler = dict(zip(held["frame_idx"], held["player_id"]))
    markers = _markers(_table(tables, schema.ACTIONS), start, end, detected=True)
    if truth is not None:
        markers += _markers(truth.actions, start, end, detected=False)
    return _animate(game, clip, players, handler, markers, Path(out), fps, stride, dpi, hold_seconds, truth is not None)


def _frame_range(tables: dict, possession: int | None, frames: str | None) -> tuple[int, int]:
    if frames:
        a, b = frames.split(":")
        return int(a), int(b)
    if possession is None:
        raise ValueError("give --possession or --frames")
    table = _table(tables, schema.POSSESSIONS)
    if table.empty:
        raise ValueError("the run has no possessions table; use --frames")
    row = table.iloc[possession]
    return int(row["start_frame"]), int(row["end_frame"])


def _table(tables: dict, kind: str) -> pd.DataFrame:
    return tables.get(kind) if tables.get(kind) is not None else schema.empty(kind)


def _markers(actions: pd.DataFrame, start: int, end: int, detected: bool) -> list[Marker]:
    keep = actions["frame_idx"].between(start, end) & actions["action_type"].isin(COLORS)
    return [Marker(a.action_type, int(a.frame_idx), detected) for a in actions[keep].itertuples()]


def _animate(game, clip, players, handler, markers, out, fps, stride, dpi, hold_seconds, has_truth) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    fig = plt.figure(figsize=(10.4, 6.75), dpi=dpi)
    ax = fig.add_axes((0.01, 0.075, 0.98, 0.835))
    draw_court(ax, game.court)
    home, away = (_team_name(game, team_id) for team_id in (game.home_team_id, game.away_team_id))
    fig.patches.append(Rectangle((0, 0.915), 1, 0.085, transform=fig.transFigure, fc=BAR, lw=0))
    bar = dict(y=0.957, va="center", fontsize=13, fontweight="bold", color=BAR_TEXT)
    fig.text(0.37, s=home.upper(), ha="right", **bar)
    fig.text(0.63, s=away.upper(), ha="left", **bar)
    fig.text(0.385, s="●", ha="center", **{**bar, "color": HOME})
    fig.text(0.615, s="●", ha="center", **{**bar, "color": AWAY})
    clock_text = fig.text(0.485, s="", ha="center", family="monospace", **{**bar, "color": RING})
    shot_clock_text = fig.text(0.575, s="", ha="center", family="monospace", **{**bar, "color": SHOT_CLOCK})

    by_frame = {idx: g for idx, g in players.groupby("frame_idx")}
    circles, labels = {}, {}
    for pid, team in zip(game.roster["player_id"], game.roster["team_id"]):
        circles[pid] = Circle(
            (0, 0), 1.3, fc=HOME if team == game.home_team_id else AWAY, ec="white", lw=1.2, zorder=5, visible=False
        )
        ax.add_patch(circles[pid])
        labels[pid] = ax.annotate(
            game.jersey_of(pid),
            (0, 0),
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=6,
            visible=False,
        )
    ring = Circle((0, 0), 1.9, fill=False, ec=RING, lw=2.2, zorder=7, visible=False)
    ax.add_patch(ring)
    ball = Circle((0, 0), 0.7, fc=BALL, ec="#3A2A1A", lw=0.8, zorder=8)
    ax.add_patch(ball)
    trail = LineCollection([], alpha=0.55, capstyle="round", zorder=7.5)
    ax.add_collection(trail)
    ball_xy = clip[["ball_x", "ball_y"]].to_numpy()
    frames = clip["frame_idx"].tolist()
    shown = [m for m in markers if m.detected != has_truth]  # what colours the ball: the labels, if given
    pulses = []
    for m in markers:
        if m.detected:
            waves = [Circle((0, 0), 0, fc=COLORS[m.kind], lw=0, zorder=7.8, visible=False) for _ in PULSE_WAVES]
            for wave in waves:
                ax.add_patch(wave)
            pulses.append((m, waves))

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=RING,
            markeredgewidth=2,
            markersize=11,
            label="Ball handler",
        ),
    ]
    for kind in COLORS:
        if any(m.kind == kind for m in markers):
            label = f"{kind.capitalize()} (SkillCorner)" if has_truth else kind.capitalize()
            handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[kind], markersize=10, label=label)
            )
    if has_truth:
        glow = dict(marker="o", color="w", markerfacecolor="#666", markeredgewidth=0, alpha=0.45)
        handles.append(Line2D([0], [0], markersize=14, label="Glow = detected", **glow))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=len(handles),
        fontsize=9.5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    fig.text(0.99, 0.008, CREDIT, ha="right", va="bottom", fontsize=8, color="#666")

    def update(i):
        row = clip.iloc[i]
        idx = int(row["frame_idx"])
        seen = set()
        for p in by_frame.get(idx, pd.DataFrame()).itertuples():
            seen.add(p.player_id)
            circles[p.player_id].center = (p.x, p.y)
            circles[p.player_id].set_visible(True)
            labels[p.player_id].set_position((p.x, p.y))
            labels[p.player_id].set_visible(True)
        for pid in circles:
            if pid not in seen:
                circles[pid].set_visible(False)
                labels[pid].set_visible(False)
        ball.center = (row["ball_x"], row["ball_y"])
        ball.radius = min(1.8, 0.7 + max(0.0, float(row["ball_z"])) * 0.09)
        h = handler.get(idx)
        if h in circles:  # None or NaN when nobody has the ball
            ring.center = circles[h].center
            ring.set_visible(True)
        else:
            ring.set_visible(False)
        lit = [m for m in shown if 0 <= idx - m.frame <= _seconds(m.kind) * game.fps]
        now = max(lit, key=lambda m: m.frame, default=None)
        ball.set_facecolor(COLORS[now.kind] if now else BALL)
        # the first shown frame at or after the action: a label can sit on a frame the loader dropped
        since = bisect.bisect_left(frames, now.frame) if now and now.kind in SECONDS else i
        path = ball_xy[max(since, i - TRAIL_FRAMES) : i + 1]
        trail.set_segments([path[k : k + 2] for k in range(len(path) - 1)])
        trail.set_linewidths([1 + 4 * (k + 1) / len(path) for k in range(len(path) - 1)])
        trail.set_color(COLORS[now.kind] if now else BALL)
        for m, waves in pulses:
            for wave, delay in zip(waves, PULSE_WAVES):
                age = ((idx - m.frame) / game.fps / PULSE_SECONDS - delay) / (1 - delay)
                wave.set_visible(0 <= age <= 1)
                if 0 <= age <= 1:
                    wave.center = ball.center
                    near = ball.radius + 0.5  # past the ball's edge, so the first frame already shows
                    wave.set_radius(near + (PULSE_RADIUS - near) * (1 - (1 - age) ** 2))  # fast, then slow
                    wave.set_alpha(0.55 * (1 - age))
        clock_text.set_text(f"Q{int(row['period'])}  {_clock(row['game_clock'])}")
        shot_clock_text.set_text("" if pd.isna(row["shot_clock"]) else f"{math.ceil(row['shot_clock']):2d}")
        return []

    order = list(range(0, len(clip), max(1, stride)))
    order += [len(clip) - 1] * int(round(hold_seconds * fps))
    anim = animation.FuncAnimation(fig, update, frames=order, interval=1000 / fps, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer="pillow" if out.suffix.lower() == ".gif" else "ffmpeg", fps=fps)
    plt.close(fig)
    return out


def _seconds(kind: str) -> float:
    return SECONDS.get(kind, SECONDS_OTHER)


def _team_name(game: Game, team_id: str) -> str:
    name = game.team_names.get(team_id, "")
    return " ".join(w for w in name.split() if w not in SPONSORS) or "Team"


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# ---------------------------------------------------------------- the court


def draw_court(ax, court: CourtGeometry, lw: float = 1.4) -> None:
    """Court lines in the data's own frame: feet, origin at center court."""
    from matplotlib.patches import Arc, Circle, Rectangle

    half_l, half_w = court.length / 2, court.width / 2
    ax.add_patch(Rectangle((-half_l, -half_w), court.length, court.width, lw=0, fc=FLOOR, zorder=0))
    ax.add_patch(Rectangle((-half_l, -half_w), court.length, court.width, lw=lw, ec=LINE, fc="none", zorder=1))
    ax.plot([0, 0], [-half_w, half_w], color=LINE, lw=lw)
    ax.add_patch(Circle((0, 0), CIRCLE, lw=lw, ec=LINE, fc="none"))
    for side in (-1, 1):
        hoop_x, baseline = side * court.hoop_x, side * half_l
        key_x = baseline if side < 0 else baseline - court.key_length
        ax.add_patch(
            Rectangle(
                (key_x, -court.key_width / 2), court.key_length, court.key_width, lw=lw, ec=LINE, fc=PAINT, zorder=0.5
            )
        )
        ft_x = baseline - side * court.key_length
        toward = 0 if side < 0 else 180
        ax.add_patch(Arc((ft_x, 0), 2 * CIRCLE, 2 * CIRCLE, theta1=toward - 90, theta2=toward + 90, lw=lw, ec=LINE))
        ax.add_patch(
            Arc((ft_x, 0), 2 * CIRCLE, 2 * CIRCLE, theta1=toward + 90, theta2=toward + 270, lw=lw, ec=LINE, ls="--")
        )
        ax.add_patch(Circle((hoop_x, 0), 0.75, lw=lw, ec=LINE, fc="none"))
        ax.plot([hoop_x - side * 1.25] * 2, [-3, 3], color=LINE, lw=lw + 0.6)
        ax.add_patch(
            Arc((hoop_x, 0), 2 * NO_CHARGE, 2 * NO_CHARGE, theta1=toward - 90, theta2=toward + 90, lw=lw, ec=LINE)
        )
        dx = math.sqrt(max(court.three_point_radius**2 - court.three_point_corner_y**2, 0.0))
        corner_x = hoop_x - side * dx
        for sign in (-1, 1):
            ax.plot([baseline, corner_x], [sign * court.three_point_corner_y] * 2, color=LINE, lw=lw)
        angle = math.degrees(math.atan2(court.three_point_corner_y, dx))
        ax.add_patch(
            Arc(
                (hoop_x, 0),
                2 * court.three_point_radius,
                2 * court.three_point_radius,
                theta1=toward - angle,
                theta2=toward + angle,
                lw=lw,
                ec=LINE,
            )
        )
    ax.set_xlim(-half_l - 1.5, half_l + 1.5)
    ax.set_ylim(-half_w - 1.5, half_w + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
