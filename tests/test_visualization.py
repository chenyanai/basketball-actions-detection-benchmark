import dataclasses

import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from runner import STAGES, load_functions, run_submission  # noqa: E402
from visualization import render  # noqa: E402


def test_render_possession_clip(tmp_path, game, feed, truth):
    result = run_submission(load_functions("baselines.naive", STAGES)[1], game, feed)
    out = render(
        game, result, tmp_path / "clip.gif", possession=3, truth=truth, stride=60, fps=5, dpi=40, hold_seconds=0
    )
    from PIL import Image

    assert Image.open(out).n_frames >= 2


def test_render_frame_range(tmp_path, game, feed, truth):
    result = run_submission(load_functions("baselines.naive", STAGES)[1], game, feed)
    out = render(game, result, tmp_path / "clip.gif", frames="2000:2300", stride=100, fps=5, dpi=40, hold_seconds=0)
    assert out.exists()


def test_render_a_label_on_a_frame_the_loader_dropped(tmp_path, game, truth):
    """SkillCorner labels frames the loader can drop (no ball or no team tracked); the clip must still render."""
    game = dataclasses.replace(game, frames=game.frames[game.frames["frame_idx"] != 2100])
    label = truth.actions.iloc[[0]].assign(action_type="pass", frame_idx=2100)
    truth = dataclasses.replace(truth, actions=pd.concat([truth.actions, label], ignore_index=True))
    out = render(game, {}, tmp_path / "clip.gif", frames="2090:2110", truth=truth, fps=5, dpi=40, hold_seconds=0)
    assert out.exists()
