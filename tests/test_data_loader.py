import urllib.request

import numpy as np
import pytest

import schema
from data_loader import load_events
from data_loader.skillcorner import available, fetch


def test_game_shape(game):
    assert game.game_id == "114099" and {game.home_team_id, game.away_team_id} == {"1315", "1316"}
    assert len(game.frames) == 9856 and len(game.players) == 10 * len(game.frames)
    assert game.frames["frame_idx"].is_monotonic_increasing
    assert game.team_of("31700") == "1315" and game.jersey_of("31700") == "12"
    assert game.absolute_time(1, 600.0) == 0.0 and game.absolute_time(2, 600.0) == 600.0


def test_positions_view(game):
    frame_ids, xy, player_ids = game.positions()
    assert xy.shape == (len(game.frames), len(player_ids), 2)
    assert np.isfinite(xy).any(axis=(1, 2)).all()


def test_ground_truth_tables(truth):
    assert len(truth.possessions) == 22
    assert truth.ball_handler["player_id"].notna().sum() == 6902
    counts = truth.actions["action_type"].value_counts().to_dict()
    assert counts == {"pass": 62, "shot": 18, "rebound": 7, "foul": 6, "turnover": 6}
    assert truth.actions.loc[truth.actions["action_type"] == "rebound", "attributable"].sum() == 5
    schema.validate(schema.ACTIONS, truth.actions)


def test_event_locations_are_in_the_tracking_frame(game, truth):
    """A shot's location must be where the shooter stands on that frame."""
    shots = truth.actions[truth.actions["action_type"] == "shot"].dropna(subset=["x"])
    pos = game.players.set_index(["frame_idx", "player_id"])
    errors = [
        np.hypot(*(pos.loc[(s.frame_idx, s.player_id), ["x", "y"]].to_numpy() - (s.x, s.y)))
        for s in shots.itertuples()
        if (s.frame_idx, s.player_id) in pos.index
    ]
    assert errors and max(errors) < 2.0


def test_feed_lists_every_annotated_event(match_dir, game, feed):
    """Including the missed shot that drew a foul, which a scoresheet leaves out."""
    events = load_events(match_dir)
    assert sum(1 for s in events["shots"] if s["fouled"] and not s["outcome"]) == 1
    assert (feed["event_type"] == "shot").sum() == len(events["shots"])
    assert feed["clock"].dtype.kind == "i"
    for _, period in feed.groupby("period"):
        assert (period["clock"].diff().dropna() <= 0).all()
    assert (
        feed["score_home"].iloc[-1]
        == feed.loc[feed["team_id"] == game.home_team_id]["points"][feed["outcome"] == "made"].sum()
    )
    free_throws = feed[feed["event_type"] == "free_throw"]
    assert free_throws["subtype"].tolist() == ["1 of 2", "2 of 2"]


def test_rebound_side_comes_from_the_shot_not_the_flag():
    """SkillCorner leaves `defensive` unset on some unsecured team rebounds; the linked shot decides."""
    from data_loader.skillcorner import rebound_subtype, shot_teams

    events = {
        "shots": [{"id": "s1", "offTeamId": "A"}],
        "free_throws": [{"id": "f1", "offTeamId": "B"}],
    }
    teams = shot_teams(events)
    unsecured_by_defense = {"shotId": "s1", "teamId": "B", "rebounded": False, "defensive": False}
    assert rebound_subtype(unsecured_by_defense, teams) == "defensive"
    assert rebound_subtype({"shotId": "s1", "teamId": "A", "rebounded": True, "defensive": False}, teams) == "offensive"
    assert rebound_subtype({"shotId": "f1", "teamId": "A", "rebounded": True, "defensive": True}, teams) == "defensive"
    # no linked shot: the flag is all there is
    assert rebound_subtype({"shotId": "missing", "teamId": "A", "defensive": True}, teams) == "defensive"
    assert rebound_subtype({"shotId": "missing", "teamId": "A", "defensive": False}, teams) == "offensive"


# ---------------------------------------------------------------- downloading, without the network


class FakeResponse:
    """What urlopen returns: read() hands out the chunks in turn, and raises the ones that are exceptions."""

    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        chunk = self.chunks.pop(0) if self.chunks else b""
        if isinstance(chunk, Exception):
            raise chunk
        return chunk


def serve(monkeypatch, *chunks):
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: FakeResponse(chunks))


def test_an_interrupted_download_is_not_kept(tmp_path, monkeypatch):
    serve(monkeypatch, b"half a file", ConnectionResetError("connection lost"))
    with pytest.raises(ConnectionResetError):
        fetch(tmp_path, ["114099"])
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []  # neither the file nor its .part

    serve(monkeypatch, b"{}")
    fetch(tmp_path, ["114099"])
    assert available(tmp_path) == ["114099"]
