"""The official ACB feed: the builder, on a handful of plays written by hand.

The real data is not redistributable, so these use the shape ACB serves, not a recorded game.
"""

import pandas as pd
import pytest

from data_loader.acb import benchmark_feed, build_acb_feed, player_map


def play(play_type, minute, second, local=True, licence=None, name="", order=0, tag=None):
    return {
        "playType": play_type,
        "quarter": 1,
        "minute": minute,
        "second": second,
        "local": local,
        "playerLicenseId": licence,
        "playerName": name,
        "scoreHome": 0,
        "scoreAway": 0,
        "order": order,
        "playTag": tag,
    }


@pytest.fixture
def raw():
    """A made three with an assist, a bad-pass turnover with a steal, a shooting foul and its two free throws."""
    return {
        "plays": [
            play(599, 10, 0),  # the starting line-up: not an event
            play(94, 9, 40, licence=1, name="Ander Gomez"),
            play(108, 9, 40, licence=2, name="Luka Peric"),
            play(106, 9, 20, local=False, licence=3, name="Marc Dubois", tag=12),
            play(103, 9, 20, licence=1, name="Ander Gomez"),
            play(161, 9, 0, local=False, licence=3, name="Marc Dubois"),
            play(110, 9, 0, licence=2, name="Luka Peric"),
            play(92, 9, 0, licence=2, name="Luka Peric"),
            play(96, 9, 0, licence=2, name="Luka Peric"),
            play(104, 8, 58, local=False, licence=4, name="Tom Novak"),
            play(600, 8, 0),  # the per-minute score marker: not an event
        ]
    }


@pytest.fixture
def game(fake_game):
    return fake_game(
        home_team_id="H",
        away_team_id="A",
        roster=[
            ("p1", "H", "Ander Gomez"),
            ("p2", "H", "Luka Peric"),
            ("p3", "A", "Marc Dubois"),
            ("p4", "A", "Tom Novak"),
        ],
    )


def test_player_map_matches_names_within_a_team(raw, game):
    assert player_map(raw, game) == {1: "p1", 2: "p2", 3: "p3", 4: "p4"}


def test_player_map_refuses_a_roster_that_is_not_this_game(raw, fake_game):
    other = fake_game(home_team_id="H", away_team_id="A", roster=[("p1", "H", "Someone Else")])
    with pytest.raises(ValueError, match="not on the SkillCorner roster"):
        player_map(raw, other)


def test_feed_rows_carry_what_the_scorer_recorded(raw, game):
    feed = build_acb_feed(raw, game)
    assert list(feed["event_type"]) == ["shot", "turnover", "foul", "free_throw", "free_throw", "rebound"]

    shot = feed.iloc[0]
    assert (shot["clock"], shot["points"], shot["outcome"]) == (580, 3, "made")
    assert (shot["player_id"], shot["player2_id"]) == ("p1", "p2")  # the assist becomes the passer

    turnover = feed.iloc[1]
    assert (turnover["subtype"], turnover["player_id"], turnover["player2_id"]) == ("bad_pass", "p3", "p1")

    foul = feed.iloc[2]
    assert (foul["subtype"], foul["player_id"], foul["player2_id"]) == ("shooting", "p3", "p2")
    assert list(feed.loc[feed["event_type"] == "free_throw", "subtype"]) == ["1 of 2", "2 of 2"]
    assert feed.iloc[5]["subtype"] == "defensive"
    assert list(feed["event_id"]) == [0, 1, 2, 3, 4, 5]


def test_a_fouled_miss_is_added_before_its_foul(raw, game):
    """ACB records the foul and two free throws; SkillCorner says a shot came first. The added row says only that,
    on the foul's clock: shooter, team and points come from the ACB foul, and it is scored exactly."""
    events = {
        "shots": [
            _sk_shot("s1", 450, "H", "p1", made=True),
            _sk_shot("s2", 1500, "H", "p2", made=False, fouled=True),
        ],
        "fouls": [
            {
                "id": "f1",
                "period": 1,
                "frame": 1510,
                "gameClock": 539.6,
                "foulerTeamId": "A",
                "foulerId": "p3",
                "fouledId": "p2",
                "shotId": "s2",
                "foulType": "shooting",
            },
        ],
        "free_throws": [
            {
                "id": "t1",
                "foulId": "f1",
                "period": 1,
                "frame": 2000,
                "gameClock": 539.6,
                "offTeamId": "H",
                "shooterId": "p2",
                "outcome": True,
            },
            {
                "id": "t2",
                "foulId": "f1",
                "period": 1,
                "frame": 2100,
                "gameClock": 539.6,
                "offTeamId": "H",
                "shooterId": "p2",
                "outcome": False,
            },
        ],
        "rebounds": [
            {
                "id": "r1",
                "shotId": "t2",
                "period": 1,
                "frame": 2200,
                "gameClock": 538.0,
                "teamId": "A",
                "rebounderId": "p4",
                "rebounded": True,
                "defensive": True,
            },
        ],
    }
    actions = pd.DataFrame(
        [
            {
                "period": 1,
                "frame_idx": 450,
                "action_type": "shot",
                "player_id": "p1",
                "team_id": "H",
                "outcome": "made",
                "subtype": None,
                "x": 1.0,
                "y": 2.0,
            },
            {
                "period": 1,
                "frame_idx": 1500,
                "action_type": "shot",
                "player_id": "p2",
                "team_id": "H",
                "outcome": "missed",
                "subtype": None,
                "x": 3.0,
                "y": 4.0,
            },
        ]
    )
    feed, answers = benchmark_feed(raw, game, events, actions)
    assert list(feed["event_type"]) == ["shot", "turnover", "shot", "foul", "free_throw", "free_throw", "rebound"]
    added = feed.iloc[2]
    assert (added["clock"], added["outcome"], added["player_id"], added["team_id"], added["points"]) == (
        540,
        "missed",
        "p2",
        "H",
        2,
    )
    assert pd.isna(added["subtype"])  # nothing from SkillCorner beyond the shot itself
    assert list(feed["event_id"]) == list(range(7))
    assert dict(zip(answers["event_id"], answers["frame_idx"])) == {0: 450, 2: 1500}


def _sk_shot(shot_id, frame, team, shooter, made, fouled=False):
    return {
        "id": shot_id,
        "period": 1,
        "startFrame": frame,
        "endGameClock": 600.0 - frame / 25,
        "offTeamId": team,
        "shooterId": shooter,
        "outcome": made,
        "fouled": fouled,
        "three": False,
        "assisted": False,
    }
