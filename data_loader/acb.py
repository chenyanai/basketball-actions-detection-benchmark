"""The official Liga ACB play-by-play: download it, and turn it into the benchmark's feed.

This is a human scorer's record, written courtside and independent of any tracking system, so a
detector reading it learns nothing about the frame-level ground truth. That is why the benchmark
uses it. The clock is the scoreboard's, in whole seconds, and it lags the event by a second or two.

ACB does not license redistribution (acb.com/es/liga/aviso-legal allows private use only), so the
data is never stored in this repository: ``python runner.py fetch-acb`` downloads it to ``data/``,
which git ignores, and everyone fetches their own copy. It needs an API key, which belongs to ACB
and is not shipped here; read the X-APIKEY request header on a live.acb.com match page and pass it
with --api-key or $ACB_API_KEY.

The benchmark's feed is this scoresheet, plus the missed shots that drew a foul, which FIBA scoring
leaves out (``benchmark_feed``). Players are linked by name within a team (ACB licence id ->
SkillCorner player id), and feed rows are linked to SkillCorner's annotations for the matching
answers, which only the evaluator reads.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from data_loader.game import Game
from data_loader.play_by_play import COLUMNS, build_feed, feed_answers

API = "https://api2.acb.com/api/matchdata"
# sha256 of each raw file's canonical JSON (see _digest): everyone scores the same feed, and a change at ACB shows.
MANIFEST = {
    "pbp_104471.json": "3ff078605952eca171e094f862fa289304cc6af6fefc100cd1d6eb5ac39eea20",
    "pbp_104479.json": "a4dd0a85919a13b302a8066c4c52affde91a063fb836cf27642e961c9ca873b9",
    "pbp_104540.json": "2a790fbb97e3c9e59ff18927c2d1c6944f60f7585e22e1e59f33f665df2ce8d7",
    "pbp_104542.json": "f4be0e77e58499431cdb22ffa45c5c5f5827c912db740f0723dba37772f648c0",
    "pbp_104615.json": "9fe7c55fdd388cc186ae6fe59015f404b7ae987e7bed9d808f2e422e2899f4d4",
    "pbp_104658.json": "db870a04d779e08afbe9aafe0f5ea5e61da62ff2fbd39970b1e29ee03065894b",
    "pbp_104689.json": "8a4cdb0c58d516fac3f3e9229c7ea4b16423f02bd17c123f7108b1ed5e635795",
    "pbp_104701.json": "2468f3fd1ea8a49ce9e29241ec78bf4af5c176ce9dfbcc86cefb0de0bfe7be28",
    "pbp_104719.json": "e45edc396b39c3e80cf1b561011c875c8f83c69be22f76a65a7562420facd2ea",
    "pbp_104761.json": "fd6c22b9588228be7640013c5b37687cc4360f169ad5db84812b6b08823c434e",
}

#: SkillCorner game id -> ACB match id, for the ten games SkillCorner published tracking for.
MATCH_IDS = {
    "114243": 104471,
    "114234": 104479,
    "114169": 104540,
    "114099": 104615,
    "114086": 104542,
    "178442": 104658,
    "179612": 104689,
    "184439": 104701,
    "188630": 104719,
    "191313": 104761,
}

# ACB play types. The names come from the enum in live.acb.com's bundle, except the steal and the
# defensive rebound, which that bundle mislabels; both were established from the running box score.
SHOTS = {93: ("made", 2, None), 94: ("made", 3, None), 100: ("made", 2, "dunk"),
         97: ("missed", 2, None), 98: ("missed", 3, None), 533: ("missed", 2, "dunk")}  # fmt: skip
FREE_THROWS = {92: "made", 96: "missed"}
REBOUNDS = {101: "offensive", 104: "defensive"}
TURNOVER, STEAL, OFFENSIVE_FOUL, FOUL_DRAWN = 106, 103, 109, 110
ASSISTS = {107, 108}
FOULS = {109, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169}
FREE_THROWS_AWARDED = {160: 1, 161: 2, 162: 3}
NO_TEAM_FOUL = {164, 169}  # a foul that does not count towards the penalty
VIOLATION_TAGS = {7, 8, 9, 10, 11, 14, 15}  # playTag on a turnover: shot clock, 8 s, backcourt...
BAD_PASS_TAG = 12
NOT_EVENTS = {599, 600}  # the starting line-up and the per-minute score marker
FOULS_TO_PENALTY = 4

LINK_ROWS, LINK_SECONDS = 4, 2.0  # an assist or a steal sits this close to the row it belongs to


# ---------------------------------------------------------------- download


def fetch_acb(out: str | Path, api_key: str, game_ids: list[str] | None = None, contact: str = "") -> list[Path]:
    """Download one raw play-by-play file per game into ``out``; files already there are kept."""
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for game_id in game_ids or list(MATCH_IDS):
        match_id = MATCH_IDS[str(game_id)]
        path = folder / f"pbp_{match_id}.json"
        if not path.exists():
            path.write_bytes(_get(f"PlayByPlay/play-by-play?matchId={match_id}", api_key, contact))
        written.append(path)
    return written


def check_acb(folder: str | Path, game_ids: list[str] | None = None) -> list[str]:
    """Which downloaded files differ from the manifest, so everyone knows they scored the same feed.

    Only the requested games are checked (all ten by default)."""
    wanted = {f"pbp_{MATCH_IDS[str(g)]}.json" for g in (game_ids or MATCH_IDS)}
    differs = []
    for name, digest in MANIFEST.items():
        if name not in wanted:
            continue
        path = Path(folder) / name
        if not path.exists() or _digest(path.read_bytes()) != digest:
            differs.append(name)
    return differs


def _digest(raw: bytes) -> str:
    """Hash the canonical JSON, so reformatting by the server is not a change of content."""
    canonical = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_acb(folder: str | Path, game_id: str) -> dict:
    """The raw play-by-play of one game, as ACB serves it."""
    path = Path(folder) / f"pbp_{MATCH_IDS[str(game_id)]}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python runner.py fetch-acb` (it needs an ACB API key)")
    return json.loads(path.read_text())


def _get(path: str, api_key: str, contact: str) -> bytes:
    agent = (
        f"basketball-actions-detection-benchmark (+{contact})" if contact else "basketball-actions-detection-benchmark"
    )
    request = urllib.request.Request(  # noqa: S310 - the host is fixed above
        f"{API}/{path}",
        headers={"X-APIKEY": api_key, "Referer": "https://live.acb.com/", "User-Agent": agent},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


# ---------------------------------------------------------------- the feed


def build_acb_feed(raw: dict, game: Game) -> pd.DataFrame:
    """The scorer's record as the benchmark's feed: one row per shot, free throw, rebound, turnover and foul."""
    player_of = player_map(raw, game)
    plays = [p for p in raw["plays"] if p["playType"] not in NOT_EVENTS]
    rows, team_fouls = [], {}

    for i, play in enumerate(plays):
        kind = play["playType"]
        row = {
            "period": int(play["quarter"]),
            "clock": play["minute"] * 60 + play["second"],
            "team_id": game.home_team_id if play["local"] else game.away_team_id,
            "player_id": player_of.get(play["playerLicenseId"]),
            "player2_id": None,
            "subtype": None,
            "outcome": None,
            "points": 0,
            "score_home": int(play["scoreHome"]),
            "score_away": int(play["scoreAway"]),
        }
        if kind in SHOTS:
            outcome, points, subtype = SHOTS[kind]
            assist = _nearby(plays, i, ASSISTS, same_team=True) if outcome == "made" else None
            passer = None if assist is None else player_of.get(assist["playerLicenseId"])
            rows.append(
                row
                | {"event_type": "shot", "outcome": outcome, "points": points, "subtype": subtype, "player2_id": passer}
            )
        elif kind in FREE_THROWS:
            rows.append(row | {"event_type": "free_throw", "outcome": FREE_THROWS[kind], "points": 1})
        elif kind in REBOUNDS:
            rows.append(row | {"event_type": "rebound", "subtype": REBOUNDS[kind]})
        elif kind == TURNOVER:
            stealer = _nearby(plays, i, {STEAL}, same_team=False)
            rows.append(
                row
                | {
                    "event_type": "turnover",
                    "subtype": _turnover_subtype(plays, i, play.get("playTag")),
                    "player2_id": None if stealer is None else player_of.get(stealer["playerLicenseId"]),
                }
            )
        elif kind in FOULS:
            fouled = _nearby(plays, i, {FOUL_DRAWN}, same_team=False)
            rows.append(
                row
                | {
                    "event_type": "foul",
                    "subtype": _foul_subtype(kind, team_fouls, play),
                    "player2_id": None if fouled is None else player_of.get(fouled["playerLicenseId"]),
                }
            )
            if kind not in NO_TEAM_FOUL:
                key = (min(int(play["quarter"]), 4), play["local"])
                team_fouls[key] = team_fouls.get(key, 0) + 1

    feed = pd.DataFrame(rows)
    _number_free_throws(feed)
    return feed.reset_index(drop=True).rename_axis("event_id").reset_index()[COLUMNS]


def _turnover_subtype(plays: list[dict], i: int, tag: int | None) -> str:
    if _nearby(plays, i, {OFFENSIVE_FOUL}, same_team=True) is not None:
        return "offensive_foul"
    if tag == BAD_PASS_TAG:
        return "bad_pass"
    return "violation" if tag in VIOLATION_TAGS else "lost_ball"


def _foul_subtype(kind: int, team_fouls: dict, play: dict) -> str:
    """Free throws in the penalty come from the bonus, not from a shooting foul."""
    if kind == OFFENSIVE_FOUL:
        return "offensive"
    awarded = FREE_THROWS_AWARDED.get(kind)
    in_penalty = team_fouls.get((min(int(play["quarter"]), 4), play["local"]), 0) >= FOULS_TO_PENALTY
    if awarded == 3 or (awarded and not in_penalty):
        return "shooting"
    return "personal"


def _nearby(plays: list[dict], i: int, kinds: set[int], same_team: bool) -> dict | None:
    """The first row of these kinds beside row ``i``: ACB writes an assist or a steal as its own row."""
    play = plays[i]
    for other in plays[max(0, i - LINK_ROWS) : i + LINK_ROWS + 1]:
        if other is play or other["playType"] not in kinds or other["quarter"] != play["quarter"]:
            continue
        gap = abs(_clock(other) - _clock(play))
        if gap <= LINK_SECONDS and (other["local"] == play["local"]) == same_team:
            return other
    return None


def _number_free_throws(feed: pd.DataFrame) -> None:
    """ACB does not number them, so consecutive attempts by one shooter become "1 of 2"."""
    group, previous = [], None
    for i in list(feed.index[feed["event_type"] == "free_throw"]) + [None]:
        same = (
            i is not None
            and previous is not None
            and (feed.at[i, "player_id"], feed.at[i, "period"], feed.at[i, "clock"])
            == (feed.at[previous, "player_id"], feed.at[previous, "period"], feed.at[previous, "clock"])
        )
        if same:
            group.append(i)
        else:
            for k, j in enumerate(group, start=1):
                feed.at[j, "subtype"] = f"{k} of {len(group)}"
            group = [] if i is None else [i]
        previous = i


def _clock(play: dict) -> float:
    return play["minute"] * 60 + play["second"]


# ---------------------------------------------------------------- the benchmark's feed


def benchmark_feed(raw: dict, game: Game, events: dict, actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The feed the benchmark scores on, and the true frame of each of its rows (for the matching task only).

    The feed is the ACB scoresheet, with one addition: a missed shot that drew a foul. FIBA scoring does not
    count it as a field-goal attempt, so the scoresheet records only the foul and its free throws, exactly as
    it records free throws awarded for the bonus, and no scorer's record can tell the two apart. SkillCorner's
    annotations say which fouls came on a shot; for each, a missed shot row goes right before the foul's ACB
    row, on that row's clock. Everything else in the row comes from ACB: the shooter is the fouled player, the
    team is the fouled team, the points are the number of free throws. The only fact taken from SkillCorner is
    that the shot happened.
    """
    acb = build_acb_feed(raw, game)
    events_feed = build_feed(events, game.home_team_id, game.away_team_id)
    exact = feed_answers(events, actions).set_index("event_id")
    missed = {int(s["startFrame"]) for s in events.get("shots", []) if s.get("fouled") and not s.get("outcome")}
    pair = _pairs(events_feed, acb)

    before: dict[
        int, list[tuple[dict, int]]
    ] = {}  # ACB row index -> shot rows to insert before it, with their true frame
    for i in events_feed.index[(events_feed["event_type"] == "shot") & (events_feed["outcome"] == "missed")]:
        frame = exact.at[events_feed.at[i, "event_id"], "frame_idx"]
        if pd.isna(frame) or int(frame) not in missed:
            continue
        j = _acb_foul_of(i, events_feed, acb, pair)
        if j is not None:
            before.setdefault(j, []).append((_fouled_shot(acb, j, events_feed.loc[i]), int(frame)))

    rows, added = [], {}
    for j in acb.index:
        for row, frame in before.get(j, []):
            added[len(rows)] = frame
            rows.append(row)
        rows.append(acb.loc[j].to_dict())
    feed = pd.DataFrame(rows, columns=COLUMNS).assign(event_id=range(len(rows)))

    truth = _scored(actions, game)
    is_shot = truth["action_type"] == "shot"
    linked = _link(feed.drop(index=list(added)), truth[~(is_shot & truth["frame_idx"].isin(missed))])
    shots = truth[is_shot].drop_duplicates("frame_idx").set_index("frame_idx")
    extra = pd.DataFrame(
        [
            {"event_id": i, "event_type": "shot", "frame_idx": f, "x": shots.at[f, "x"], "y": shots.at[f, "y"]}
            for i, f in added.items()
            if f in shots.index
        ],
        columns=ANSWER_COLUMNS,
    )
    answers = pd.concat([linked, extra], ignore_index=True).sort_values("event_id", ignore_index=True)
    return feed, answers


def _acb_foul_of(i: int, events_feed: pd.DataFrame, acb: pd.DataFrame, pair: dict) -> int | None:
    """The ACB row of the foul on SkillCorner's shot ``i``.

    First the ACB partner of SkillCorner's own foul on the shot: a foul by the other team within three rows
    either side (SkillCorner may list it before the shot), preferring one on the shooter. Else the ACB foul by
    the other team between the ACB partners of the nearest aligned rows around the shot, again preferring one
    on the shooter.
    """
    shooter, team, period = events_feed.at[i, "player_id"], events_feed.at[i, "team_id"], events_feed.at[i, "period"]
    near = [
        k
        for k in range(max(0, i - 3), min(i + 4, len(events_feed)))
        if k != i
        and events_feed.at[k, "period"] == period
        and events_feed.at[k, "event_type"] == "foul"
        and events_feed.at[k, "team_id"] != team
    ]
    near.sort(key=lambda k: (events_feed.at[k, "player2_id"] != shooter, abs(k - i)))
    for k in near:
        if k in pair:
            return pair[k]
    lo = max((pair[k] for k in pair if k < i and events_feed.at[k, "period"] == period), default=-1)
    hi = min((pair[k] for k in pair if k > i and events_feed.at[k, "period"] == period), default=len(acb))
    fouls = acb[(acb.index > lo) & (acb.index < hi) & (acb["event_type"] == "foul") & (acb["team_id"] != team)]
    if not len(fouls):
        return None
    on_shooter = fouls[fouls["player2_id"] == shooter]
    return int((on_shooter if len(on_shooter) else fouls).index[0])


def _fouled_shot(acb: pd.DataFrame, j: int, shot: pd.Series) -> dict:
    """The missed shot, written from the foul's ACB row; SkillCorner's own row fills only what ACB leaves blank."""
    foul = acb.loc[j]
    attempts = acb[(acb.index > j) & (acb["event_type"] == "free_throw") & (acb["clock"] == foul["clock"])]
    total = str(attempts["subtype"].iloc[0]).split(" of ")[-1] if len(attempts) else ""
    points = int(total) if total in ("2", "3") else int(shot["points"])
    shooter = foul["player2_id"] if not pd.isna(foul["player2_id"]) else shot["player_id"]
    team = (
        next(t for t in acb["team_id"].unique() if t != foul["team_id"])
        if acb["team_id"].nunique() == 2
        else shot["team_id"]
    )
    return {
        "period": foul["period"],
        "clock": foul["clock"],
        "event_type": "shot",
        "subtype": None,
        "points": points,
        "team_id": team,
        "player_id": shooter,
        "player2_id": None,
        "outcome": "missed",
        "score_home": foul["score_home"],
        "score_away": foul["score_away"],
    }


def _pairs(events_feed: pd.DataFrame, acb: pd.DataFrame) -> dict[int, int]:
    """SkillCorner row -> ACB row, by aligning the two records in order within each period.

    Rows are compared as (event type, team, player) only, never by time; the longest identical runs are
    paired first, then the stretches on either side, like a text diff. A row either record lacks is skipped
    without shifting the rest.
    """
    out = {}
    for period in sorted(set(events_feed["period"])):
        a = events_feed.index[events_feed["period"] == period]
        b = acb.index[acb["period"] == period]
        matcher = SequenceMatcher(None, _keys(events_feed.loc[a]), _keys(acb.loc[b]), autojunk=False)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                out[int(a[block.a + k])] = int(b[block.b + k])
    return out


def _keys(rows: pd.DataFrame) -> list[tuple]:
    return [(r.event_type, str(r.team_id), "" if pd.isna(r.player_id) else str(r.player_id)) for r in rows.itertuples()]


# ---------------------------------------------------------------- matching answers

ANSWER_COLUMNS = ["event_id", "event_type", "frame_idx", "x", "y"]
ANSWER_SECONDS = 8.0  # how far the scorer's clock may lag the event it describes


def _scored(actions: pd.DataFrame, game: Game) -> pd.DataFrame:
    clock = game.frames.set_index("frame_idx")["game_clock"]
    truth = actions[actions["action_type"].isin(("shot", "rebound", "turnover", "foul"))].copy()
    truth["clock"] = truth["frame_idx"].map(clock)
    return truth[truth["clock"].notna()]


def _link(feed: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Each scorer row to the SkillCorner annotation it describes, for the matching answers.

    The two share no identifiers, so a row goes to the nearest unused annotation of the same type, team,
    player and outcome within 8 s. The scoreboard can drift (up to 9 s in game 114099), so each period is
    linked twice, the second time with that period's median offset removed. A row that cannot be linked is
    left out: the two records disagree about it, so there is nothing to score it against.
    """
    first = _link_pass(feed, truth, {})
    offsets = first.groupby("period")["offset"].median().to_dict()
    return _link_pass(feed, truth, offsets)[ANSWER_COLUMNS]


def _link_pass(feed: pd.DataFrame, truth: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    used, rows = set(), []
    for r in feed[feed["event_type"] != "free_throw"].itertuples(index=False):
        shifted = r.clock + offsets.get(r.period, 0.0)
        c = truth[
            (truth["action_type"] == r.event_type)
            & (truth["period"] == r.period)
            & (truth["team_id"] == str(r.team_id))
            & ~truth.index.isin(used)
            & ((truth["clock"] - shifted).abs() <= ANSWER_SECONDS)
        ]
        if r.event_type == "shot":
            c = c[c["outcome"] == r.outcome]
        elif r.event_type == "rebound":
            c = c[c["subtype"] == r.subtype]
        if not pd.isna(r.player_id):
            c = c[c["player_id"] == str(r.player_id)]
        if len(c):
            j = (c["clock"] - (shifted + 0.5)).abs().idxmin()  # the clock is floored: the event is within the second
            used.add(j)
            rows.append(
                {
                    "event_id": r.event_id,
                    "event_type": r.event_type,
                    "frame_idx": int(truth.at[j, "frame_idx"]),
                    "x": truth.at[j, "x"],
                    "y": truth.at[j, "y"],
                    "period": r.period,
                    "offset": truth.at[j, "clock"] - r.clock,
                }
            )
    return pd.DataFrame(rows, columns=ANSWER_COLUMNS + ["period", "offset"])


# ---------------------------------------------------------------- players


def player_map(raw: dict, game: Game, threshold: float = 0.72) -> dict[int, str]:
    """ACB licence id -> SkillCorner player id, matched on name within a team, never across teams.

    Raises when a player cannot be matched: a hole here would put events on the wrong player.
    """
    rosters = {
        True: {r.player_id: r.name for r in game.roster.itertuples() if r.team_id == game.home_team_id},
        False: {r.player_id: r.name for r in game.roster.itertuples() if r.team_id == game.away_team_id},
    }
    acb: dict[bool, dict[int, str]] = {True: {}, False: {}}
    for play in raw["plays"]:
        if play["playerLicenseId"]:
            acb[bool(play["local"])][play["playerLicenseId"]] = play["playerName"]

    mapping, unmatched = {}, []
    for home, players in acb.items():
        pairs = sorted(
            (
                (_similarity(acb_name, name), licence, player_id)
                for licence, acb_name in players.items()
                for player_id, name in rosters[home].items()
            ),
            key=lambda pair: -pair[0],
        )
        taken_licence, taken_player = set(), set()
        for score, licence, player_id in pairs:  # most confident first, so a near tie cannot steal a slot
            if score >= threshold and licence not in taken_licence and player_id not in taken_player:
                mapping[licence] = player_id
                taken_licence.add(licence)
                taken_player.add(player_id)
        unmatched += [
            f"{name} (licence {licence})" for licence, name in players.items() if licence not in taken_licence
        ]
    if unmatched:
        raise ValueError("these ACB players are not on the SkillCorner roster: " + "; ".join(unmatched))
    return mapping


def _similarity(a: str, b: str) -> float:
    first, second = _tokens(a), _tokens(b)
    if not first or not second:
        return 0.0
    score = SequenceMatcher(None, " ".join(first), " ".join(second)).ratio()
    if first[-1] == second[-1]:  # same surname
        score = max(score, 0.90)
    if set(first) & set(second):  # any shared name token
        score = max(score, 0.80)
    return score


def _tokens(name: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", plain).split()
