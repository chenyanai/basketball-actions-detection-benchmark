# The Data

The benchmark uses two independent sources: SkillCorner's tracking and labels, and the official Liga
ACB play-by-play. This repository stores neither; both are downloaded into `data/`, which git
ignores.

## Tracking and Ground Truth

[SkillCorner's open basketball data](https://github.com/SkillCorner/opendata-basketball), under the
MIT license: 25 fps broadcast tracking (player x and y, ball x, y and z, and the clocks) for ten
Liga ACB games, and SkillCorner's Dynamic Events, which give the frame and location of every event
and serve as the ground truth.

`python runner.py fetch` downloads three files per game into `data/matches/<gameId>/`, about 360 MB
in all. It needs no git, git-lfs or account. If the download fails, clone SkillCorner's repository,
run `git lfs pull`, and pass `--data path/to/opendata-basketball/data/matches` to any command.

When loading a game, the benchmark:

- keeps only frames where the ball is tracked and both teams are on the floor, about half of them;
- turns SkillCorner's event locations, which are normalized to the attacking direction, back into
  the tracking's coordinates;
- builds the ball-handler ground truth from SkillCorner's touches;
- takes a rebound's side, offensive or defensive, from the shot it follows.

[`data_loader/skillcorner.py`](../data_loader/skillcorner.py) has the details.

## The Play-by-Play Feed

The play-by-play feed is the official Liga ACB scoresheet, from the league's match-data API, plus
one kind of added row, explained below.

**What it contains.** One row per shot, free throw, rebound, turnover and foul, with team, player
and outcome, and the scoreboard clock in whole seconds. A scorer writes it courtside, independently
of the tracking and of SkillCorner's labels, so it carries no frame-level timing.

**How accurate its clock is.** In these ten games the scorer's second typically lags the event by
one to three seconds. In one game, 114099, the scoreboard drifts by up to 9 s in the first half.

**How to get it.** This repository does not redistribute ACB data; each user downloads their own
copy. See [ACB's legal notice](https://www.acb.com/es/liga/aviso-legal) for its terms.

```bash
export ACB_API_KEY=...            # the X-APIKEY header on a live.acb.com match page
python runner.py fetch-acb        # raw ACB -> data/acb/, feed -> data/play_by_play/, matching answers -> data/answers/
```

- **The API key** belongs to ACB, not to this project, so it is not shipped. Read it from the
  X-APIKEY request header on a match page at [live.acb.com](https://live.acb.com/), with the
  browser's network tab open.
- **Pinned content.** `fetch-acb` checks every downloaded file against a hash pinned in
  [`data_loader/acb.py`](../data_loader/acb.py), so everyone scores the same feed, and a change at
  the source is visible rather than silent.
- **The matching answers** go to their own folder, never next to the feed a detector reads.

## The Added Rows: Missed Shots That Drew a Foul

Under FIBA scoring, a missed shot that drew a foul is not a field-goal attempt: the scoresheet keeps
the foul and its free throws and drops the shot. Here is one from the first quarter of game 114099,
as ACB records it:

```text
Q1 1:07  shot        made     2 pts  away  Raul Neto
Q1 0:52  foul        personal        away  Luke Fischer  -> fouled Nikola Maric
Q1 0:52  free_throw  1 of 2   missed home  Nikola Maric
Q1 0:52  free_throw  2 of 2   made   home  Nikola Maric
Q1 0:32  shot        made     2 pts  away  Luke Fischer
```

SkillCorner's labels show that Maric went up for a layup, missed, and was fouled. That shot is in
the ground truth, but nothing in the scoresheet says it happened. Its rows look exactly like two free
throws awarded for the bonus, since home was already in the penalty. No reading of the scoresheet
can tell the two apart, so a detector driven by it would never search for the shot: 111 shots
across the ten games would be unreachable.

So the feed adds each of them, just before its foul:

```text
Q1 1:07  shot        made     2 pts  away  Raul Neto
Q1 0:52  shot        missed   2 pts  home  Nikola Maric      <- added
Q1 0:52  foul        personal        away  Luke Fischer  -> fouled Nikola Maric
Q1 0:52  free_throw  1 of 2   missed home  Nikola Maric
Q1 0:52  free_throw  2 of 2   made   home  Nikola Maric
Q1 0:32  shot        made     2 pts  away  Luke Fischer
```

Everything in the added row comes from the scoresheet:

- the clock is the foul's, 0:52;
- the shooter and team are the fouled player's, Maric for home;
- the points are the number of free throws, two.

The one fact taken from SkillCorner is that a shot happened, which a scorer knows but FIBA scoring
does not record. Across all 111 added rows, the shooter, team and points written from the
scoresheet agree with SkillCorner's labels. The row says nothing only SkillCorner knows, such as the
shot type or the frame.

Two alternatives were measured and rejected:

| feed | scored events with a row | fouled shots the heuristic finds |
|---|---|---|
| the scoresheet alone | 93.7% (90.1% of shots) | 19% |
| SkillCorner's events on the scoresheet's clock | 100% | 49% |
| **the scoresheet plus the added rows** | **97.5%** (97.7% of shots) | **79%** |

SkillCorner's events on the scoresheet's clock would take identity, subtypes and every event the
scorer missed from the labels, which is information from the ground truth, and it scored no better:
its fouled shots are placed by interpolating their neighbours' clocks. With the added rows, a fouled
shot takes its foul's clock, the moment the shot was stopped.
