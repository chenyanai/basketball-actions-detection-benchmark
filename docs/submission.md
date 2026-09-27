# Submitting a Detector

A submission is a Python file, or a dotted module such as `baselines.naive`, that defines any subset
of the benchmark's functions. The runner calls them, validates what they return against
[`schema.py`](../schema.py), and writes the tables to a run folder that `evaluate` scores.

## The Three Stages

```python
def detect_possessions(game, pbp):                                    # period, start_frame, end_frame, team_id
def detect_ball_handler(game, pbp, possessions=None):                 # frame_idx, player_id (empty = nobody)
def detect_actions(game, pbp, possessions=None, ball_handler=None):   # one row per action
```

They run in this order, and each receives the tables of the stages before it.

- `game` is the tracking of one game:
  - `game.frames` has one row per frame: the clocks, the ball's x, y and z, and a `live` flag;
  - `game.players` has one row per player per frame;
  - `game.roster`, `game.court` and `game.positions()` (a dense array) are also available.
- `pbp` is the play-by-play feed, one row per event: `event_id, period, clock, event_type, subtype,
  points, team_id, player_id, player2_id, outcome, score_home, score_away`.

A function may ignore any argument it does not use.

**A stage you skip passes `None` to the stages after it, never the ground truth.** Filling it with
the ground truth would hand over the answers: SkillCorner's possessions start on the frame of each
defensive rebound, and its ball-handler table changes on the frame of each pass. To build on a
baseline's possessions or ball handler, call its stage functions, for example
`heuristic.detect_possessions(game, pbp)`.

## The Action Table

Each action is one row: `period, frame_idx, end_frame, action_type, subtype, player_id, player2_id,
team_id, x, y, outcome, confidence`.

- Frames use SkillCorner's game-wide index at 25 fps.
- Coordinates are in feet, with the origin at center court.
- `subtype` and `outcome` are optional.
- If `confidence` (between 0 and 1) varies, the report also gives average precision.

| action_type | shape | player_id | player2_id | subtype | outcome |
|---|---|---|---|---|---|
| pass | instant, at release | passer | receiver | handoff | complete / incomplete |
| shot | instant, at release | shooter | | | made / missed |
| rebound | instant | rebounder | | offensive / defensive | |
| turnover | instant | player who lost the ball | stealer | | |
| foul | instant | fouler | fouled player | shooting / personal / offensive / flagrant | |
| screen | instant | screener | teammate it frees | on_ball / off_ball | |
| drive, isolation, post | interval (`end_frame` set) | ball handler | | | |
| closeout | interval | defender | ball handler | | |

A submission can return any subset of these types. A type it does not return is reported as not
attempted, rather than scored as zero.

## Play-by-Play Matching

The second task, scored on its own:

```python
def match_pbp(game, pbp):   # event_id, frame_idx, x, y (x and y optional)
```

```bash
python runner.py match-pbp baselines.heuristic    # writes runs/heuristic-pbp/
python runner.py evaluate runs/heuristic-pbp      # writes results/heuristic-pbp.json
```

- Every shot, rebound, turnover and foul row with a known true frame is scored.
- A row you do not return counts as a miss.
- Free-throw rows are there for context and are not scored.
- The matcher does not receive the true possessions or ball handler, since their boundaries would
  reveal the answers.

The feed and SkillCorner share no identifiers, so `fetch-acb` links each row to the annotation it
describes: the same type, team, player and outcome, nearest in time. 96.1% of rows are linked; the
rest, where the two records disagree, are not scored.

## Run Folders and Other Languages

A run folder holds `run.json`, describing the run, and one sub-folder per game with the
submission's own tables as CSV. Evaluation reads only the run folder, so a model written in another
language can be scored by writing those files itself.

## The Heuristic's Settings

All of them are in [`baselines/heuristic/config.yaml`](../baselines/heuristic/config.yaml). Edit the
file directly (`git checkout` restores it), or write a yaml with only the settings you want to
change and pass `heuristic.HeuristicConfig.from_yaml("mine.yaml")` as `config=`.

## Examples

| Example | What it shows |
|---|---|
| [`template.py`](../examples/template.py) | all three stages, returning empty tables |
| [`extend_heuristic.py`](../examples/extend_heuristic.py) | the heuristic with your own shots, plus two types no baseline detects yet: screens (instant) and drives (interval) |
