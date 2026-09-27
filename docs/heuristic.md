# The Heuristic Baseline

The heuristic follows the Q-Ball detection process step for step. It is built around possessions: a
play-by-play feed says *what* happened in a possession and *who* did it, and tracking finds *when*
and *where*. Its settings are in
[`baselines/heuristic/config.yaml`](../baselines/heuristic/config.yaml).

## How It Works

1. **Frames.** Detectors see live play only: one frame per game-clock reading, so a stopped clock
   collapses to a single frame. The last second before the clock restarts is kept too, because that
   is where the inbound pass is thrown (`pre_inbound_seconds`).
2. **Possessions.** The feed is read in order, tracking which team has the ball after every event:
   a made shot or a turnover gives it to the other team, a rebound to the rebounder's team, a foul
   to the team that inbounds. A change of team ends a possession, and every event is filed with the
   possession it happened in. Each boundary is then moved from the feed's whole second to the
   shot-clock reset, or the dead ball, that tracking shows near it.
3. **Ball handler.** The possession team's player within 3 ft of the ball; the previous handler
   keeps it while still in reach. Between possessions anyone can be the handler.
4. **Actions, one possession at a time.** Each possession is searched on its own frames plus 2 s on
   either side, in this order:
   - **shots**: for the feed's shooter, a high ball he last handled, a rising ball in his hands, or
     a high ball next to him; the release is traced back from there. Tips have their own rule near
     the rim.
   - **rebounds**: after the relevant shot, the first frames where the ball has peaked, come down,
     and the feed's rebounder is next to it.
   - **putbacks**: a tip the shot search missed, right after the same player's offensive rebound.
   - **turnovers**: back from the feed's time to the frame where the ball left the player's hands.
     A bad pass also gives an incomplete pass.
   - **fouls**: the feed's time, placed where the fouler stood.
   - **passes**: tracking alone, a change of handler between teammates on the possession's own
     frames.

   A detection is kept only if it falls near its own possession's time span, and an event that was
   already reported is never reported again.

Every action carries the number of the possession it was found in (the `possession` column, also in
the possessions table). To see what the feed gives each possession, who shot, rebounded, lost the
ball or fouled in it:

```python
from baselines import heuristic
from baselines.heuristic.inputs import frames_from_game
from baselines.heuristic.possessions import events_table, find_possessions

config = heuristic.default_config()
possessions = find_possessions(game, frames_from_game(game, config), feed, config)
events_table(possessions)  # possession, period, team_id, event_id, event_type, subtype, player_id, outcome
```

## Known Limitations

- Fouls are its weakest type, and the only one where reading the feed's clock alone does better
  (49.0% F1 against 65.1%). A foul keeps the feed's second and is placed at the closest frame, which
  on a stoppage lands after the whistle; widening the matching window to 2 s barely helps.
- Turnovers are placed where the ball left the player's hands, while SkillCorner marks the
  interception or the whistle, so they land early by about 0.2 s.
- The first event of each period and rebounds inside a free-throw sequence are never filed with a
  possession, so they are never detected. A turnover or foul more than 0.5 s outside its snapped
  possession boundary is dropped.
- Possession boundaries are frequently misplaced, which, together with missed passes, limits the
  complete-possession score.
- Inbound passes are missed more often than other passes, because the game clock is stopped during
  inbound plays.
- Handoff passes are detected but not labelled as handoffs.

Per-type figures are in [`results/heuristic.json`](../results/heuristic.json); `python runner.py evaluate runs/heuristic` prints them as tables.

## SportVU Parity

The same code runs on NBA SportVU data, loaded by
[`data_loader/sportvu.py`](../data_loader/sportvu.py), and
`tests/test_sportvu_parity.py` checks that it reproduces the original pipeline's possessions and
actions exactly (with `pre_inbound_seconds: 0`, which the original does not have).
