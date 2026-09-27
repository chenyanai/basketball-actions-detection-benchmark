# Matching Rules and Every Metric

How detections are matched to SkillCorner's annotations, and every number the reports contain.
The README's [Evaluation](../README.md#evaluation) section covers what is scored; this page covers how.

## Live Play

A frame is live while the game clock is running, so free throws, timeouts and the interval
preceding an inbound play are excluded from frame-level metrics. Actions are scored when they occur
within 1 s of live play, which keeps whistles and inbound passes in.

## Actions

Actions are matched blind, never by identity: same period and type, closest first, within 1 s for
instant actions or with temporal IoU of at least 0.5 for actions with a duration. The player does
not need to be correct for a match; *strict recall* is the version that also requires the right
player.

For each type the report gives precision, recall, F1, timing error (mean, median, 90th percentile,
MSE, RMSE and signed), location error, player, second-player and subtype accuracy, and recall on
subsets such as made and missed shots, fouled shots, inbound and handoff passes, and team events.
F1 is also given at 0.5 s and 2 s; a type whose F1 rises with the window is found, but late. The
leaderboard uses 1 s and lists only reports evaluated with the default settings.

## Possessions

The share of live frames assigned to the team in possession, plus interval matching (same team, IoU
of at least 0.5) with precision, recall, F1 and the error at the start and end. The leaderboard shows
the frame share and the F1.

## Ball Handler

The share of live frames that identify the player in control of the ball, or no player while the
ball is loose. The report also breaks the errors down: the wrong teammate, the wrong team, and no
handler named when someone had the ball.

## Complete Possessions

A true possession is complete when a detected possession of the same team overlaps it with IoU of
at least 0.5, and every true action in it, of the types the submission attempts, is matched by type
and player within 2 s (`--complete-window`).

## Play-by-Play Matching

A row placed within 1 s of the annotated frame is correct. Precision is the share of the rows the
matcher returns that are correct, recall the share of all scored rows (the same as "within 1 s"),
and F1 combines the two, overall and per event type. A matcher that returns every row has equal
precision, recall and F1; one that leaves out rows it is unsure of can raise its precision. Reports
also give the share within 0.5 s and 2 s, the median and 90th-percentile timing error, and the
median location error.

## Notes

Games are pooled by concatenating their tables, so an aggregate is the metric over all events, not
the average of per-game rates. SkillCorner stamps a bad-pass turnover at the interception, not at
the release. The first half of game 114099 runs up to 9 s off the official ACB clock; this does not
affect scoring, which uses SkillCorner's frame index throughout.
