import pandas as pd
import pytest

import schema


def test_empty_tables_validate():
    for kind in (schema.POSSESSIONS, schema.BALL_HANDLER, schema.ACTIONS):
        assert len(schema.validate(kind, schema.empty(kind))) == 0


def test_ids_become_strings_and_missing_stays_missing():
    table = pd.DataFrame({"frame_idx": [1, 2, 3], "player_id": [59188, None, 28336.0]})
    out = schema.validate(schema.BALL_HANDLER, table)
    assert out["player_id"].tolist()[0] == "59188" and out["player_id"].tolist()[2] == "28336"
    assert pd.isna(out["player_id"].iloc[1])


def test_actions_defaults_and_column_order():
    table = pd.DataFrame([{"period": 1, "frame_idx": 10, "action_type": "shot", "player_id": "1", "team_id": "9"}])
    out = schema.validate(schema.ACTIONS, table)
    assert list(out.columns) == list(schema.COLUMNS[schema.ACTIONS])
    assert out["confidence"].iloc[0] == 1.0 and pd.isna(out["end_frame"].iloc[0])


def test_actions_reject_bad_rows():
    base = {"period": 1, "frame_idx": 10, "team_id": "9", "player_id": "1"}
    with pytest.raises(schema.SchemaError, match="unknown action_type"):
        schema.validate(schema.ACTIONS, pd.DataFrame([{**base, "action_type": "dunk_contest"}]))
    with pytest.raises(schema.SchemaError, match="interval action without end_frame"):
        schema.validate(schema.ACTIONS, pd.DataFrame([{**base, "action_type": "drive"}]))
    with pytest.raises(schema.SchemaError, match="instant action with end_frame"):
        schema.validate(schema.ACTIONS, pd.DataFrame([{**base, "action_type": "shot", "end_frame": 20}]))
    with pytest.raises(schema.SchemaError, match="confidence"):
        schema.validate(schema.ACTIONS, pd.DataFrame([{**base, "action_type": "shot", "confidence": 1.5}]))
    with pytest.raises(schema.SchemaError, match="missing required column"):
        schema.validate(schema.ACTIONS, pd.DataFrame([{"period": 1, "frame_idx": 1}]))


def test_possessions_reject_backwards_interval():
    with pytest.raises(schema.SchemaError, match="row 0"):
        schema.validate(
            schema.POSSESSIONS, pd.DataFrame([{"period": 1, "start_frame": 10, "end_frame": 5, "team_id": "9"}])
        )


def test_one_ball_handler_per_frame_and_one_match_per_feed_row():
    """A repeated frame or feed row would be counted twice by the evaluator."""
    with pytest.raises(schema.SchemaError, match="duplicate frame_idx at row 1"):
        schema.validate(schema.BALL_HANDLER, pd.DataFrame({"frame_idx": [7, 7], "player_id": ["1", "2"]}))
    with pytest.raises(schema.SchemaError, match="duplicate event_id at row 2"):
        schema.validate(schema.PBP_MATCHES, pd.DataFrame({"event_id": [0, 1, 0], "frame_idx": [10, 20, 30]}))


def test_concat_keeps_extra_columns():
    a = pd.DataFrame(
        [{"period": 1, "frame_idx": 1, "action_type": "pass", "player_id": "1", "team_id": "9", "note": "x"}]
    )
    out = schema.concat(schema.ACTIONS, [a, schema.empty(schema.ACTIONS)])
    assert len(out) == 1 and "note" in out.columns


def test_vocabulary_shapes():
    assert schema.ACTION_TYPES["drive"].shape == schema.INTERVAL
    assert schema.ACTION_TYPES["pass"].player2 == "receiver"
    assert all(spec.shape in (schema.INSTANT, schema.INTERVAL) for spec in schema.ACTION_TYPES.values())
