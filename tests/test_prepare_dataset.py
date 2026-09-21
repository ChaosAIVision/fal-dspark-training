import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_record_shape_and_json_normalization():
    row = {"system_prompt": "system", "input": "input", "thinking": "reason",
           "output": "```json\n[ {\"id\": 1} ]\n```"}
    record = MODULE.build_record("example", row, 2)
    assert record["id"] == "example-00000002"
    assert record["conversations"][-1] == {
        "role": "assistant", "reasoning_content": "reason", "content": '[{"id":1}]'
    }


def test_shards_are_deterministic(tmp_path):
    records = [{"id": str(i)} for i in range(7)]
    first, second = tmp_path / "a", tmp_path / "b"
    one = MODULE.write_shards(records.copy(), first, 3, 42)
    two = MODULE.write_shards(records.copy(), second, 3, 42)
    assert one == two
    assert [x["rows"] for x in one["shards"]] == [3, 3, 1]
    for item in one["shards"]:
        assert (first / item["file"]).read_text() == (second / item["file"]).read_text()


def test_output_must_be_json_array():
    try:
        MODULE.canonical_json_output('{"not":"an array"}')
    except ValueError as error:
        assert "array" in str(error)
    else:
        raise AssertionError("object output should be rejected")
