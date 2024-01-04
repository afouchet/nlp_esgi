from data.presto import parse_presto_labels


def test_parse_presto_labels():
    sentence = "Can you write a note for 9h ?"
    target = "Create_note ( trigger_time « 9h » )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        "sentence": sentence,
        "words": sentence.split(),
        "labels": [0, 0, 0, 0, 0, 0, "trigger_time", 0],
        "task": "Create_note",
    }
