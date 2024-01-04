from data.presto import parse_presto_labels


def test_parse_presto_labels():
    sentence = "Can you write a note for 9h ?"
    target = "Create_note ( trigger_time « 9h » )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        "sentence": "Can you write a note for 9h",
        "words": ["Can", "you", "write", "a", "note", "for", "9h"],
        "labels": [0, 0, 0, 0, 0, 0, "trigger_time"],
        "task": "Create_note",
    }

    sentence = "Create a shopping note ज़रा in Keep."
    target = "Create_note ( app « Keep » label « shopping » )"

    res = parse_presto_labels(sentence, target)

    assert res["labels"] == [0, 0, "label", 0, 0, 0, "app"]

    sentence = "Make un recordatorio."
    target = "Create_note ( )"

    res = parse_presto_labels(sentence, target)

    assert res["labels"] == [0, 0, 0]

    sentence = "Create a reminder to buy fromage frais and crème brûlée."
    target = "Create_note ( content « buy fromage frais and crème brûlée » note_feature « reminder » )"

    res = parse_presto_labels(sentence, target)

    assert res["labels"] == [0, 0, "note_feature", 0, "content", "content", "content", "content", "content", "content"]
