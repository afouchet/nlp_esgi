from data.presto import parse_presto_labels


def test_parse_presto_labels__case_1():
    sentence = "Can you write a note for 9h ?"
    target = "Create_note ( trigger_time « 9h » )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        "sentence": "Can you write a note for 9h ?",
        "words": ["Can", "you", "write", "a", "note", "for", "9h", "?"],
        "labels": [0, 0, 0, 0, 0, 0, "trigger_time", 0],
        "task": "Create_note",
    }


def test_parse_presto_labels__case_2():
    sentence = "Create a shopping note ज़रा in Keep."
    target = "Create_note ( app « Keep » label « shopping » )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        'sentence': 'Create a shopping note ज़रा in Keep.',
        'words': ['Create', 'a', 'shopping', 'note', 'ज़रा', 'in', 'Keep', '.'],
        'labels': [0, 0, 'label', 0, 0, 0, 'app', 0],
        'task': 'Create_note',
    }


def test_parse_presto_labels__case_3():
    sentence = "Make un recordatorio."
    target = "Create_note ( )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        'sentence': 'Make un recordatorio.',
        'words': ['Make', 'un', 'recordatorio', '.'],
        'labels': [0, 0, 0, 0],
        'task': 'Create_note',
    }


def test_parse_presto_labels__case_4():
    sentence = "Create a reminder to buy fromage frais and crème brûlée."
    target = "Create_note ( content « buy fromage frais and crème brûlée » note_feature « reminder » )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        'sentence': 'Create a reminder to buy fromage frais and crème brûlée.',
        'words': ['Create', 'a', 'reminder', 'to', 'buy', 'fromage', 'frais', 'and', 'crème', 'brûlée', '.'],
        'labels': [0, 0, 'note_feature', 0, 'content', 'content', 'content', 'content', 'content', 'content', 0],
        'task': 'Create_note',
    }


def test_parse_presto_labels__case_5():
    sentence = "Tweet um tweet 'hello ;)'"
    target = "Post_message ( medium « tweet » message « hello ;) » )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        'sentence': "Tweet um tweet 'hello ;)'",
        'words': ['Tweet', 'um', 'tweet', "'", 'hello', ';)', "'"],
        'labels': [0, 0, 'medium', 0, 'message', 'message', 0],
        'task': 'Post_message',
    }


def test_parse_presto_labels__case_6():
    sentence = "Do not be late!"
    target = "Create_note ( content « Do not be late! » note_assignee InferFromContext )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        'sentence': 'Do not be late!',
        'words': ['Do', 'not', 'be', 'late', '!'],
        'labels': ['content', 'content', 'content', 'content', 'content'],
        'task': "Create_note",
    }


    
def test_parse_presto_labels__nested_labels():
    # 2-levels nest
    sentence = "Get me the message with the content are you going to yoga class today?."
    target = "Get_message_content ( message Electronic_message ( content « are you going to yoga class today? » medium « message » ) )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        'sentence': 'Get me the message with the content are you going to yoga class today?.',
        'words': ['Get', 'me', 'the', 'message', 'with', 'the', 'content', 'are', 'you', 'going', 'to', 'yoga', 'class', 'today', '?', '.'],
        'labels': [0, 0, 0, 'message__medium', 0, 0, 0, 'message__content', 'message__content', 'message__content', 'message__content', 'message__content', 'message__content', 'message__content', 'message__content', 0],
        'task': 'Get_message_content',
    }


def test_parse_presto_labels__nested_labels__case_2():
    # 4-levels nest
    sentence = "Read me the first 2 messages from 課長."
    target = "Get_message_content ( message Electronic_message ( medium « messages » quantity NonNegativeSimpleNumber ( Number « 2 » ) sender Personal_contact ( person « 課長 » ) ) modality « Read » )"

    res = parse_presto_labels(sentence, target)

    assert res == {
        'sentence': 'Read me the first 2 messages from 課長.',
        'words': ['Read', 'me', 'the', 'first', '2', 'messages', 'from', '課長', '.'],
        'labels': ['modality', 0, 0, 0, 'message__quantity__Number', 'message__medium', 0, 'message__sender__person', 0],
        'task': 'Get_message_content',
    }
