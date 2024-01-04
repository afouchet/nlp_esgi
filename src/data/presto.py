def parse_presto_labels(sentence, target):
    words = sentence.split()
    task = target.split()[0]

    text_with_label = _extract_text_with_labels(target)
    labels = [0] * len(words)

    for txt, label in text_with_label:
        indexes = _find_indexes(txt, words)
        for idx in indexes:
            labels[idx] = label

    res = {
        "sentence": sentence,
        "words": words,
        "labels": labels,
        "task": task,
    }

    return res


def _find_indexes(txt, words):
    txt_words = txt.split()
    nb_words = len(txt_words)
    i_start = next(i for i in range(len(words)) if words[i:i+nb_words] == txt_words)
    return range(i_start, i_start + nb_words)
    


def _extract_text_with_labels(target):
    return [("9h", "trigger_time")]
