def parse_presto_labels(sentence, target):
    sentence = _remove_end_punctuation(sentence)
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
    labels_start = target.find("(")
    labels_end = len(target) - list(reversed(target)).index(")")

    target = target[labels_start+1:labels_end-1].strip()

    text_with_label = []
    while target:
        txt_label_start = target.find("«")
        txt_label_end = target.find("»")

        label = target[:txt_label_start].strip()
        txt_label = target[txt_label_start+1:txt_label_end].strip()

        text_with_label.append([txt_label, label])

        target = target[txt_label_end+1:]

    return text_with_label

def _remove_end_punctuation(sentence):
    punctuations = ".?!"

    if sentence[-1] in punctuations:
        return sentence[:-1].strip()
    else:
        return sentence
