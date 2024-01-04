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
    target = _find_content_in_parenthesis(target)

    text_with_label = []
    while target:
        txt_label_start = target.find("«")
        txt_label_end = target.find("»")

        if "(" in target[:txt_label_start]:
            # There is nested content
            sub_text_with_label = _extract_text_with_labels(target)

            txt_label_start = target.find("(")
            label = target[:txt_label_start].split()[0].strip()

            this_text_with_label = [
                (txt, f"{label}__{sub_label}")
                for txt, sub_label in sub_text_with_label
            ]
            text_with_label += this_text_with_label

            # find last ")"
            txt_in_parenthesis = _find_content_in_parenthesis(target)
            label_txt = txt_label_start + len(txt_in_parenthesis) + 2
            end_parenthesis = target[label_txt:].find(")") + label_txt
            txt_label_end = end_parenthesis

        else:
            label = target[:txt_label_start].split()[0].strip()
            txt_label = target[txt_label_start+1:txt_label_end].strip()

            text_with_label.append([txt_label, label])

        target = target[txt_label_end+1:]

    return text_with_label


def _find_content_in_parenthesis(txt):
    content_start = None
    nb_parenthesis = 0

    for i, char in enumerate(txt):
        if char == "(":
            nb_parenthesis += 1
            if content_start is None:
                content_start = i
        elif char == ")":
            nb_parenthesis -= 1
            if nb_parenthesis == 0:
                content_end = i
                break
    
    return txt[content_start+1:content_end-1].strip()


def _remove_end_punctuation(sentence):
    punctuations = ".?!"

    if sentence[-1] in punctuations:
        return sentence[:-1].strip()
    else:
        return sentence
