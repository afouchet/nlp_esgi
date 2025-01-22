def parse_presto_labels(sentence, target):
    words = _split_sentence_into_words(sentence)
    
    task = target.split()[0]

    ner_labels = _get_words_labels(words, target)

    return {
        "sentence": sentence,
        "words": words,
        "labels": ner_labels,
        "task": task,
    }


def _get_words_labels(words, target):
    """
    For a a list words and Google's target,
    (like
    words = ['Tweet', 'um', 'tweet', "'", 'hello', ';)', "'"]
    target = "Post_message ( medium « tweet » message « hello ;) » )"
    )

    Returns the list of each word's label
    (in the example, ner_label = [0, 0, 'medium', 0, 'message', 'message', 0])
    """
    labels = _find_content_in_parenthesis(target)

    texts_with_label = _get_texts_with_label(labels)

    ner_labels = _get_label_by_word(words, texts_with_label)

    return ner_labels


def _get_texts_with_label(labels_txt):
    """
    Given a Google's target
    (like "medium « tweet » message « hello ;) »")

    Returns the list of tuples (text, label)
    (here [("tweet", "medium"), ("hello ;)", "message")])

    In the case of nested labels
    (like
    "message ( content « are you going  » medium « insta » )"
    )
    Will return
    [("are you going", "message__content"), ("insta", "message__medium")]
    """
    texts_with_label = []

    while labels_txt:
        if _has_nested_parenthesis(labels_txt):
            this_texts_with_label, text_left = _parse_nested_labels(labels_txt)
            texts_with_label += this_texts_with_label
            labels_txt = text_left
        else:
            if "»" not in labels_txt:
                break
            this_label_txt, labels_txt = labels_txt.split("»", 1)
            label, text = this_label_txt.split("«")
            texts_with_label.append((text.strip(), label.strip()))

    return texts_with_label


def _has_nested_parenthesis(labels_txt):
    start_label = labels_txt.find("«")
    return "(" in labels_txt[:start_label]


def _parse_nested_labels(labels_txt):
    label = labels_txt.split()[0]

    nested_label_text = _find_content_in_parenthesis(labels_txt)

    nested_texts_with_labels = _get_texts_with_label(nested_label_text)

    this_texts_with_label = [(text, f"{label}__{this_label}") for (text, this_label) in nested_texts_with_labels]

    txt_start = labels_txt.find(nested_label_text)
    text_left = labels_txt[txt_start+len(nested_label_text)+2:]

    return this_texts_with_label, text_left


def _get_label_by_word(words, texts_with_label):
    ner_labels = [0] * len(words)

    for text, label in texts_with_label:
        text_words = _split_sentence_into_words(text)
        for i in range(len(words)):
            if words[i: i+len(text_words)] == text_words:
                for j in range(len(text_words)):
                    ner_labels[i + j] = label

    return ner_labels


def _find_content_in_parenthesis(txt):
    content_start = None
    is_in_quote = False
    nb_parenthesis = 0

    for i, char in enumerate(txt):
        if is_in_quote:
            if char == "»":
                is_in_quote = False
        else:
            if char == "(":
                nb_parenthesis += 1
                if content_start is None:
                    content_start = i
            elif char == ")":
                nb_parenthesis -= 1
                if nb_parenthesis == 0:
                    content_end = i
                    break
            elif char == "«":
                is_in_quote = True
                    
    
    return txt[content_start+1:content_end-1].strip()


def _split_sentence_into_words(sentence):
    words = []
    special_chars = ".!?':"

    for word in sentence.split():
        start_words = []
        end_words = []
        if len(word) > 1:
            while word[0] in special_chars:
                start_words.append(word[0])
                word = word[1:]

            while word[-1] in special_chars:
                end_words = [word[-1]] + end_words
                word = word[:-1]

        if start_words:
            words += start_words

        words.append(word)

        if end_words:
            words += end_words

    return words
