import json
import pandas as pd
from tqdm import tqdm


def parse_dataset(dataset_name):
    """From path to a presto file, return a df with columns:
    - sentence: sentence in the presto dataset
    - words: sentence split into words
    - labels: label for each word
    - task: the task if this presto line

    Filter rows in English,
    from tasks create_note, post_message or get_message_content
    for label "note_assignee", "person" or "content"
    "note_assignee" is renamed as "person"
    """
    df = pd.read_json(f"data/raw/presto/presto_{dataset_name}.jsonl", lines=True)

    df["lang"] = df["metadata"].apply(lambda d: d["locale"])
    df = df.query("lang == 'en-US'")

    df["job"] = df["targets"].apply(lambda txt: txt.split()[0])
    sub_df = df[df["job"].isin(["Create_note", "Post_message", "Get_message_content"])]
    bad_indexes = []

    lines = []
    for i, row in tqdm(sub_df.iterrows()):
        try:
            res = parse_presto_labels(row.inputs, row.targets)
        except Exception as e:
            bad_indexes.append(i)
        else:
            lines.append(res)

    for line in lines:
        line["labels"] = clean_labels(line["labels"])

    df = pd.DataFrame(lines)

    df["words"] = df["words"].apply(json.dumps)
    df["labels"] = df["labels"].apply(json.dumps)

    return df


def clean_label(label):
    if label == 0:
        return label
        
    label = label.split("__")[-1]
    if label == "note_assignee":
        label = "person"
        
    if label in {"person", "content"}:
        return label
    else:
        return 0

    
def clean_labels(labels):
    return [clean_label(lab) for lab in labels]


def parse_presto_labels(sentence, target):
    words = _split_sentence_in_words(sentence)
    task = target.split()[0]

    labels = _get_labels(words, target)
    res = {
        "sentence": sentence,
        "words": words,
        "labels": labels,
        "task": task,
    }

    return res


def _split_sentence_in_words(sentence):
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


def _get_labels(words, target):
    text_with_label = _extract_text_with_labels(target)

    labels = _spread_label_in_text(words, text_with_label)

    return labels


def _spread_label_in_text(words, text_with_label):
    labels = [0] * len(words)

    for txt, label in text_with_label:
        indexes = _find_indexes(txt, words)
        for idx in indexes:
            labels[idx] = label

    return labels

def _find_indexes(txt, words):
    txt_words = _split_sentence_in_words(txt)
    nb_words = len(txt_words)
    i_start = next(i for i in range(len(words)) if words[i:i+nb_words] == txt_words)
    return range(i_start, i_start + nb_words)


def _extract_text_with_labels(target):
    target = _find_content_in_parenthesis(target)

    text_with_label = []
    while target:
        txt_label_start = target.find("«")
        txt_label_end = target.find("»")

        if txt_label_start == -1:
            # No text to label
            break


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
