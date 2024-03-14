import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer

from api import api
from data import presto
from utils import change_list_to_tensors

MODEL_NAME = "foucheta/nlp_esgi_td4_ner"

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_NAME, add_prefix_space=True,
)

MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

LABEL_DICT = {
    0: 0,
    1: 'person',
    2: 'content',
}



def run(sentence):
    words = presto._split_sentence_in_words(sentence)
    labels = predict_word_labels(words)

    call_json = translate_labels_to_api_call(words, labels[0])

    return api.send_message(receiver=call_json["person"], message=call_json["content"])


def predict_word_labels(words, tokenizer=TOKENIZER, model=MODEL):
    labels = [0] * len(words)

    tokenized_inputs = tokenize_and_align_labels([words], [labels], tokenizer)
    tokenized_inputs = change_list_to_tensors(tokenized_inputs)

    pred_ids = model(**tokenized_inputs)

    pred_labels = pred_to_label(pred_ids, tokenized_inputs["labels"])

    return pred_labels


def translate_labels_to_api_call(words, labels):
    res = {"person": [], "content": []}

    for word, label in zip(words, labels):
        if label != 0:
            res[label].append(word)

    return {key: " ".join(words) for key, words in res.items()}


def tokenize_and_align_labels(sentences, ner_tags, tokenizer):
    tokenized_inputs = tokenizer(
        sentences,
        truncation=True,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(ner_tags):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        # Set the special tokens to -100.
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def pred_to_label(predictions, labels):
    predictions = np.argmax(predictions.logits.detach().numpy(), axis=2)
    true_predictions = [
        [LABEL_DICT[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_predictions

