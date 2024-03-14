import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import presto
from utils import change_list_to_tensors

MODEL_NAME = "foucheta/nlp_esgi_td5_classification"

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_NAME, add_prefix_space=True,
)

MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABEL_DICT = {
    0: "question_rag",
    1: "send_message",
}

def identify_task(sentence, tokenizer=TOKENIZER, model=MODEL):
    words = presto._split_sentence_in_words(sentence)

    tokenized_inputs = tokenizer([sentence])
    tokenized_inputs = change_list_to_tensors(tokenized_inputs)

    pred_ids = model(**tokenized_inputs)

    predictions = np.argmax(pred_ids.logits.detach().numpy(), axis=1)

    return LABEL_DICT[predictions[0]]
