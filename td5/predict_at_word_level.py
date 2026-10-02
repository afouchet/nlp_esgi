import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def predict_at_word_level(
    words: list[str],
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
) -> list[int]:
    inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True)
    
    logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)

    word_labels = []
    word_ids = inputs.word_ids()
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            word_labels.append(predictions[0][idx].item())
            previous_word_idx = word_idx
    
    return word_labels
