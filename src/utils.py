import torch


def change_list_to_tensors(tokenized_inputs):
    for field in ["input_ids", "attention_mask", "labels"]:
        if field in tokenized_inputs:
            tokenized_inputs[field] = torch.tensor(tokenized_inputs[field])

    return tokenized_inputs
