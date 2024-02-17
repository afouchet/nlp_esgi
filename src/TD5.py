import torch
import numpy as np
from pandas import read_csv
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments, pipeline

# En gros il va falloir faire un pipeline en réentrainant un distill bert qui va devoir classifier en deux parties.
# L'une des deux classes concerne la question à un quelqu'un (prof de python par exemple) et je vais donc répondre ce qui suit.
# L'autre concerne une question à nous, il faudra l'envoyer au RAG alors.

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_classifier():
    df = read_csv("data/raw/question_classif.csv")
    msk = np.random.rand(len(df)) <= 0.7
    train = df[msk]
    test = df[~msk]
    train_texts = train['question'].values.tolist()
    train_labels = train['label'].values.tolist()
    test_texts = test['question'].values.tolist()
    test_labels = test['label'].values.tolist()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    for name, param in model.base_model.named_parameters():
        if (
                any(layer_name in name for layer_name in ["layer.5.ffn"])
                and "attention" not in name
        ):
            param.requires_grad = False

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        logging_dir='./logs',
        max_steps=len(train_dataset) // 8 * 10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda p: {'accuracy': (p.predictions == p.label_ids).mean()}
    )
    trainer.train()
    model.save_pretrained("./model_directory_TD5")
    tokenizer.save_pretrained("./model_directory_TD5")
    """
    inputs = tokenizer(
        "Does the introduction to machine learning course require prior knowledge of advanced mathematics?",
        return_tensors="pt")
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits).item()
    print(predictions)
    """


# Label : 0=rien ; 1=person; 2=content
def get_label(sentence):
    label_person = ""
    label_content = ""
    classifier = pipeline("token-classification", model="foucheta/nlp_esgi_td4_ner")
    dict_classified = classifier(sentence)

    labels_predicted = [words["entity"] for words in dict_classified]
    words_predicted = [words["word"] for words in dict_classified]
    for i, _ in enumerate(words_predicted):
        if labels_predicted[i] == "LABEL_1":
            label_person += words_predicted[i] + " "
        elif labels_predicted[i] == "LABEL_2":
            label_content += words_predicted[i] + " "

    label_person.strip()
    label_content.strip()
    # print(label_person)
    # print(label_content)
    return label_content, label_person


# Quand on reçoit une question pour quelqu'un, il va falloir séparer le content, des personnes, de rien et renvoyer la réponse.
if __name__ == '__main__':
    train_classifier()
    get_label("Write to the friend inviting them to join a fitness class together")
