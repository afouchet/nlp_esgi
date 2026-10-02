# NLP TD 5:

Dans ce TD, nous allons coder un assistant virtuel, capable de transformer:

"Ask the python teacher when is the next class?"

en un json:

```
{
   "task": "send_message",
   "receiver": "the python teacher",
   "content": "when is the next class?",
}
```
et

"Does the React course cover the use of hooks?"

en un json:

```
{
    "task": "ask_RAG",
    "reply": "asked_to_rag: Does the React course cover the use of hooks?",
}
```

## Part 1: Parser une query pour envoyer un message

Nous allons entraîner un premier modèle "message" -> service à utiliser ("send_message" ou "ask_RAG")

Télécharger [ce dataset](https://docs.google.com/spreadsheets/d/1ryDizBb7QunbWXmCs8MdaZ-GYgd-HO39T8459jB3dE0/edit?usp=sharing) user_query -> service à utiliser.

Fine-tuner un DistilBert de SequenceClassification classifiant les queries entre "question_rag" et "send_message"

Etant donné le peu d'exemples dans le dataset, on ne pourra pas apprendre beaucoup de couches...

Uploader le modèle sur HuggingFace.

## Part 2: Parser receiver et content pour send_message

Voici [une partie du dataset Presto parsée](https://drive.google.com/file/d/1-7-esuAMBDzjN2DQsUD9Up7z7bIRwahL/view?usp=sharing). Il ne contient des user queries en anglais, qui contiennent des mots labellisés "person" (la personne à qui envoyer) et "content" (le message à envoyer).

Fine-tuner un DistilBert de TokenClassification reconnaîssant les tokens "person" et "content", en utilisant du transfert learning.
Uploader le modèle sur HuggingFace.

Faîtes une pipeline "parse_message" qui, pour une query, repère les tokens "person" et "content", et renvoie le json:
```
{
   "receiver": {tokens labellisés "person"}, 
   "content": {tokens labellisés "content"}, 
}
```
Par exemple:
```
>> parse_message("Ask the python teacher when is the next class")
{"receiver": "the python teacher", "content": "when is the next class"}
```

J'ai ajouté, dans td5/predict_at_word_level.py, une fonction "predict_at_word_level" permettant d'obtenir, au niveau "mot", les predictions du modèle niveau token.

## Part 3: Putting it all together!

Il s'agît du modèle classifiant "ask the python teacher when is the next class" -> "receiver": "the python teacher", "content": "when is the next class")

Renvoyer le code d'un virtual assistant.
Le virtual_assistant.main(user_query):
- classifiera la user_query en tant que "question_rag" ou "send_message"
- si elle est classifiée "question_rag", main renvoie {"task": "ask_RAG", "reply": f"asked_to_rag: {user_query}"}
- si elle est classifiée "send_message", main renvoie le json
```
{
   "task": "send_message"
   "receiver": {tokens labellisés "person"}, 
   "content": {tokens labellisés "content"}, 
}
```

(ceci, évidemment à l'aide de vos modèles uploadés sur HuggingFace)

Exemples:
```
>> call_virtual_assistant("Does the React course cover the use of hooks?"
{
    "task": "ask_RAG",
    "reply": "asked_to_rag: Does the React course cover the use of hooks?",
}

>> call_virtual_assistant("Ask the python teacher when is the next class"
{
    "task": "send_message",
    "receiver": "the python teacher",
    "content": "when is the next class",
}
```

# A rendre

- classifier.txt: le nom de votre modèle HuggingFace sequence -> {"ask_RAG" ou "send_message"}
- ner.txt: le nom de votre modèle HuggingFace sentence -> label ["person", "content"] by token
- un fichier virtual_assistant.py avec une fonction "call_virtual_assistant(user_query: str) -> dict"


Envoyer ces documents via MyGES.