# NLP TD 1: classification

L'objectif de ce TD est de créer un modèle "nom de vidéo" -> "is_comic" (is_comic vaut 1 si c'est une chronique humouristique, 0 sinon).

Dans ce TD, on s'intéresse surtout à la démarche. Pour chaque tâche:
- Bien poser le problème
- Avoir une baseline
- Experimenter diverses features et modèles
- Garder une trace écrite des expérimentations dans un rapport. Dans le rapport, on s'intéresse plus au sens du travail effectué (quelles expérimentations ont été faites, pourquoi, quelles conclusions) qu'à la liste de chiffres.
- Avoir une codebase clean, permettant de reproduire les expérimentations.

On se contentera de méthodes pré-réseaux de neurones. Nos features sont explicables et calculables "à la main".

La codebase doit fournir les entry points suivant:
- Un entry point pour train, prenant en entrée le path aux données de train et dumpant le modèle dans "model_dump" 
```
python src/main.py train --input_filename=data/raw/train.csv --model_dump_filename=models/model.json
```
- Un entry point pour predict, prenant en entrée le path au modèle dumpé, le path aux données à prédire et outputtant dans un csv les prédictions
```
python src/main.py predict --input_filename=data/raw/test.csv --model_dump_filename=models/model.json --output_filename=data/processed/prediction.csv
```
- Un entry point pour evaluer un modèle, prenant en entrée le path aux données de train.
```
python src/main.py evaluate --input_filename=data/raw/train.csv
```


## Dataset

Dans [ce lien](https://docs.google.com/spreadsheets/d/1HBs08WE5DLcHEfS6MqTivbyYlRnajfSVnTiKxKVu7Vs/edit?usp=sharing), on a un CSV avec 2 colonnes:
- video_name: le nom de la video
- is_comic: est-ce une chronique humoristique

## Partie 1: Text classification: prédire si la vidéo est une chronique comique

### Tasks

- Créer une pipeline train, qui:
  - load le CSV
  - transforme les titres de videos en one-hot-encoded words (avec sklearn: CountVectorizer)
  - train un modèle (linéaire ou random forest)
  - dump le model
- Créer la pipeline predict, qui:
  - prend le modèle dumpé
  - prédit sur de nouveaux noms de video
  <br\>(comment cette partie one-hot encode les mots ? ERREUR à éviter: l'encoding en "predict" ne pointe pas les mots vers les mêmes index. Par exemple, en train, un nom de video avec le mot chronique aurait 1 dans la colonne \#10, mais en predict, il aurait 1 dans la colonne \#23)
- Créer une pipeline "evaluate" qui fait la cross-validation du modèle pour connaître ses performances)
- Transformer les noms de video avec différentes opérations de NLTK (Stemming, remove stop words) ou de CountVectorizer (min / max document frequency)
- Envoyer ce code (à la fin du cour)
- Itérer avec les différentes features / différents modèles pour trouver le plus performant
- Faire un rapport avec les différentes itérations faites, et les conclusions
- Envoyer le rapport et le code entraînant le meilleur modèle (avant le 27 octobre)

# NLP TD 2: Entraînement de réseaux de neurones

voir notebooks/TD2.ipynb

# NLP TD 3: Transfer learning for named-entity recognition

Dans ce TD, on va fine-tune un modèle BERT pour identifier des noms de personnes dans du texte en français. <br/>
Nous l'utiliserons ensuite sur nos videos France Inter.

Dans le notebooks/TD3_transfer_learning.ipynb, vous trouverez le code pour:
- Extraire d'un fichier MultiNERD English une serie de phrase, dont les mots sont labelisés 1 si le mot est un nom de personne, 0 sinon.
- Fine-tune le modèle DistilBert en gelant la 1ère couche.

Après avoir vérifié que ça marche, vous devez:
- Adapter ce code en français (données MultiNERD FR, modèle CamemBERT ou autre)
- Créer une fonction (text_split_in_words, model, tokenizer) -> labels <br/>
text_split_in_words est la liste des mots d'un texte. <br/>
Par exemple, la video_name "Bonjour class d'ESGI" sera le text_split_in_words: ["Bonjour", "class", "d'", "ESGI"]
- Uploader votre modèle sur HuggingFace.
- Fournir un code:

```
def predict(texts_split_into_words: list[list[str]]) -> list[list[int]]:
    model = AutoModelForTokenClassification.from_pretrained(your_uploaded_model_name)
    tokenizer = AutoTokenizer.from_pretrained(your_uploaded_model_name)

    labels = []
    for text_split_into_words in texts_split_into_words:
        word_labels = predict_is_name(text_split_into_words, model, tokenizer)
	labels.append(word_labels)

    return labels
```
- Expérimenter pour produire le meilleur modèle à identifier les noms de personne sur les noms de videos France Inter.<br/>
Trouver [sur ce lien](https://drive.google.com/file/d/1ZEuK3JYIgXhG90rKUyq2rLAZW4VexD5J/view?usp=drive_link) un dataset avec les noms de video, et le label pour chaque token. <br/>
(Remarque: le modèle peut être entraîné sur MultiNERD, puis le dataset France Inter).

# NLP TD 4:

Dans ce TD, nous allons coder un assistant virtuel, capable de transformer:

"Ask the python teacher when is the next class?"

en un json:

```

   "job": "send_message",
   "receiver": "the python teacher",
   "content": "when is the next class?",
}
```


Pour cela, nous allons utiliser [le PRESTO dataset](https://github.com/google-research-datasets/presto). <br/>
Le bot fonctionnera sur des phrases en anglais (car le dataset contient plus de contenu en anglais).

## Parser le PRESTO dataset

J'ai créé un fichier de test "tests/data/test_presto.py" avec différents cas de "inputs / targets" extraits du dataset PRESTO. <br/>
Faites la fonction "parse_presto_labels" qui passe les tests.

Cette fonction doit m'être envoyée avant le 5 décembre 16:00.


# NLP TD 5: Prompt Engineering

On revient au problème d'identifier les noms de comiques dans des noms de video France Inter.

On veut développer une prompt pour ChatGPT GPT-4o-mini donnant un ou plusieurs titres de video, et le LLM répondant les noms de comique contenus dans ces titres.

Vous allez expérimenter plusieurs prompts, en intégrant au fur et à mesure les guidelines de [ce site](https://www.promptingguide.ai/) et [OpenAI](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api).

Vous allez aussi essayer des techniques comme Chain-Of-Thought.

Vous allez rendre un rapport avec vos différentes expérimentations. Quelles difficultés rencontrées ? Quelles méthodes ont amélioré l'efficacité de la prompt.

Vous enverrez aussi votre prompt, ainsi que le code pour parser la réponse de ChatGPT et avoir une fonction:
list(titres de videos) -> list(noms de comiques)

# NLP TD 6: RAG

Vous allez créer un RAG pour une école d'informatique. </br>
Le RAG répond aux questions des étudiants sur les cours en se servant des fiches descriptives de chaque cours. </br>
Tout est dans le notebook notebook/RAG.ipynb

Voici la [liste de questions](https://drive.google.com/file/d/14hZ0hTx5dM1WgJYewZsn9BkHzEReq-pj/view?usp=sharing) que je poserai au RAG. </br>
A rendre:
- Le notebook de votre RAG
- un CSV avec question,embedding,rag_reply
- un CSV avec chunk,embedding

# NLP TD 7: Virtual assistant [end]

Nous construisons un assistant virtuel pour l'ESGI avec 2 fonctions:
- envoyer un message "Ask the DevOps teacher when is next class" -> {"job": "send_message", "receiver": "the python teacher", "content": "when is the next class?"}
- poser une question à un RAG "What do we learn in the project management course" -> use RAG to answer the question

## Part 1: Parser une query pour envoyer un message

Voici [une partie du dataset Presto parsée](https://drive.google.com/file/d/1-7-esuAMBDzjN2DQsUD9Up7z7bIRwahL/view?usp=sharing). Il ne contient des user queries en anglais, qui contiennent des mots labellisés "person" (la personne à qui envoyer) et "content" (le message à envoyer).

Fine-tuner un DistilBert de TokenClassification reconnaîssant les tokens "person" et "content", en utilisant du transfert learning.
Uploader le modèle sur github.

Faîtes une pipeline qui, pour une query, repère les tokens "person" et "content", et renvoie le json:
```
{
   "receiver": {tokens labellisés "person"}, 
   "content": {tokens labellisés "content"}, 
}
```

## Part 2: Classify user query

Télécharger [ce dataset](https://docs.google.com/spreadsheets/d/1ryDizBb7QunbWXmCs8MdaZ-GYgd-HO39T8459jB3dE0/edit?usp=sharing) user_query -> service à utiliser.

Fine-tuner un DistilBert de SequenceClassification classifiant les queries entre "question_rag" et "send_message"

Etant donné le peu d'exemples dans le dataset, on ne pourra pas apprendre beaucoup de couches...

Uploader le modèle sur HuggingFace.

## Part 3: Putting it all together!

Renvoyer le code d'un virtual assistant.
Le virtual_assistant.main(user_query):
- classifiera la user_query en tant que "question_rag" ou "send_message"
- si elle est classifiée "question_rag", main renvoie f"asked_to_rag: {user_query}"
- si elle est classifiée "send_message", main renvoie le json
```
{
   "receiver": {tokens labellisés "person"}, 
   "content": {tokens labellisés "content"}, 
}
```

(ceci, évidemment à l'aide de vos modèles uploadés sur HuggingFace)