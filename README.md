# NLP TD 1: classification

L'objectif de ce TD est de créer un modèle "nom de vidéo" -> "is_comic" (is_comic vaut 1 si c'est une chronique humouristique, 0 sinon).

Il s'agît d'un problème d'apprentissage supervisé classique, à ceci près qu'on doit extraire les features du texte. <br/>
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
- [Optionel mais recommandé] Un entry point pour evaluer un modèle, prenant en entrée le path aux données de train.
```
python src/main.py evaluate --input_filename=data/raw/train.csv
```


## Dataset

Dans [ce lien](https://docs.google.com/spreadsheets/d/1HBs08WE5DLcHEfS6MqTivbyYlRnajfSVnTiKxKVu7Vs/edit?usp=sharing), on a un CSV avec 2 colonnes:
- video_name: le nom de la video
- is_comic: est-ce une chronique humoristique

## Text classification: prédire si la vidéo est une chronique comique

- Créer une pipeline train, qui:
  - load le CSV
  - transforme les titres de videos en one-hot-encoded words (avec sklearn: CountVectorizer)
  - train un modèle (linéaire ou random forest)
  - dump le model
- Créer la pipeline predict, qui:
  - prend le modèle dumpé
  - prédit sur de nouveaux noms de video
  <br\>(comment cette partie one-hot encode les mots ? ERREUR à éviter: l'encoding en "predict" ne pointe pas les mots vers les mêmes index. Par exemple, en train, un nom de video avec le mot chronique aurait 1 dans la colonne \#10, mais en predict, il aurait 1 dans la colonne \#23)
- (optionel mais recommandé: créer une pipeline "evaluate" qui fait la cross-validation du modèle pour connaître ses performances)
- Transformer les noms de video avec différentes opérations de NLTK (Stemming, remove stop words) ou de CountVectorizer (min / max document frequency)
- Itérer avec les différentes features / différents modèles pour trouver le plus performant

## A Rendre

Envoyez le code à foucheta@gmail.com. <br/>
Le mail aura comme object [ESGI][NLP] TD1. <br/>
Si vous avez fait le TD en groupe de 2, ajoutez l'autre membre dans le CC du mail.

# NLP TD 2: Transfer learning for named-entity recognition

## Part 1: Named-entity recognition

Dans ce TD, on va fine-tune un modèle BERT pour identifier des noms de personnes dans du texte en français. <br/>
Nous l'utiliserons ensuite sur nos videos France Inter.

Dans le notebooks/TD2_transfer_learning.ipynb, vous trouverez le code pour:
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
Vous devriez atteindre 98.5%+ d'accuracy.

Trouver [sur ce lien](https://drive.google.com/file/d/1ZEuK3JYIgXhG90rKUyq2rLAZW4VexD5J/view?usp=drive_link) un dataset avec les noms de video, et le label pour chaque token. <br/>
(Remarque: le modèle peut être entraîné sur MultiNERD, puis le dataset France Inter).

## Part 2: Full-pipeline trouver noms de comiques dans les videos.

Avec un modèle:
- titre de video -> is_comic
- titre de video -> nom de personne dans le titre

Faire une pipeline [titres de video] -> [(nom de comiques, liste des videos où il apparaît)]

# NLP TD 3: Prompt Engineering

On revient au problème d'identifier les noms de comiques dans des noms de video France Inter.

On veut développer une prompt pour ChatGPT GPT-4o-mini donnant un ou plusieurs titres de video, et le LLM répondant les noms de comique contenus dans ces titres.

Vous allez expérimenter plusieurs prompts, en intégrant au fur et à mesure les guidelines de [ce site](https://www.promptingguide.ai/) et [OpenAI](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api).

Vous allez aussi essayer des techniques comme Chain-Of-Thought.

Vous allez rendre un rapport avec vos différentes expérimentations. Quelles difficultés rencontrées ? Quelles méthodes ont amélioré l'efficacité de la prompt.

Vous enverrez aussi votre prompt, ainsi que le code pour parser la réponse de ChatGPT et avoir une fonction:
list(titres de videos) -> list(noms de comiques)

# NLP TD 4: RAG

Vous allez créer un RAG pour une école d'informatique. </br>
Le RAG répond aux questions des étudiants sur les cours en se servant des fiches descriptives de chaque cours. </br>
Tout est dans le notebook notebook/RAG.ipynb

Voici la [liste de questions](https://drive.google.com/file/d/14hZ0hTx5dM1WgJYewZsn9BkHzEReq-pj/view?usp=sharing) que je poserai au RAG. </br>
A rendre:
- Le notebook de votre RAG
- un CSV avec question,embedding,rag_reply
- un CSV avec chunk,embedding

# NLP TD 5-6:

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

## TD 5: Parser le PRESTO dataset

J'ai créé un fichier de test "tests/data/test_presto.py" avec différents cas de "inputs / targets" extraits du dataset PRESTO. <br/>
Faites la fonction "parse_presto_labels" qui passe les tests.

## TD 6: Virtual assistant, suite

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

J'ai ajouté, dans src/models.py, une fonction "predict_at_word_level" permettant d'obtenir, au niveau "mot", les predictions du modèle niveau token.

## TD 7: Assistant virtual (fin)

## Part 1: Parser une query pour envoyer un message

Télécharger [ce dataset](https://docs.google.com/spreadsheets/d/1ryDizBb7QunbWXmCs8MdaZ-GYgd-HO39T8459jB3dE0/edit?usp=sharing) user_query -> service à utiliser.

Fine-tuner un DistilBert de SequenceClassification classifiant les queries entre "question_rag" et "send_message"

Etant donné le peu d'exemples dans le dataset, on ne pourra pas apprendre beaucoup de couches...

Uploader le modèle sur HuggingFace.

## Part 2: Putting it all together!

(Si vous n'avez pas le modèle du TD7, vous pouvez utilisez ce modèle HuggingFace: foucheta/nlp_esgi_td4_ner et la fonction "predict_at_word_level" dans src/models.py </br>
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

A rendre: un fichier virtual_assistant.py avec une fonction "call_virtual_assistant(user_query: str) -> dict"

## TD 8: Travail collectif: RAG sur les films

Nous allons développer un nouveau RAG, qui répondra aux questions sur les films. <br/>
Je fournis:
- un [fichier zip](https://drive.google.com/file/d/19udLiCp6HdEEzsq_NQBeImXSmoBKTBov/view?usp=sharing) avec les pages wikipedia de divers films.
- un dataframe avec un set de question - text à trouver dans les sources - réponse attendue
- le code src_rag/ avec un modèle de RAG et un script evaluate.py qui évalue le RAG et pousse les résultats sur mlflow
- un [dashboard databricks](https://dbc-264ce65d-5aec.cloud.databricks.com/ml/experiments/1145748448592611?viewStateShareKey=798eb27a57612968762ab3b78541252abaf2a8a5576f027a77db23833b574b94&compareRunsMode=TABLE&o=1111593904032546) avec le résultat des experimentations ML-Flow
- un [board Trello](https://trello.com/b/8hT0M8L8/esgi-iabd-jan-rag-movies) avec toutes les idées pour améliorer le RAG

L'idée est de se répartir les "idées d'amélioration". <br/>
Lorsqu'un groupe a amélioré le RAG, il peut en informer les autres qui intègrent son travail. <br/>
On cherchera à avoir le meilleur MRR sur ce sujet.

A faire:
- Changer config.yml.example en config.yml. Y ajouter une api_key Groq pour pouvoir générer le texte
- Run the code
```
from src_rag import evaluate
evaluate.run_evaluate_retrieval(config={})
```
Ceci doit marcher et vous pousser une expérimentation ML-Flow locale
- Changer la fonction "_load_ml_flow(conf)"
```
def _load_ml_flow(conf):
    os.environ["DATABRICKS_HOST"] = conf["databricks_url"]
    os.environ["DATABRICKS_TOKEN"] = conf["databricks_key"]

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(conf["mlflow_experiment"])
```
Pour pousser vos expérimentations sur le databricks

## TD 9: Travail collectif: Agents

On va étendre le RAG fait au TD précédent en y ajoutant un agent capable de faire des requêtes SQL. <br/>
Je fournis (un CSV)[https://drive.google.com/file/d/1pb2TONwr-GdqiH32btOhjBxRl6hAGOxM/view?usp=sharing] avec des informations sur les films: réalisateurs, acteurs, genre, **note moyenne**. <br/>
Je fournis (un zip)[https://drive.google.com/file/d/1dKAVimkciabC2cl0J1kCwKuOOcEIoCJI/view?usp=sharing] avec la fiche wikipedia, en markdown, des films notés dans le CSV.

Vous allez utiliser un LLM pour pouvoir traduire des questions d'utilisateur en requête SQL, puis vous servir du résultat de la requête pour envoyer une réponse en langage naturel.

Aide: une fois que vous avez extrait la requête SQL de la réponse du LLM, vous pouvez l'exécuter sur le CSV en utilsant polar

```python
import polars

ctx = polars.SQLContext(MOVIES=df)

ctx.execute("""
SELECT title, director, vote_average
  FROM MOVIES
  WHERE genres LIKE \'%Animation%\'
  AND vote_average >= 7.5
  ORDER BY vote_average DESC
  LIMIT 10;""", eager=True).to_pandas()
```

Votre bot doit pouvoir répondre aux questions suivantes:
- "Quels sont les 3 films les mieux notés dans le genre action ?"
- "Dans quels films Leonardo DiCaprio a-t-il joué ?"
- "Quel film réalisé par Michael Bay a la meilleure note ?"
- "Lister tous les thrillers réalisés par Quentin Tarantino, triés par date de sortie."
- "Trouver tous les films qui présentent à la fois Léonardo DirCaprio et Brad Pit dans le casting."  (fautes d'orthographe)
- "J'aime les films d'action avec un retournement de situation. Pourriez-vous me recommander quelque chose ?" (Dans l'idée, l'IA liste des films avec la requête SQL, cherche avec RAG s'il y a retournement de situation, puis refiltre)