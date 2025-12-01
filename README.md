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

## Installation

### Sur Mac/Linux :
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
uv run python src/main.py train --input_filename=data/raw/train.csv --model_dump_filename=models/model.json
```

### Sur Windows:

#### WSL (recommandé) -> Utilisez l'installation Mac/Linux

#### Windows terminal

```bash
scripts\setup.bat
uv run python src/main.py train --input_filename=data/raw/train.csv --model_dump_filename=models/model.json
```

### PyCharm

Si vous avez PyCharm, cliquez droit sur le dossier "src/" -> "Mark directory as" -> "Sources Root"

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

## TODO

1. Faire tourner le code
2. Comme en conditions réelles, splitter les données en 80% train (dans un autre train.csv) et 20% test (dans test.csv)
3. Pour l'instant, les features X sont 0 pour toutes les videos. <br/>
Utiliser sklearn CountVectorizer pour faire video_name -> encoded words
4. Coder l'entry point predict pour loader votre modèle, et predire sur de nouvelles données
5. Vérifier que toute votre pipeline fonctionne (train -> model.json -> predict -> predictions avec accuracy > 90%)
6. Utiliser NLTK et scikit-learn pour améliorer les features et améliorer l'accuracy du modèle.

## !! Timeline !! (**Points en moins si non respectée**)

### Après 30 minutes

La commande "python src/main.py train" doit tourner sur votre machine.<br/>
**-1 point si non fait après 30 minutes**<br/>
**0 au TD si non fait après 1 heure**

### Après 1 heure

Vous devez avoir un modèle avec une accuracy > 90% en test<br/>
**-1 point si non fait après 1 heure**

## A Rendre

Envoyez le code à foucheta@gmail.com. <br/>
Le mail aura comme object [ESGI][NLP] TD1. <br/>
Si vous avez fait le TD en groupe de 2, ajoutez l'autre membre dans le CC du mail.

# NLP TD 2: Transfer learning for named-entity recognition

## Installation

### Utilisateurs de Git

```bash
git pull
uv sync
```

### Autres

Télécharger le nouveau pyproject.toml et notebook/TD2.ipynb
puis
```bash
uv sync
```
**M'appeler si "uv sync" ne marche pas**

**Si l'installation est très difficile, faire le TD sur Google Colab**

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

## TODO

1. Run le notebook sur multiNERD en anglais
2. Prendre un modèle CamemBERT et run sur multiNERD en français. Avoir 99%+ accuracy
3. Faire fonction predict at word level
4. Prédire sur le dataset France Inter
5. Uploader votre modèle sur HuggingFace

## !! Timeline !! (**Points en moins si non respectée**)

### Après 30 minutes

Le notebook TD2_transfer_leaning fonctionne sur votre ordinateur
**-1 point si non fait après 30 minutes**<br/>
**0 au TD si non fait après 1 heure**

### Après 1 heure

Vous avez entraîné un modèle français prédisant si un mot est un nom de personnes avec une accuracy > 98.5%+ en test<br/>
**-1 point si non fait après 1 heure**

## A Rendre

- Votre fonction predict(texts_split_into_words: list[list[str]]) -> list[list[int]]
- Vos prediction sur le jeu de données France Inter
- Le nom de votre modèle sur HuggingFace

# NLP TD 3: Prompt Engineering

On revient au problème d'identifier les noms de comiques dans des noms de video France Inter.

On veut développer une prompt pour LLM donnant un ou plusieurs titres de video, et le LLM répondant les noms de comique contenus dans ces titres.

Vous allez expérimenter plusieurs prompts, en intégrant au fur et à mesure les guidelines de [ce site](https://www.promptingguide.ai/) et [OpenAI](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api).

Vous allez aussi essayer des techniques comme Chain-Of-Thought.

Vous allez rendre un rapport avec vos différentes expérimentations. Quelles difficultés rencontrées ? Quelles méthodes ont amélioré l'efficacité de la prompt.

## Installation

[Groq](https://groq.com/) permet d'utiliser gratuitement des LLMs (sous une certaine limite d'utilisation).

Créer un compte et créer une clé API.

Mettez là dans le script src/llm_call.py et faites tourner

```bash
uv run python src/llm_call.py
```

## Etapes

1. Faites une fonction video_name: str -> comic_name: str qui extrait les noms de comiques d'un nom de video. <br/>
En utilisant "structured output".<br/>
Le résultat peut être une liste de vide.

2. Faites une fonction video_names: list[str] -> comic_names: list[str] <br/>
En utilisant "start your reply with ```csv\nvideo_name;comic_names"<br/>
i.e. *sans* utiliser le structured output, mais permettant de donner une structure à la sortie de LLM

3. Récupérer, dans la réponse du LLM, le nombre de input / output tokens. <br/>
Calculer le prix et temps de calcul de votre prompt.

4. Testez différentes bonnes pratiques, et documentez, dans un rapport, ce qui a marché ou non.

## Approche

Au début, *n'essayez pas* d'optimiser la prompt.<br/>
Faites juste "un LLM call qui marche" pour les étapes 1 et 2.

Par la suite, faite un dataset "video_name -> expected answer" pour bien évaluer l'accuracy de la prompt.

## !! Timeline !! (**Points en moins si non respectée**)

### Après 30 minutes

Vous avez une fonction video_name: str -> comic_names: list[str] qui marche<br/>
**-1 point si non fait après 30 minutes**<br/>
**0 au TD si non fait après 1 heure**

### Après 1 heure

Vous avez un jeu de données pour évaluer votre fonction<br/>
**-1 point si non fait après 1 heure**

# A rendre

Vous enverrez aussi votre prompt, ainsi que le code pour parser la réponse de ChatGPT et avoir une fonction:
list(titres de videos) -> list(noms de comiques)

Vous enverrez un CSV video_name,comic_names sortant les prédictions de votre few-shot learner.

Vous enverrez votre rapport montrant les expérimentations et ce qui a marché.

# NLP TD 4: RAG

Vous allez créer un RAG pour une école d'informatique. </br>
Le RAG répond aux questions des étudiants sur les cours en se servant des fiches descriptives de chaque cours. </br>
Tout est dans le notebook notebook/RAG.ipynb

Voici la [liste de questions](https://drive.google.com/file/d/14hZ0hTx5dM1WgJYewZsn9BkHzEReq-pj/view?usp=sharing) que je poserai au RAG. </br>

### Après 30 minutes

Vous avez testé  plusieurs chunking size<br/>
**-1 point si non fait après 30 minutes**<br/>
**0 au TD si non fait après 1 heure**

### Après 1 heure

Vous avez implémenté Small2Big<br/>
**-1 point si non fait après 1 heure**

# A rendre
- Le notebook de votre RAG
- un CSV avec question,embedding,rag_reply
- un CSV avec chunk,embedding

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

J'ai ajouté, dans src/models.py, une fonction "predict_at_word_level" permettant d'obtenir, au niveau "mot", les predictions du modèle niveau token.

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

- Vendredi 10 octobre, 23:59: le nom de votre modèle HuggingFace sequence -> {"ask_RAG" ou "send_message"}
- Dimanche 12 octobre, 23:59: le nom de votre modèle HuggingFace sentence -> label ["person", "content"] by token
- Dimanche 26 octobre, 23:59: un fichier virtual_assistant.py avec une fonction "call_virtual_assistant(user_query: str) -> dict"

## TD 6: Travail collectif: RAG sur les films

Nous allons développer un nouveau RAG, qui répondra aux questions sur les films. <br/>
Je fournis:
- un [fichier zip](https://drive.google.com/file/d/19udLiCp6HdEEzsq_NQBeImXSmoBKTBov/view?usp=sharing) avec les pages wikipedia de divers films.
- un dataframe avec un set de question - text à trouver dans les sources - réponse attendue
- le code src_rag/ avec un modèle de RAG et un script evaluate.py qui évalue le RAG et pousse les résultats sur mlflow

L'idée est de se répartir les "idées d'amélioration". <br/>
Formez des groupes, d'au moins 4 et max 8 <br/>
Lorsque quelqu'un a amélioré le RAG, il peut en informer les autres qui intègrent son travail. <br/>
Itérez pour avoir le meilleur MRR

A faire:
- Changer config.yml.example en config.yml. Y ajouter une api_key Groq pour pouvoir générer le texte
- Run the code
```
from src_rag import evaluate
evaluate.run_evaluate_retrieval(config={})
```
Ceci doit marcher et vous pousser une expérimentation ML-Flow locale
- Pour votre groupe, ayez votre clone github du cours, ou vous pourrez pousser de nouvelles features

Changer la taille des chunks, overlap, small2Big, embedding de sorte à avoir le meilleur MRR / reply accuracy

## TD 7: Agents

On va étendre le RAG fait au TD précédent en y ajoutant un agent capable de faire des requêtes SQL. <br/>
Je fournis [un CSV](https://docs.google.com/spreadsheets/d/1VFJeNt0Z0ZGhBT1bS7idUC4mwc-SBoXQ/edit?usp=sharing&ouid=105481616530780956748&rtpof=true&sd=true) avec des informations sur les films: réalisateurs, acteurs, genre, **note moyenne**. <br/>
Je fournis [une archive tar.gz](https://drive.google.com/file/d/1UOcWA7m7-wYHsqGRfMe5DDI9t3pEFJKS/view?usp=sharing) avec la fiche wikipedia, en markdown, des films notés dans le CSV.

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
- Agent SQL
  - "Quels sont les 3 films les mieux notés dans le genre action ?"
  - "Dans quels films Leonardo DiCaprio a-t-il joué ?"
  - "Quel film réalisé par Michael Bay a la meilleure note ?"
  - "Lister tous les thrillers réalisés par Quentin Tarantino, triés par date de sortie."
  - "Trouver tous les films qui présentent à la fois Léonardo DirCaprio et Brad Pit dans le casting."  (fautes d'orthographe)
- Agent multi-outils, pouvant faire requête SQL et RAG
  - "Quel est le scenario de Pulp Fiction

- Agent ReAct. Vous pouvez vous inspirer du [notebook agent ReAct](https://github.com/neural-maze/agentic-patterns-course/blob/main/notebooks/planning_pattern.ipynb). Dans l'idée, l'IA peut faire des requêtes SQL, chercher des infos avec le RAG, et elle organise son travail.
  - "Quels sont les 3 meilleurs films d'action futuristes ?"
  - "J'aime les films d'action avec un retournement de situation. Pourriez-vous me recommander quelque chose ?"

Pour l'agent ReAct, vous ne pouvez pas utiliser des toolbox (LangChain, HayStack). Je souhaite que vous implémentiez vous-même le pattern.

### A remettre

- CSV agent_replies.csv avec colonnes question, reply. Les questions sont celles mentionnées juste au dessus
- Le code de votre agent.