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

1. Faites une fonction video_name: str -> comic_names: list[str] qui extrait les noms de comiques d'un nom de video. <br/>
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

Envoyer ces documents via MyGES.