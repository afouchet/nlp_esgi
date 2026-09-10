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

# A rendre

Vous enverrez aussi votre prompt, ainsi que le code pour parser la réponse de ChatGPT et avoir une fonction:
list(titres de videos) -> list(noms de comiques)

Vous enverrez un CSV video_name,comic_names sortant les prédictions de votre few-shot learner.

Vous enverrez votre rapport montrant les expérimentations et ce qui a marché.

Envoyer ces documents via MyGES.