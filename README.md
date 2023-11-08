# NLP TD 1

L'objectif de ce TD est de créer un modèle "nom de vidéo" -> "nom du comique" si c'est la chronique d'un comique, None sinon. On formulera le problème en 2 tâches d'apprentissage:
- une de text classification pour savoir si la vidéo est une chronique comique
- une de named-entity recognition pour reconnaître les noms dans un texte
En assemblant les deux, nous obtiendrons notre modèle.

Dans ce TD, on s'intéresse surtout à la démarche. Pour chaque tâche:
- Bien poser le problème
- Avoir une baseline
- Experimenter diverses features et modèles
- Garder une trace écrite des expérimentations dans un rapport. Dans le rapport, on s'intéresse plus au sens du travail effectué (quelles expérimentations ont été faites, pourquoi, quelles conclusions) qu'à la liste de chiffres.
- Avoir une codebase clean, permettant de reproduire les expérimentations.

On se contentera de méthodes pré-réseaux de neurones. Nos features sont explicables et calculables "à la main".

La codebase doit fournir les entry points suivant:
- Un entry point pour train sur une "task", prenant en entrée le path aux données de train et dumpant le modèle dans "model_dump" 
```
python src/main.py train --task=is_comic_video --input_filename=data/raw/train.csv --model_dump_filename=models/model.json
```
- Un entry point pour predict sur une "task", prenant en entrée le path au modèle dumpé, le path aux données à prédire et outputtant dans un csv les prédictions
```
python src/main.py predict --task=is_comic_video --input_filename=data/raw/test.csv --model_dump_filename=models/model.json --output_filename=data/processed/prediction.csv
```
- Un entry point pour evaluer un modèle sur une "task", prenant en entrée le path aux données de train.
```
python src/main.py evaluate --task=is_comic_video --input_filename=data/raw/train.csv
```

Les "tasks":
- "is_comic_video": prédit si la video est une chronique comique
- "is_name": prédit si le mot est un nom de personne
- "find_comic_name": si la video est une chronique comique, sort le nom du comique

## Dataset

Dans [ce lien](https://docs.google.com/spreadsheets/d/1x6MITsoffSq7Hs3mDIe1YLVvpvUdcsdUBnfWYgieH7A/edit?usp=sharing), on a un CSV avec 3 colonnes:
- video_name: le nom de la video
- is_comic: est-ce une chronique humoristique
- is_name: une liste de taille (nombre de mots dans video_name) valant 1 si le mot est le nom d'une personne, 0 sinon
- comic: le nom du comique si c'est une chronique humoristique

## Partie 1: Text classification: prédire si la vidéo est une chronique comique

### Tasks

- Run la pipeline evaluate "python src/main.py evaluate". Elle devrait marcher sur la task "is_comic_video"
- Essayed d'optimiser en ajoutant / optimisant les features faites dans "make_features":
    - Regarder les features disponibles dans sklearn.feature_extraction.text. Lesquelles semblent adaptées ?
    - Regarder NLTK. télécharger le corpus français. La librairie permettra de retirer les stopwords et de stemmer les mots
- Essayer d'autres modèles (regression logistic, Bayesian, etc)
- Ecrire le rapport (dans report/td1.{your choice}) avec:
   - Les a-priori que vous aviez sur les features & modèles utiles ou non
   - Quels ont été les apports individuels de chacune de ces variation ?
   - Conclusion sur le bon modeling (en l'état)
- Adapter "src/" pour que les pipelines "train" et "predict" marchent

## Partie 2: Named-entity recognition: Reconnaître les noms de personne dans le texte

### Tasks

- Adapter "src/" pour que la pipeline "evaluate" marche sur la task "is_name", avec un modèle constant (fournissant la baseline)
- Comment definir les features pour un mot ?
    - Est-ce que ce sont les mots avant ? après ?
    - Comment traite-t-on la ponctuation ?
    - On peut définir des features "is_final_word", "is_starting_word", "is_capitalized"
    - On peut aussi définir des balises pour repérer les choses importantes. Par exemple, je peux transformer "L'humeur de Marina Rollman" en "<START> <MAJ> l'humeur de <MAJ> marina <MAJ>rollman <END>" en utilisant les balises <START> pour identifier le début d'une phrase, <END> fin d'une phrase, <MAJ> si la première lettre est en majuscule
    - (optionel) pour cette tâche, un "pos_tagger" (part-of-speech tagger: détermine, pour chaque mot, sa classe: nom, verbe, adjectif, etc). NLTK n'en fournit pas en français, mais on peut utiliser le POS tagger de Standford https://nlp.stanford.edu/software/tagger.shtml#About
- Ecrire le rapport (dans report/td1.{your choice}) avec:
   - Les a-priori que vous aviez sur les features & modèles utiles ou non
   - Quels ont été les apports individuels de chacune de ces variation ?
   - Conclusion sur le bon modeling (en l'état)
- Adapter "src/" pour que les pipelines "train" et "predict" marchent

## Partie 3: Assembler les modèles

- Adapter "src/" pour que la pipeline "evaluate" marche sur la task "find_comic_name", avec un modèle constant (fournissant la baseline)
- Assembler les 2 modèles précédents. Quelle performance ?
- Essayer une autre façon de résoudre le problème. Par exemple, un modèle named-entity recognition donne les noms qu'il a trouvé. Pour chaque nom, on associe la liste des videos où il apparaît. On entraîne un autre prédicteur "liste videos où nom apparaît" -> est-ce le nom d'un comique
- Adapter "src/" pour que les pipelines "train" et "predict" marchent
- Terminer le rapport

# Troubleshooting

Quelques problèmes rencontrés et leur solution

## ImportError "No module name src"

Python n'a pas src dans son path.
Pour l'instant, on n'a pas trouvé de solution  qui marche dans tous les cas.

Solution 1:
Dans le root folder (avant src/)
```
conda develop .
```

Solution 2: change "from src.data..." to "from data..."
(Dans le "pythonpath" (là où python va chercher le code), il est normal de regarder "src/")

Solution 3: PyCharm
Definir le dossier root "nlp_esgi" comme source de code (clic droit sur le dossier, "set as source root")

# NLP TD 2:

Dans ce TD, nous allons créer et optimiser un réseau de neurones récurrents (RNN) avec PyTorch dans la partie 1
Dans la partie 2, nous allons utiliser ce RNN pour une tâche de named-entity recognition
Ensuite, vous devrez utiliser le modèle de named-entity recognition à notre problème de reconnaissance de nom de comiques dans les videos youtube (la partie 2, named-entity recognition)

Evaluation:
- Rapport sur les itérations pour le RNN pour named-entity recognition
- Test du nouveau modèle "Named-entity recognition" sur le jeu de données test des videos youtube.



