# NLP TD 2: Transfer learning for named-entity recognition

## Installation

⚠️ Si vous avez des problèmes d'install ou de GPU, utilisez __Google Colab__

```bash
uv sync
```

⚠️ **Utilisateurs Mac (M1/M2/M3, GPU MPS)** : l'entraînement plante avec l'erreur
`scaled_dot_product_attention for MPS does not support dropout`. <br/>
Solution : charger le modèle avec `attn_implementation="eager"` :

```python
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=2, attn_implementation="eager",
)
```


**M'appeler si "uv sync" ne marche pas**

## Part 1: Named-entity recognition

Dans ce TD, on va fine-tune un modèle BERT pour identifier des noms de personnes dans du texte en français. <br/>
Nous l'utiliserons ensuite sur nos videos France Inter.

Dans le notebooks/TD2_transfer_learning.ipynb, vous trouverez le code pour:
- Extraire d'un fichier MultiNERD English une serie de phrase, dont les mots sont labelisés 1 si le mot est un nom de personne, 0 sinon.
- Fine-tune le modèle DistilBert en gelant la 1ère couche.

A faire:
- Faire tourner le notebook sur le MultiNERD EN, avec le modèle DistilBERT, vérifier que ça marche
- Adapter ce code en français (données MultiNERD FR, modèle CamemBERT ou autre). Si vous avez de mauvaises performances (< 98.5%) il y a des éléments à modifier.
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

Trouver data/raw/train_named_entity_recognition.csv un dataset avec les noms de video, et le label pour chaque token. <br/>
(Remarque: le modèle peut être entraîné sur MultiNERD, puis le dataset France Inter).

## TODO

1. Run le notebook sur multiNERD en anglais
2. Prendre un modèle CamemBERT et run sur multiNERD en français. Avoir 99%+ accuracy
3. Faire fonction predict at word level
4. Prédire sur le dataset France Inter, disponible, dans le zip, dans data/raw/train_named_entity_recognition.csv ou sur [ce lien](https://drive.google.com/file/d/1-7-esuAMBDzjN2DQsUD9Up7z7bIRwahL/view?usp=sharing)
5. Uploader votre modèle sur HuggingFace

## !! Timeline !! (**Points en moins si non respectée**)

### Après 30 minutes

Le notebook TD2_transfer_leaning fonctionne sur votre ordinateur
**-1 point si non fait après 30 minutes**<br/>
**0 au TD si non fait après 1 heure**

## Après 1 heure

Vous avez entraîné un modèle français prédisant si un mot est un nom de personnes avec une accuracy > 98.5%+ en test<br/>

## A Rendre

- Votre fonction predict_at_word_level(text_split_into_words: list[str], model, tokenizer) -> list[int]
__dans un fichier src/predict_at_word_level.py__
- Vos prediction sur le jeu de données France Inter
- Le nom de votre modèle sur HuggingFace

Créer 1 archive zip avec tous vos fichiers. <br/>
**Vérifier qu'elle contient tous les fichiers!** Dézipper là dans un autre dossier et vérifier que le code fonctionne avec seulement les fichiers du zip. <br/>
