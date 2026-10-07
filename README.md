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

⚠️ En passant de 5 documents à 100, on voit déjà qu'il est bien plus dur de faire un bon RAG ⚠️

⚠️  
Le fait d'encoder les chunks prend du temps.  
1. Commencer par peu de documents. Vous pouvez coder vos différentes méthodes de chunking et embedding.
2. Utilisez du cache pour "(text, embedding_config) -> embedding"  
Dans votre embedding config, il peut y avoir:
- Le modèle embedder
- S'il y a un pré-traitement du texte (HyDE, FAQ,...)
⚠️

Je fournis un CSV questions.csv, ne s'intéressant qu'aux 5 films précisés dans evaluate.py.
Puis un CSV questions_long.csv, s'intéressant à tous les films
Puis un CSV questions_test.csv, avec des questions sans la réponse attendue

A faire:
- Changer .env.example en .env. Y ajouter une api_key Groq / OpenRouter pour pouvoir générer le texte
- Run the code
```
from src_rag import evaluate
evaluate.run_evaluate_retrieval(config={})
```
Ceci doit marcher et vous pousser une expérimentation ML-Flow locale
- Pour votre groupe, ayez votre clone github du cours, ou vous pourrez pousser de nouvelles features

Changer la taille des chunks, overlap, small2Big, embedding de sorte à avoir le meilleur MRR / reply accuracy

A rendre:
- Code dans src_rag/
- Rapport sur les différentes méthodes utilisées, leurs performances (MRR, reply similarity)
- un chunk.parquet avec les chunks des documents et leur embedding
- un question.parquet avec les question de questions_test, leur embedding, et la réponse de votre RAG

Créer 1 archive zip avec tous vos fichiers. <br/>
Envoyer cette archive zip via MyGES.