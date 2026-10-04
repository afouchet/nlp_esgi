from pathlib import Path
import mlflow
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import sleep
import yaml

from src_rag import models

from FlagEmbedding import FlagModel

CONF = yaml.safe_load(open("config.yml"))

FOLDER = Path("data") / "raw" / "movies" / "wiki"
FILENAMES = [
    FOLDER / title for title in ["Inception.md", "The Dark Knight.md", "Deadpool.md", "Fight Club.md", "Pulp Fiction.md"]
]
DF = pd.read_csv("data/raw/movies/questions.csv", sep=";") 

ENCODER = SentenceTransformer('all-MiniLM-L6-v2')


def _load_ml_flow(conf):
    mlflow.set_experiment("RAG_Movies_clean")


_load_ml_flow(CONF)

def run_evaluate_retrieval(config, rag=None):
    rag = rag or models.get_model(config)
    score = evaluate_retrieval(rag, FILENAMES, DF.dropna())

    description = str(config["model"])
    _push_mlflow_result(score, config, description)
    
    return rag


def run_evaluate_reply(config, rag=None):
    rag = rag or models.get_model(config)
    indexes = range(2, len(DF), 10)
    score = evaluate_reply(rag, FILENAMES, DF.iloc[indexes])

    description = str(config["model"])
    _push_mlflow_result(score, config, description)
    
    return rag


def _push_mlflow_result(score, config, description=None):
    with mlflow.start_run(description=description):
        df = score.pop("df_result")
        mlflow.log_table(df, artifact_file="df.json")
        mlflow.log_metrics(score)

        config_no_key = {
            key: val for key, val in config.items() if not key.endswith("_key")
        }

        mlflow.log_dict(config_no_key, "config.json")


def evaluate_reply(rag, filenames, df):
    rag.load_files(filenames)

    replies = []
    for question in tqdm(df["question"]):
        replies.append(rag.reply(question))
        # Not to many requests to groq
        sleep(2)

    df["reply"] = replies
    df["sim"] = df.apply(lambda row: calc_semantic_similarity(row["reply"], row["expected_reply"]), axis=1)
    df["is_correct"] = df["sim"] > .7

    return {
        "reply_similarity": df["sim"].mean(),
        "percent_correct": df["is_correct"].mean(),
        "df_result": df[["question", "reply", "expected_reply", "sim", "is_correct"]],
    }


def evaluate_retrieval(rag, filenames, df_question):
    rag.load_files(filenames)
    ranks = []
    for _, row  in df_question.iterrows():
        chunks = rag._get_context(row.question)
        try:
            rank = next(i for i, c in enumerate(chunks) if row.text_answering in c)
        except StopIteration:
            rank = 0

        ranks.append(rank)
        
    df_question["rank"] = ranks
            
    mrr = np.mean([0 if r == 0 else 1 / r for r in ranks])

    return {
        "mrr": mrr,
        "nb_chunks": len(rag.get_chunks()),
        "df_result": df_question[["question", "text_answering", "rank"]],
    }


def calc_acceptable_chunks(chunks, text_to_find):
    acceptable_chunks = []
    for answer in text_to_find:
        chunks_ok = set(i for i, chunk in enumerate(chunks) if answer in chunk)
        acceptable_chunks.append(chunks_ok)

    return acceptable_chunks


def calc_mrr(sim_score, acceptable_chunks, top_n=5):
    ranks = []
    for this_score, this_acceptable_chunks in zip(sim_score, acceptable_chunks):
        indexes = reversed(np.argsort(this_score))
        try:
            rank = 1 + next(i for i, idx in enumerate(indexes) if idx in this_acceptable_chunks)
        except StopIteration:
            rank = len(this_score) + 1
        
        ranks.append(rank)
        
    return {
        "mrr": sum(1 / r if r < top_n + 1 else 0 for r in ranks) / len(ranks),
        "ranks": ranks,
    }


def calc_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    """
    Calculate semantic similarity between generated and reference answers.
    
    Args:
        generated_answer: The answer produced by the RAG system
        reference_answer: The expected or ground-truth answer
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    # Generate embeddings for both texts
    embeddings = ENCODER.encode([generated_answer, reference_answer])
    generated_embedding = embeddings[0].reshape(1, -1)
    reference_embedding = embeddings[1].reshape(1, -1)
    similarity = cosine_similarity(generated_embedding, reference_embedding)[0][0]
    return float(similarity)


if __name__ == "__main__":
    model_config = {"chunk_size": 512}
    # run_evaluate_retrieval({"model": model_config})
    run_evaluate_reply({"model": model_config})

