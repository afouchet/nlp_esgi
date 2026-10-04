"""Disk-cached sentence embeddings: (text, embedding_config) -> embedding.

Uses joblib.Memory so results survive across runs. `joblib` keys a cached call
on its arguments, so we memoize one text at a time (`_embed_one`). To keep the
fast batched model call, `embed` first computes every cache miss in a single
`FlagModel.encode`, parks the vectors in `_PENDING`, then calls `_embed_one` so
joblib stores each text separately.
"""

from pathlib import Path

import joblib
import numpy as np
from FlagEmbedding import FlagModel


MEMORY = joblib.Memory(Path(".cache") / "embeddings", verbose=0)

_MODELS = {}
_PENDING = {}


def _get_model(model_name, model_kwargs):
    key = (model_name, model_kwargs)
    if key not in _MODELS:
        _MODELS[key] = FlagModel(model_name, **dict(model_kwargs))

    return _MODELS[key]


@MEMORY.cache
def _embed_one(model_name, model_kwargs, text):
    key = (model_name, model_kwargs, text)
    if key in _PENDING:
        return _PENDING.pop(key)

    return _get_model(model_name, model_kwargs).encode([text])[0]


def embed(texts, model_name, **model_kwargs):
    """Embed `texts`, reusing the cache for texts already seen with this config."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    model_kwargs = tuple(sorted(model_kwargs.items()))

    missing = [
        text
        for text in texts
        if not _embed_one.check_call_in_cache(model_name, model_kwargs, text)
    ]

    if missing:
        vectors = _get_model(model_name, model_kwargs).encode(missing)
        for text, vector in zip(missing, vectors):
            _PENDING[(model_name, model_kwargs, text)] = vector

    return np.vstack([_embed_one(model_name, model_kwargs, text) for text in texts])
