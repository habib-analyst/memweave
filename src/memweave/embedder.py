"""Text embedder backed by all-MiniLM-L6-v2 (CPU).

Uses sentence-transformers when available and falls back to a manual
transformers mean-pooling implementation. Embeddings are L2-normalized so
cosine similarity is a dot product.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = os.environ.get(
    "MEMWEAVE_MODEL_PATH",
    str(
        Path.home()
        / "workspace/goals/github-portfolio-linkedin-presence-buildout"
        / "hidden_files/builds/promptgate/models/all-MiniLM-L6-v2"
    ),
)


class Embedder:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._st_model = None
        self._hf_model = None
        self._tokenizer = None
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_path)
        except Exception:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._hf_model = AutoModel.from_pretrained(self.model_path)
            self._hf_model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self._st_model is not None:
            emb = self._st_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(emb, dtype=np.float32)
        import torch

        enc = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out = self._hf_model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = (summed / counts).numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        return emb / norms
