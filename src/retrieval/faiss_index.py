"""Dense semantic retrieval index (FAISS).

Owner:    Raahim Irfan (SSOT §2)
Contract: C — FAISSIndex (SSOT §6C)
Phase:    2 — Hybrid Retrieval (IMPLEMENTATION.md §2.2)
STATUS:   core

Encodes papers with the configured embedding model and serves
nearest-neighbour search via a FAISS index. The model and dimension come from
src/app/config.py (EMBEDDING_MODEL, EMBEDDING_DIM) — currently SPECTER2,
768-dim. SPECTER2 loads via the `adapters` library, not a plain
SentenceTransformer.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL, FAISS_DIR, FAISS_NLIST, FAISS_NPROBE

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Dense index. Contract C — the public signatures are fixed (SSOT §6C)."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self.index = None
        self.paper_ids: list[str] = []
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        """Load SPECTER2 base model and activate the proximity adapter."""
        if self._model is not None:
            return
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        logger.info("Loading SPECTER2 base model from %s", EMBEDDING_MODEL)
        self._tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self._model = AutoAdapterModel.from_pretrained(EMBEDDING_MODEL)

        # Load the proximity adapter — must call set_active_adapters separately;
        # set_active=True on load_adapter() alone does not engage the adapter
        # for the forward pass (silent "no adapter activated" warning).
        self._model.load_adapter(
            "allenai/specter2",
            source="hf",
            load_as="proximity",
            set_active=True,
        )
        self._model.set_active_adapters("proximity")
        self._model.to(self._device)
        self._model.eval()
        logger.info("SPECTER2 loaded on %s", self._device)

    def _encode_corpus(
        self, papers: list[dict], batch_size: int
    ) -> tuple[list[str], np.ndarray]:
        """Encode all papers; return (paper_ids, embeddings float32 L2-normed)."""
        import torch.nn.functional as F
        from tqdm import tqdm

        self._load_model()
        paper_ids: list[str] = [p["arxiv_id"] for p in papers]
        texts = [f"{p['title']} {p['abstract']}" for p in papers]

        all_embeddings: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # CLS token — do not use mean pooling for SPECTER2
            vecs = outputs.last_hidden_state[:, 0, :]
            vecs = F.normalize(vecs, p=2, dim=1)
            all_embeddings.append(vecs.cpu().numpy())

        return paper_ids, np.vstack(all_embeddings).astype(np.float32)

    def _save_embeddings(
        self, paper_ids: list[str], embeddings: np.ndarray
    ) -> None:
        """Persist paper_ids and raw embeddings so BERTopic can reuse them."""
        import json

        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(FAISS_DIR / "embeddings.npy", embeddings)
        with open(FAISS_DIR / "paper_ids.json", "w") as f:
            json.dump(paper_ids, f)
        logger.info("Saved %d embeddings to %s", len(paper_ids), FAISS_DIR)

    def _build_ivf_index(self, embeddings: np.ndarray) -> "faiss.Index":
        """Build an IVFFlat index, training on GPU if available."""
        import faiss

        n = len(embeddings)
        # nlist: rule of thumb ≈ sqrt(N); config upper-bound is 1024
        nlist = min(FAISS_NLIST, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        cpu_index = faiss.IndexIVFFlat(
            quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT
        )

        if self._device == "cuda" and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_index.train(embeddings)
            gpu_index.add(embeddings)
            cpu_index = faiss.index_gpu_to_cpu(gpu_index)
            logger.info("IVFFlat trained and populated on GPU (nlist=%d, n=%d)", nlist, n)
        else:
            cpu_index.train(embeddings)
            cpu_index.add(embeddings)
            logger.info("IVFFlat trained and populated on CPU (nlist=%d, n=%d)", nlist, n)

        return cpu_index

    def build(self, papers: list[dict], batch_size: int = 128) -> None:
        """Encode papers and build + persist the FAISS index."""
        paper_ids, embeddings = self._encode_corpus(papers, batch_size)
        self._save_embeddings(paper_ids, embeddings)
        self.index = self._build_ivf_index(embeddings)
        self.paper_ids = paper_ids

        import faiss as _faiss
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        _faiss.write_index(self.index, str(FAISS_DIR / "vectors.index"))
        logger.info("FAISS index saved to %s", FAISS_DIR / "vectors.index")

    def load(self) -> None:
        """Load the persisted index from config.FAISS_DIR."""
        # TODO(raahim, phase-2): read the index; optionally move it to GPU.
        raise NotImplementedError

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(arxiv_id, cosine_similarity), ...] sorted descending."""
        # TODO(raahim, phase-2): encode query, search, drop -1 slots.
        raise NotImplementedError

    def search_by_embedding(
        self, embedding: np.ndarray, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """'More Like This' — nearest neighbours of a precomputed vector."""
        # TODO(raahim, enhancement): powers the More Like This feature (SSOT §10).
        raise NotImplementedError
