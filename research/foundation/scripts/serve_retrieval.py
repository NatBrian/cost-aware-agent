"""Retrieval server over the rescued Search-R1 index (F1; exercised at I2 smoke).

E5-base-v2 query encoder + FAISS flat index (64G, 768-dim float32, 21M wiki-18
passages — dims verified: 64559075373 bytes / 21,015,324 rows = 3072 = 768*4).

Usage (GPU box, after `eval $(gpu_acquire.sh 1)` if using CUDA for encoding):
  .venv/bin/python scripts/serve_retrieval.py [--cpu] [--port 8001]
Heavy deps (torch, faiss, transformers, fastapi/uvicorn) are imported lazily so
CPU test envs never need them.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import FOUNDATION_ROOT, load_config

E5_MODEL = "intfloat/e5-base-v2"


def build_corpus_offsets(corpus_path: Path, offsets_path: Path) -> list[int]:
    """Byte offset per line so passages load lazily (14G file, don't hold in RAM)."""
    if offsets_path.exists():
        return json.loads(offsets_path.read_text())
    offsets, pos = [], 0
    with open(corpus_path, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    offsets_path.write_text(json.dumps(offsets))
    return offsets


class Retriever:
    def __init__(self, index_dir: Path, cpu: bool = False):
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        # faiss segfaults in its SIMD kernels with the default 192 OMP threads
        # on this box (kernel log 2026-07-22); 16 threads is stable, ~4s/query
        # cold. The old cassi torch_retrieval_server was the same bug's workaround.
        faiss.omp_set_num_threads(16)

        self.corpus_path = index_dir / "wiki-18.jsonl"
        self.offsets = build_corpus_offsets(self.corpus_path,
                                            index_dir / "wiki-18.offsets.json")
        self.index = faiss.read_index(str(index_dir / "e5_Flat.index"))
        assert self.index.ntotal == len(self.offsets), \
            f"index rows {self.index.ntotal} != corpus lines {len(self.offsets)}"
        self.device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(E5_MODEL)
        self.model = AutoModel.from_pretrained(E5_MODEL).to(self.device).eval()
        self._torch = torch

    def encode(self, query: str):
        import torch.nn.functional as F
        with self._torch.no_grad():
            batch = self.tok(f"query: {query}", return_tensors="pt",
                             truncation=True, max_length=256).to(self.device)
            out = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            emb = (out * mask).sum(1) / mask.sum(1)          # mean pooling
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32")

    def passage(self, row: int) -> dict:
        with open(self.corpus_path, "rb") as f:
            f.seek(self.offsets[row])
            rec = json.loads(f.readline())
        title, text = "", rec.get("contents", "")
        if "\n" in text:
            title, text = text.split("\n", 1)
        return {"title": rec.get("title", title.strip('"')), "text": text.strip()}

    def search(self, query: str, top_k: int) -> list[dict]:
        # The similarity SCORE is returned, not discarded. S1 (2026-07-31) found
        # retrieval productivity among the strongest gold-free predictors of
        # eventual failure, and the score is the most direct form of it: "is this
        # query finding anything?" is exactly the quit signal FOUNDATION-2 needs.
        # It was thrown away here (`_, idx = ...`) for the whole first run.
        dist, idx = self.index.search(self.encode(query), top_k)
        out = []
        for d, i in zip(dist[0], idx[0]):
            if i < 0:
                continue
            out.append(self.passage(int(i)) | {"score": float(d)})
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    from fastapi import FastAPI
    import uvicorn
    from pydantic import BaseModel

    cfg = load_config()
    port = args.port or int(cfg["retrieval"]["endpoint"].rsplit(":", 1)[1])
    retr = Retriever(FOUNDATION_ROOT / cfg["retrieval"]["index_dir"], cpu=args.cpu)

    app = FastAPI()

    class Q(BaseModel):
        query: str
        top_k: int = cfg["retrieval"]["top_k"]

    @app.post("/search")
    def search(q: Q):
        return {"results": retr.search(q.query, q.top_k)}

    @app.get("/health")
    def health():
        return {"ok": True, "ntotal": retr.index.ntotal}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
