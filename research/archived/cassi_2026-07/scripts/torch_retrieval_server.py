#!/usr/bin/env python3
"""Drop-in replacement for Search-R1's retrieval_server.py with torch-GPU exact search.

WHY THIS EXISTS (2026-07-22, see HANDOFF machine note): on this box the pinned
server's two search paths both fail —
  * CPU faiss: brute-force over the 61GB flat index costs ~35 s/query
    (memory-bandwidth-bound; ~77K queries needed for round-0 collection alone);
  * GPU faiss: the pip `faiss-gpu-cu12` wheel has no kernel image for the
    L20X ("CUDA error 209").

This server keeps EVERYTHING scientifically identical — same E5 encoder code
(imported from the pinned Search-R1 checkout, not copied), same corpus, same
EXACT inner-product top-k over the same vectors (reconstructed from the same
faiss flat index) — and only swaps the search implementation: the 21M x 768
matrix sits on the GPU in fp16 and a query is one chunked matmul (~tens of ms).
fp16 storage matches what the pinned server itself uses on GPU
(GpuMultipleClonerOptions.useFloat16 = True), so precision is not worse than
the reference GPU path.

API contract (identical to the pinned server, including the response shapes):
  POST /retrieve  {"queries": [...], "topk": k, "return_scores": bool}
  -> {"result": [[{"document": {...}, "score": s}, ...] ...]}   (scores on)
  -> {"result": [[{...doc...}, ...] ...]}                        (scores off)
(The upstream unpack bug for return_scores=false does NOT exist here.)

Run (GPU expected; index load takes several minutes):
  CUDA_VISIBLE_DEVICES=6 python scripts/torch_retrieval_server.py \
      --index_path data/searchr1_index/e5_Flat.index \
      --corpus_path data/searchr1_index/wiki-18.jsonl \
      --retriever_model intfloat/e5-base-v2 --topk 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

CASSI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CASSI_ROOT / "third_party" / "Search-R1"))

import faiss                      # noqa: E402  (CPU faiss — only used to READ the index)
import numpy as np                # noqa: E402
import torch                      # noqa: E402
import uvicorn                    # noqa: E402
from fastapi import FastAPI       # noqa: E402
from pydantic import BaseModel    # noqa: E402

# Reuse the pinned encoder/corpus code verbatim (never modified — §19 manifest rule).
from search_r1.search.retrieval_server import Encoder, load_corpus, load_docs  # noqa: E402

RECONSTRUCT_CHUNK = 1_000_000     # rows per CPU->GPU upload chunk
MATMUL_CHUNK = 4_000_000          # corpus rows per matmul chunk (bounds temp memory)


class TorchFlatRetriever:
    def __init__(self, index_path: str, corpus_path: str, model_path: str,
                 model_name: str = "e5", topk: int = 3):
        print(f"[torch-retriever] reading faiss index {index_path} (CPU, minutes)...",
              flush=True)
        index = faiss.read_index(index_path)
        n, d = index.ntotal, index.d
        print(f"[torch-retriever] {n} vectors x {d} dims -> GPU fp16 "
              f"({n * d * 2 / 1e9:.1f} GB)", flush=True)
        self.matrix = torch.empty((n, d), dtype=torch.float16, device="cuda")
        for start in range(0, n, RECONSTRUCT_CHUNK):
            cnt = min(RECONSTRUCT_CHUNK, n - start)
            chunk = index.reconstruct_n(start, cnt)              # np fp32
            self.matrix[start:start + cnt] = torch.from_numpy(chunk).to(
                "cuda", dtype=torch.float16, non_blocking=True)
        del index
        print("[torch-retriever] loading corpus...", flush=True)
        self.corpus = load_corpus(corpus_path)
        self.encoder = Encoder(model_name=model_name, model_path=model_path,
                               pooling_method="mean", max_length=256, use_fp16=True)
        self.topk = topk
        # warmup matmul
        self._topk_scores(torch.zeros((1, d), dtype=torch.float16, device="cuda"), 1)
        print("[torch-retriever] READY", flush=True)

    @torch.no_grad()
    def _topk_scores(self, q: torch.Tensor, k: int):
        """Exact IP top-k via chunked matmul; returns (scores, idxs) on CPU."""
        best_s, best_i = None, None
        n = self.matrix.shape[0]
        for start in range(0, n, MATMUL_CHUNK):
            block = self.matrix[start:start + MATMUL_CHUNK]
            s = (q @ block.T).float()                            # (B, chunk)
            k_eff = min(k, s.shape[1])
            cs, ci = torch.topk(s, k_eff, dim=1)
            ci = ci + start
            if best_s is None:
                best_s, best_i = cs, ci
            else:
                cat_s = torch.cat([best_s, cs], dim=1)
                cat_i = torch.cat([best_i, ci], dim=1)
                best_s, sel = torch.topk(cat_s, min(k, cat_s.shape[1]), dim=1)
                best_i = torch.gather(cat_i, 1, sel)
        return best_s.cpu().numpy(), best_i.cpu().numpy()

    def batch_search(self, queries: List[str], topk: Optional[int] = None):
        k = topk or self.topk
        emb = self.encoder.encode(queries, is_query=True)        # np fp32, normalized
        q = torch.from_numpy(emb).to("cuda", dtype=torch.float16)
        scores, idxs = self._topk_scores(q, k)
        results, out_scores = [], []
        for row_i, row_s in zip(idxs, scores):
            docs = load_docs(self.corpus, row_i.tolist())
            results.append(docs)
            out_scores.append([float(s) for s in row_s])
        return results, out_scores


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
retriever: TorchFlatRetriever = None  # set in main


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    results, scores = retriever.batch_search(request.queries, request.topk)
    resp = []
    for i, single in enumerate(results):
        if request.return_scores:
            resp.append([{"document": doc, "score": s}
                         for doc, s in zip(single, scores[i])])
        else:
            resp.append(single)
    return {"result": resp}


def main():
    global retriever
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--corpus_path", required=True)
    ap.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    ap.add_argument("--retriever_name", default="e5")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    retriever = TorchFlatRetriever(args.index_path, args.corpus_path,
                                   args.retriever_model, args.retriever_name,
                                   args.topk)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
