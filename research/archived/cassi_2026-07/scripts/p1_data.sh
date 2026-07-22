#!/usr/bin/env bash
# =============================================================================
# P1 — Data & environments (paper_plan_v2 §16 P1, week 1)
#
# Goal: stage NQ/HotpotQA/MuSiQue train samples + dev sets, the OOD/control eval
# sets (Bamboogle, 2Wiki, MATH-500, AIME 2025, GAIA text-only 103 dev,
# BrowseComp-Plus), build the Search-R1 wiki retrieval index, run the §5.6
# decontamination pass, and emit the dataset manifest.
#
# Done-criterion (§16 P1, quoted):
#   "✅ Done: dataset manifest with counts + split hashes committed."
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

banner "P1 — data & environments"
require_cmd python3 "runs the data drivers"
python3 -c "import datasets" 2>/dev/null \
    || pending "requires 'datasets' (pip install -r requirements-cpu.txt)"

mkdir -p "${DATA_DIR}"

# ------------------------------------------- 1. HF datasets (train / dev / eval)
banner "P1.1 — download + sample HF datasets (seed 42; frozen subsamples chosen once)"
python3 "${CASSI_ROOT}/scripts/download_data.py" \
    || echo "[p1] some sets missing (gated GAIA needs 'huggingface-cli login' + accepted terms) — rerun after fixing; continuing with what exists"

# ----------------------------------------------- 2. Search-R1 wiki index (§19)
# Recipe (§19 retriever row): E5 + Wikipedia-21M dump, identical to all baselines;
# BM25 fallback; one Qwen3-Embedding-0.6B ablation for robustness (P9).
banner "P1.2 — Search-R1 retrieval index (documented recipe)"
if [ -d "${THIRD_PARTY}/Search-R1" ]; then
    cat <<EOF
Search-R1 recipe (run from ${THIRD_PARTY}/Search-R1; ~70GB disk, GPU for E5 encoding):

  # 1) corpus + prebuilt E5 index (Search-R1's own download script):
  python scripts/download.py --save_path ${DATA_DIR}/searchr1_index

  # 2) launch the local retriever server (keep running during P2-P9 rollouts):
  #    TODO(GPU): needs 1 GPU for the E5 encoder; BM25 mode is CPU-only.
  eval \$(${GPU_ACQUIRE} 1)
  python search_r1/search/retrieval_server.py \\
      --index_path ${DATA_DIR}/searchr1_index/e5_Flat.index \\
      --corpus_path ${DATA_DIR}/searchr1_index/wiki-18.jsonl \\
      --retriever_name e5 --topk 3 --port 8000
  # (BM25 fallback: --retriever_name bm25 --index_path .../bm25)
  # release when the server is torn down: ${GPU_RELEASE}

NOTE: exact script paths follow the pinned Search-R1 checkout — verify against
      ${THIRD_PARTY}/Search-R1/README.md (pins: configs/cassi.yaml).
EOF
    if [ ! -d "${DATA_DIR}/searchr1_index" ]; then
        echo "TODO(GPU): index not built yet — run the recipe above on the GPU node."
    fi
else
    echo "PENDING: third_party/Search-R1 not cloned — run scripts/p0_setup.sh first (index recipe printed then)."
fi

# ------------------------------------------------- 3. ALFWorld task list (P0 clone)
banner "P1.3 — ALFWorld"
if [ -d "${THIRD_PARTY}/verl-agent" ]; then
    cat <<EOF
ALFWorld train/eval task lists ship with verl-agent (GiGPO official harness, §19):
  pip install alfworld && alfworld-download          # game files (~1.6GB)
  # task splits: ${THIRD_PARTY}/verl-agent (see its ALFWorld example configs)
EOF
else
    echo "PENDING: third_party/verl-agent not cloned — run scripts/p0_setup.sh first."
fi

# -------------------------------- 4. GAIA + BrowseComp-Plus staging (documented)
banner "P1.4 — GAIA text-only + BrowseComp-Plus (OOD transfer, §5.1)"
cat <<EOF
GAIA text-only 103-Q dev  : downloaded by download_data.py (gated — needs
  'huggingface-cli login' and accepting terms at hf.co/datasets/gaia-benchmark/GAIA).
  Validation-used-as-test is DISCLOSED in the paper (§5.1); test set is hidden.

BrowseComp-Plus (830 Qs + fixed 100K-doc corpus, ACL 2026): staged SEPARATELY —
  1) fetch the released queries + 100K-doc corpus (TODO(P1-verify): exact hub id /
     release URL from the BrowseComp-Plus paper artifacts);
  2) build a LOCAL retriever over the corpus (same E5/BM25 tooling as P1.2 — the
     local-retrieval tool type is what makes executor transfer well-posed, §5.1);
  3) store under ${DATA_DIR}/browsecomp_plus/{queries.jsonl,corpus/,index/}.
Used at P9 (E2 transfer) only — staging may lag until W7 without blocking P2-P6.
EOF

# --------------------------------------------------- 5. decontamination (§5.6)
banner "P1.5 — decontamination: train prompts vs ALL eval sets (13-gram + MinHash)"
TRAIN_FILES=()
for f in nq_train hotpotqa_train musique_train; do
    [ -f "${DATA_DIR}/${f}.jsonl" ] && TRAIN_FILES+=("${DATA_DIR}/${f}.jsonl")
done
EVAL_FILES=()
for f in hotpotqa_dev_frozen musique_dev_frozen bamboogle 2wikimultihopqa_dev \
         math500 aime2025 gaia_dev_textonly; do
    [ -f "${DATA_DIR}/${f}.jsonl" ] && EVAL_FILES+=("${DATA_DIR}/${f}.jsonl")
done
if [ "${#TRAIN_FILES[@]}" -eq 0 ] || [ "${#EVAL_FILES[@]}" -eq 0 ]; then
    pending "train or eval JSONLs missing — fix P1.1 downloads first"
fi
python3 "${CASSI_ROOT}/scripts/decontaminate.py" \
    --train "${TRAIN_FILES[@]}" \
    --eval "${EVAL_FILES[@]}" \
    --report "${DATA_DIR}/decontamination_report.json" \
    --drop
echo "[p1] training uses the *.decontaminated.jsonl files from here on (§5.6 protocol step 1)"

# ------------------------------------------------------ 6. manifest (done-criterion)
banner "P1.6 — dataset manifest (P1 done-criterion)"
python3 "${CASSI_ROOT}/scripts/dataset_manifest.py"

banner "P1 DONE — commit experiments/results/dataset_manifest.csv"
echo "Note: BrowseComp-Plus corpus + Search-R1 index are large local artifacts (gitignored);"
echo "their identity is captured by the manifest + pins, not by git."
