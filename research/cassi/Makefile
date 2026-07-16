# CASSI figures & tables — paper_plan_v2 §16 P10.
# `make figures tables` (or `make all`) regenerates every F1–F6 PDF and T1–T5
# .tex from experiments/results/*.csv. No hand-edited numbers: each artifact is
# produced by exactly one script; a script whose results CSV is missing prints
# a skip message and exits 0 (only f2's --demo flag may synthesize data, and
# the Makefile never passes it).
#
# Expected inputs (produced at P9 — schemas documented in each script's docstring):
#   experiments/results/f2_trajectory.csv       one labeled trajectory (t, u_t, cont)
#   experiments/results/f3_pareto.csv           per method×domain×knob frontier points
#   experiments/results/f4_internalization.csv  per iteration×monitor arm
#   experiments/results/f5_label_study.csv      per label_type (E4)
#   experiments/results/f6_hacking.csv          per training step (divergence dashboard)
#   experiments/results/t1_headline.csv … t5_transfer.csv
# Outputs: paper/figures/*.pdf, paper/tables/*.tex

PY      ?= python3
RESULTS ?= experiments/results
FIGS    ?= paper/figures
TABS    ?= paper/tables

.PHONY: all figures tables test

all: figures tables

figures:
	$(PY) analysis/figures/f1_pipeline.py --out $(FIGS)/f1_pipeline.pdf
	$(PY) analysis/figures/f2_shaping_intuition.py --results $(RESULTS)/f2_trajectory.csv --out $(FIGS)/f2_shaping_intuition.pdf
	$(PY) analysis/figures/f3_pareto.py --results $(RESULTS)/f3_pareto.csv --out $(FIGS)/f3_pareto.pdf
	$(PY) analysis/figures/f4_internalization.py --results $(RESULTS)/f4_internalization.csv --out $(FIGS)/f4_internalization.pdf
	$(PY) analysis/figures/f5_label_study.py --results $(RESULTS)/f5_label_study.csv --out $(FIGS)/f5_label_study.pdf
	$(PY) analysis/figures/f6_hacking_diagnostic.py --results $(RESULTS)/f6_hacking.csv --out $(FIGS)/f6_hacking_diagnostic.pdf

tables:
	$(PY) analysis/tables/t1_headline.py --results $(RESULTS)/t1_headline.csv --out $(TABS)/t1_headline.tex
	$(PY) analysis/tables/t2_baselines.py --results $(RESULTS)/t2_baselines.csv --out $(TABS)/t2_baselines.tex
	$(PY) analysis/tables/t3_ablations.py --results $(RESULTS)/t3_ablations.csv --out $(TABS)/t3_ablations.tex
	$(PY) analysis/tables/t4_overhead.py --results $(RESULTS)/t4_overhead.csv --out $(TABS)/t4_overhead.tex
	$(PY) analysis/tables/t5_transfer.py --results $(RESULTS)/t5_transfer.csv --out $(TABS)/t5_transfer.tex

test:
	$(PY) -m pytest tests/ -q
