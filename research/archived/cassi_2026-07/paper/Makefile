# CASSI paper build (paper_plan_v2 §16 P11 step 5: "Compile loop: latexmk -pdf").
# `make`       -> main.pdf (compiles TODAY via the article fallback in main.tex;
#                 drop the ICLR 2027 style files into this directory at P11 step 1)
# `make clean` -> remove all build artifacts

LATEXMK ?= latexmk

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.pdf)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
	rm -f sections/*.aux
