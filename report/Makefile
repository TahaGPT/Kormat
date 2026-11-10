# Build the report. Requires a TeX distribution with IEEEtran (e.g. TeX Live).
# No local TeX? Upload report/ (+ ../assets/logo.png if used) to Overleaf and set main.tex as the main file.
all:
	latexmk -pdf -interaction=nonstopmode main.tex
clean:
	latexmk -C
.PHONY: all clean
