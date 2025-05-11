FILENAME=main

all:
	latexmk -pdf -bibtex -pdflatex="pdflatex -interaction=nonstopmode -shell-escape" -use-make -f -r .latexmkrc $(FILENAME).tex
	
.PHONY: clean
clean:
	latexmk -CA
