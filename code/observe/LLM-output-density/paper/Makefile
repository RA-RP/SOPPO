.PHONY: en supplement zh checklist clean

PDFLATEX ?= pdflatex
BIBTEX ?= bibtex

en:
	cd en && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main_en.tex
	cd en && $(BIBTEX) main_en
	cd en && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main_en.tex
	cd en && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main_en.tex

supplement:
	cd en && $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplement_en.tex
	cd en && $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplement_en.tex

zh:
	cd zh && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main_zh.tex
	cd zh && $(BIBTEX) main_zh
	cd zh && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main_zh.tex
	cd zh && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main_zh.tex

checklist:
	cd checklist && $(PDFLATEX) -interaction=nonstopmode -halt-on-error ReproducibilityChecklist.tex
	cd checklist && $(PDFLATEX) -interaction=nonstopmode -halt-on-error ReproducibilityChecklist.tex

clean:
	$(RM) en/*.aux en/*.bbl en/*.blg en/*.log en/*.out en/*.pdf
	$(RM) zh/*.aux zh/*.bbl zh/*.blg zh/*.log zh/*.out zh/*.pdf
	$(RM) checklist/*.aux checklist/*.log checklist/*.out checklist/*.pdf
