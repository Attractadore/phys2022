report.pdf: report.tex stamp.txt
	latexmk -M -MF report.d -MP -pdf -quiet report.tex

stamp.txt: process_data.py
	python3 process_data.py && touch stamp.txt

.PHONY: clean-report
clean-report:
	rm -f report.d report.dvi report.fdb_latexmk report.fls report.log report.toc report.aux report.pdf

.PHONY: clean-process
clean-process:
	rm -f data.pgf stamp.txt

.PHONY: clean
clean: clean-report clean-process

-include report.d
