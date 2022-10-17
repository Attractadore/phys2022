report.pdf: report.tex stamp.txt
	latexmk -M -MF report.d -MP -pdf -quiet report.tex

stamp.txt: process_data.py balmer.txt neon.txt mercury.txt
	python3 process_data.py && touch stamp.txt

.PHONY: clean-report
clean-report:
	rm -f report.d report.dvi report.fdb_latexmk report.fls report.log report.toc report.aux report.pdf

.PHONY: clean-process
clean-process:
	rm -f grad.pgf R.tex balmer.tab.tex energies.tab.tex iodine.tab.tex stamp.txt neon.tab.tex mercury.tab.tex

.PHONY: clean
clean: clean-report clean-process

-include report.d
