.PHONY: figures paper-en paper-supplement paper-zh checklist test-fast verify

figures:
	python paper/scripts/plot_aaai27_figures.py

paper-en:
	$(MAKE) -C paper en

paper-supplement:
	$(MAKE) -C paper supplement

paper-zh:
	$(MAKE) -C paper zh

checklist:
	$(MAKE) -C paper checklist

test-fast:
	OMP_NUM_THREADS=1 python -m unittest discover -s experiments/opd_sft_h1/tests -p 'test_*.py'

verify:
	python scripts/verify_repository.py
