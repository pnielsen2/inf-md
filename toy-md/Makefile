PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: setup run smoke help

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

smoke:
	$(PY) run_toy_md.py --samplers pc_ula --K 10 --kl-runs 200 --sweep-runs 200 --refresh-L 2 --points-per-seed 10

run:
	$(PY) run_toy_md.py --samplers pc_ula,pc_mala

help:
	@echo "make setup   - create venv and install deps"
	@echo "make smoke   - quick test producing a few outputs"
	@echo "make run     - full default run (ULA+MALA)"
