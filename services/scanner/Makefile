# Spatial Scanner — common tasks.
PYTHON := .venv/bin/python
UV     := $(shell command -v uv 2>/dev/null)

.PHONY: help setup run test lint scan clean

help:
	@echo "make setup   create .venv and install pinned dependencies"
	@echo "make run     start the API + web viewer on http://127.0.0.1:8765"
	@echo "make test    run the test suite"
	@echo "make lint    run ruff"
	@echo "make scan    process the bundled sample scan from the command line"
	@echo "make clean   remove caches and generated output"

setup:
ifeq ($(UV),)
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt
else
	$(UV) venv -p 3.12 .venv
	$(UV) pip install -p $(PYTHON) -r requirements-dev.txt
endif
	@echo "Done. Start the service with: make run"

run:
	./run.sh

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

scan:
	$(PYTHON) -m engine.pipeline

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache engine/output
