# StructuralSync — common tasks. `make help` lists them.
.DEFAULT_GOAL := help
PY := .venv/bin/python

help:           ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:          ## create .venv, install pinned deps, download model weights (~13 GB incl. harmoniser)
	bash setup_gpu.sh --download-models

setup-lite:     ## create .venv + deps only (models download on first use)
	bash setup_gpu.sh

dev-deps:       ## add test/lint tools
	uv pip install -p $(PY) -r requirements-dev.txt

run:            ## start backend + frontend on http://127.0.0.1:8000
	./run_web.sh

run-public:     ## same, reachable from the network (set STRUCTSYNC_TOKEN first)
	./run_web.sh --public

test:           ## run tests (no models needed)
	$(PY) -m pytest

lint:           ## ruff
	$(PY) -m ruff check .

docker-up:      ## build + run with Docker (NVIDIA toolkit required)
	docker compose up -d --build

docker-down:    ## stop the Docker container
	docker compose down

.PHONY: help setup setup-lite dev-deps run run-public test lint docker-up docker-down
