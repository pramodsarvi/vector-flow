# Repository-level shortcuts. Each service also has its own Makefile; these
# just save you the `cd`.
SCANNER := services/scanner
COMPOSE := docker compose -f deploy/docker-compose.yml

.PHONY: help setup run test lint docker-build docker-up docker-down clean

help:
	@echo "Scanner (CPU):"
	@echo "  make setup          create the scanner venv and install dependencies"
	@echo "  make run            start the scanner on http://127.0.0.1:8765"
	@echo "  make test           run the scanner test suite"
	@echo "  make lint           lint the scanner"
	@echo ""
	@echo "Containers:"
	@echo "  make docker-build   build the scanner image"
	@echo "  make docker-up      start the scanner in the background"
	@echo "  make docker-down    stop it"
	@echo ""
	@echo "  make clean          remove caches and generated output"
	@echo ""
	@echo "StructuralSync has its own Makefile: cd services/structsync && make help"

setup:
	$(MAKE) -C $(SCANNER) setup

run:
	$(MAKE) -C $(SCANNER) run

test:
	$(MAKE) -C $(SCANNER) test

lint:
	$(MAKE) -C $(SCANNER) lint

docker-build:
	$(COMPOSE) build scanner

docker-up:
	$(COMPOSE) up -d scanner
	@echo "Scanner -> http://127.0.0.1:8765/viewer/"

docker-down:
	$(COMPOSE) down

clean:
	$(MAKE) -C $(SCANNER) clean
	find . -type d -name __pycache__ -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
