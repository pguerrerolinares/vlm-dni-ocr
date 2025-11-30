PYTHON ?= python
PIP ?= $(PYTHON) -m pip
SERVER_HOST ?= 0.0.0.0
SERVER_PORT ?= 8000
SERVER_ARGS ?=
TUNNEL_SERVER_HOST ?= 0.0.0.0
TUNNEL_FORWARD_HOST ?= 127.0.0.1

.PHONY: install install-dev test lint format serve serve-tunnel clean

install:
	$(PIP) install -e .

install-dev: install
	$(PIP) install -e .[dev,api,ui]

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check --fix src tests

serve:
	$(PYTHON) -m dni_pipeline.server --host $(SERVER_HOST) --port $(SERVER_PORT) $(SERVER_ARGS)

serve-tunnel:
	SERVER_HOST=$(TUNNEL_SERVER_HOST) NGROK_FORWARD_HOST=$(TUNNEL_FORWARD_HOST) SERVER_PORT=$(SERVER_PORT) PYTHON=$(PYTHON) bash scripts/serve_with_ngrok.sh $(SERVER_ARGS)

clean:
	rm -rf .pytest_cache .ruff_cache dist build
