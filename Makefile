PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev test lint format serve clean

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
	$(PYTHON) -m dni_pipeline.server

clean:
	rm -rf .pytest_cache .ruff_cache dist build
