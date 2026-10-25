# Synthia build targets. Every target runs through uv so the same interpreter is used everywhere.
UV ?= uv
PKG = synthia

.PHONY: setup lint test bench serve report docker clean

setup:
	$(UV) sync --all-groups

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test:
	$(UV) run pytest -q

bench:
	$(UV) run python -m $(PKG).bench --check --readme README.md

report:
	$(UV) run $(PKG) report --out reports/patients_copula.html

serve:
	$(UV) run uvicorn $(PKG).api.app:app --host 0.0.0.0 --port 8000

docker:
	docker build -t synthia:latest .

clean:
	rm -rf .pytest_cache .ruff_cache .hypothesis reports/*.html
