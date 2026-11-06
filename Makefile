.PHONY: install install-all test lint format run quick gate demo docker

install:            ## core + dev (lightweight: numpy, pandas, matplotlib, rank-bm25)
	pip install -e ".[dev]"

install-all:        ## everything: spaCy NER, dense retrieval, local LLM, API SDKs, demo
	pip install -e ".[all,dev]"
	python -m spacy download en_core_web_sm

test:
	pytest

lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

run:                ## full benchmark (GPU recommended for the local LLM arm)
	contract-assistant run --config configs/default.json --out results

quick:              ## 5 contracts, BM25 only, no LLM: runs on a laptop CPU in seconds
	contract-assistant run --config configs/quick.json --out runs/quick --no-api

gate:               ## fail if committed metrics cross the deployment thresholds
	contract-assistant gate --metrics results/tables/table_metrics.csv --rules configs/gate.json

demo:
	contract-assistant demo --config configs/default.json

docker:
	docker build -t private-contract-assistant .
