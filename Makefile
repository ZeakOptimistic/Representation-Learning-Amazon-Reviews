.PHONY: setup test lint notebook

setup:
	python -m pip install -U pip
	pip install -r requirements.txt
	pre-commit install

test:
	pytest -q

lint:
	pre-commit run --all-files

notebook:
	jupyter notebook
