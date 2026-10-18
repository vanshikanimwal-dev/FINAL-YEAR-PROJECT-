PYTHON ?= python

.PHONY: data train detect app test all

data:
	$(PYTHON) -m src.generate_data
	$(PYTHON) -m src.preprocess
	$(PYTHON) -m src.features

train:
	$(PYTHON) -m src.train
	$(PYTHON) -m src.evaluate
	$(PYTHON) -m src.explain

detect:
	$(PYTHON) -m src.detect

app:
	$(PYTHON) -m streamlit run app/dashboard.py

test:
	$(PYTHON) -m pytest -q tests

all: data train detect test
