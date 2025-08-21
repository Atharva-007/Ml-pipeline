.PHONY: setup train test clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	mkdir -p data/raw data/processed models plots logs

train:
	python app.py --data_path data/raw/data.csv --model_path models/final_model.joblib

test:
	python app.py --test

clean:
	rm -rf __pycache__ */__pycache__
	rm -rf models/* plots/* logs/*

clean-all: clean
	rm -rf venv'
	