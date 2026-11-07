.PHONY: setup test lint demo api dashboard

PY = .venv/bin/python

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -q -r requirements.txt pyflakes==3.2.0

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m pyflakes src scripts dashboard tests

demo:
	$(PY) scripts/demo.py --place downtown

api:
	.venv/bin/uvicorn src.api.main:app --port 8000

dashboard:
	.venv/bin/streamlit run dashboard/app.py
