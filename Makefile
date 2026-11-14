.PHONY: setup test lint demo api dashboard viewer viewer-data

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

# 3D viewer: re-export the pipeline outputs, then serve docs/viewer on http://localhost:8765
viewer-data:
	$(PY) scripts/export_viewer.py

viewer: viewer-data
	cd docs/viewer && ../../$(PY) -m http.server 8765
