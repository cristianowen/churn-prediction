PY ?= python

.PHONY: install train train-fast test api app docker

install:
	$(PY) -m pip install -e ".[api,app,dev]"

train:
	$(PY) -m churn.train

train-fast:
	$(PY) -m churn.train --fast

test:
	$(PY) -m pytest -q

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py

docker: models/churn_model.joblib
	docker build -t churn-api .
	docker run --rm -p 8000:8000 churn-api

models/churn_model.joblib:
	$(PY) -m churn.train
