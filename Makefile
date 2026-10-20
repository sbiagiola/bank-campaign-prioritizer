.PHONY: setup test train app clean

setup:
	uv sync --locked

test:
	PYTHONPATH=. uv run pytest -v

train:
	PYTHONPATH=. uv run python -m src.pipeline

app:
	PYTHONPATH=. uv run streamlit run app/main.py

clean:
	rm -rf .pytest_cache artifacts/plots/*.png
