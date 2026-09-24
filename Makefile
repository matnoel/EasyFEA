MD = $(shell git ls-files '*.md')

.PHONY: format lint

format:
	black .
	ruff check --fix .
	mdformat $(MD)

lint:
	black --check .
	ruff check .
	mdformat --check $(MD)
