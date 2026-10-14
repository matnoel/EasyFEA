MD = $(shell git ls-files '*.md')

.PHONY: format lint test-mpi

format:
	black .
	ruff check --fix .
	mdformat $(MD)

lint:
	black --check .
	ruff check .
	lint-imports
	mdformat --check $(MD)

MPI_TESTS = tests/Utilities

test-mpi:
	mpirun -n 2 pytest -m mpi -n0 -p no:cacheprovider $(MPI_TESTS)
