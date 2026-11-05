MD = $(shell git ls-files '*.md')

.PHONY: format lint test-mpi

format:
	ruff check --fix .
	ruff format .
	mdformat $(MD)

lint:
	ruff format --check .
	ruff check .
	lint-imports
	mdformat --check $(MD)

MPI_TESTS = tests/Utilities tests/FEM/mesh_test.py tests/Simulations/incompressible_hyperelastic_test.py

test-mpi:
	mpirun -n 2 pytest -m mpi -n0 -p no:cacheprovider $(MPI_TESTS)
