.PHONY: all install dev qhybrid-build qhybrid-develop test clean

all: dev

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

qhybrid-build:
	cd quantumforge/python && maturin build

qhybrid-develop:
	cd quantumforge/python && maturin develop

test:
	pytest -q

clean:
	rm -rf build/ dist/ *.egg-info/ quantumforge/target/ quantumforge/python/target/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
