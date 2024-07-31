#!/bin/bash

EXCLUDE="venv"

echo "Running Black..."
black . --line-length 78 --exclude "$EXCLUDE"

echo "Running isort..."
isort . --skip "$EXCLUDE"

echo "Running flake8..."
flake8 . --exclude "$EXCLUDE"

echo "Running mypy..."
mypy . --exclude "$EXCLUDE" --disable-error-code import-untyped --disallow-untyped-defs --disallow-incomplete-defs --warn-unreachable --warn-unused-ignores --strict

echo "Linting and formatting complete!"