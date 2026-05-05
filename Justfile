set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    just --list

run:
    uv run python main.py

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

typecheck:
    uv run mypy .

test:
    uv run pytest

check: lint typecheck

hooks-install:
    uv run pre-commit install

hooks:
    uv run pre-commit run --all-files
