DEV_TOOLS := mypy pytest ruff

# A hijacked release is usually found and pulled within days, so update-deps
# takes nothing younger than this.
MIN_AGE_DAYS := 3

# Newest stable CPython uv can download, as major.minor. A pre-release such as
# 3.16.0a2 has letters after the patch number, so the pattern skips it.
NEWEST_PYTHON = uv python list --only-downloads | awk -F'[-.]' '/^cpython-[0-9]+\.[0-9]+\.[0-9]+-/ {print $$2"."$$3}' | sort -u -t. -k1,1n -k2,2n | tail -n1

# Newest uv_build on the index; --refresh stops uv from answering from a stale cache.
NEWEST_UV_BUILD = uv run --no-project --refresh --exclude-newer "$(MIN_AGE_DAYS) days" --with uv-build python -c 'import importlib.metadata as m; print(m.version("uv-build"))'

.PHONY: update-deps
.PHONY: test eval clean

# The eval is part of the test suite, so test runs it too. --locked makes a
# lockfile that no longer matches pyproject.toml fail instead of re-resolving.
test: ## Run the tests and the eval
	uv run --locked pytest

eval: ## Run only the eval: a freshly built wheel over the recorded cases in testdata/eval/
	uv run --locked pytest tests/test_eval.py

# The eval builds its wheel in a pytest temp directory, so the tree holds only tool caches.
clean: ## Remove the pytest, mypy, and ruff caches and every __pycache__ outside .venv
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +

# uv has no age limit for Python downloads, so the Python pin takes the newest
# minor without the wait. The uv steps after it read exclude-newer from
# pyproject.toml, which uv lock records and CI's uv sync --locked checks.
# uv add keeps a dev tool's existing == pin, so each tool is removed and added
# back. No uv command edits [build-system], and perl -pi works the same on
# macOS and Linux where sed -i does not.
# pinact runs as a one-shot so nothing is installed; the token keeps it under the API rate limit.
update-deps: ## Move Python to the newest minor, and the build backend, dev tool pins, lockfile, and action pins to the newest release at least MIN_AGE_DAYS days old
	uv python pin "$$($(NEWEST_PYTHON))"
	perl -pi -e 's/^exclude-newer = .*/exclude-newer = "$(MIN_AGE_DAYS) days"/' pyproject.toml
	version="$$($(NEWEST_UV_BUILD))" && perl -pi -e "s/uv_build==[0-9.]+/uv_build==$$version/" pyproject.toml
	uv remove --dev $(DEV_TOOLS) && uv add --dev --bounds exact $(DEV_TOOLS) && uv lock --upgrade
	GITHUB_TOKEN="$$(gh auth token)" go run github.com/suzuki-shunsuke/pinact/v3/cmd/pinact@latest run --update --min-age $(MIN_AGE_DAYS)
