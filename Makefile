DEV_TOOLS := mypy pytest ruff

# Newest stable CPython uv can download, as major.minor. A pre-release such as
# 3.16.0a2 has letters after the patch number, so the pattern skips it.
NEWEST_PYTHON = uv python list --only-downloads | awk -F'[-.]' '/^cpython-[0-9]+\.[0-9]+\.[0-9]+-/ {print $$2"."$$3}' | sort -u -t. -k1,1n -k2,2n | tail -n1

# Newest uv_build on the index; --refresh stops uv from answering from a stale cache.
NEWEST_UV_BUILD = uv run --no-project --refresh --with uv-build python -c 'import importlib.metadata as m; print(m.version("uv-build"))'

.PHONY: update-deps

# uv add keeps a dev tool's existing == pin, so each tool is removed and added
# back. No uv command edits [build-system], and perl -pi works the same on
# macOS and Linux where sed -i does not.
# pinact runs as a one-shot so nothing is installed; the token keeps it under the API rate limit.
update-deps: ## Move Python, the build backend, the dev tool pins, the lockfile, and action pins to the newest release
	uv python pin "$$($(NEWEST_PYTHON))"
	version="$$($(NEWEST_UV_BUILD))" && perl -pi -e "s/uv_build==[0-9.]+/uv_build==$$version/" pyproject.toml
	uv remove --dev $(DEV_TOOLS) && uv add --dev --bounds exact $(DEV_TOOLS) && uv lock --upgrade
	GITHUB_TOKEN="$$(gh auth token)" go run github.com/suzuki-shunsuke/pinact/v3/cmd/pinact@latest run --update
