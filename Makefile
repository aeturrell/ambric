# Builds the documentation for ambric locally.
# In CI, docs are built and deployed by .github/workflows/release.yml.
.PHONY: all site

all: site

# Build the GitHub Pages site with great-docs.
# Output is written to great-docs/_site/ (ephemeral build directory).
site:
	uv pip install -e .
	uv pip install great-docs
	uv run great-docs build
	uv run nbstripout docs/*.ipynb
	uv run pre-commit run --all-files
