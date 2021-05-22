# docs/conf.py
"""Sphinx configuration."""
project = "movieclassifier"
author = "Fallou Tall"
copyright = f"2021, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
html_static_path = ["_static"]
