"""Sphinx configuration for the PyTTV-photodyn documentation.

Autodoc imports the real ``pyttv_photodyn`` package (and its scientific
dependencies), so the package must be installed in the environment that
builds these docs (``pip install -e '.[docs]'``).
"""

from pyttv_photodyn.version import __version__

# -- Project information -----------------------------------------------------

project = "PyTTV-photodyn"
author = "Judith Korth"
copyright = "2026, Judith Korth"

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc / Napoleon ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autoclass_content = "class"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"PyTTV-photodyn {release}"
