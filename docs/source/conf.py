# Configuration file for the Sphinx documentation builder.
import os
import sys
import sphinx_autodoc_typehints

# Make project importable for autodoc
sys.path.insert(0, os.path.abspath("../../"))

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'SIDISH'
copyright = 'McGill Ding Lab, 2025'
author = 'Yasmin Jolasun'

try:
    from importlib.metadata import version as _pkg_version  # py>=3.8
except ImportError:  # pragma: no cover
    from importlib_metadata import version as _pkg_version  # type: ignore

try:
    release = _pkg_version("SIDISH")
except Exception:
    release = "1.0.0"
version = release


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "nbsphinx",
]

# If you include notebooks, this avoids executing them on RTD (safer, faster).
nbsphinx_execute = "never"

# Mock optional / heavy deps so autodoc can import modules without them.
autodoc_mock_imports = [
    "torch_geometric",
    "torch_geometric.nn",
    "torch_geometric.data",
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
]


intersphinx_mapping = dict(
    python=("https://docs.python.org/3/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    networkx=("https://networkx.org/documentation/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    torch=("https://pytorch.org/docs/stable/", None),
    ignite=("https://pytorch.org/ignite/", None),
    plotly=("https://plotly.com/python-api-reference/", None),
)


qualname_overrides = {
    "anndata._core.anndata.AnnData": "anndata.AnnData",
    "matplotlib.axes._axes.Axes": "matplotlib.axes.Axes",
    "networkx.classes.graph.Graph": "networkx.Graph",
    "networkx.classes.digraph.DiGraph": "networkx.DiGraph",
    "networkx.classes.multigraph.MultiGraph": "networkx.MultiGraph",
    "networkx.classes.multidigraph.MultiDiGraph": "networkx.MultiDiGraph",
    "numpy.random.mtrand.RandomState": "numpy.random.RandomState",
    "pandas.core.frame.DataFrame": "pandas.DataFrame",
    "scipy.sparse.base.spmatrix": "scipy.sparse.spmatrix",
    "seaborn.axisgrid.JointGrid": "seaborn.JointGrid",
    "torch.device": "torch.torch.device",
    "torch.nn.modules.module.Module": "torch.nn.Module",
}

intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_show_sourcelink = True
set_type_checking_flag = True
typehints_fully_qualified = True
napoleon_use_rtype = False
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True
autodoc_default_options = {"autosummary": True}

html_theme = "sphinx_rtd_theme"
epub_show_urls = "footnote"

# -- Ensure Read the Docs can build the project ------------------------------

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    html_static_path = []



