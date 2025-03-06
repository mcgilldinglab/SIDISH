API Documentation
=================

The SIDISH framework integrates single-cell and bulk RNA-seq data to identify high-risk cells and potential biomarkers. This section provides detailed API documentation for all public functions and classes in the ``SIDISH`` module.

Quick Start
-----------

To use the SIDISH framework, first import the module:

.. code-block:: python

    from SIDISH import SIDISH

SIDISH Class
------------

.. autoclass:: SIDISH.SIDISH
   :members:
   :undoc-members:
   :show-inheritance:

Functions in SIDISH
-------------------

.. autofunction:: SIDISH.SIDISH.init_Phase1

.. autofunction:: SIDISH.SIDISH.init_Phase2

.. autofunction:: SIDISH.SIDISH.train

.. autofunction:: SIDISH.SIDISH.getEmbedding

.. autofunction:: SIDISH.SIDISH.plotUMAP

.. autofunction:: SIDISH.SIDISH.annotateCells

.. autofunction:: SIDISH.SIDISH.reload

.. autofunction:: SIDISH.SIDISH.get_percentille

.. autofunction:: SIDISH.SIDISH.get_embedding

.. autofunction:: SIDISH.SIDISH.set_adata

.. autofunction:: SIDISH.SIDISH.plot_HighRisk_UMAP

.. autofunction:: SIDISH.SIDISH.plot_CellType_UMAP

.. autofunction:: SIDISH.SIDISH.get_MarkerGenes

.. autofunction:: SIDISH.SIDISH.analyze_perturbation_effects

.. autofunction:: SIDISH.SIDISH.run_Perturbation

.. autofunction:: SIDISH.SIDISH.plot_KM
