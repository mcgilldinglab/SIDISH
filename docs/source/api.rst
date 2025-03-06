API Documentation
=================

The SIDISH framework integrates single-cell and bulk RNA-seq data to identify high-risk cells and potential biomarkers. This section provides detailed API documentation for all public functions and classes in the ``SIDISH`` module.

Quick Start
-----------

To use the SIDISH framework, first import the module:

.. code-block:: python

    from SIDISH import SIDISH

Initialise SIDISH
-----------------
.. module:: SIDISH.SIDISH
.. currentmodule:: SIDISH
.. autosummary::
   :toctree: api

   SIDISH


Functions in SIDISH
-------------------
.. module:: SIDISH.SIDISH.SIDISH
.. currentmodule:: SIDISH
.. autosummary::
   :toctree: api

   SIDISH.SIDISH.SIDISH.init_Phase1
   SIDISH.SIDISH.SIDISH.init_Phase2
   SIDISH.SIDISH.SIDISH.train
   SIDISH.SIDISH.SIDISH.getEmbedding
   SIDISH.SIDISH.SIDISH.plotUMAP
   SIDISH.SIDISH.SIDISH.annotateCells
   SIDISH.SIDISH.SIDISH.reload
   SIDISH.SIDISH.SIDISH.get_percentille
   SIDISH.SIDISH.SIDISH.get_embedding
   SIDISH.SIDISH.SIDISH.set_adata
   SIDISH.SIDISH.SIDISH.plot_HighRisk_UMAP
   SIDISH.SIDISH.SIDISH.plot_CellType_UMAP
   SIDISH.SIDISH.SIDISH.get_MarkerGenes
   SIDISH.SIDISH.SIDISH.analyze_perturbation_effects
   SIDISH.SIDISH.SIDISH.run_Perturbation
   SIDISH.SIDISH.SIDISH.plot_KM
