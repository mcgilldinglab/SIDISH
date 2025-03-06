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
.. autosummary::
   :toctree: api

   SIDISH.init_Phase1
   SIDISH.init_Phase2
   SIDISH.train
   SIDISH.getEmbedding
   SIDISH.plotUMAP
   SIDISH.annotateCells
   SIDISH.reload
   SIDISH.get_percentille
   SIDISH.get_embedding
   SIDISH.set_adata
   SIDISH.plot_HighRisk_UMAP
   SIDISH.plot_CellType_UMAP
   SIDISH.get_MarkerGenes
   SIDISH.analyze_perturbation_effects
   SIDISH.run_Perturbation
   SIDISH.plot_KM
