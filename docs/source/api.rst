API Documentation
=================

The SIDISH framework integrates single-cell and bulk RNA-seq data for identifying high-risk cells. This section provides detailed API documentation for all public functions and classes in the ``SIDISH`` module.

Quick Start
-----------

To use the SIDISH framework, first import the module:

.. code-block:: python

    from SIDISH import SIDISH

Initial Setup
-------------

.. module:: SIDISH.SIDISH
.. currentmodule:: SIDISH

.. autosummary::
   :toctree: api

   SIDISH.SIDISH.__init__
   SIDISH.SIDISH.init_Phase1
   SIDISH.SIDISH.init_Phase2
   SIDISH.SIDISH.train
   SIDISH.SIDISH.getEmbedding
   SIDISH.SIDISH.plotUMAP
   SIDISH.SIDISH.annotateCells
   SIDISH.SIDISH.reload
   SIDISH.SIDISH.get_percentile
   SIDISH.SIDISH.get_embedding
   SIDISH.SIDISH.set_adata
   SIDISH.SIDISH.plot_HighRisk_UMAP
   SIDISH.SIDISH.plot_CellType_UMAP
   SIDISH.SIDISH.get_MarkerGenes
   SIDISH.SIDISH.analyze_perturbation_effects
   SIDISH.SIDISH.run_Perturbation
   SIDISH.SIDISH.plot_KM


