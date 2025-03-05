API Documentation
=================

The SIDISH framework integrates single-cell and bulk RNA-seq data for identifying high-risk cells. This section provides detailed API documentation for all public functions and classes in the ``SIDISH`` module.

Quick Start
-----------

To use the SIDISH framework, first import the module:

.. code-block:: python

    from SIDISH.SIDISH import SIDISH

Initial Setup
-------------

.. autosummary::
   :toctree: api
   :recursive:

   # First, list the class itself
   SIDISH.SIDISH.SIDISH

   # Then list methods inside that class
   SIDISH.SIDISH.SIDISH.init_Phase1
   SIDISH.SIDISH.SIDISH.init_Phase2
   SIDISH.SIDISH.SIDISH.train
   SIDISH.SIDISH.SIDISH.getEmbedding
   SIDISH.SIDISH.SIDISH.plotUMAP
   SIDISH.SIDISH.SIDISH.annotateCells
   SIDISH.SIDISH.SIDISH.reload
   SIDISH.SIDISH.SIDISH.get_percentile
   SIDISH.SIDISH.SIDISH.get_embedding
   SIDISH.SIDISH.SIDISH.set_adata
   SIDISH.SIDISH.SIDISH.plot_HighRisk_UMAP
   SIDISH.SIDISH.SIDISH.plot_CellType_UMAP
   SIDISH.SIDISH.SIDISH.get_MarkerGenes
   SIDISH.SIDISH.SIDISH.analyze_perturbation_effects
   SIDISH.SIDISH.SIDISH.run_Perturbation
   SIDISH.SIDISH.SIDISH.plot_KM

----------------------------

.. automodule:: SIDISH.SIDISH
   :members:
   :undoc-members:
   :show-inheritance:
