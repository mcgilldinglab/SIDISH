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
   SIDISH.SIDISH.SIDISH.init_Phase1
   SIDISH.SIDISH.SIDISH.init_Phase2


Train SIDISH Model
-------------------
.. module:: SIDISH.SIDISH.SIDISH
.. currentmodule:: SIDISH
.. autosummary::
   :toctree: api

   SIDISH.SIDISH.SIDISH.train


Reload Trained SIDISH Model
---------------------------
.. module:: SIDISH.SIDISH.SIDISH
.. currentmodule:: SIDISH
.. autosummary::
    SIDISH.SIDISH.SIDISH.reload



Reload Trained SIDISH Model
---------------------------
.. module:: SIDISH.SIDISH.SIDISH
.. currentmodule:: SIDISH
.. autosummary::
    SIDISH.SIDISH.SIDISH.reload


Plotting Functions
------------------
.. module:: SIDISH.SIDISH.SIDISH
.. currentmodule:: SIDISH
.. autosummary::
    SIDISH.SIDISH.SIDISH.plotUMAP
    SIDISH.SIDISH.SIDISH.plot_KM
    SIDISH.SIDISH.SIDISH.plot_HighRisk_UMAP
    SIDISH.SIDISH.SIDISH.plot_CellType_UMAP



Perturbation 
------------
.. module:: SIDISH.SIDISH.SIDISH
.. currentmodule:: SIDISH
.. autosummary::
    SIDISH.SIDISH.SIDISH.run_Perturbation
    SIDISH.SIDISH.SIDISH.analyze_perturbation_effects
    
