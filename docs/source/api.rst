SIDISH API Documentation
========================

This section provides detailed API documentation for all public functions and classes in ``SIDISH``.

Importing SIDISH
================

To use the SIDISH framework, first import the `SIDISH` module:

.. code-block:: python

    from SIDISH import SIDISH


.. automodule:: SIDISH
    :members:
    :undoc-members:
    :show-inheritance:

Workflow Overview
-----------------
1. **Initialize the SIDISH framework**.
2. **Prepare and configure Phase 1**: Train the Variational Autoencoder (VAE) on single-cell RNA-seq data.
3. **Prepare and configure Phase 2**: Train the Deep Cox model on bulk RNA-seq data using transfer learning from the VAE encoder.
4. **Train SIDISH iteratively** to refine high-risk cell identification.
5. **Extract the final iteration embedding** for downstream analysis.
6. **Visualize high-risk cell subpopulations** using UMAP.
7. **Visualize and analyze identified cell types**.
8. **Run gene perturbation simulations** to understand the biological significance of high-risk cell populations.

---

Core API
--------

### **1. Initialize SIDISH**
**Before running any model, SIDISH must be initialized with the single-cell (scRNA-seq) and bulk RNA-seq datasets.**
.. autoclass:: SIDISH.SIDISH
    :members:
    :undoc-members:
    :show-inheritance:

### **2. Initialize Phase 1: Train Variational Autoencoder (VAE)**
**Phase 1 trains the VAE on scRNA-seq data, creating a meaningful latent space.**
.. automethod:: SIDISH.SIDISH.init_Phase1

### **3. Initialize Phase 2: Train Deep Cox Model with Transfer Learning**
**Phase 2 trains the Deep Cox model using bulk RNA-seq survival data while leveraging the VAE encoder.**
.. automethod:: SIDISH.SIDISH.init_Phase2

### **4. Train SIDISH (Iterative Learning)**
**The `train` function executes the full SIDISH training loop, iteratively refining high-risk cell identification.**
.. automethod:: SIDISH.SIDISH.train

---

### **Post-Training Analysis**
After training, several core functionalities can be used to **interpret results**.

### **5. Extract Final Embedding**
**Retrieve the final latent representation of the cells after iterative learning.**
.. automethod:: SIDISH.SIDISH.getEmbedding

### **6. Visualize SIDISH-Identified High-Risk Cells**
**Plot UMAP projections highlighting the cells classified as high-risk by SIDISH.**
.. automethod:: SIDISH.SIDISH.plot_HighRisk_UMAP

### **7. Visualize Cell Type Distributions**
**Plot UMAP projections to visualize the clustering of different cell types in the dataset.**
.. automethod:: SIDISH.SIDISH.plot_CellType_UMAP

### **8. Run Gene Perturbation Simulations**
**Perform in-silico gene perturbations to assess how knocking out specific genes impacts the high-risk population.**
.. automethod:: SIDISH.SIDISH.run_Perturbation

### **9. Analyze Perturbation Effects**
**Statistically evaluate the impact of perturbations on high-risk cell populations.**
.. automethod:: SIDISH.SIDISH.analyze_perturbation_effects
