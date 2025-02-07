.. SIDISH documentation master file, created by
   sphinx-quickstart on Wed Feb  5 11:50:30 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SIDISH Documentation !
=================================

``SIDISH``  (Semi-supervised Iterative Deep learning for Identifying Single-cell High-Risk populations) is an advanced deep learning framework designed to revolutionize biomarker discovery and therapeutic target identification. By seamlessly integrating the comprehensive breadth of bulk RNA sequencing (bulk RNA-seq) with the granular depth of single-cell RNA sequencing (scRNA-seq), SIDISH empowers researchers to uncover High-Risk disease-associated cell populations, prognostic biomarkers, and novel therapeutic targets—laying the foundation for transformative advancements in precision medicine. For more information, explore our `paper <#>`_.

What is SIDISH?
===============

At its core, SIDISH is a semi-supervised iterative learning framework that bridges molecular data with clinical outcomes. It leverages cutting-edge machine learning techniques to address one of the most pressing challenges in biomedical research: identifying the cellular and genetic drivers of poor patient outcomes.

SIDISH combines several key components to achieve this goal:

- **High-Risk Cell Identification:** Detects cell populations strongly associated with poor clinical outcomes through iterative refinement.
- **Prognostic Biomarker Discovery:** Uncovers key genetic markers linked to disease progression and patient survival.
- **In Silico Therapeutic Simulation:** Simulates gene perturbations to predict the impact of targeting specific genes, aiding in drug discovery and target validation.

SIDISH Framework Overview
==========================

The SIDISH workflow is structured around four key phases:

1. **Extraction of Cellular Heterogeneity:** Utilizing a variational autoencoder (VAE) to capture hidden cellular states from scRNA-seq data, preserving the biological complexity of tumor microenvironments.
2. **Survival Prediction with Transfer Learning:** Applying deep Cox regression models trained on bulk RNA-seq data to predict patient survival risks, effectively transferring learned representations across data modalities.
3. **Risk Stratification:** Identifying High-Risk cells and patient groups based on survival predictions, supported by robust statistical models for accurate risk categorization.
4. **Iterative Learning and Weight Updates:** Incorporating feedback loops using feature attribution methods (e.g., SHAP values) to iteratively refine gene and cell-level predictions.

Additionally, SIDISH features an in silico perturbation module, enabling virtual gene knockouts to simulate therapeutic interventions, predict drug responses, and prioritize targets for experimental validation.

.. image:: ../../SIDISH_9.jpg
   :width: 800
   :alt: Method Overview


SIDISH is run in four phases: 

*a,* Phase 1: Extraction of Cellular Heterogeneity via a Variational Autoencoder (VAE) trained on scRNA-seq data to capture key biological patterns.

*b,* Phase 2: Survival Prediction Using Transfer Learning through a deep Cox regression model optimized for patient survival risk assessment. 

*c,* Phase 3: Risk Prediction and Stratification using survival scores to identify High-Risk cells and patient groups.  

*d,* Phase 4: Iterative Weight Updates leveraging SHAP-based feature attribution to enhance model performance.  

*e,* In Silico Perturbation simulating gene knockouts to predict the impact of potential therapeutic targets.  

*f,* Core Functionalities of SIDISH encompassing biomarker discovery, High-Risk cell identification, and precision medicine applications.

Key Features
=============

- **Iterative Learning Framework:** Continuously improves model accuracy in identifying High-Risk cell populations through a robust semi-supervised iterative approach.
- **Deep Survival Analysis:** Employs deep Cox regression models for high-precision survival risk prediction, directly linking molecular data to patient outcomes.
- **In Silico Gene Perturbation:** Simulates gene knockouts computationally to identify potential therapeutic targets and predict treatment efficacy.
- **Multi-Omics Integration:** Bridges bulk RNA-seq and single-cell RNA-seq data to provide a holistic view of disease mechanisms at both the population and cellular levels.
- **Scalable & Versatile:** Demonstrated effectiveness across multiple cancer types, including pancreatic ductal adenocarcinoma (PDAC), breast cancer (BRCA), and lung adenocarcinoma (LUAD).
- **Precision Medicine-Driven Insights:** SIDISH is uniquely designed to power precision medicine initiatives, offering tools to tailor therapeutic strategies based on individual patient profiles, cellular dynamics, and genetic vulnerabilities.

Why Choose SIDISH?
==================

SIDISH revolutionizes biomarker discovery by unifying detailed single-cell insights with large-scale clinical data. It outperforms traditional approaches through:

- **Enhanced Precision:** Identifies rare High-Risk subpopulations often missed by conventional methods.
- **Clinical Relevance:** Directly links cellular features to patient survival outcomes, facilitating translational research.
- **Therapeutic Discovery:** Predicts druggable targets through computational perturbation analysis, accelerating therapeutic development.
- **Personalized Insights:** Powers precision medicine initiatives by uncovering patient-specific biomarkers for targeted therapies.

SIDISH is designed for researchers, clinicians, and data scientists dedicated to advancing personalized medicine and improving patient outcomes.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Get Started with SIDISH:

   installation
   tutorials
   api.md
   release
   credits
   contact
   

