# **SIDISH**  
**SIDISH Identifies High-Risk Disease-Associated Cells and Biomarkers by Integrating Single-Cell Depth and Bulk Breadth**

## Table of Contents
- [Key Capabilities](#key-capabilities)
- [Methods Overview](#methods-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Contact](#contact)

## Key Capabilities
- **Multi-Scale Data Integration** – Combines **single-cell** and **bulk** RNA-seq data to enhance disease and biomarker insights.  
- **High-Risk Cell Identification** – Detects disease-associated cell populations linked to poor survival outcomes.  
- **Biomarker Discovery** – Utilizes iterative deep learning and SHAP-based feature selection to identify clinically significant genes.  
- **In-Silico Perturbation** – Simulates **gene knockouts** to prioritize **therapeutic targets** and assess their potential impact on high-risk cell populations.  
- **Precision Medicine Applications** – Enables patient stratification and therapeutic prioritization for diseases such as Pancreatic Ductal Adenocarcinoma (PDAC), and triple-negative Breast Cancer (TNBC).  
- **Scalable & Generalizable** – Adapts to large datasets and diverse disease types, ensuring robust and clinically meaningful analyses.  


## Methods Overview
![SIDISH Overview](SIDISH_9.jpg)

Understanding disease mechanisms at both cellular and clinical levels remains a major challenge in biomedical research. Single-cell RNA sequencing (scRNA-seq) provides high-resolution insights into cellular heterogeneity but is costly and lacks large-scale clinical context. Conversely, bulk RNA sequencing (bulk RNA-seq) enables large-cohort studies but obscures critical cellular-level variations by averaging gene expression across thousands of cells.  

**SIDISH (Semi-supervised Iterative Deep Learning for Identifying Single-cell High-risk Populations)** overcomes these limitations by integrating scRNA-seq and bulk RNA-seq through an advanced deep learning framework.  By iteratively refining high-risk cell predictions using Variational Autoencoders (VAE), Deep Cox Regression, and SHAP-based feature selection, SIDISH uncovers cellular subpopulations linked to poor survival while enabling robust patient-level risk assessment. In addition to identifying high-risk cells, SIDISH employs in silico perturbation to simulate gene knockouts, ranking potential therapeutic target based on their impact on disease progression. This dual ability—disease risk assessment and therapeutic prioritization—positions SIDISH as a transformative tool in precision medicine, biomarker discovery, and drug development.  

Explore comprehensive details, including API references, usage examples, and tutorials (in [Jupyter notebook](https://jupyter.org/) format), in our [full documentation](https://sidish.readthedocs.io/en/latest/api.html) and the README below.


## Prerequisites
First, install [Anaconda](https://www.anaconda.com/). You can find specific instructions for different operating systems [here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Second, create a new conda environment and activate it:
```
conda create -n sidish python=3.9
```
```
conda activate sidish
```
Finally, install the version of PyTorch compatible with your devices by following the [instructions on the official website](https://pytorch.org/get-started/locally/). 

## Installation

 There are 2 steps to install SIDISH.  
* __Step 1: Install Dependencies__   
Before installing SIDISH, ensure all required dependencies are installed. Run the following command in your terminal:
    ```shell
    pip install -r requirements.txt
    ```

* __Step 2: Install SIDISH from PyPI__ 
For the latest stable release, install SIDISH directly from PyPI:
    ```shell
    pip install SIDISH==1.0.0
    ```

## Tutorials:
To download the Lung Adenocarcinoma single-cell data as well as the bulk and paired survival data used in the tutorial, follow this [link](https://drive.google.com/file/d/1myrifg9f4fvFgunwpDzkPhlZ9AZUxLuX/view?usp=sharing).

### Preprocessing single-cell and bulk data
[Tutorial 0: Preprocess LUAD single-cell data and paired bulk RNA-seq and survival data, for use to train SIDISH](https://github.com/mcgilldinglab/SIDISH/blob/main/tutorials/tutorial_0_data_preprocessing.ipynb)


### Running SIDISH on lung cancer dataset and visualization of results
[Tutorial 1: Train SIDISH using lung cancer dataset and saving the results](https://github.com/mcgilldinglab/SIDISH/blob/main/tutorials/tutorial_1_initializing_and_training_SIDISH.ipynb)

### Running SIDISH's In silico perturbation feature lung cancer dataset and visualization of results
[Use SIDISH's in silico perturbation feature using lung cancer dataset and visualize your results](https://github.com/mcgilldinglab/SIDISH/blob/main/TUTORIAL/tutorial%203.ipynb)

### Running SIDISH on individual patients in Breast Cancer single-cell dataset and visualization of results
[Analyse patient-level High-Risk cells using SIDISH breast cancer single-cell dataset and visualize your results](https://github.com/mcgilldinglab/SIDISH/blob/main/TUTORIAL/tutorial%202.ipynb)

If you find the tool is useful to your study, please consider citing the SIDISH [manuscript](https://www.researchsquare.com/article/rs-5921999/v1).

## Contact
[Yasmin Jolasun](mailto:yasmin.jolasun@mail.mcgill.ca) and [Jun Ding](mailto:jun.ding@mcgill.ca)
