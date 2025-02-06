from setuptools import setup, find_packages

setup(
    name="SIDISH",              # Choose a name for your package
    version="1.0.0",            # Version number
    description='SIDISH package 1.0.0',
    author="Yasmin Jolasun",
    author_email="yasmin.jolasun@mail.mcgill.ca",
    url="https://github.com/mcgilldinglab/SIDISH",
    packages=find_packages(),   # Automatically find Python packages
    classifiers=['Programming Language :: Python :: 3.9'],
    install_requires=['numpy==1.26.4',
        'bioinfokit==2.1.3',
        "imbalanced-learn==0.12.4",
        "leidenalg==0.10.2",
        'matplotlib==3.9.2',
        "pandas==2.2.3",
        'scanpy==1.10.4',
        'scikit-learn==1.5.2',
        'scipy==1.14.1',
        'seaborn==0.13.2',
        'shap==0.46.0',
        'statsmodels==0.14.4',
        'anndata==0.11.1',
        'igraph==0.10.15',
        'imblearn==0.0'])
