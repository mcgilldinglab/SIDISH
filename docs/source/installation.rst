Installation
============
This page includes instructions for installing SIDISH.

Prerequisites
-------------

First, install `Anaconda <https://www.anaconda.com/>`_ for your operating system if you have not. You can find specific instructions for different operating systems `here <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_.

Second, create a new conda environment and activate it::

    conda create -n SIDISH python=3.10
    conda activate SIDISH

Finally, install the version of PyTorch compatible with your devices by following the `instructions on the official website <https://pytorch.org/get-started/locally/>`_.

Installing SIDISH
------------------

There are 2 options to install SIDISH.

Option 1: Install from download directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download SIDISH from the `GitHub repository <https://github.com/mcgilldinglab/SIDISH>`_, go to the downloaded SIDISH root directory and use pip tool to install::

    pip install -e .

Option 2: Install from Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    pip install --upgrade https://github.com/mcgilldinglab/SIDISH/zipball/main

The installation should take less than 2 minutes.
The `environment.txt <environment.txt>`_ file includes information about the environment that we used to test SIDISH.
