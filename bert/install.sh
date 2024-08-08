#!/usr/bin/env bash

# Install script for first time user

pip_packages="nltk"

conda_packages="\
    conda-forge \
    sentence-transformers"

conda install -c ${conda_packages}
pip install --user -U ${pip_packages}
