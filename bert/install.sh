#!/usr/bin/env bash

# Install script for first time user

pip_packages="nltk"
conda_packages="\
    conda-forge \
    sentence-transformers"

# prepare nltk data folder
mkdir -p "$HOME/.local/share/nltk_data"

echo "set \$NLTK_DATA environment variable"
# export NLTK_DATA="$HOME/.local/share/nltk_data"

# pip install --user -U ${pip_packages}
# conda install -c ${conda_packages}
