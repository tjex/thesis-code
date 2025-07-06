#! /usr/bin/env bash

# An example script of how the simdiss.json results can be integrated into
# downstream custom user processes, returns the title of a note and passess it
# to `zk edit`. zk and fzf are required for this to work.

script_dir=$(dirname "$(realpath "$0")")
data_dir="$script_dir/../data"

zk edit $(cat ${data_dir}/simdiss.json | jq -r '[.most_similar[]]? | .[] | "\(.title)\t\(.path)"' | fzf --with-nth=1 --delimiter=$'\t' | cut -f)

