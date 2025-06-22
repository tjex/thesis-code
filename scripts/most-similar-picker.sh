#! /usr/bin/env bash

script_dir=$(dirname "$(realpath "$0")")
data_dir="$script_dir/../data"

zk edit $(cat ${data_dir}/simdiss.json | jq -r '[.most_similar[]]? | .[] | "\(.title)\t\(.path)"' | fzf --with-nth=1 --delimiter=$'\t' | cut -f)

