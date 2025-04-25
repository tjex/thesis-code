#! /usr/bin/env bash

script_dir=$(dirname "$(realpath "$0")")
data_dir="$script_dir/../data"

cat "$data_dir/simdiss.json" | jq '.most_similar[].title'
