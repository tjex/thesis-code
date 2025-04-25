#! /usr/bin/env bash

script_dir=$(dirname "$(realpath "$0")")
data_dir="$script_dir/../data"

input_title=$(cat "$data_dir/simdiss.json" | jq '.title' | sed 's/"//g')

most_similar=$(cat "$data_dir/simdiss.json" | jq '.most_similar[].title' | sed 's/"//g')
least_similar=$(cat "$data_dir/simdiss.json" | jq '.least_similar[].title' | sed 's/"//g')

echo "Input title: $input_title"
echo ""
echo "--- Most Similar ---"
echo "$most_similar"
echo ""
echo "--- Least Similar ---"
echo "$least_similar"
