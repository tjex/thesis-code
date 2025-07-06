#!/usr/bin/env bash

# Return word count data for a Zettelksaten repository.

shopt -s globstar # Enable recursive globbing with **
max_count=0
cumulative_count=0
max_file=""
file_count=0
zk_dir=$1

for file in $zk_dir/**; do # Remove quotes so glob expands
    if [ -f "$file" ]; then
        count=$(wc -w <"$file")
        cumulative_count=$((cumulative_count + count))
        file_count=$((file_count + 1))
        if [ "$count" -gt "$max_count" ]; then
            max_count=$count
            max_file=$file
        fi
    fi
done

if [ "$file_count" -gt 0 ]; then
    average_wc=$((cumulative_count / file_count))
else
    average_wc=0
fi

echo "File with highest word count: $max_file"
echo "Max word count: $max_count"
echo "Average word count: $average_wc"
