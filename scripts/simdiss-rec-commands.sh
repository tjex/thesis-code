#!/usr/bin/env bash

# Runs all command functionality (except training).
# Useful to check the entire program is working as expected 
# and for recording the asciienma cast

countdown() {
  local seconds=5
  printf "\e[33mCommand:\e[0m $1\n"
  while [ $seconds -gt 0 ]; do
    printf "\rExecuting in %02d..." $((seconds % 60))
    sleep 1
    ((seconds--))
  done
  printf "\nExecuting...\n"
}


countdown "python main.py sl compare --title 'Zettelkasten is an interface for thought'"
python main.py sl compare --title 'Zettelkasten is an interface for thought'
nvim data/simdiss.json
printf "\n"

countdown "python main.py sl cluster --clusters 15"
python main.py sl cluster --clusters 15
printf "\n"

countdown "python main.py tm list --topics"
python main.py tm list --topics
printf "\n"

countdown "python main.py tm list --docs-for-topic 2"
python main.py tm list --docs-for-topic 2
printf "\n"

countdown "python main.py tm list --related 'Zettelkasten is an interface for thought'"
python main.py tm list --related "Zettelkasten is an interface for thought"
printf "\n"

countdown "python main.py tm search 'creative technologies'"
python main.py tm search "creative technologies"
printf "\n"
